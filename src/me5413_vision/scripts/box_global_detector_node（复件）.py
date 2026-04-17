#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
box_global_detector_node.py

架构：
  已有地图 + AMCL 定位（发布 map→odom→base_link TF）
            ↓
  ┌────────────────────────┐     ┌─────────────────────────┐
  │  LiDAR 点云 /mid/points│     │  相机 /front/image_raw  │
  │  每帧做：              │     │  每帧做：               │
  │  1. TF变换到map帧      │     │  1. 轮廓检测候选区域    │
  │  2. 高度滤波           │     │  2. 模板匹配主干识别    │
  │  3. 体素下采样         │     │  3. CNN专职复核(8转6)   │
  │  4. 2D聚类             │     └─────────────────────────┘
  │  5. 尺寸筛选           │               ↑
  │  → 候选箱子全局坐标列表│               │
  └────────────────────────┘               │
              ↓                            │
         投影匹配：把3D质心投影到图像坐标系 ─┘
              ↓
       (全局坐标 x,y) + (识别数字 number)
              ↓
        语义地图去重 + 加权融合
              ↓
  发布 /vision/box_map       (std_msgs/String, JSON, latch=True)
  发布 /vision/box_markers   (visualization_msgs/MarkerArray, RViz可视化)
"""

import json
import math
import os
import threading

import cv2
import numpy as np
import rospy
import rospkg
import tf2_ros
import tf2_geometry_msgs

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError

# 引入 ONNX Runtime 进行轻量化推理
import onnxruntime as ort

# ==============================================================================
# 2D Euclidean 聚类（纯 numpy，无需 scipy）
# ==============================================================================
def euclidean_clustering(pts_2d, radius=0.25, min_pts=8):
    n = len(pts_2d)
    if n < min_pts:
        return []

    diff = pts_2d[:, np.newaxis, :] - pts_2d[np.newaxis, :, :]  
    dist_sq = (diff * diff).sum(axis=2)                          
    adj = dist_sq < (radius * radius)                            

    visited = np.zeros(n, dtype=bool)
    clusters = []

    for seed in range(n):
        if visited[seed]:
            continue

        queue = [seed]
        visited[seed] = True
        members = []

        while queue:
            cur = queue.pop()
            members.append(cur)
            neighbors = np.where(adj[cur] & ~visited)[0]
            visited[neighbors] = True
            queue.extend(neighbors.tolist())

        if len(members) >= min_pts:
            clusters.append(np.array(members, dtype=np.int32))

    return clusters

# ==============================================================================
# 模板匹配数字识别器
# ==============================================================================
class TemplateDigitMatcher:
    MATCH_SIZE = (64, 64)  

    def __init__(self, number_dir):
        self.templates = {}   
        for i in range(1, 10):
            path = os.path.join(number_dir, f'{i}.jpg')
            if not os.path.exists(path):
                rospy.logwarn(f"[Matcher] 模板不存在: {path}")
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.MATCH_SIZE)
            img = self._clahe(img)
            self.templates[i] = img.astype(np.float32)

        rospy.loginfo(f"[Matcher] 加载模板: {sorted(self.templates.keys())}")

    @staticmethod
    def _clahe(gray):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        return clahe.apply(gray)

    def match(self, roi_bgr):
        if roi_bgr is None or roi_bgr.size == 0:
            return None, -1.0

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self.MATCH_SIZE)
        gray = self._clahe(gray)
        roi_f = gray.astype(np.float32)

        best_digit = None
        best_score = -1.0

        for digit, tmpl in self.templates.items():
            result = cv2.matchTemplate(roi_f, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(result[0, 0])
            if score > best_score:
                best_score = score
                best_digit = digit

        return best_digit, best_score

# ==============================================================================
# 语义地图：去重 + 加权平均融合
# ==============================================================================
class SemanticMapManager:
    def __init__(self, merge_radius=1.2):
        self.boxes = []            
        self.merge_radius = merge_radius
        self._lock = threading.Lock()

    def update(self, number, new_x, new_y):
        with self._lock:
            for box in self.boxes:
                if math.hypot(box['x'] - new_x, box['y'] - new_y) < self.merge_radius:
                    if box['number'] == number:
                        n = box['count']
                        box['x'] = (box['x'] * n + new_x) / (n + 1)
                        box['y'] = (box['y'] * n + new_y) / (n + 1)
                        box['count'] += 1
                    return False, len(self.boxes)

            self.boxes.append({'number': number, 'x': new_x, 'y': new_y, 'count': 1})
            return True, len(self.boxes)

    def to_json(self):
        with self._lock:
            return json.dumps([
                {'number': b['number'],
                 'x': round(b['x'], 3),
                 'y': round(b['y'], 3),
                 'count': b['count']}
                for b in self.boxes
            ])

    def summary(self):
        with self._lock:
            rospy.loginfo("=" * 55)
            rospy.loginfo(f"[BoxMap] 共确认 {len(self.boxes)} 个箱子:")
            for b in self.boxes:
                rospy.loginfo(
                    f"  #{b['number']:2d}  X={b['x']:.3f}  Y={b['y']:.3f}"
                    f"  (融合 {b['count']} 次观测)"
                )
            rospy.loginfo("=" * 55)

# ==============================================================================
# 主检测节点
# ==============================================================================
class BoxGlobalDetector:

    FX, FY = 381.36, 381.36
    CX, CY = 320.0, 240.0
    IMG_W, IMG_H = 640, 480

    LIDAR_TOPIC = '/mid/points'         
    PC_Z_MIN    = 0.10                   
    PC_Z_MAX    = 1.30                   
    VOXEL_SIZE  = 0.05                   
    CLUSTER_R   = 0.25                   
    CLUSTER_MINPTS = 8                   
    BOX_XY_MIN  = 0.35                   
    BOX_XY_MAX  = 1.60                   

    PROJ_PIX_THRESH = 40                 
    CNN_CONF_THRESH = 0.25               

    VOTE_MERGE_RADIUS = 1.2              
    CONFIRM_VOTES    = 3                 
    CONFIRM_RATIO    = 0.6               

    PUBLISH_HZ = 2.0

    def __init__(self):
        rospy.init_node('box_global_detector_node', anonymous=False)

        self.bridge = CvBridge()
        self.map_manager = SemanticMapManager(merge_radius=1.2)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('me5413_vision')
        number_dir = os.path.join(pkg_path, 'number')
        if not os.path.isdir(number_dir):
            rospy.logfatal(f"[BoxDetector] 模板目录不存在: {number_dir}")
            raise FileNotFoundError(number_dir)
        self.matcher = TemplateDigitMatcher(number_dir)
        rospy.loginfo("[BoxDetector] 模板匹配器初始化完成")

        # --- 初始化 ONNX 神经网络 (仅用于 8转6 校验) ---
        self.ONNX_PATH = "/home/marmot/zhouxin/ME5413_Final_Project/src/me5413_vision/weights/cnn.onnx"
        if os.path.exists(self.ONNX_PATH):
            self.ort_session = ort.InferenceSession(self.ONNX_PATH)
            self.ort_input_name = self.ort_session.get_inputs()[0].name
            rospy.loginfo("✅ ONNX 神经网络加载成功 (专职复核 8)")
        else:
            self.ort_session = None
            rospy.logwarn("❌ 找不到 ONNX 模型，复核功能关闭")
        # -----------------------------------------------

        self._cam_detections = []
        self._cam_lock = threading.Lock()

        self._candidates = []
        self._cand_lock = threading.Lock()

        self.box_map_pub = rospy.Publisher('/vision/box_map', String, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher('/vision/box_markers', MarkerArray, queue_size=1, latch=True)
        self.candidate_pub = rospy.Publisher('/vision/box_candidates', MarkerArray, queue_size=1, latch=True)

        rospy.Subscriber(self.LIDAR_TOPIC, PointCloud2, self._lidar_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/front/image_raw', Image, self._camera_cb, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.PUBLISH_HZ), self._publish_timer_cb)

        rospy.on_shutdown(self._on_shutdown)
        rospy.loginfo(
            f"[BoxDetector] 节点启动\n"
            f"  LiDAR: {self.LIDAR_TOPIC}\n"
            f"  Camera: /front/image_raw\n"
            f"  发布: /vision/box_map  /vision/box_markers"
        )

    # ==========================================================================
    # LiDAR 回调
    # ==========================================================================
    def _lidar_cb(self, cloud_msg):
        try:
            pts = np.array(
                [[p[0], p[1], p[2]] for p in pc2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True)],
                dtype=np.float64
            )
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[BoxDetector] 点云读取失败: {e}")
            return

        if len(pts) < 50:
            return

        try:
            tf_s = self.tf_buffer.lookup_transform('map', cloud_msg.header.frame_id, rospy.Time(0), rospy.Duration(0.2))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(3.0, f"[BoxDetector] TF 失败: {e}")
            return

        pts_map = self._transform_points(pts, tf_s)
        mask = (pts_map[:, 2] > self.PC_Z_MIN) & (pts_map[:, 2] < self.PC_Z_MAX)
        pts_filt = pts_map[mask]
        if len(pts_filt) < self.CLUSTER_MINPTS:
            return

        pts_ds = self._voxel_downsample(pts_filt, self.VOXEL_SIZE)
        clusters = euclidean_clustering(pts_ds[:, :2], radius=self.CLUSTER_R, min_pts=self.CLUSTER_MINPTS)
        if not clusters:
            return

        with self._cam_lock:
            cam_dets = list(self._cam_detections)

        for idx_arr in clusters:
            cluster = pts_ds[idx_arr]
            x_span = cluster[:, 0].max() - cluster[:, 0].min()
            y_span = cluster[:, 1].max() - cluster[:, 1].min()

            if not (self.BOX_XY_MIN < x_span < self.BOX_XY_MAX and self.BOX_XY_MIN < y_span < self.BOX_XY_MAX):
                continue

            cx = float(cluster[:, 0].mean())
            cy = float(cluster[:, 1].mean())
            cz = float(cluster[:, 2].mean())

            number = self._match_number_by_projection(cx, cy, cz, cam_dets)
            
            if number is not None:
                self._vote_and_confirm(cx, cy, number)
            else:
                is_near_known = False
                with self.map_manager._lock:
                    for b in self.map_manager.boxes:
                        if math.hypot(b['x'] - cx, b['y'] - cy) < 1.0:
                            is_near_known = True
                            break
                
                if not is_near_known:
                    self._add_and_publish_candidate(cx, cy, cz)

    def _add_and_publish_candidate(self, x, y, z):
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = rospy.Time.now()
        m.ns = 'candidates'
        m.id = int((x + 50) * 100 + (y + 50)) 
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.scale.x = m.scale.y = m.scale.z = 0.4
        m.color.a = 0.6
        m.color.r = 1.0 
        m.color.g = 1.0
        m.color.b = 0.0
        m.lifetime = rospy.Duration(2.0) 
        ma.markers.append(m)
        self.candidate_pub.publish(ma)

    def _vote_and_confirm(self, x, y, number):
        with self._cand_lock:
            target = None
            for cand in self._candidates:
                if math.hypot(cand['x'] - x, cand['y'] - y) < self.VOTE_MERGE_RADIUS:
                    target = cand
                    break

            if target is None:
                target = {'x': x, 'y': y, 'votes': {number: 1}, 'total': 1}
                self._candidates.append(target)
                rospy.logdebug(f"[Vote] 新候选 #{number} 位置=({x:.2f},{y:.2f})  票数: 1/{self.CONFIRM_VOTES}")
            else:
                n = target['total']
                target['x'] = (target['x'] * n + x) / (n + 1)
                target['y'] = (target['y'] * n + y) / (n + 1)
                target['total'] += 1
                target['votes'][number] = target['votes'].get(number, 0) + 1
                rospy.logdebug(f"[Vote] 候选 #{number} 位置=({target['x']:.2f},{target['y']:.2f})  票数: {target['votes']}")

            best_num = max(target['votes'], key=target['votes'].get)
            best_cnt = target['votes'][best_num]
            ratio    = best_cnt / target['total']

            if best_cnt >= self.CONFIRM_VOTES and ratio >= self.CONFIRM_RATIO:
                cx, cy = target['x'], target['y']
                is_new, total = self.map_manager.update(best_num, cx, cy)

                if is_new:
                    rospy.loginfo(
                        f"[BoxDetector] ✔ 确认新箱子! "
                        f"#{best_num}  全局坐标=({cx:.2f}, {cy:.2f})"
                        f"  投票: {target['votes']}  共确认 {total} 个箱子"
                    )
                    self._do_publish()
                else:
                    rospy.loginfo(
                        f"[BoxDetector] ✔ 再次确认 #{best_num}"
                        f" 位置=({cx:.2f}, {cy:.2f})  投票: {target['votes']}"
                    )

                target['votes'] = {}
                target['total'] = 0

    # ==========================================================================
    # 相机回调与检测
    # ==========================================================================
    def _camera_cb(self, rgb_msg):
        try:
            img = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        except CvBridgeError:
            return

        dets = self._detect_digits(img)
        with self._cam_lock:
            self._cam_detections = dets

    def _detect_digits(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if not (20 < w < 250 and 30 < h < 250 and 0.4 < w / h < 1.8):
                continue

            u_c = x + w // 2
            v_c = y + h // 2

            pw, ph = int(w * 0.4), int(h * 0.4)
            roi = img[max(0, y - ph):min(self.IMG_H, y + h + ph),
                      max(0, x - pw):min(self.IMG_W, x + w + pw)]
            if roi.size == 0:
                continue

            # 1. 模板匹配主干预测
            digit, score = self.matcher.match(roi)
            rospy.logdebug_throttle(0.5, f"[Match] 预测=#{digit}  NCC={score:.3f}")
            
            # =======================================================
            # 🚀 专项校验逻辑：只有模板匹配认为是 8 时，才劳驾 CNN 出马
            # =======================================================
            if digit == 8 and self.ort_session is not None and score >= self.CNN_CONF_THRESH:
                cnn_digit, cnn_conf = self._cnn_infer_onnx(roi)
                if cnn_digit == 6:
                    rospy.loginfo_throttle(1.0, f"🔍 专职复核：模板误认为 8，神经网络确认为 6 (置信度:{cnn_conf:.2f})，已纠正为 6！")
                    digit = 6
            # =======================================================

            if digit is not None and score >= self.CNN_CONF_THRESH:
                results.append((u_c, v_c, digit, score))

        return results

    def _cnn_infer_onnx(self, roi_bgr):
        """调用 ONNX 模型进行数字推理，返回类别和置信度"""
        img_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (64, 64))
        
        img_data = resized.astype(np.float32) / 255.0
        img_data = np.transpose(img_data, (2, 0, 1))
        img_data = np.expand_dims(img_data, axis=0)

        outputs = self.ort_session.run(None, {self.ort_input_name: img_data})
        logits = outputs[0][0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        pred_class = int(np.argmax(probs)) + 1  
        confidence = float(probs[np.argmax(probs)])
        
        return pred_class, confidence

    # ==========================================================================
    # 投影匹配
    # ==========================================================================
    def _match_number_by_projection(self, map_x, map_y, map_z, cam_dets):
        if not cam_dets:
            return None

        pt_map = PointStamped()
        pt_map.header.frame_id = 'map'
        pt_map.header.stamp = rospy.Time(0)   
        pt_map.point.x = map_x
        pt_map.point.y = map_y
        pt_map.point.z = map_z

        try:
            pt_cam = self.tf_buffer.transform(
                pt_map, 'front_camera_optical', rospy.Duration(0.1))
        except Exception:
            return None

        if pt_cam.point.z <= 0.05:
            return None

        u_proj = self.FX * pt_cam.point.x / pt_cam.point.z + self.CX
        v_proj = self.FY * pt_cam.point.y / pt_cam.point.z + self.CY

        if not (0 <= u_proj < self.IMG_W and 0 <= v_proj < self.IMG_H):
            return None

        best_dist = self.PROJ_PIX_THRESH
        best_number = None
        for (u_det, v_det, number, _conf) in cam_dets:
            d = math.hypot(u_proj - u_det, v_proj - v_det)
            if d < best_dist:
                best_dist = d
                best_number = number

        return best_number

    # ==========================================================================
    # 工具方法
    # ==========================================================================
    @staticmethod
    def _transform_points(pts, tf_stamped):
        t = tf_stamped.transform.translation
        r = tf_stamped.transform.rotation
        qx, qy, qz, qw = r.x, r.y, r.z, r.w

        R = np.array([
            [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
            [    2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qw*qx)],
            [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
        ], dtype=np.float64)
        T = np.array([t.x, t.y, t.z], dtype=np.float64)

        return (R @ pts.T).T + T   

    @staticmethod
    def _voxel_downsample(pts, voxel_size):
        if len(pts) == 0:
            return pts
        keys = np.floor(pts / voxel_size).astype(np.int32)
        unique_map = {}
        for i, k in enumerate(map(tuple, keys)):
            unique_map[k] = i
        return pts[list(unique_map.values())]

    # ==========================================================================
    # 发布
    # ==========================================================================
    def _on_shutdown(self):
        self.map_manager.summary()
        with self._cand_lock:
            pending = [c for c in self._candidates if c['total'] > 0]
        if pending:
            rospy.loginfo(f"[BoxDetector] 未确认候选 {len(pending)} 个（票数未达到 {self.CONFIRM_VOTES}）:")
            for c in pending:
                best = max(c['votes'], key=c['votes'].get) if c['votes'] else '?'
                rospy.loginfo(f"  位置=({c['x']:.2f},{c['y']:.2f})  投票={c['votes']}  最可能=#{best}")

    def _publish_timer_cb(self, _event):
        self._do_publish()

    def _do_publish(self):
        if not self.map_manager.boxes:
            return
        msg = String()
        msg.data = self.map_manager.to_json()
        self.box_map_pub.publish(msg)
        self._publish_markers()

    def _publish_markers(self):
        ma = MarkerArray()
        now = rospy.Time.now()

        for i, box in enumerate(self.map_manager.boxes):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'box_spheres'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = box['x']
            m.pose.position.y = box['y']
            m.pose.position.z = 0.4
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.5
            m.color.a = 0.8
            m.color.r = 0.1
            m.color.g = 0.9
            m.color.b = 0.4
            m.lifetime = rospy.Duration(0)   

            t = Marker()
            t.header.frame_id = 'map'
            t.header.stamp = now
            t.ns = 'box_labels'
            t.id = i + 1000
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = box['x']
            t.pose.position.y = box['y']
            t.pose.position.z = 1.0
            t.pose.orientation.w = 1.0
            t.scale.z = 0.5
            t.color.a = 1.0
            t.color.r = 1.0
            t.color.g = 1.0
            t.color.b = 0.0
            t.text = f"#{box['number']}  (n={box['count']})"
            t.lifetime = rospy.Duration(0)

            ma.markers += [m, t]

        self.marker_pub.publish(ma)


if __name__ == '__main__':
    try:
        BoxGlobalDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
