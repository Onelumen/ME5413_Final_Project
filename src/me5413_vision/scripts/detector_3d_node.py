#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import onnxruntime as ort
import os
import math
import rospkg

import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError

import tf2_ros
import tf2_geometry_msgs


# ==========================================
# 🗺️ Advanced Semantic Map Manager (Handles volumetric collision and deduplication)
# ==========================================
class SemanticMapManager:
    def __init__(self, box_size=0.8):
        # Memory bank: Stores all discovered boxes
        self.boxes = []

        # Box size is 0.8m. Coordinates of different surfaces of the same box can differ by ~0.8m.
        # Set a 1.2m merge radius to safely associate all observations of the same box together.
        self.merge_radius = 1.2

    def update_map(self, number, new_x, new_y):
        """
        Updates the map with newly detected coordinates, handling volumetric collision detection and deduplication.
        Returns: is_new_box (bool), total_count (current total)
        """
        for box in self.boxes:
            # 1. Calculate physical distance (Euclidean) between new coordinates and known boxes
            distance = math.hypot(box['x'] - new_x, box['y'] - new_y)

            # 2. If it falls within the volumetric range of a known box
            if distance < self.merge_radius:
                # Considered as a box at the same location
                if box['number'] == number:
                    # [Weighted Average Fusion]: The more observations, the more accurate the center coordinates!
                    old_count = box['count']
                    box['x'] = (box['x'] * old_count + new_x) / (old_count + 1)
                    box['y'] = (box['y'] * old_count + new_y) / (old_count + 1)
                    box['count'] += 1

                    return False, len(self.boxes)  # Not a new box
                else:
                    # Very close but different number (occasional CNN misclassification).
                    # Trust initial detection, ignore erroneous frames.
                    return False, len(self.boxes)

        # 3. Iterated through all known boxes with no collisions; this is a brand new box!
        new_box = {
            'number': number,
            'x': new_x,
            'y': new_y,
            'count': 1  # Initial observation count
        }
        self.boxes.append(new_box)
        return True, len(self.boxes)

    def print_final_map(self):
        """Prints the final perfect map upon node shutdown"""
        rospy.loginfo("=====================================")
        rospy.loginfo(f"🏁 Task Summary: Confirmed {len(self.boxes)} boxes on the map!")
        for i, box in enumerate(self.boxes):
            rospy.loginfo(
                f"  [{i + 1}] Box #{box['number']} -> X: {box['x']:.2f}m, Y: {box['y']:.2f}m (Fused {box['count']} observations)")
        rospy.loginfo("=====================================")


# ==========================================
# 👁️ Core Node for 3D Vision and Localization
# ==========================================
class VisionDetector3D:
    def __init__(self):
        rospy.init_node('detector_3d_node', anonymous=True)
        self.bridge = CvBridge()

        # 1. Instantiate the map manager (Set box size to 0.8m)
        self.map_manager = SemanticMapManager(box_size=0.8)

        # 2. TF coordinate transform listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 3. Find and load the ONNX model
        rospack = rospkg.RosPack()
        try:
            pkg_path = rospack.get_path('me5413_vision')
        except rospkg.common.ResourceNotFound:
            rospy.logerr("Cannot find the 'me5413_vision' package. Did you source devel/setup.bash?")
            return

        onnx_path = os.path.join(pkg_path, "weights", "simple_cnn.onnx")

        if not os.path.exists(onnx_path):
            rospy.logerr(f"Model not found: {onnx_path}")
            return

        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        rospy.loginfo("🚀 ONNX model loaded successfully! Vision node with multi-range detection started...")

        # 4. Camera Intrinsics (Calculated based on URDF perspective)
        self.cx = 320.0
        self.cy = 240.0
        self.fx = 381.36
        self.fy = 381.36

        # 5. Synchronized subscription to RGB and depth images
        image_sub = message_filters.Subscriber("/front/image_raw", Image)
        depth_sub = message_filters.Subscriber("/front/depth/image_raw", Image)

        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)

    def preprocess(self, img_crop):
        """Preprocessing to match the training set"""
        img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64, 64))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def sync_callback(self, rgb_msg, depth_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # 🚀 Upgrade 1: Relaxed binarization threshold (40 -> 85) to catch darker/distant digits
        _, thresh = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # 🚀 Upgrade 2: Vastly expanded size constraints to accommodate both tiny (far) and huge (near) boxes
            if 10 < w < 400 and 15 < h < 400 and 0.3 < w / h < 2.5:

                # ==========================================
                # Physical Filter - Calculate 3D coords first to drop false targets on the ground
                # ==========================================
                u = x + w // 2
                v = y + h // 2

                # Extract median depth, bypass all-NaN cases
                depth_roi = depth_image[max(0, v - 2):min(480, v + 3), max(0, u - 2):min(640, u + 3)]
                if np.all(np.isnan(depth_roi)):
                    continue

                z_val = np.nanmedian(depth_roi)

                # 🚀 Upgrade 3: Extended radar range (Max depth increased from 6.0m to 15.0m)
                if math.isnan(z_val) or z_val <= 0.1 or z_val > 15.0:
                    continue

                # Calculate 3D coordinates in the camera frame
                x_cam = (u - self.cx) * z_val / self.fx
                y_cam = (v - self.cy) * z_val / self.fy
                z_cam = z_val

                point_cam = PointStamped()
                point_cam.header.frame_id = "front_camera_optical"
                point_cam.header.stamp = rospy.Time(0)
                point_cam.point.x = x_cam
                point_cam.point.y = y_cam
                point_cam.point.z = z_cam

                try:
                    # Transform to global map coordinates
                    point_map = self.tf_buffer.transform(point_cam, "map", rospy.Duration(0.1))
                    map_x = point_map.point.x
                    map_y = point_map.point.y
                    map_z = point_map.point.z  # Absolute physical height in the world!

                    # Physical height filter: Ground shadows have Z near 0.
                    # We only identify targets with height between 0.15m and 1.0m!
                    if map_z < 0.15 or map_z > 1.0:
                        continue  # Discard directly, don't waste CNN computation

                except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException):
                    continue  # Skip current frame if TF is not ready

                # ==========================================
                # Feed to CNN only if it passes the physical height test
                # ==========================================

                # 🚀 Upgrade 4: Dynamic Padding. Max padding capped at 30 pixels to prevent capturing background for tiny boxes
                pad_w = min(int(w * 0.3), 30)
                pad_h = min(int(h * 0.3), 30)

                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(cv_image.shape[1], x + w + pad_w)
                y2 = min(cv_image.shape[0], y + h + pad_h)

                roi = cv_image[y1:y2, x1:x2]

                # Prevent crash if ROI is entirely outside the image bounds
                if roi.shape[0] == 0 or roi.shape[1] == 0:
                    continue

                # Real-time display of CNN's perspective (Optional: can comment out for performance)
                cv2.imshow("What CNN Sees", roi)

                onnx_input = self.preprocess(roi)
                outputs = self.session.run(None, {self.input_name: onnx_input})
                logits = outputs[0]
                pred_idx = np.argmax(logits)
                confidence = np.max(self.softmax(logits))

                # 🚀 Upgrade 5: Lowered confidence threshold (0.90 -> 0.75) due to blurring on distant objects
                if confidence > 0.75:
                    number = pred_idx + 1

                    # Call map manager for deduplication and fusion
                    is_new, total_count = self.map_manager.update_map(number, map_x, map_y)

                    if is_new:
                        rospy.loginfo(
                            f"🎉 [NEW TARGET] Found box number {total_count}! (Digit #{number}, X={map_x:.2f}, Y={map_y:.2f}, Z={map_z:.2f})")

                    # Draw bounding box and target information on the image
                    cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(cv_image, f"#{number} D:{z_val:.1f}m Z:{map_z:.2f}m", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.circle(cv_image, (u, v), 4, (0, 0, 255), -1)

        # Display main vision image
        cv2.imshow("3D Vision Localizer", cv_image)
        cv2.waitKey(1)

    def on_shutdown(self):
        """Print final inventory results upon node shutdown"""
        self.map_manager.print_final_map()


if __name__ == '__main__':
    try:
        detector = VisionDetector3D()
        rospy.on_shutdown(detector.on_shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()