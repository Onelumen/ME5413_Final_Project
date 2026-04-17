#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import os
import math
import rospkg

import message_filters
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PoseStamped
from cv_bridge import CvBridge, CvBridgeError

import tf2_ros
import tf
from tf.transformations import quaternion_from_euler


# ==========================================
# 🗺️ Advanced Semantic Map Manager (With Mission Tally Logic)
# ==========================================
class SemanticMapManager:
    def __init__(self, box_size=0.8):
        self.boxes = []
        self.merge_radius = 1.2

    def update_map(self, number, new_x, new_y):
        for box in self.boxes:
            distance = math.hypot(box['x'] - new_x, box['y'] - new_y)
            if distance < self.merge_radius:
                if box['number'] == number:
                    old_count = box['count']
                    box['x'] = (box['x'] * old_count + new_x) / (old_count + 1)
                    box['y'] = (box['y'] * old_count + new_y) / (old_count + 1)
                    box['count'] += 1
                    return False, len(self.boxes)
                else:
                    return False, len(self.boxes)

        new_box = {
            'number': number,
            'x': new_x,
            'y': new_y,
            'count': 1
        }
        self.boxes.append(new_box)
        return True, len(self.boxes)

    def get_mission_result(self):
        """🚀 Core Mission Logic: Find the digit(s) with the minimum occurrences"""
        if not self.boxes:
            return [], 0, {}

        # 1. Tally up the occurrences of each digit
        tally = {}
        for box in self.boxes:
            num = box['number']
            tally[num] = tally.get(num, 0) + 1

        # 2. Find the minimum count
        min_count = min(tally.values())

        # 3. Find all digits that share this minimum count (handles ties)
        target_numbers = [num for num, count in tally.items() if count == min_count]

        return target_numbers, min_count, tally

    def print_final_map(self):
        """Prints the final summary report upon node shutdown"""
        rospy.loginfo("=====================================")
        rospy.loginfo(f"🏁 Exploration Complete! Locked onto {len(self.boxes)} distinct boxes on the map.")

        for i, box in enumerate(self.boxes):
            rospy.loginfo(f"  [{i + 1}] Found Box #{box['number']} -> X: {box['x']:.2f}m, Y: {box['y']:.2f}m")

        rospy.loginfo("-------------------------------------")

        targets, min_count, tally = self.get_mission_result()
        if targets:
            tally_str = ", ".join([f"Digit {k}: {v}" for k, v in tally.items()])
            rospy.loginfo(f"📊 Field Inventory: [{tally_str}]")

            if len(targets) == 1:
                rospy.loginfo(
                    f"🏆 [FINAL MISSION TARGET] The least frequent digit is: >>>> {targets[0]} <<<< (Appeared only {min_count} time(s)!)")
            else:
                rospy.loginfo(
                    f"🏆 [FINAL MISSION TARGET] Tied for least frequent digits: {targets} (Appeared only {min_count} time(s)!)")
        else:
            rospy.logwarn("⚠️ Oh no! The robot did not find any digit boxes!")

        rospy.loginfo("=====================================")


# ==========================================
# 👁️ Core Node for 3D Vision & Autonomous Navigation
# ==========================================
class VisionDetector3D:
    def __init__(self):
        rospy.init_node('detector_3d_node', anonymous=True)
        self.bridge = CvBridge()
        self.map_manager = SemanticMapManager(box_size=0.8)

        # TF listeners
        self.tf_listener = tf.TransformListener()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Navigation goal publisher
        self.goal_pub = rospy.Publisher("/detected_goal_pose", PoseStamped, queue_size=1)

        # Load Templates
        rospack = rospkg.RosPack()
        try:
            pkg_path = rospack.get_path('me5413_vision')
        except rospkg.common.ResourceNotFound:
            rospy.logerr("Cannot find 'me5413_vision' package.")
            return

        template_dir = os.path.join(pkg_path, "number")
        self.templates = {}

        rospy.loginfo("Loading digital templates from: " + template_dir)
        for i in range(1, 10):
            img_path = os.path.join(template_dir, f"{i}.png")
            if not os.path.exists(img_path):
                continue

            temp_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            _, temp_bin = cv2.threshold(temp_img, 127, 255, cv2.THRESH_BINARY_INV)

            coords = cv2.findNonZero(temp_bin)
            if coords is not None:
                tx, ty, tw, th = cv2.boundingRect(coords)
                tight_temp = temp_bin[ty:ty + th, tx:tx + tw]
                self.templates[i] = cv2.resize(tight_temp, (64, 64))

        rospy.loginfo(f"🚀 Loaded {len(self.templates)} templates! Vision + Navigation node started...")

        self.cx, self.cy = 320.0, 240.0
        self.fx, self.fy = 381.36, 381.36

        image_sub = message_filters.Subscriber("/front/image_raw", Image)
        depth_sub = message_filters.Subscriber("/front/depth/image_raw", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)

    def publish_navigation_goal(self, target_map_x, target_map_y):
        """Calculates robot heading and publishes the driving goal"""
        try:
            robot_pose = PoseStamped()
            robot_pose.header.stamp = rospy.Time(0)
            robot_pose.header.frame_id = "base_link"
            self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(0.5))
            robot_pose_map = self.tf_listener.transformPose("map", robot_pose)

            robot_x = robot_pose_map.pose.position.x
            robot_y = robot_pose_map.pose.position.y

            # Calculate Yaw
            dx = target_map_x - robot_x
            dy = target_map_y - robot_y
            yaw = np.arctan2(dy, dx)

            quat = quaternion_from_euler(0, 0, yaw)

            goal_pose = PoseStamped()
            goal_pose.header.stamp = rospy.Time.now()
            goal_pose.header.frame_id = "map"

            # Smart stopping mechanism: stop 1.0 meter in front of the box
            stop_distance = 1.0
            dist_to_box = math.hypot(dx, dy)

            if dist_to_box > stop_distance:
                goal_pose.pose.position.x = robot_x + dx * ((dist_to_box - stop_distance) / dist_to_box)
                goal_pose.pose.position.y = robot_y + dy * ((dist_to_box - stop_distance) / dist_to_box)
            else:
                goal_pose.pose.position.x = robot_x
                goal_pose.pose.position.y = robot_y

            goal_pose.pose.position.z = 0.0
            goal_pose.pose.orientation.x = quat[0]
            goal_pose.pose.orientation.y = quat[1]
            goal_pose.pose.orientation.z = quat[2]
            goal_pose.pose.orientation.w = quat[3]

            self.goal_pub.publish(goal_pose)
            rospy.loginfo(
                f"🚗 [NAVIGATION] Generated Goal -> Driving to X:{goal_pose.pose.position.x:.2f}, Y:{goal_pose.pose.position.y:.2f} facing the box!")

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"TF Error in navigation target calculation: {e}")

    def sync_callback(self, rgb_msg, depth_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        except CvBridgeError as e:
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if 15 < w < 400 and 20 < h < 400 and 0.3 < w / h < 2.5:
                u = x + w // 2
                v = y + h // 2

                depth_roi = depth_image[max(0, v - 2):min(480, v + 3), max(0, u - 2):min(640, u + 3)]
                if np.all(np.isnan(depth_roi)):
                    continue

                z_val = np.nanmedian(depth_roi)
                if math.isnan(z_val) or z_val <= 0.1 or z_val > 15.0:
                    continue

                x_cam = (u - self.cx) * z_val / self.fx
                y_cam = (v - self.cy) * z_val / self.fy
                z_cam = z_val

                point_cam = PointStamped()
                point_cam.header.frame_id = "front_camera_optical"
                point_cam.point.x = x_cam
                point_cam.point.y = y_cam
                point_cam.point.z = z_cam

                try:
                    point_map = self.tf_buffer.transform(point_cam, "map", rospy.Duration(0.1))
                    map_x = point_map.point.x
                    map_y = point_map.point.y
                    map_z = point_map.point.z

                    if map_z < 0.15 or map_z > 1.0:
                        continue
                except:
                    continue

                roi_bin = thresh[y:y + h, x:x + w]
                if roi_bin.shape[0] == 0 or roi_bin.shape[1] == 0:
                    continue

                roi_resized = cv2.resize(roi_bin, (64, 64))

                best_match_num = -1
                highest_score = -1.0

                for num, template in self.templates.items():
                    res = cv2.matchTemplate(roi_resized, template, cv2.TM_CCOEFF_NORMED)
                    score = res[0][0]
                    if score > highest_score:
                        highest_score = score
                        best_match_num = num

                if highest_score > 0.45:
                    is_new, total_count = self.map_manager.update_map(best_match_num, map_x, map_y)

                    if is_new:
                        rospy.loginfo(f"🎉 [NEW TARGET] Found box number {total_count}! (Digit #{best_match_num})")
                        self.publish_navigation_goal(map_x, map_y)

                    cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(cv_image, f"#{best_match_num} ({highest_score:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.circle(cv_image, (u, v), 4, (0, 0, 255), -1)

        cv2.imshow("3D Vision Localizer", cv_image)
        cv2.waitKey(1)

    def on_shutdown(self):
        self.map_manager.print_final_map()


if __name__ == '__main__':
    try:
        detector = VisionDetector3D()
        rospy.on_shutdown(detector.on_shutdown)
        rospy.spin()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()