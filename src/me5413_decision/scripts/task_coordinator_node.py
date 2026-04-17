#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
import math
import json  
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray
from dynamic_reconfigure.client import Client

class TaskState:
    """States for the Finite State Machine (FSM)"""
    INIT = "INIT"
    EXPLORE_LOWER = "EXPLORE_LOWER"
    CALCULATE_TARGET = "CALCULATE_TARGET"
    UNBLOCK_GATE = "UNBLOCK_GATE"
    CROSS_GATE = "CROSS_GATE"
    APPROACH_RAMP = "APPROACH_RAMP"
    CLIMB_RAMP = "CLIMB_RAMP"
    RETREAT_RAMP = "RETREAT_RAMP"  
    NAV_UPPER_ROOM = "NAV_UPPER_ROOM"
    SEARCH_FINAL_TARGET = "SEARCH_FINAL_TARGET"
    DONE = "DONE" # 删除了冗余的 ARRIVE_AND_FINISH

class TaskCoordinator:
    def __init__(self):
        rospy.init_node('task_coordinator_node', anonymous=False)
        self.state = TaskState.INIT
        
        # ================== 1. Initialize ROS Interfaces ==================
        rospy.loginfo("[Coordinator] Waiting for move_base action server...")
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.move_base_client.wait_for_server() 
        rospy.loginfo("[Coordinator] Successfully connected to move_base!")

        self.unblock_pub = rospy.Publisher('/cmd_unblock', Bool, queue_size=1)

        # Dynamic Reconfigure Client for Speed Control
        try:
            self.reconfig_client = Client("move_base/TebLocalPlannerROS", timeout=5.0)
            rospy.loginfo("[Coordinator] Connected to Dynamic Reconfigure server.")
        except rospy.ROSException:
            rospy.logwarn("[Coordinator] Dynamic Reconfigure server not found. Speed control disabled.")
            self.reconfig_client = None

        self.robot_x = 0.0
        self.robot_y = 0.0
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_callback)

        # Vision Subscriptions
        self.confirmed_boxes = [] 
        rospy.Subscriber('/vision/box_map', String, self.box_map_callback)
        
        self.candidates = [] 
        rospy.Subscriber('/vision/box_candidates', MarkerArray, self.candidate_callback)

        # ================== 2. Business Data & Control Variables ==================
        self.box_counts = {} 
        self.target_number = None 
        self.final_box_pose = (0.0, 0.0)

        # Error Handling Variables
        self.ramp_retry_count = 0
        self.MAX_RAMP_RETRIES = 3
        self.lower_wp_retry = 0
        
        # [新增] 突击尝试记录器
        self.nav_upper_attempt = 1
        
        self.visited_candidates = []
        
        self.lower_waypoints = [
            (3.5, 0.0, 1.57), (3.5, 7.5, 1.57), (3.5, 16.5, 0.0),
            (11.5, 16.5, -1.57), (11.5, 7.5, -1.57), (11.5, 2.0, 0.0),
            (13.5, 2.0, 1.57), (13.5, 7.5, 1.57), (13.5, 16.5, 0.0), 
            (21.5, 16.5, -1.57), (21.5, 7.5, -1.57), (21.5, -1.5, 3.14),
            (13.5, -1.5, 1.57), (13.5, 2.0, 3.14), (11.5, 2.0, -1.57),
            (11.5, -1.5, 3.14), (7.5, -1.5, -1.57)
        ]
        
        self.upper_gate_waypoints = [
            (31.0, 14.6, 3.14), (31.0, 9.6, 3.14), 
            (31.0, 4.6, 3.14), (31.0, -0.4, 3.14)  
        ]
        self.gate_index = 0
        self.current_wp_index = 0
        self.active_candidate_goal = None 

    # ================== 3. Callbacks & Helpers ==================
    def amcl_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def box_map_callback(self, msg):
        try:
            self.confirmed_boxes = json.loads(msg.data)
            self.box_counts = {b['number']: b['count'] for b in self.confirmed_boxes if b['number'] > 0}
        except Exception as e:
            rospy.logwarn(f"[Coordinator] BoxMap parsing failed: {e}")

    def candidate_callback(self, msg):
        allowed_states = [TaskState.EXPLORE_LOWER, TaskState.SEARCH_FINAL_TARGET]
        if self.state not in allowed_states:
            return
        
        for marker in msg.markers:
            cx, cy = marker.pose.position.x, marker.pose.position.y
            dist = math.hypot(self.robot_x - cx, self.robot_y - cy)
            
            if dist < 4.5:
                is_known = any(math.hypot(bx['x'] - cx, bx['y'] - cy) < 1.2 for bx in self.confirmed_boxes)
                is_visited = any(math.hypot(vx - cx, vy - cy) < 1.0 for vx, vy in self.visited_candidates)
                
                if not is_known and not is_visited:
                    self.active_candidate_goal = (cx, cy)
                    break 

    def send_nav_goal(self, x, y, yaw, timeout_secs=0.0):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.target_pose.pose.orientation.w = math.cos(yaw / 2.0)
        
        rospy.loginfo(f"[Coordinator] Sending Goal: ({x:.2f}, {y:.2f}, yaw={yaw:.2f}) [Timeout: {timeout_secs}s]")
        self.move_base_client.send_goal(goal)
        
        if timeout_secs > 0.0:
            finished_within_time = self.move_base_client.wait_for_result(rospy.Duration(timeout_secs))
            if not finished_within_time:
                rospy.logwarn("[Coordinator] Navigation Timeout! Canceling current goal to prevent deadlock.")
                self.move_base_client.cancel_goal()
                return actionlib.GoalStatus.ABORTED
        else:
            self.move_base_client.wait_for_result()
            
        return self.move_base_client.get_state()

    def calc_offset_pose(self, target_x, target_y, stop_distance=0.6):
        dx = target_x - self.robot_x
        dy = target_y - self.robot_y
        dist = math.hypot(dx, dy)
        
        if dist <= stop_distance:
            return self.robot_x, self.robot_y, math.atan2(dy, dx)
            
        ratio = (dist - stop_distance) / dist
        stop_x = self.robot_x + dx * ratio
        stop_y = self.robot_y + dy * ratio
        stop_yaw = math.atan2(dy, dx)
        
        return stop_x, stop_y, stop_yaw

    def set_speed(self, max_vel_x=0.6, max_vel_theta=0.5):
        if self.reconfig_client:
            try:
                self.reconfig_client.update_configuration({
                    "max_vel_x": max_vel_x,
                    "max_vel_theta": max_vel_theta
                })
                rospy.loginfo(f"[Coordinator] Speed limits updated: x={max_vel_x}, theta={max_vel_theta}")
            except Exception as e:
                rospy.logwarn(f"[Coordinator] Failed to update speed: {e}")

    def set_strict_clearance(self, is_strict=True):
        """[新增] 动态修改TEB避障半径，强迫小车不许挤缝隙"""
        if self.reconfig_client:
            try:
                dist = 0.55 if is_strict else 0.25 
                self.reconfig_client.update_configuration({
                    "min_obstacle_dist": dist
                })
                mode = "STRICT (Fat)" if is_strict else "NORMAL"
                rospy.loginfo(f"[Coordinator] Clearance mode updated to {mode}: {dist}m")
            except Exception as e:
                rospy.logwarn(f"[Coordinator] Failed to update clearance: {e}")

    # ================== 4. Main FSM Loop ==================
    def run(self):
        rate = rospy.Rate(10) # 10Hz
        while not rospy.is_shutdown():
            
            if self.state == TaskState.INIT:
                rospy.loginfo("[State] Initialization complete. Starting lower-level exploration...")
                rospy.sleep(2) 
                self.state = TaskState.EXPLORE_LOWER

            elif self.state == TaskState.EXPLORE_LOWER:
                if self.active_candidate_goal:
                    cx, cy = self.active_candidate_goal
                    rospy.loginfo(f"[State] 🔎 Sighted Candidate at ({cx:.2f}, {cy:.2f})! Deviating to align...")
                    
                    stop_x, stop_y, stop_yaw = self.calc_offset_pose(cx, cy, stop_distance=1.5)
                    self.send_nav_goal(stop_x, stop_y, stop_yaw, timeout_secs=15.0)
                    
                    rospy.sleep(2.0)
                    self.visited_candidates.append((cx, cy))
                    self.active_candidate_goal = None 
                else:
                    rospy.loginfo(f"[State] Exploring Lower: {self.current_wp_index + 1}/{len(self.lower_waypoints)}")
                    wp = self.lower_waypoints[self.current_wp_index]
                    
                    state_result = self.send_nav_goal(wp[0], wp[1], wp[2], timeout_secs=30.0)
                    
                    if state_result == actionlib.GoalStatus.SUCCEEDED:
                        self.current_wp_index += 1
                        self.lower_wp_retry = 0 
                        if self.current_wp_index >= len(self.lower_waypoints):
                            rospy.loginfo("[State] Lower exploration complete!")
                            self.state = TaskState.CALCULATE_TARGET
                    else:
                        self.lower_wp_retry += 1
                        if self.lower_wp_retry >= 2:
                            rospy.logwarn("[State] Waypoint failed twice. Skipping.")
                            self.current_wp_index += 1
                            self.lower_wp_retry = 0

            elif self.state == TaskState.CALCULATE_TARGET:
                if self.box_counts:
                    self.target_number = min(self.box_counts, key=self.box_counts.get)
                    rospy.loginfo(f"[State] Box Counts: {self.box_counts}")
                    rospy.loginfo(f"[State] >>> TARGET NUMBER CONFIRMED: {self.target_number} <<<")
                else:
                    rospy.logwarn("[State] WARNING: No boxes detected! Defaulting to 9.")
                    self.target_number = 9
                    
                self.state = TaskState.UNBLOCK_GATE

            elif self.state == TaskState.UNBLOCK_GATE:
                unblock_msg = Bool()
                unblock_msg.data = True
                self.unblock_pub.publish(unblock_msg)
                rospy.loginfo("[State] Unblock command published! Gate will vanish in 10s...")
                rospy.sleep(0.5) 
                self.state = TaskState.CROSS_GATE

            elif self.state == TaskState.CROSS_GATE:
                gate_x, gate_y, gate_yaw = 7.5, -4.0, 0.0
                state_result = self.send_nav_goal(gate_x, gate_y, gate_yaw, timeout_secs=20.0)
                if state_result == actionlib.GoalStatus.SUCCEEDED:
                    self.state = TaskState.APPROACH_RAMP
                else:
                    rospy.logerr("[State] Crossing gate failed or timed out! Retrying unlocking...")
                    self.state = TaskState.UNBLOCK_GATE

            elif self.state == TaskState.APPROACH_RAMP:
                ramp_bottom_x, ramp_bottom_y, ramp_bottom_yaw = 24.0, -4.0, 0.0 
                state_result = self.send_nav_goal(ramp_bottom_x, ramp_bottom_y, ramp_bottom_yaw, timeout_secs=30.0)
                if state_result == actionlib.GoalStatus.SUCCEEDED:
                    self.state = TaskState.CLIMB_RAMP

            elif self.state == TaskState.CLIMB_RAMP:
                rospy.loginfo(f"[State] Climbing Ramp! (Attempt {self.ramp_retry_count+1}/{self.MAX_RAMP_RETRIES})")
                ramp_top_x, ramp_top_y, ramp_top_yaw = 41.5, -4.0, 1.57 
                state_result = self.send_nav_goal(ramp_top_x, ramp_top_y, ramp_top_yaw, timeout_secs=90.0)
                
                if state_result == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo("[State] Successfully reached the upper level!")
                    # 确保重置尝试次数
                    self.nav_upper_attempt = 1
                    self.state = TaskState.NAV_UPPER_ROOM
                else:
                    self.ramp_retry_count += 1
                    if self.ramp_retry_count < self.MAX_RAMP_RETRIES:
                        rospy.logwarn("[State] Climb failed! Slipping or drift detected. Transitioning to Retreat state...")
                        self.state = TaskState.RETREAT_RAMP
                    else:
                        rospy.logerr("[State] Climb failed 3 times! Terminating climbing attempts.")
                        pass
            
            elif self.state == TaskState.RETREAT_RAMP:
                rospy.loginfo("[State] Retreating to ramp bottom to reset position...")
                ramp_bottom_x, ramp_bottom_y, ramp_bottom_yaw = 24.0, -4.0, 0.0
                state_result = self.send_nav_goal(ramp_bottom_x, ramp_bottom_y, ramp_bottom_yaw, timeout_secs=90.0)
                
                if state_result == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo("[State] Successfully retreated to bottom. Restarting climb...")
                    self.state = TaskState.CLIMB_RAMP
                else:
                    rospy.logerr("[State] Failed to retreat properly. Forcing climb attempt anyway...")
                    self.state = TaskState.CLIMB_RAMP
                    
            elif self.state == TaskState.NAV_UPPER_ROOM:
                upper_room_x, upper_room_y, upper_room_yaw = 31.0, 7.3, 3.14
                
                # =======================================================
                # ⚠️ 关键设置：请在 RViz 中获取这两个“开口中心”的精确坐标
                # 这两个点必须放置在墙壁缺口的最窄处！
                # =======================================================
                gap_upper_pose = (34.4, 12.3) # 请修改这里！
                gap_lower_pose = (34.4, 2.3) # 请修改这里！

                if self.nav_upper_attempt == 1:
                    rospy.loginfo("[State] Reconnaissance: Attempting UPPER gap...")
                    # [战术变胖] 强迫小车不许挤缝隙
                    self.set_strict_clearance(is_strict=True)
                    
                    # 极短的超时时间 (15s)！如果不通畅，TEB 必定卡死超时
                    state_result = self.send_nav_goal(gap_upper_pose[0], gap_upper_pose[1], 3.14, timeout_secs=45.0)
                    
                    if state_result == actionlib.GoalStatus.SUCCEEDED:
                        rospy.loginfo("[State] UPPER gap is CLEAR! Proceeding to room corridor...")
                        # [恢复正常体型]
                        self.set_strict_clearance(is_strict=False)
                        self.send_nav_goal(upper_room_x, upper_room_y, upper_room_yaw, timeout_secs=20.0)
                        self.state = TaskState.SEARCH_FINAL_TARGET
                    else:
                        rospy.logwarn("[State] UPPER gap blocked or squeezing detected! Aborting and switching to LOWER gap...")
                        self.nav_upper_attempt = 2 # 切换方案

                elif self.nav_upper_attempt == 2:
                    rospy.loginfo("[State] Reconnaissance: Attempting LOWER gap...")
                    # 依然保持[战术变胖]状态
                    self.set_strict_clearance(is_strict=True)
                    
                    state_result = self.send_nav_goal(gap_lower_pose[0], gap_lower_pose[1], 3.14, timeout_secs=15.0)
                    
                    if state_result == actionlib.GoalStatus.SUCCEEDED:
                        rospy.loginfo("[State] LOWER gap is CLEAR! Proceeding to room corridor...")
                        # [恢复正常体型]
                        self.set_strict_clearance(is_strict=False)
                        self.send_nav_goal(upper_room_x, upper_room_y, upper_room_yaw, timeout_secs=20.0)
                        self.state = TaskState.SEARCH_FINAL_TARGET
                    else:
                        rospy.logerr("[State] BOTH gaps failed! This shouldn't happen. Retrying upper...")
                        # 极端情况防死锁，退回尝试1
                        self.set_strict_clearance(is_strict=False)
                        self.nav_upper_attempt = 1 

            elif self.state == TaskState.SEARCH_FINAL_TARGET:
                if self.active_candidate_goal:
                    cx, cy = self.active_candidate_goal
                    rospy.loginfo(f"[State] 🔎 Sighted Candidate in Final Phase at ({cx:.2f}, {cy:.2f})!")
                    self.visited_candidates.append((cx, cy))
                    self.active_candidate_goal = None 

                if self.gate_index >= len(self.upper_gate_waypoints):
                    rospy.logerr("[State] All rooms checked but TARGET NOT FOUND! Restarting search...")
                    self.gate_index = 0
                    continue

                wp = self.upper_gate_waypoints[self.gate_index]
                rospy.loginfo(f"[State] 🔎 Checking Room {self.gate_index + 1} at ({wp[0]}, {wp[1]}) for Box #{self.target_number}")
                
                state_result = self.send_nav_goal(wp[0], wp[1], wp[2], timeout_secs=25.0)
                
                if state_result == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo(f"[State] At Gate {self.gate_index + 1}. Gazing for recognition...")
                    rospy.sleep(2.0) 
                    
                    found_in_this_room = None
                    for box in self.confirmed_boxes:
                        if box['number'] == self.target_number:
                            # [重要修复] 必须加上 box['x'] > 23.0 屏蔽一楼幽灵盒子！
                            if abs(box['y'] - wp[1]) < 2.0:
                                found_in_this_room = box
                                break
                    
                    if found_in_this_room:
                        self.final_box_pose = (found_in_this_room['x'], found_in_this_room['y'])
                        rospy.loginfo(f"[State] ✔ MATCH! Target found in Room {self.gate_index + 1}.")
                        
                        self.set_speed(max_vel_x=1.2, max_vel_theta=1.0)
                        rospy.sleep(0.5) 
                        
                        rospy.loginfo("[State] Charging directly to the target stop point at X=24.5...")
                        state_result = self.send_nav_goal(24.5, found_in_this_room['y'], 0.0, timeout_secs=20.0)
                        
                        if state_result == actionlib.GoalStatus.SUCCEEDED:
                            rospy.loginfo("[State] *** MISSION COMPLETE! *** Stopped safely at target.")
                            self.set_speed(max_vel_x=0.6, max_vel_theta=0.5)
                            self.state = TaskState.DONE
                        else:
                            rospy.logwarn("[State] Final charge to 24.5 failed or obstructed. Retrying...")
                    else:
                        rospy.loginfo(f"[State] Not found in Room {self.gate_index + 1}. Moving to next...")
                        self.gate_index += 1
                else:
                    rospy.logwarn(f"[State] Failed to reach Gate {self.gate_index + 1}. Skipping.")
                    self.gate_index += 1

            elif self.state == TaskState.DONE:
                pass
            
            rate.sleep()

if __name__ == '__main__':
    try:
        coordinator = TaskCoordinator()
        coordinator.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Task Coordinator Node Terminated.")