#!/usr/bin/env python3
print("[Goal Manager] SCRIPT IS STARTING...")

import rospy
import math
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import dynamic_reconfigure.client

class SeniorGoalManager:
    def __init__(self):
        print("--------------------------------------------------")
        print("[Goal Manager] INITIALIZING...")
        rospy.init_node('sequential_goal_manager', anonymous=True)

        # 1. Define Waypoints
        self.waypoints = [
            {"x": 26.5, "y": -4.5, "z": 0.0, "w": 1.0, "mode": "RAMP"},
            {"x": 41.2, "y": -4.35, "z": 0.0, "w": 1.0, "mode": "STANDARD"}
        ]
        
        self.current_wp_index = 0
        self.current_pose = None
        self.goal_published = False
        self.task_completed = False
        
        self.pub_goal = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        
        try:
            self.amcl_client = dynamic_reconfigure.client.Client("amcl", timeout=1)
            self.local_costmap_client = dynamic_reconfigure.client.Client("move_base/local_costmap/obstacle_layer", timeout=1)
            self.teb_client = dynamic_reconfigure.client.Client("move_base/TebLocalPlannerROS", timeout=1)
        except:
            self.amcl_client = None
            self.local_costmap_client = None
            self.teb_client = None

        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.pose_callback)
        
        print("[Goal Manager] READY. Waiting for /amcl_pose message...")
        print("--------------------------------------------------")
        self.main_loop()

    def pose_callback(self, msg):
        self.current_pose = msg.pose.pose

    def apply_config(self, mode):
        print(">>>> [Goal Manager] APPLYING MODE: " + mode)
        try:
            if mode == "RAMP":
                if self.amcl_client: self.amcl_client.update_configuration({"odom_alpha3": 1.5})
                if self.teb_client: self.teb_client.update_configuration({"max_vel_x": 0.4, "weight_obstacle": 150.0})
            elif mode == "STANDARD":
                if self.amcl_client: self.amcl_client.update_configuration({"odom_alpha3": 0.2})
                if self.teb_client: self.teb_client.update_configuration({"max_vel_x": 0.6, "weight_obstacle": 50.0})
        except Exception as e:
            print("[Goal Manager] Reconfigure Failed: " + str(e))

    def publish_current_goal(self):
        if self.current_wp_index >= len(self.waypoints):
            self.task_completed = True
            return

        wp = self.waypoints[self.current_wp_index]
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = wp["x"]
        goal.pose.position.y = wp["y"]
        goal.pose.orientation.z = wp["z"]
        goal.pose.orientation.w = wp["w"]

        print("[Goal Manager] TARGETING WP %d: (%.2f, %.2f)" % (self.current_wp_index+1, wp["x"], wp["y"]))
        self.pub_goal.publish(goal)
        self.goal_published = True

    def main_loop(self):
        rate = rospy.Rate(5)
        nag_counter = 0
        startup_delay_done = False
        first_pose_time = None
        
        while not rospy.is_shutdown() and not self.task_completed:
            if self.current_pose is None:
                nag_counter += 1
                if nag_counter % 20 == 0:
                    print("[Goal Manager] WAITING FOR AMCL POSE...")
                rate.sleep()
                continue
            
            # Delay start until move_base is fully ready
            if not startup_delay_done:
                if first_pose_time is None:
                    first_pose_time = rospy.get_time()
                
                if rospy.get_time() - first_pose_time < 5.0:
                    print("[Goal Manager] POSE OK! Waiting 5s for costmap stability... %.1f" % (rospy.get_time() - first_pose_time))
                    rate.sleep()
                    continue
                else:
                    startup_delay_done = True
                    print("[Goal Manager] SYSTEM STABLE. LAUNCHING SEQUENCE...")

            if not self.goal_published:
                self.publish_current_goal()
            
            # Periodically re-publish if robot is not moving (heartbeat)
            nag_counter += 1
            if nag_counter % 50 == 0: # Every 10 seconds refresh the goal if not arrived
                self.publish_current_goal()
            
            wp = self.waypoints[self.current_wp_index]
            dx = self.current_pose.position.x - wp["x"]
            dy = self.current_pose.position.y - wp["y"]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < 2.0:
                print("[Goal Manager] REACHED WP %d. Switching to %s." % (self.current_wp_index+1, wp["mode"]))
                self.apply_config(wp["mode"])
                self.current_wp_index += 1
                self.goal_published = False
                nag_counter = 0 # Reset heartbeat
                
            rate.sleep()
        print("[Goal Manager] SEQUENCE FINISHED.")

if __name__ == '__main__':
    try:
        SeniorGoalManager()
    except rospy.ROSInterruptException:
        pass
