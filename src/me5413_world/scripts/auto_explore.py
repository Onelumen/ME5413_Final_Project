#!/usr/bin/env python3
import rospy
import numpy as np
import actionlib
from nav_msgs.msg import OccupancyGrid
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus


class FrontierExplorer:
    def __init__(self):
        rospy.init_node('frontier_explorer')
        self.map_data = None
        self.map_info = None
        self.goal_start_time = None
        self.goal_timeout = rospy.Duration(20.0)  # 20秒卡住则换目标
        self.failed_goals = set()  # 记录失败的目标点，避免反复尝试

        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base...")
        self.client.wait_for_server(rospy.Duration(30.0))
        rospy.loginfo("Connected to move_base.")

        rospy.Subscriber('/map', OccupancyGrid, self.map_callback, queue_size=1)
        rospy.Timer(rospy.Duration(5.0), self.explore_timer)
        rospy.loginfo("Frontier explorer ready.")

    def map_callback(self, msg):
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)

    def find_frontiers(self):
        data = self.map_data
        free = (data >= 0) & (data <= 50)
        unknown = (data == -1)

        # Vectorized: free cells adjacent to unknown cells (4-connectivity)
        frontier = np.zeros_like(free)
        frontier[1:,  :] |= free[1:,  :] & unknown[:-1, :]
        frontier[:-1, :] |= free[:-1, :] & unknown[1:,  :]
        frontier[:,  1:] |= free[:,  1:] & unknown[:, :-1]
        frontier[:, :-1] |= free[:, :-1] & unknown[:, 1: ]

        ys, xs = np.where(frontier)
        return list(zip(xs.tolist(), ys.tolist()))

    def cell_to_world(self, cx, cy):
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        return ox + (cx + 0.5) * res, oy + (cy + 0.5) * res

    def send_goal(self, wx, wy):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = wx
        goal.target_pose.pose.position.y = wy
        goal.target_pose.pose.orientation.w = 1.0
        self.client.send_goal(goal)
        self.goal_start_time = rospy.Time.now()
        rospy.loginfo("Goal sent: (%.2f, %.2f)", wx, wy)

    def explore_timer(self, _event):
        if self.map_data is None:
            rospy.logwarn("No map received yet.")
            return

        state = self.client.get_state()

        # 超时检测：卡住超过20秒则取消当前目标
        if state in (GoalStatus.ACTIVE, GoalStatus.PENDING):
            if self.goal_start_time and \
               (rospy.Time.now() - self.goal_start_time) > self.goal_timeout:
                rospy.logwarn("Goal timed out, cancelling and trying next frontier.")
                self.client.cancel_goal()
                # 记录失败目标避免重复
                result = self.client.get_goal_status_text()
                rospy.logwarn("Cancelled goal: %s", result)
            else:
                return  # 正常导航中，等待

        # 目标被 ABORTED 也记录失败
        if state == GoalStatus.ABORTED:
            rospy.logwarn("Goal aborted by move_base, selecting new frontier.")

        frontiers = self.find_frontiers()
        if not frontiers:
            rospy.loginfo("No frontiers found - exploration complete!")
            return

        rospy.loginfo("Found %d frontier cells.", len(frontiers))

        h, w = self.map_data.shape
        cx, cy = w // 2, h // 2
        arr = np.array(frontiers)

        # 过滤掉之前失败过的目标点（1m范围内）
        def not_failed(pt):
            wx, wy = self.cell_to_world(int(pt[0]), int(pt[1]))
            return all(abs(wx - fx) > 1.0 or abs(wy - fy) > 1.0
                      for fx, fy in self.failed_goals)

        if len(self.failed_goals) > 0:
            arr = np.array([p for p in arr.tolist() if not_failed(p)])
            if len(arr) == 0:
                rospy.logwarn("All frontiers were previously failed, clearing history.")
                self.failed_goals.clear()
                arr = np.array(frontiers)

        # 采样加速
        if len(arr) > 300:
            idx = np.random.choice(len(arr), 300, replace=False)
            arr = arr[idx]

        dists = (arr[:, 0] - cx) ** 2 + (arr[:, 1] - cy) ** 2
        best = arr[np.argmax(dists)]
        wx, wy = self.cell_to_world(int(best[0]), int(best[1]))

        # 如果是 ABORTED，记录失败点
        if state == GoalStatus.ABORTED:
            self.failed_goals.add((wx, wy))

        self.send_goal(wx, wy)


if __name__ == '__main__':
    try:
        FrontierExplorer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
