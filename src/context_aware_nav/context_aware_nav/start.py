import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import matplotlib.pyplot as plt
import time


class SmartDemo(Node):
    def __init__(self):
        super().__init__('smart_demo_node')

        # Topics
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/cpr_a200_0000/sensors/lidar2d_0/scan',
            self.scan_callback,
            10)

        self.cmd_pub = self.create_publisher(
            Twist,
            '/cpr_a200_0000/cmd_vel',
            10)

        # Arm action
        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/cpr_a200_0000/arm_0_joint_trajectory_controller/follow_joint_trajectory'
        )

        # Data logging for graph
        self.distances = []
        self.timestamps = []
        self.start_time = time.time()

        self.state = "MOVE"
        self.get_logger().info("Smart Demo Node Started")

    def scan_callback(self, msg):
        num_rays = len(msg.ranges)
        center = num_rays // 2
        
        # Calculate a 60-degree slice size (assuming a 270-degree Hokuyo LiDAR)
        # Adjust 'slice_size' if your LiDAR has a different resolution
        slice_size = num_rays // (360//10) 

        # Split the front view into three distinct sectors
        right_sector = msg.ranges[center - int(1.5*slice_size) : center - int(0.5*slice_size)]
        center_sector = msg.ranges[center - int(0.5*slice_size) : center + int(0.5*slice_size)]
        left_sector = msg.ranges[center + int(0.5*slice_size) : center + int(1.5*slice_size)]

        # Filter out infinite/invalid readings for each sector
        def get_min_dist(sector):
            valid = [r for r in sector if 0.1 < r < 10.0]
            return min(valid) if valid else 10.0

        min_right = get_min_dist(right_sector)
        min_center = get_min_dist(center_sector)
        min_left = get_min_dist(left_sector)

        # Find the absolute closest obstacle for logging
        min_dist = min(min_right, min_center, min_left)
        
        self.distances.append(min_dist)
        self.timestamps.append(time.time() - self.start_time)

        twist = Twist()

        # Behavior Logic: Sector-based decision making
        if min_dist > 2:
            self.get_logger().info("Path Clear - Moving Forward")
            twist.linear.x = 0.5
            twist.angular.z = 0.0
            self.state = "FORWARD"

        elif min_center <= 2 and min_center > 0.7:
            # Obstacle directly ahead, decide which way to turn based on peripheral space
            if min_left > min_right:
                self.get_logger().warn(f"Center blocked ({min_center:.2f}m) - Turning LEFT")
                twist.linear.x = 0.1
                twist.angular.z = 0.5  # Positive is left
            else:
                self.get_logger().warn(f"Center blocked ({min_center:.2f}m) - Turning RIGHT")
                twist.linear.x = 0.1
                twist.angular.z = -0.5 # Negative is right
            self.state = "AVOID"

        elif min_right <= 2 and min_right > 0.7:
            self.get_logger().warn(f"Right blocked ({min_right:.2f}m) - Steering LEFT")
            twist.linear.x = 0.2
            twist.angular.z = 0.4
            self.state = "AVOID"

        elif min_left <= 1.5 and min_left > 0.7:
            self.get_logger().warn(f"Left blocked ({min_left:.2f}m) - Steering RIGHT")
            twist.linear.x = 0.2
            twist.angular.z = -0.4
            self.state = "AVOID"

        else:
            # Danger Zone - Stop and Deploy Arm
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            if self.state != "ARM_ACTION"  and min_dist < 0.7:
                self.get_logger().error(f"EMERGENCY STOP ({min_dist:.2f}m) - Triggering Arm")
                self.send_arm_goal()
                self.state = "ARM_ACTION"
        self.cmd_pub.publish(twist)

    def send_arm_goal(self):
        # 1. Check server availability
        if not self.arm_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Arm server not available! Check namespace (0000 vs 0001).")
            return

        self.get_logger().info("Deploying Arm: Executing 'Scan and Point' Sequence...")
        goal_msg = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        
        # URDF Joint Names
        traj.joint_names = [
            'arm_0_shoulder_pan_joint',
            'arm_0_shoulder_lift_joint',
            'arm_0_elbow_joint',
            'arm_0_wrist_1_joint',
            'arm_0_wrist_2_joint',
            'arm_0_wrist_3_joint'
        ]

        # POINT 1: Stand Tall / Ready Stance (Arrive at 2.0 seconds)
        p1 = JointTrajectoryPoint()
        p1.positions = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        p1.time_from_start = Duration(sec=2, nanosec=0)
        traj.points.append(p1)

        # POINT 2: Look Left (Arrive at 4.0 seconds)
        p2 = JointTrajectoryPoint()
        p2.positions = [0.78, -1.57, 1.57, -1.57, -1.57, 0.0]
        p2.time_from_start = Duration(sec=4, nanosec=0)
        traj.points.append(p2)

        # POINT 3: Look Right (Arrive at 6.0 seconds)
        p3 = JointTrajectoryPoint()
        p3.positions = [-0.78, -1.57, 1.57, -1.57, -1.57, 0.0]
        p3.time_from_start = Duration(sec=6, nanosec=0)
        traj.points.append(p3)

        # POINT 4: Dramatic Point Forward (Arrive at 8.0 seconds)
        # Lowers the shoulder and extends the elbow
        p4 = JointTrajectoryPoint()
        p4.positions = [0.0, -0.5, 0.5, -1.57, -1.57, 0.0]
        p4.time_from_start = Duration(sec=8, nanosec=0)
        traj.points.append(p4)

        p5 = JointTrajectoryPoint()
        p5.positions = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        p5.time_from_start = Duration(sec=10, nanosec=0)
        traj.points.append(p5)

        goal_msg.trajectory = traj
        
        # Send the goal
        future = self.arm_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("❌ Arm Goal rejected by controller.")
            return

        self.get_logger().info("✅ Arm Goal accepted! Executing trajectory...")
        
        # Optional: You can attach a callback here to know exactly when the 8-second motion finishes
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.get_logger().info("🤖 Arm trajectory complete. Starting recovery...")
        
        # Back up for 1.5 seconds, then resume normal scanning
        recovery = Twist()
        recovery.linear.x = -0.3
        recovery.angular.z = 0.0
        
        # Publish reverse for ~1.5s using a one-shot timer
        self._recovery_count = 0
        self._recovery_timer = self.create_timer(0.1, self._recovery_step)
    def _recovery_step(self):
        self._recovery_count += 1
        if self._recovery_count <= 15:   # 15 × 0.1s = 1.5s
            recovery = Twist()
            recovery.linear.x = -0.3
            self.cmd_pub.publish(recovery)
        else:
            self._recovery_timer.cancel()
            self._recovery_timer.destroy()
            self.state = "MOVE"          # ← the critical reset
            self.get_logger().info("✅ Recovery complete, resuming navigation")

    def destroy_node(self):
        # Plot graph when stopping
        if len(self.timestamps) > 5:
            plt.plot(self.timestamps, self.distances)
            plt.xlabel("Time (s)")
            plt.ylabel("Distance (m)")
            plt.title("Obstacle Distance Over Time")
            plt.grid()
            plt.savefig("distance_plot.png")
            self.get_logger().info("📊 Graph saved as distance_plot.png")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SmartDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
