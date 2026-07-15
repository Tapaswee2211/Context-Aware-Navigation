import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class ArmTest(Node):
    def __init__(self):
        super().__init__('arm_test_node')

        self.client = ActionClient(
            self,
            FollowJointTrajectory,
            '/cpr_a200_0001/arm_0_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.timer = self.create_timer(5.0, self.send_goal)
        self.toggle = 0

        self.get_logger().info("🤖 Arm Test Node Ready")

    def send_goal(self):
        if not self.client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error("Arm server not available")
            return

        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        #traj.header.stamp = self.get_clock().now().to_msg()
        traj.header.stamp = Duration(sec=0, nanosec=0).to_msg() 



        # ✅ USE CORRECT JOINT NAMES HERE
        #traj.joint_names = [
        #    'arm_0_shoulder_pan_joint',
        #    'arm_0_shoulder_lift_joint',
        #    'arm_0_elbow_joint',
        #    'arm_0_wrist_1_joint',
        #    'arm_0_wrist_2_joint',
        #    'arm_0_wrist_3_joint'
        #]
        traj.joint_names = ['arm_0_shoulder_pan_joint', 'arm_0_shoulder_lift_joint', 'arm_0_elbow_joint', 'arm_0_wrist_1_joint', 'arm_0_wrist_2_joint', 'arm_0_wrist_3_joint']

        point = JointTrajectoryPoint()

        
        #if self.toggle == 1:
        #    self.get_logger().info(f"Step: {self.toggle}")
        #    point.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #elif self.toggle == 2:
        #    self.get_logger().info(f"Step: {self.toggle}")
        #    point.positions = [0.0, -1.0, 0.0, 0.0, 0.0, 0.0]
        #elif self.toggle == 3:
        #    self.get_logger().info(f"Step: {self.toggle}")
        #    point.positions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        #elif self.toggle == 4:
        #    self.get_logger().info(f"Step: {self.toggle}")
        #    point.positions = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        #elif self.toggle == 5:
        ##    self.get_logger().info(f"Step: {self.toggle}")
        #    point.positions = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        #elif self.toggle == 6:
        #    self.get_logger().info(f"Step: {self.toggle}")
        #    point.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        #else:
        #    self.toggle = 0
        point.positions = [0.0, -0.4, 1.5, -1.2, 1.6, 1.5]
        point.velocities = [0.0] * len(point.positions)
        point.accelerations = [0.0] * len(point.positions)
        self.toggle += 1

        point.time_from_start = Duration(sec=3)
        traj.points.append(point)

        goal.trajectory = traj

        self.get_logger().info("🚀 Sending arm goal")
        self.client.send_goal_async(goal)


def main(args=None):
    rclpy.init(args=args)
    node = ArmTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
