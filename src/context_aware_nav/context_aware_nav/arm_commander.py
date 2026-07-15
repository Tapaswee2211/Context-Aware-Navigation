import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class ArmCommander(Node):
    def __init__(self):
        super().__init__('arm_commander_node')
        # Connect directly to the Clearpath Kinova controller
        self._action_client = ActionClient(
            self, 
            FollowJointTrajectory, 
            '/cpr_a200_0000/arm_0_joint_trajectory_controller/follow_joint_trajectory')

    def send_goal(self):
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        goal_msg = FollowJointTrajectory.Goal()
        trajectory = JointTrajectory()
        # Kinova Gen3 6DOF Joint Names
        trajectory.joint_names = [
            'arm_0_joint_1', 'arm_0_joint_2', 'arm_0_joint_3', 
            'arm_0_joint_4', 'arm_0_joint_5', 'arm_0_joint_6'
        ]

        # Target Pose (e.g., reaching forward)
        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.5, 1.0, 0.0, 1.0, 0.0]  # Radians
        point.time_from_start = Duration(sec=3, nanosec=0) # Take 3 seconds to move
        trajectory.points.append(point)

        goal_msg.trajectory = trajectory
        self.get_logger().info('Sending arm trajectory goal...')
        self._action_client.send_goal_async(goal_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ArmCommander()
    node.send_goal()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
