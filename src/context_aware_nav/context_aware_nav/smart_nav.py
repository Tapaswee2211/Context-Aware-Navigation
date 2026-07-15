#smart_nav.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import matplotlib
matplotlib.use('Agg')  # non-interactive backend, safe for ROS nodes
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import time
import os
import math

import json
from std_msgs.msg import String


class SmartDemo(Node):
    def __init__(self):
        super().__init__('smart_demo_node')

        # ── Topics ──────────────────────────────────────────────────────────
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/cpr_a200_0000/sensors/lidar2d_0/scan',
            self.scan_callback, 10)

        self.odom_sub = self.create_subscription(
            Odometry,
            '/cpr_a200_0000/platform/odom/filtered',
            self.odom_callback, 10)

        self.cmd_pub = self.create_publisher(
            Twist, '/cpr_a200_0000/cmd_vel', 10)

        # ── Arm action ──────────────────────────────────────────────────────
        self.arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/cpr_a200_0000/arm_0_joint_trajectory_controller/follow_joint_trajectory')

        # ── State machine ───────────────────────────────────────────────────
        self.state = "FORWARD"
        #  FORWARD | TURN_COMMIT | EMERGENCY_STOP | ARM_ACTION | RECOVERY

        # Turn commitment: once we decide a direction, hold it for N seconds
        self.turn_direction = 0.0       # +1 left, -1 right
        self.turn_start_time = None
        self.TURN_COMMIT_SECS = 1.8     # hold turn for this long before re-evaluating

        # Recovery after arm
        self._recovery_count = 0
        self._recovery_timer = None

        # ── Logging for graphs ──────────────────────────────────────────────
        self.log_dir = os.path.expanduser("~/clearpath_ws/logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self.start_time = self.get_clock().now()

        # Distance log
        self.distances = []
        self.timestamps = []

        # State log  (for state timeline graph)
        self.state_log = []          # list of (time, state_int)
        self.STATE_MAP = {
            "FORWARD": 0, "TURN_COMMIT": 1,
            "EMERGENCY_STOP": 2, "ARM_ACTION": 3, "RECOVERY": 4
        }
        self.yolo_sub = self.create_subscription(
            String,
            '/cpr_a200_0000/yolo/detections',
            self._yolo_callback, 10)
        
        self.yolo_stop = False    # YOLO says stop (person/animal in path)
        self.yolo_slow = False    # YOLO says slow (vehicle/object in path)
        self.yolo_info = []       # latest detection list for logging


        # Velocity log
        self.vel_linear = []
        self.vel_angular = []

        # Odometry / path
        self.path_x = []
        self.path_y = []

        # Sector distance log (right, center, left)
        self.sector_right = []
        self.sector_center = []
        self.sector_left = []

        # Periodic graph save every 30 s
        self.graph_timer = self.create_timer(30.0, self.save_all_graphs)

        self.get_logger().info("SmartDemo node started — logs → " + self.log_dir)

    # ── Odometry callback ────────────────────────────────────────────────────
    def odom_callback(self, msg):
        self.path_x.append(msg.pose.pose.position.x)
        self.path_y.append(msg.pose.pose.position.y)
        
    def _yolo_callback(self, msg):
        try:
            data = json.loads(msg.data)
            self.yolo_stop = data.get("stop", False)
            self.yolo_slow = data.get("slow", False)
            self.yolo_info = data.get("detections", [])
            if self.yolo_stop:
                self.get_logger().warn(
                    f"YOLO: STOP — "
                    f"{[d['class'] for d in self.yolo_info if d['center']]}")
        except Exception as e:
            self.get_logger().error(f"YOLO parse error: {e}")

    # ── Scan callback ─────────────────────────────────────────────────────────
    # ── Scan callback ─────────────────────────────────────────────────────────
    def scan_callback(self, msg):
        current_time = self.get_clock().now()
        now = (current_time - self.start_time).nanoseconds / 1e9

        # ── Sector extraction ───────────────────────────────────────────────
        num_rays = len(msg.ranges)
        center = num_rays // 2
        slice_size = max(1, num_rays // 36)   # ~10° slice

        def get_min(sector):
            valid = [r for r in sector if 0.1 < r < 10.0]
            return min(valid) if valid else 10.0

        # Three forward-facing sectors
        r_sec = msg.ranges[center - 3*slice_size : center - slice_size]
        c_sec = msg.ranges[center - slice_size   : center + slice_size]
        l_sec = msg.ranges[center + slice_size   : center + 3*slice_size]

        # Wider side sectors to detect wall-following needs
        far_r = msg.ranges[center - 5*slice_size : center - 3*slice_size]
        far_l = msg.ranges[center + 3*slice_size : center + 5*slice_size]

        min_r   = get_min(r_sec)
        min_c   = get_min(c_sec)
        min_l   = get_min(l_sec)
        min_far_r = get_min(far_r)
        min_far_l = get_min(far_l)
        min_all = min(min_r, min_c, min_l)

        # ── Logging ─────────────────────────────────────────────────────────
        self.distances.append(min_all)
        self.timestamps.append(now)
        self.sector_right.append(min_r)
        self.sector_center.append(min_c)
        self.sector_left.append(min_l)
        self.state_log.append((now, self.STATE_MAP.get(self.state, -1)))

        twist = Twist()

        # ── State machine ────────────────────────────────────────────────────

        # 1. Let ongoing states finish before re-evaluating
        if self.state in ("ARM_ACTION", "RECOVERY"):
            self.cmd_pub.publish(twist)   # zero velocity during arm/recovery
            return

        # 2. EMERGENCY: something dangerously close
        if min_all < 0.5:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            if self.state != "EMERGENCY_STOP" and self.state != "ARM_ACTION":
                self.get_logger().error(
                    f"EMERGENCY STOP ({min_all:.2f}m) — deploying arm")
                self._set_state("ARM_ACTION") # Lock state to prevent loop
                self.send_arm_goal()
            self.cmd_pub.publish(twist)
            return

        # 3. TURN_COMMIT: we decided a direction — hold it
        if self.state == "TURN_COMMIT":
            elapsed = (self.get_clock().now() - self.turn_start_time).nanoseconds / 1e9
            if elapsed < self.TURN_COMMIT_SECS and min_all > 0.4:
                # Fix 2: Point turn (zero linear velocity) for efficient clearing
                twist.linear.x = 0.0
                twist.angular.z = float(self.turn_direction)
                self._log_vel(twist)
                self.cmd_pub.publish(twist)
                return
            else:
                self._set_state("FORWARD")

        # 4. YOLO Stop path
        if self.yolo_stop and self.state not in ("ARM_ACTION", "RECOVERY"):
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            self._set_state("FORWARD")   # stay ready but don't move
            self._log_vel(twist)
            self.cmd_pub.publish(twist)
            self.get_logger().warn("Stopped by YOLO context")
            return

        # 5. OBSTACLE DETECTED — decide turn direction
        # Fix 1: Check obstacles first, eliminating the dead zone
        if min_c <= 2.0 or min_r <= 1.5 or min_l <= 1.5:
            open_left  = min_l + min_far_l
            open_right = min_r + min_far_r

            if open_left >= open_right:
                self.turn_direction = 0.6   # left
                self.get_logger().warn(
                    f"obstacle ({min_c:.2f}m) — committing left "
                    f"(l:{open_left:.2f} r:{open_right:.2f})")
            else:
                self.turn_direction = -0.6  # right
                self.get_logger().warn(
                    f"obstacle ({min_c:.2f}m) — committing right "
                    f"(l:{open_left:.2f} r:{open_right:.2f})")

            self.turn_start_time = self.get_clock().now()
            self._set_state("TURN_COMMIT")
            
            twist.linear.x = 0.0 # Point turn init
            twist.angular.z = float(self.turn_direction)

        # 6. FORWARD: open path (Catch-all)
        else:
            speed = 0.25 if self.yolo_slow else 0.5
            twist.linear.x = float(speed)
            twist.angular.z = 0.0
            self._set_state("FORWARD")

        self._log_vel(twist)
        self.cmd_pub.publish(twist)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _set_state(self, new_state):
        if self.state != new_state:
            self.get_logger().info(f"state: {self.state} → {new_state}")
            self.state = new_state

    def _log_vel(self, twist):
        self.vel_linear.append(twist.linear.x)
        self.vel_angular.append(twist.angular.z)

    # ── arm ──────────────────────────────────────────────────────────────────
    def send_arm_goal(self):
        if not self.arm_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("arm server not available!")
            self._set_state("RECOVERY")
            self._start_recovery()
            return

        goal_msg = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = [
            'arm_0_shoulder_pan_joint', 'arm_0_shoulder_lift_joint',
            'arm_0_elbow_joint',        'arm_0_wrist_1_joint',
            'arm_0_wrist_2_joint',      'arm_0_wrist_3_joint'
        ]

        def pt(pos, sec):
            p = JointTrajectoryPoint()
            p.positions = pos
            p.time_from_start = Duration(sec=int(sec), nanosec=0)
            return p

        traj.points = [
            pt([0.0,  -1.57, 1.57, -1.57, -1.57, 0.0], 2),
            pt([0.78, -1.57, 1.57, -1.57, -1.57, 0.0], 4),
            pt([-0.78,-1.57, 1.57, -1.57, -1.57, 0.0], 6),
            pt([0.0,  -0.5,  0.5,  -1.57, -1.57, 0.0], 8),
            pt([0.0,  -1.57, 1.57, -1.57, -1.57, 0.0], 10),
        ]
        goal_msg.trajectory = traj
        future = self.arm_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("arm goal rejected!")
            self._set_state("RECOVERY")
            self._start_recovery()
            return
        self.get_logger().info("arm goal accepted, executing...")
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.get_logger().info("arm complete — starting recovery backup")
        self._set_state("RECOVERY")
        self._start_recovery()

    def _start_recovery(self):
        self._recovery_count = 0
        if self._recovery_timer is not None:
            try:
                self._recovery_timer.cancel()
                self._recovery_timer.destroy()
            except Exception:
                pass
        self._recovery_timer = self.create_timer(0.1, self._recovery_step)

    def _recovery_step(self):
        self._recovery_count += 1
        twist = Twist()
        if self._recovery_count <= 20:        # 2 s back
            twist.linear.x = -0.25
        elif self._recovery_count <= 30:      # 1 s rotate to open side
            twist.angular.z = 0.6
        else:
            self._recovery_timer.cancel()
            self._recovery_timer.destroy()
            self._recovery_timer = None
            self._set_state("FORWARD")
            self.get_logger().info("recovery complete — resuming")
            return
        self.cmd_pub.publish(twist)

    # ── graph saving ──────────────────────────────────────────────────────────
    def save_all_graphs(self):
        if len(self.timestamps) < 10:
            return
        self._save_distance_graph()
        self._save_state_timeline()
        self._save_velocity_graph()
        self._save_sector_graph()
        self._save_path_graph()
        self.get_logger().info(f"graphs saved to {self.log_dir}")

    def _save_distance_graph(self):
        fig, ax = plt.subplots(figsize=(10, 4))
        t = self.timestamps
        ax.plot(t, self.distances, color='steelblue', linewidth=1.2,
                label='min distance (all sectors)')
        ax.axhline(0.5, color='red',    linestyle='--', linewidth=1, label='emergency (0.5m)')
        ax.axhline(2.0, color='orange', linestyle='--', linewidth=1, label='caution (2.0m)')
        ax.set_xlabel("time (s)")
        ax.set_ylabel("distance (m)")
        ax.set_title("closest obstacle distance over time")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "01_distance.png"), dpi=120)
        plt.close(fig)

    def _save_state_timeline(self):
        if len(self.state_log) < 2:
            return
        fig, ax = plt.subplots(figsize=(10, 3))
        times  = [s[0] for s in self.state_log]
        states = [s[1] for s in self.state_log]
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#3498db']
        labels = list(self.STATE_MAP.keys())
        for i in range(len(times) - 1):
            ax.barh(0, times[i+1] - times[i], left=times[i],
                    color=colors[states[i]], height=0.5, align='center')
        patches = [mpatches.Patch(color=colors[i], label=labels[i])
                   for i in range(len(labels))]
        ax.legend(handles=patches, loc='upper right', fontsize=8)
        ax.set_xlabel("time (s)")
        ax.set_yticks([])
        ax.set_title("robot state timeline")
        ax.grid(axis='x', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "02_state_timeline.png"), dpi=120)
        plt.close(fig)

    def _save_velocity_graph(self):
        n = min(len(self.vel_linear), len(self.timestamps))
        if n < 2:
            return
        t = self.timestamps[:n]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        ax1.plot(t, self.vel_linear[:n], color='#2980b9', linewidth=1.2)
        ax1.set_ylabel("linear vel (m/s)")
        ax1.axhline(0, color='gray', linewidth=0.8)
        ax1.grid(True, alpha=0.3)
        ax2.plot(t, self.vel_angular[:n], color='#e67e22', linewidth=1.2)
        ax2.set_ylabel("angular vel (rad/s)")
        ax2.set_xlabel("time (s)")
        ax2.axhline(0, color='gray', linewidth=0.8)
        ax2.grid(True, alpha=0.3)
        fig.suptitle("command velocities over time")
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "03_velocity.png"), dpi=120)
        plt.close(fig)

    def _save_sector_graph(self):
        n = min(len(self.sector_center), len(self.timestamps))
        if n < 2:
            return
        t = self.timestamps[:n]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, self.sector_right[:n],  color='#e74c3c', linewidth=1,
                label='right sector', alpha=0.8)
        ax.plot(t, self.sector_center[:n], color='#2ecc71', linewidth=1.4,
                label='center sector')
        ax.plot(t, self.sector_left[:n],   color='#3498db', linewidth=1,
                label='left sector', alpha=0.8)
        ax.axhline(0.5, color='black', linestyle=':', linewidth=1, label='emergency')
        ax.set_xlabel("time (s)")
        ax.set_ylabel("distance (m)")
        ax.set_title("lidar sector distances over time")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 5)
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "04_sectors.png"), dpi=120)
        plt.close(fig)

    def _save_path_graph(self):
        if len(self.path_x) < 2:
            return
        fig, ax = plt.subplots(figsize=(6, 6))
        # color path by time (older = faded)
        points = np.array([self.path_x, self.path_y]).T.reshape(-1, 1, 2)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        from matplotlib.collections import LineCollection
        from matplotlib.cm import get_cmap
        cmap = get_cmap('plasma')
        n = len(segs)
        colors = [cmap(i / max(n, 1)) for i in range(n)]
        lc = LineCollection(segs, colors=colors, linewidth=1.5)
        ax.add_collection(lc)
        ax.plot(self.path_x[0],  self.path_y[0],  'go', markersize=8, label='start')
        ax.plot(self.path_x[-1], self.path_y[-1],  'rs', markersize=8, label='current')
        ax.set_xlim(min(self.path_x)-1, max(self.path_x)+1)
        ax.set_ylim(min(self.path_y)-1, max(self.path_y)+1)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("robot path (odometry)")
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "05_path.png"), dpi=120)
        plt.close(fig)

    def destroy_node(self):
        self.get_logger().info("shutting down — saving final graphs...")
        self.save_all_graphs()
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
