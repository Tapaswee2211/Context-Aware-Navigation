# smart_nav_node.py
# Location: ~/clearpath_ws/src/context_aware_nav/context_aware_nav/smart_demo.py
#
# ╔══════════════════════════════════════════════════════════════╗
# ║  PARAMETERS TO TUNE FOR YOUR SIMULATION — all in one place  ║
# ╚══════════════════════════════════════════════════════════════╝

# ── Navigation ──────────────────────────────────────────────────────────────
FORWARD_SPEED      = 0.45    # m/s  — cruise speed on clear path
TURN_SPEED         = 0.60    # rad/s — angular speed when avoiding
SLOW_SPEED         = 0.15    # m/s  — speed near obstacles
TURN_COMMIT_SECS   = 2.0     # s    — how long to hold a turn before re-evaluating
OBSTACLE_CAUTION   = 2.0     # m    — start avoiding at this distance
OBSTACLE_SIDE      = 1.5     # m    — side sector alert distance
EMERGENCY_DIST     = 0.45    # m    — hard stop distance
GOAL_TOLERANCE     = 0.50    # m    — how close = "arrived at waypoint"
BLOCKED_TIMEOUT    = 3.0     # s    — seconds stuck before arm deploys

# ── LiDAR rear mask ─────────────────────────────────────────────────────────
# Your Hokuyo UST has a 270° scan. The arm is directly behind the LiDAR.
# Rays are indexed 0..N-1, left-to-right when facing forward.
# Mask the rear ~60° (±30° from directly behind) to ignore arm self-hits.
# If you see phantom obstacles, widen this (e.g. 0.30).
REAR_MASK_FRACTION = 0.22    # fraction of total rays to blank each side edge

# ── Waypoints — ADD YOUR OWN (x, y) in metres, robot-frame origin = spawn ──
WAYPOINTS = [
    ( 3.0,  0.0),   # waypoint 1
    ( 3.0,  3.0),   # waypoint 2
    ( 0.0,  3.0),   # waypoint 3
    (5.0, 5.0),
    ( 0.0,  0.0),   # return home
]

# ── Arm joint positions (radians) ────────────────────────────────────────────
# Format: [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
# Measure these in your sim with:  ros2 topic echo /cpr_a200_0000/platform/joint_states
STOW_POS       = [ 0.00, -2.80,  2.40, -1.57, -1.57,  0.00]  # arm folded tight
HOME_READY_POS = [ 0.00, -1.57,  1.57, -1.57, -1.57,  0.00]  # upright, centred
LOOK_LEFT_POS  = [ 1.20, -1.57,  1.57, -1.57, -1.57,  0.00]
LOOK_RIGHT_POS = [-1.20, -1.57,  1.57, -1.57, -1.57,  0.00]
POINT_FWD_POS  = [ 0.00, -0.40,  0.40, -1.57, -1.57,  0.00]
SALUTE_1_POS   = [ 0.00, -1.00,  0.80, -2.20, -1.57,  0.00]  # wind-up
SALUTE_2_POS   = [ 0.50, -0.60,  0.50, -1.80, -1.57,  1.57]  # sweep right
SALUTE_3_POS   = [-0.50, -0.60,  0.50, -1.80, -1.57, -1.57]  # sweep left
SALUTE_4_POS   = [ 0.00, -0.30,  0.20, -1.57, -1.57,  0.00]  # raise high
SALUTE_5_POS   = [ 0.00, -1.57,  1.57, -1.57, -1.57,  0.00]  # back to ready
GRASP_OPEN_POS = [ 0.00, -0.50,  0.80, -0.80, -1.57,  0.00]  # arm extended, ready
GRASP_CLOSE_POS= [ 0.00, -0.50,  0.80, -0.80, -1.57,  1.57]  # wrist rotate = close

# ──────────────────────────────────────────────────────────────────────────────

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
import math
import time
import os


# ─────────────────────────────────────────────────────────────────────────────
#  State machine states
# ─────────────────────────────────────────────────────────────────────────────
class State:
    STARTUP          = "STARTUP"           # arm grand salute, then stow
    NAVIGATING       = "NAVIGATING"        # driving toward current waypoint
    TURN_COMMIT      = "TURN_COMMIT"       # holding a turn for TURN_COMMIT_SECS
    BLOCKED          = "BLOCKED"           # stopped, counting blocked time
    ARM_BLOCKED      = "ARM_BLOCKED"       # arm deploy due to being blocked
    RECOVERY         = "RECOVERY"          # backing up after arm/emergency
    WAYPOINT_ARRIVED = "WAYPOINT_ARRIVED"  # at waypoint, running arm sequence
    MISSION_COMPLETE = "MISSION_COMPLETE"  # all waypoints done


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: build a JointTrajectoryPoint
# ─────────────────────────────────────────────────────────────────────────────
def _pt(positions, sec, nanosec=0):
    p = JointTrajectoryPoint()
    p.positions = [float(x) for x in positions]
    p.time_from_start = Duration(sec=int(sec), nanosec=int(nanosec))
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  Main node
# ─────────────────────────────────────────────────────────────────────────────
class SmartNavNode(Node):

    JOINT_NAMES = [
        'arm_0_shoulder_pan_joint',
        'arm_0_shoulder_lift_joint',
        'arm_0_elbow_joint',
        'arm_0_wrist_1_joint',
        'arm_0_wrist_2_joint',
        'arm_0_wrist_3_joint',
    ]

    STATE_INT = {
        State.STARTUP:          0,
        State.NAVIGATING:       1,
        State.TURN_COMMIT:      2,
        State.BLOCKED:          3,
        State.ARM_BLOCKED:      4,
        State.RECOVERY:         5,
        State.WAYPOINT_ARRIVED: 6,
        State.MISSION_COMPLETE: 7,
    }
    STATE_COLORS = [
        '#95a5a6',  # STARTUP       — gray
        '#2ecc71',  # NAVIGATING    — green
        '#f39c12',  # TURN_COMMIT   — amber
        '#e67e22',  # BLOCKED       — orange
        '#9b59b6',  # ARM_BLOCKED   — purple
        '#3498db',  # RECOVERY      — blue
        '#1abc9c',  # ARRIVED       — teal
        '#e74c3c',  # COMPLETE      — red
    ]

    def __init__(self):
        super().__init__('smart_nav_node')

        # ── Publishers / Subscribers ─────────────────────────────────────────
        self.cmd_pub = self.create_publisher(
            Twist, '/cpr_a200_0000/cmd_vel', 10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/cpr_a200_0000/sensors/lidar2d_0/scan',
            self._scan_cb, 10)

        self.odom_sub = self.create_subscription(
            Odometry,
            '/cpr_a200_0000/platform/odom/filtered',
            self._odom_cb, 10)

        # ── Arm action client ────────────────────────────────────────────────
        self.arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/cpr_a200_0000/arm_0_joint_trajectory_controller/'
            'follow_joint_trajectory')

        # ── Gripper — uses same controller, wrist_3 rotation as proxy ────────
        # If you have a real gripper controller, swap the topic here:
        # self.gripper_client = ActionClient(self, FollowJointTrajectory,
        #     '/cpr_a200_0000/arm_0_gripper_controller/follow_joint_trajectory')

        # ── State machine ────────────────────────────────────────────────────
        self.state           = State.STARTUP
        self.waypoint_idx    = 0
        self.turn_direction  = 0.0
        self.turn_start_time = None
        self.blocked_since   = None   # time we first got stuck
        self._arm_busy       = False  # True while a trajectory is executing

        # ── Odometry ─────────────────────────────────────────────────────────
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0

        # ── Recovery timer ───────────────────────────────────────────────────
        self._recovery_count = 0
        self._recovery_timer = None

        # ── Logging ──────────────────────────────────────────────────────────
        self.log_dir = os.path.expanduser("~/clearpath_ws/logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.t0            = time.time()
        self.timestamps    = []
        self.distances     = []
        self.vel_linear    = []
        self.vel_angular   = []
        self.sector_right  = []
        self.sector_center = []
        self.sector_left   = []
        self.state_log     = []   # (time, state_int)
        self.path_x        = []
        self.path_y        = []
        self.waypoint_times = []  # times when each waypoint was reached

        self.graph_timer = self.create_timer(30.0, self.save_all_graphs)

        self.get_logger().info(
            f"SmartNav started — {len(WAYPOINTS)} waypoints loaded")
        self.get_logger().info(
            f"Waypoints: {WAYPOINTS}")

        # Kick off the startup arm sequence
        self._run_startup_sequence()

    # ─────────────────────────────────────────────────────────────────────────
    #  Odometry
    # ─────────────────────────────────────────────────────────────────────────
    def _odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny, cosy)
        self.path_x.append(self.robot_x)
        self.path_y.append(self.robot_y)

    # ─────────────────────────────────────────────────────────────────────────
    #  LiDAR scan callback — main decision loop
    # ─────────────────────────────────────────────────────────────────────────
    def _scan_cb(self, msg):
        now = time.time() - self.t0

        # ── Apply rear mask to ignore arm self-detections ─────────────────
        ranges = list(msg.ranges)
        n      = len(ranges)
        mask_n = int(n * REAR_MASK_FRACTION)
        # Zero out the first and last mask_n rays (rear arc of Hokuyo 270° scan)
        for i in range(mask_n):
            ranges[i]         = float('inf')
            ranges[n-1-i]     = float('inf')

        # ── Sector extraction ─────────────────────────────────────────────
        center     = n // 2
        slice_size = max(1, n // 36)   # ≈10° per slice

        def get_min(sector_slice):
            valid = [r for r in sector_slice if 0.15 < r < 10.0]
            return min(valid) if valid else 10.0

        c_sec   = ranges[center - slice_size     : center + slice_size]
        r_sec   = ranges[center - 3*slice_size   : center - slice_size]
        l_sec   = ranges[center + slice_size     : center + 3*slice_size]
        far_r   = ranges[center - 5*slice_size   : center - 3*slice_size]
        far_l   = ranges[center + 3*slice_size   : center + 5*slice_size]

        min_c     = get_min(c_sec)
        min_r     = get_min(r_sec)
        min_l     = get_min(l_sec)
        min_far_r = get_min(far_r)
        min_far_l = get_min(far_l)
        min_all   = min(min_c, min_r, min_l)

        # ── Logging ───────────────────────────────────────────────────────
        self.timestamps.append(now)
        self.distances.append(min_all)
        self.sector_center.append(min_c)
        self.sector_right.append(min_r)
        self.sector_left.append(min_l)
        self.state_log.append((now, self.STATE_INT.get(self.state, 0)))

        twist = Twist()

        # ── States that block scan-driven motion ──────────────────────────
        if self.state in (State.STARTUP,
                          State.ARM_BLOCKED,
                          State.RECOVERY,
                          State.WAYPOINT_ARRIVED,
                          State.MISSION_COMPLETE):
            self._pub(twist)
            return

        # ── Mission complete guard ────────────────────────────────────────
        if self.waypoint_idx >= len(WAYPOINTS):
            self._set_state(State.MISSION_COMPLETE)
            self._pub(twist)
            self.get_logger().info("All waypoints complete — mission done!")
            self.save_all_graphs()
            return

        # ── Check arrival at current waypoint ─────────────────────────────
        wp   = WAYPOINTS[self.waypoint_idx]
        dist = math.hypot(wp[0] - self.robot_x, wp[1] - self.robot_y)

        if dist < GOAL_TOLERANCE:
            self.get_logger().info(
                f"Arrived at waypoint {self.waypoint_idx+1}/{len(WAYPOINTS)} "
                f"({wp[0]:.1f}, {wp[1]:.1f})")
            self.waypoint_times.append(now)
            self._set_state(State.WAYPOINT_ARRIVED)
            self._pub(twist)
            self._run_waypoint_sequence()
            return

        # ── Emergency stop ────────────────────────────────────────────────
        if min_all < EMERGENCY_DIST:
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            if self.state != State.BLOCKED:
                self.blocked_since = time.time()
            self._set_state(State.BLOCKED)
            # Check if blocked long enough to deploy arm
            if (self.blocked_since is not None and
                    time.time() - self.blocked_since > BLOCKED_TIMEOUT and
                    not self._arm_busy):
                self.get_logger().warn(
                    f"Blocked for >{BLOCKED_TIMEOUT}s — deploying arm")
                self._set_state(State.ARM_BLOCKED)
                self._run_blocked_sequence()
            self._log_vel(twist)
            self._pub(twist)
            return

        # If we were blocked but now have space, reset blocked timer
        if min_all >= EMERGENCY_DIST and self.state == State.BLOCKED:
            self.blocked_since = None
            self._set_state(State.NAVIGATING)

        # ── TURN_COMMIT: hold the decided turn ────────────────────────────
        if self.state == State.TURN_COMMIT:
            elapsed = time.time() - self.turn_start_time
            if elapsed < TURN_COMMIT_SECS and min_all > EMERGENCY_DIST + 0.1:
                speed = float(np.clip(SLOW_SPEED * (min_all / 1.0),
                                      0.04, SLOW_SPEED))
                twist.linear.x  = speed
                twist.angular.z = float(self.turn_direction * TURN_SPEED)
                self._log_vel(twist)
                self._pub(twist)
                return
            else:
                self.blocked_since = None
                self._set_state(State.NAVIGATING)

        # ── NAVIGATING ────────────────────────────────────────────────────
        # Goal-directed: bias turn toward waypoint when path is clear
        angle_to_wp = self._angle_to_waypoint()

        if min_all > OBSTACLE_CAUTION:
            # Path clear — head toward waypoint
            steer = float(np.clip(1.5 * angle_to_wp, -TURN_SPEED, TURN_SPEED))
            twist.linear.x  = FORWARD_SPEED
            twist.angular.z = steer
            self._set_state(State.NAVIGATING)
            self.blocked_since = None

        elif min_c <= OBSTACLE_CAUTION or min_r <= OBSTACLE_SIDE or min_l <= OBSTACLE_SIDE:
            # Obstacle — pick most open side, weighted by waypoint direction
            open_left  = min_l + min_far_l
            open_right = min_r + min_far_r

            # Bonus for the side the waypoint is on
            if angle_to_wp > 0:
                open_left  += 0.5
            else:
                open_right += 0.5

            if open_left >= open_right:
                self.turn_direction = TURN_SPEED
                side = "LEFT"
            else:
                self.turn_direction = -TURN_SPEED
                side = "RIGHT"

            self.get_logger().warn(
                f"Obstacle c={min_c:.2f}m — committing {side} "
                f"(L:{open_left:.2f} R:{open_right:.2f})")

            speed = float(np.clip(
                SLOW_SPEED * (min_c / OBSTACLE_CAUTION), 0.0, SLOW_SPEED))
            twist.linear.x  = speed
            twist.angular.z = float(self.turn_direction)
            self.turn_start_time = time.time()
            self._set_state(State.TURN_COMMIT)

            # Start blocked timer only if nearly stopped
            if speed < 0.05:
                if self.blocked_since is None:
                    self.blocked_since = time.time()
            else:
                self.blocked_since = None

        self._log_vel(twist)
        self._pub(twist)

    # ─────────────────────────────────────────────────────────────────────────
    #  Navigation helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _angle_to_waypoint(self):
        """Signed angle (rad) from robot heading to current waypoint."""
        if self.waypoint_idx >= len(WAYPOINTS):
            return 0.0
        wp    = WAYPOINTS[self.waypoint_idx]
        dx    = wp[0] - self.robot_x
        dy    = wp[1] - self.robot_y
        goal_yaw = math.atan2(dy, dx)
        angle    = goal_yaw - self.robot_yaw
        return float((angle + math.pi) % (2 * math.pi) - math.pi)

    def _pub(self, twist):
        self.cmd_pub.publish(twist)

    def _log_vel(self, twist):
        self.vel_linear.append(twist.linear.x)
        self.vel_angular.append(twist.angular.z)

    def _set_state(self, new_state):
        if self.state != new_state:
            self.get_logger().info(f"  state: {self.state} → {new_state}")
            self.state = new_state

    # ─────────────────────────────────────────────────────────────────────────
    #  Arm trajectory sender
    # ─────────────────────────────────────────────────────────────────────────
    def _send_trajectory(self, points, done_callback):
        """
        points: list of (positions_list, time_sec)
        done_callback: called when trajectory completes
        """
        if not self.arm_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("Arm server not available!")
            done_callback()
            return

        goal  = FollowJointTrajectory.Goal()
        traj  = JointTrajectory()
        traj.joint_names = self.JOINT_NAMES
        traj.points = [_pt(pos, sec) for pos, sec in points]
        goal.trajectory = traj

        self._arm_busy = True
        future = self.arm_client.send_goal_async(goal)
        future.add_done_callback(
            lambda f: self._on_goal_accepted(f, done_callback))

    def _on_goal_accepted(self, future, done_callback):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("Arm goal rejected!")
            self._arm_busy = False
            done_callback()
            return
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._on_trajectory_done(f, done_callback))

    def _on_trajectory_done(self, future, done_callback):
        self._arm_busy = False
        self.get_logger().info("Arm trajectory complete")
        done_callback()

    # ─────────────────────────────────────────────────────────────────────────
    #  ARM SEQUENCE 1 — Grand Startup Salute
    #  Plays ONCE before navigation begins
    # ─────────────────────────────────────────────────────────────────────────
    def _run_startup_sequence(self):
        self.get_logger().info("=== STARTUP SEQUENCE — Grand Salute ===")
        # fmt: off
        points = [
            # Rise to ready
            (HOME_READY_POS,  2),
            # Wind up — pull back dramatically
            (SALUTE_1_POS,    4),
            # Sweep right — grand arc
            (SALUTE_2_POS,    6),
            # Sweep left — other side
            (SALUTE_3_POS,    8),
            # Raise high — triumphant
            (SALUTE_4_POS,   10),
            # Back to ready
            (SALUTE_5_POS,   12),
            # Point forward — "let's go"
            (POINT_FWD_POS,  14),
            # STOW before driving — critical for LiDAR clearance
            (STOW_POS,       17),
        ]
        # fmt: on
        self._send_trajectory(points, self._startup_done)

    def _startup_done(self):
        self.get_logger().info("Startup complete — beginning navigation")
        self._set_state(State.NAVIGATING)

    # ─────────────────────────────────────────────────────────────────────────
    #  ARM SEQUENCE 2 — Waypoint Arrived: Area Scan + Gripper Demo
    # ─────────────────────────────────────────────────────────────────────────
    def _run_waypoint_sequence(self):
        wp_num = self.waypoint_idx + 1
        is_last = (self.waypoint_idx == len(WAYPOINTS) - 1)
        self.get_logger().info(
            f"=== WAYPOINT {wp_num} SEQUENCE "
            f"{'(FINAL GOAL)' if is_last else ''} ===")

        # fmt: off
        points = [
            # Rise to scanning position
            (HOME_READY_POS,   2),
            # Scan left
            (LOOK_LEFT_POS,    4),
            # Scan right
            (LOOK_RIGHT_POS,   7),
            # Back to centre
            (HOME_READY_POS,   9),
            # Lower arm into grasp-ready position
            (GRASP_OPEN_POS,  11),
            # Simulate grasp — close gripper (wrist_3 rotate)
            (GRASP_CLOSE_POS, 13),
            # Open again
            (GRASP_OPEN_POS,  15),
            # Close again — second grasp
            (GRASP_CLOSE_POS, 17),
            # Open and retract
            (GRASP_OPEN_POS,  19),
            # Point upward if final goal, else ready
            (SALUTE_4_POS if is_last else HOME_READY_POS, 21),
            # Stow for next navigation leg
            (STOW_POS,        24),
        ]
        # fmt: on
        self._send_trajectory(points,
            lambda: self._waypoint_sequence_done(is_last))

    def _waypoint_sequence_done(self, is_last):
        if is_last:
            self.get_logger().info(
                "=== MISSION COMPLETE — all waypoints visited ===")
            self._set_state(State.MISSION_COMPLETE)
            self.save_all_graphs()
        else:
            self.waypoint_idx += 1
            self.get_logger().info(
                f"Moving to waypoint {self.waypoint_idx+1}: "
                f"{WAYPOINTS[self.waypoint_idx]}")
            self.blocked_since = None
            self._set_state(State.NAVIGATING)

    # ─────────────────────────────────────────────────────────────────────────
    #  ARM SEQUENCE 3 — Blocked Sequence (frustrated scan)
    # ─────────────────────────────────────────────────────────────────────────
    def _run_blocked_sequence(self):
        self.get_logger().warn("=== BLOCKED SEQUENCE — frustrated scan ===")
        # fmt: off
        points = [
            (HOME_READY_POS,   2),
            (LOOK_LEFT_POS,    4),
            (LOOK_RIGHT_POS,   7),
            (HOME_READY_POS,   9),
            (STOW_POS,        11),
        ]
        # fmt: on
        self._send_trajectory(points, self._blocked_sequence_done)

    def _blocked_sequence_done(self):
        self.get_logger().info("Blocked sequence done — starting recovery")
        self._set_state(State.RECOVERY)
        self.blocked_since = None
        self._start_recovery()

    # ─────────────────────────────────────────────────────────────────────────
    #  Recovery: back up then rotate to open side
    # ─────────────────────────────────────────────────────────────────────────
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

        if self._recovery_count <= 25:       # 2.5 s reverse
            twist.linear.x  = -0.25
            twist.angular.z =  0.0
        elif self._recovery_count <= 40:     # 1.5 s rotate toward open side
            twist.linear.x  =  0.0
            twist.angular.z =  0.7   # rotate left by default after block
        else:
            self._recovery_timer.cancel()
            self._recovery_timer.destroy()
            self._recovery_timer = None
            self.blocked_since = None
            self._set_state(State.NAVIGATING)
            self.get_logger().info("Recovery complete — resuming navigation")
            return

        self.cmd_pub.publish(twist)

    # ─────────────────────────────────────────────────────────────────────────
    #  Graph saving
    # ─────────────────────────────────────────────────────────────────────────
    def save_all_graphs(self):
        if len(self.timestamps) < 10:
            return
        self.get_logger().info(f"Saving graphs to {self.log_dir}...")
        self._graph_distance()
        self._graph_state_timeline()
        self._graph_velocity()
        self._graph_sectors()
        self._graph_path()
        self.get_logger().info("All graphs saved.")

    def _graph_distance(self):
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(self.timestamps, self.distances,
                color='steelblue', lw=1.2, label='Min distance (all sectors)')
        ax.axhline(EMERGENCY_DIST, color='red',    ls='--', lw=1,
                   label=f'Emergency ({EMERGENCY_DIST}m)')
        ax.axhline(OBSTACLE_CAUTION, color='orange', ls='--', lw=1,
                   label=f'Caution ({OBSTACLE_CAUTION}m)')
        for t in self.waypoint_times:
            ax.axvline(t, color='green', ls=':', lw=1.2, alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (m)")
        ax.set_title("Closest Obstacle Distance Over Time")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "01_distance.png"), dpi=130)
        plt.close(fig)

    def _graph_state_timeline(self):
        if len(self.state_log) < 2:
            return
        fig, ax = plt.subplots(figsize=(11, 2.5))
        times  = [s[0] for s in self.state_log]
        states = [s[1] for s in self.state_log]
        for i in range(len(times) - 1):
            ax.barh(0, times[i+1]-times[i], left=times[i],
                    color=self.STATE_COLORS[states[i]],
                    height=0.6, align='center')
        labels  = list(self.STATE_INT.keys())
        patches = [mpatches.Patch(color=self.STATE_COLORS[i], label=labels[i])
                   for i in range(len(labels))]
        ax.legend(handles=patches, loc='upper right',
                  fontsize=7, ncol=4)
        ax.set_xlabel("Time (s)")
        ax.set_yticks([])
        ax.set_title("Robot State Timeline")
        ax.grid(axis='x', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "02_state_timeline.png"), dpi=130)
        plt.close(fig)

    def _graph_velocity(self):
        n = min(len(self.vel_linear), len(self.timestamps))
        if n < 2:
            return
        t = self.timestamps[:n]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        ax1.plot(t, self.vel_linear[:n],  color='#2980b9', lw=1.2)
        ax1.set_ylabel("Linear vel (m/s)")
        ax1.axhline(0, color='gray', lw=0.8)
        ax1.grid(True, alpha=0.3)
        ax2.plot(t, self.vel_angular[:n], color='#e67e22', lw=1.2)
        ax2.set_ylabel("Angular vel (rad/s)")
        ax2.set_xlabel("Time (s)")
        ax2.axhline(0, color='gray', lw=0.8)
        ax2.grid(True, alpha=0.3)
        fig.suptitle("Command Velocities Over Time")
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "03_velocity.png"), dpi=130)
        plt.close(fig)

    def _graph_sectors(self):
        n = min(len(self.sector_center), len(self.timestamps))
        if n < 2:
            return
        t = self.timestamps[:n]
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(t, self.sector_right[:n],  color='#e74c3c', lw=1,
                alpha=0.8, label='Right')
        ax.plot(t, self.sector_center[:n], color='#2ecc71', lw=1.4,
                label='Center')
        ax.plot(t, self.sector_left[:n],   color='#3498db', lw=1,
                alpha=0.8, label='Left')
        ax.axhline(EMERGENCY_DIST,  color='red',    ls=':', lw=1)
        ax.axhline(OBSTACLE_CAUTION, color='orange', ls=':', lw=1)
        for t_wp in self.waypoint_times:
            ax.axvline(t_wp, color='green', ls=':', lw=1.2, alpha=0.7)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (m)")
        ax.set_title("LiDAR Sector Distances Over Time")
        ax.legend(loc='upper right')
        ax.set_ylim(0, 5)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "04_sectors.png"), dpi=130)
        plt.close(fig)

    def _graph_path(self):
        if len(self.path_x) < 2:
            return
        fig, ax = plt.subplots(figsize=(7, 7))
        pts  = np.array([self.path_x, self.path_y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs,
                              cmap='plasma',
                              linewidth=1.5,
                              array=np.linspace(0, 1, len(segs)))
        ax.add_collection(lc)
        # Mark waypoints
        for i, (wx, wy) in enumerate(WAYPOINTS):
            ax.plot(wx, wy, 'g^', markersize=10)
            ax.annotate(f"WP{i+1}", (wx, wy),
                        textcoords="offset points",
                        xytext=(6, 4), fontsize=8)
        ax.plot(self.path_x[0],  self.path_y[0],
                'go', markersize=10, label='Start')
        ax.plot(self.path_x[-1], self.path_y[-1],
                'rs', markersize=10, label='Current')
        margin = 1.0
        ax.set_xlim(min(self.path_x + [w[0] for w in WAYPOINTS]) - margin,
                    max(self.path_x + [w[0] for w in WAYPOINTS]) + margin)
        ax.set_ylim(min(self.path_y + [w[1] for w in WAYPOINTS]) - margin,
                    max(self.path_y + [w[1] for w in WAYPOINTS]) + margin)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Robot Path with Waypoints (odometry)")
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.colorbar(lc, ax=ax, label='Time progression')
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "05_path.png"), dpi=130)
        plt.close(fig)

    def destroy_node(self):
        self.get_logger().info("Shutting down — saving final graphs...")
        self.save_all_graphs()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = SmartNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
