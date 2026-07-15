# smart_nav_vision.py
# Location: ~/clearpath_ws/src/context_aware_nav/context_aware_nav/smart_nav_node.py
#
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║           PARAMETERS — edit these for your simulation                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Navigation ───────────────────────────────────────────────────────────────
FORWARD_SPEED     = 0.45   # m/s   cruise speed on clear path
FORWARD_SPEED_SLOW= 0.25   # m/s   cruise speed when YOLO says slow
TURN_SPEED        = 0.65   # rad/s angular speed during committed turn
TURN_COMMIT_SECS  = 2.2    # s     how long to hold a turn before re-evaluating
OBSTACLE_CAUTION  = 2.0    # m     start avoiding at this distance
OBSTACLE_SIDE     = 1.5    # m     side-sector alert distance
EMERGENCY_DIST    = 0.45   # m     hard stop / arm-deploy threshold
GOAL_TOLERANCE    = 0.50   # m     arrival radius for each waypoint
BLOCKED_TIMEOUT   = 3.0    # s     seconds blocked before arm frustration deploy

# ── LiDAR rear mask ──────────────────────────────────────────────────────────
# Hokuyo UST = 270° scan.  Arm sits behind the LiDAR on the same top plate.
# Mask the outermost rays (the rear arc) so the arm isn't seen as an obstacle.
# Increase to 0.28 if phantom detections persist; decrease to 0.16 if you lose
# too much side coverage.
REAR_MASK_FRACTION = 0.22

# ── Waypoints  (x, y) in metres from spawn origin ────────────────────────────
# Check real coordinates with: ros2 topic echo /cpr_a200_0000/platform/odom/filtered
WAYPOINTS = [
    ( 3.0,  0.0),
    ( 3.0,  3.0),
    ( 0.0,  3.0),
    ( 4.0,  2.0),
    ( 0.0,  0.0),   # return home
]

# ── Arm joint positions  [pan, lift, elbow, wrist1, wrist2, wrist3] radians ──
# Verify in sim: ros2 topic echo /cpr_a200_0000/platform/joint_states
STOW_POS        = [ 0.00, -2.80,  2.40, -1.57, -1.57,  0.00]  # tight fold, clears LiDAR
HOME_READY_POS  = [ 0.00, -1.57,  1.57, -1.57, -1.57,  0.00]  # upright centre
LOOK_LEFT_POS   = [ 1.20, -1.57,  1.57, -1.57, -1.57,  0.00]
LOOK_RIGHT_POS  = [-1.20, -1.57,  1.57, -1.57, -1.57,  0.00]
POINT_FWD_POS   = [ 0.00, -0.40,  0.40, -1.57, -1.57,  0.00]
SALUTE_1_POS    = [ 0.00, -1.00,  0.80, -2.20, -1.57,  0.00]  # wind-up
SALUTE_2_POS    = [ 0.50, -0.60,  0.50, -1.80, -1.57,  1.57]  # sweep right
SALUTE_3_POS    = [-0.50, -0.60,  0.50, -1.80, -1.57, -1.57]  # sweep left
SALUTE_4_POS    = [ 0.00, -0.30,  0.20, -1.57, -1.57,  0.00]  # raise high
SALUTE_5_POS    = [ 0.00, -1.57,  1.57, -1.57, -1.57,  0.00]  # back to ready
GRASP_OPEN_POS  = [ 0.00, -0.50,  0.80, -0.80, -1.57,  0.00]  # extended ready
GRASP_CLOSE_POS = [ 0.00, -0.50,  0.80, -0.80, -1.57,  1.57]  # wrist rotate = close

# ── Vision logging ────────────────────────────────────────────────────────────
# Save an annotated camera frame every N seconds during navigation
VISION_LOG_INTERVAL = 5.0   # seconds between saved frames
# Max detection frames to keep per waypoint arrival event
MAX_FRAMES_PER_EVENT = 6

# ─────────────────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
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
import json
import csv

try:
    import cv2
    from cv_bridge import CvBridge
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
#  State constants
# ─────────────────────────────────────────────────────────────────────────────
class S:
    STARTUP          = "STARTUP"
    NAVIGATING       = "NAVIGATING"
    TURN_COMMIT      = "TURN_COMMIT"
    BLOCKED          = "BLOCKED"
    ARM_BLOCKED      = "ARM_BLOCKED"
    RECOVERY         = "RECOVERY"
    WAYPOINT_ARRIVED = "WAYPOINT_ARRIVED"
    MISSION_COMPLETE = "MISSION_COMPLETE"

    INT = {
        STARTUP: 0, NAVIGATING: 1, TURN_COMMIT: 2, BLOCKED: 3,
        ARM_BLOCKED: 4, RECOVERY: 5, WAYPOINT_ARRIVED: 6, MISSION_COMPLETE: 7,
    }
    COLORS = [
        '#95a5a6',  # STARTUP
        '#2ecc71',  # NAVIGATING
        '#f39c12',  # TURN_COMMIT
        '#e67e22',  # BLOCKED
        '#9b59b6',  # ARM_BLOCKED
        '#3498db',  # RECOVERY
        '#1abc9c',  # WAYPOINT_ARRIVED
        '#e74c3c',  # MISSION_COMPLETE
    ]


def _pt(positions, sec):
    """Build a JointTrajectoryPoint."""
    p = JointTrajectoryPoint()
    p.positions = [float(x) for x in positions]
    p.time_from_start = Duration(sec=int(sec), nanosec=0)
    return p


# ─────────────────────────────────────────────────────────────────────────────
#  Node
# ─────────────────────────────────────────────────────────────────────────────
class SmartNavNode(Node):

    JOINT_NAMES = [
        'arm_0_shoulder_pan_joint', 'arm_0_shoulder_lift_joint',
        'arm_0_elbow_joint',        'arm_0_wrist_1_joint',
        'arm_0_wrist_2_joint',      'arm_0_wrist_3_joint',
    ]

    def __init__(self):
        super().__init__('smart_nav_node')

        # ── ROS interfaces ───────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(
            Twist, '/cpr_a200_0000/cmd_vel', 10)

        self.create_subscription(
            LaserScan, '/cpr_a200_0000/sensors/lidar2d_0/scan',
            self._scan_cb, 10)

        self.create_subscription(
            Odometry, '/cpr_a200_0000/platform/odom/filtered',
            self._odom_cb, 10)

        # YOLO detections (JSON string)
        self.create_subscription(
            String, '/cpr_a200_0000/yolo/detections',
            self._yolo_det_cb, 10)

        # YOLO annotated image (for saving to disk)
        if CV2_AVAILABLE:
            self.create_subscription(
                Image, '/cpr_a200_0000/yolo/image',
                self._yolo_img_cb, 5)
            self.bridge = CvBridge()
        else:
            self.get_logger().warn(
                "cv2/cv_bridge not found — vision frame saving disabled")

        # Raw camera (for saving non-YOLO frames if YOLO node is not running)
        if CV2_AVAILABLE:
            self.create_subscription(
                Image, '/cpr_a200_0000/sensors/camera_0/color/image',
                self._raw_img_cb, 5)

        self.arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/cpr_a200_0000/arm_0_joint_trajectory_controller/'
            'follow_joint_trajectory')

        # ── State ────────────────────────────────────────────────────────────
        self.state           = S.STARTUP
        self.waypoint_idx    = 0
        self.turn_direction  = 0.0
        self.turn_start_ros  = None   # rclpy Time object
        self.blocked_since   = None   # wall-clock float
        self._arm_busy       = False

        # ── Odometry ─────────────────────────────────────────────────────────
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.robot_yaw = 0.0

        # ── Recovery timer ───────────────────────────────────────────────────
        self._recovery_count = 0
        self._recovery_timer = None

        # ── YOLO context ─────────────────────────────────────────────────────
        self.yolo_stop       = False
        self.yolo_slow       = False
        self.yolo_detections = []
        self._latest_yolo_frame  = None   # annotated BGR image (numpy)
        self._latest_raw_frame   = None   # raw BGR image

        # ── Logging dirs ─────────────────────────────────────────────────────
        self.log_dir    = os.path.expanduser("~/clearpath_ws/logs")
        self.vision_dir = os.path.join(self.log_dir, "vision")
        self.frames_dir = os.path.join(self.log_dir, "frames")
        for d in (self.log_dir, self.vision_dir, self.frames_dir):
            os.makedirs(d, exist_ok=True)

        self.t0 = time.time()

        # Scalar time-series
        self.timestamps    = []
        self.distances     = []
        self.vel_linear    = []
        self.vel_angular   = []
        self.sector_right  = []
        self.sector_center = []
        self.sector_left   = []
        self.state_log     = []   # (t, state_int)
        self.path_x        = []
        self.path_y        = []
        self.waypoint_times = []  # wall-clock t when each waypoint arrived

        # Detection event log  [{time, wp_idx, detections, frame_path}, ...]
        self.detection_log  = []
        self._last_vision_save = 0.0   # wall-clock t of last periodic frame save

        # CSV for detection events
        self._csv_path = os.path.join(self.log_dir, "detections.csv")
        with open(self._csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['time_s', 'waypoint', 'state',
                 'n_objects', 'classes', 'frame_path'])

        # Periodic graph save
        self.create_timer(30.0, self.save_all_graphs)

        # Periodic frame save during navigation
        self.create_timer(VISION_LOG_INTERVAL, self._periodic_frame_save)

        self.get_logger().info(
            f"SmartNav ready — {len(WAYPOINTS)} waypoints | "
            f"logs → {self.log_dir}")
        self._run_startup_sequence()

    # ─────────────────────────────────────────────────────────────────────────
    #  Callbacks — odometry
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
    #  Callbacks — YOLO
    # ─────────────────────────────────────────────────────────────────────────
    def _yolo_det_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.yolo_stop       = data.get("stop", False)
            self.yolo_slow       = data.get("slow", False)
            self.yolo_detections = data.get("detections", [])
            if self.yolo_stop:
                classes = [d['class'] for d in self.yolo_detections
                           if d.get('center')]
                self.get_logger().warn(f"YOLO STOP — {classes}")
        except Exception as e:
            self.get_logger().error(f"YOLO parse error: {e}")

    def _yolo_img_cb(self, msg):
        """Store the latest annotated frame from the YOLO node."""
        if not CV2_AVAILABLE:
            return
        try:
            self._latest_yolo_frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception:
            pass

    def _raw_img_cb(self, msg):
        """Store the latest raw camera frame (fallback if YOLO node is off)."""
        if not CV2_AVAILABLE:
            return
        try:
            self._latest_raw_frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    #  Vision logging helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _best_frame(self):
        """Return annotated YOLO frame if available, else raw frame."""
        if self._latest_yolo_frame is not None:
            return self._latest_yolo_frame.copy()
        if self._latest_raw_frame is not None:
            return self._latest_raw_frame.copy()
        return None

    def _save_frame(self, subdir, label, detections=None):
        """
        Save a camera frame to disk.
        Returns the saved file path, or None if no frame available.
        """
        if not CV2_AVAILABLE:
            return None
        frame = self._best_frame()
        if frame is None:
            return None

        ts      = time.time() - self.t0
        fname   = f"{label}_{ts:.1f}s.jpg"
        fpath   = os.path.join(subdir, fname)

        # Burn telemetry onto frame
        overlay = frame.copy()
        lines = [
            f"t={ts:.1f}s  state={self.state}",
            f"wp={self.waypoint_idx+1}/{len(WAYPOINTS)}  "
            f"pos=({self.robot_x:.2f},{self.robot_y:.2f})",
        ]
        if detections:
            dstr = ", ".join(
                f"{d['class']}({d['conf']:.2f})" for d in detections[:4])
            lines.append(f"det: {dstr}")

        y0 = 22
        for line in lines:
            cv2.putText(overlay, line, (8, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),   2)
            cv2.putText(overlay, line, (8, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            y0 += 20

        cv2.imwrite(fpath, overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return fpath

    def _log_detection_event(self, label, detections, save_frames=1):
        """Log a detection event to CSV and save N frames."""
        ts   = time.time() - self.t0
        paths = []
        for i in range(save_frames):
            p = self._save_frame(self.vision_dir, f"{label}_{i}", detections)
            if p:
                paths.append(p)

        classes_str = "|".join(d['class'] for d in detections)
        row = [
            f"{ts:.2f}",
            self.waypoint_idx + 1,
            self.state,
            len(detections),
            classes_str,
            ";".join(paths),
        ]
        with open(self._csv_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

        self.detection_log.append({
            "time":       ts,
            "waypoint":   self.waypoint_idx + 1,
            "state":      self.state,
            "detections": detections,
            "frames":     paths,
        })

    def _periodic_frame_save(self):
        """Save a frame every VISION_LOG_INTERVAL seconds during navigation."""
        if self.state not in (S.NAVIGATING, S.TURN_COMMIT, S.BLOCKED):
            return
        self._save_frame(self.frames_dir, f"nav_{self.state.lower()}")

    # ─────────────────────────────────────────────────────────────────────────
    #  LiDAR scan callback — main decision loop
    # ─────────────────────────────────────────────────────────────────────────
    def _scan_cb(self, msg):
        now = time.time() - self.t0

        # ── Rear mask: blank arm self-detections ─────────────────────────
        ranges = list(msg.ranges)
        n      = len(ranges)
        mask_n = int(n * REAR_MASK_FRACTION)
        for i in range(mask_n):
            ranges[i]       = float('inf')
            ranges[n-1-i]   = float('inf')

        # ── Sector extraction ────────────────────────────────────────────
        center     = n // 2
        slice_size = max(1, n // 36)   # ≈10° per slice

        def get_min(sl):
            valid = [r for r in sl if 0.15 < r < 10.0]
            return min(valid) if valid else 10.0

        c_sec = ranges[center - slice_size   : center + slice_size]
        r_sec = ranges[center - 3*slice_size : center - slice_size]
        l_sec = ranges[center + slice_size   : center + 3*slice_size]
        far_r = ranges[center - 5*slice_size : center - 3*slice_size]
        far_l = ranges[center + 3*slice_size : center + 5*slice_size]

        min_c     = get_min(c_sec)
        min_r     = get_min(r_sec)
        min_l     = get_min(l_sec)
        min_far_r = get_min(far_r)
        min_far_l = get_min(far_l)
        min_all   = min(min_c, min_r, min_l)

        # ── Logging ──────────────────────────────────────────────────────
        self.timestamps.append(now)
        self.distances.append(min_all)
        self.sector_center.append(min_c)
        self.sector_right.append(min_r)
        self.sector_left.append(min_l)
        self.state_log.append((now, S.INT.get(self.state, 0)))

        twist = Twist()

        # ── Guard: states where scan drives nothing ──────────────────────
        if self.state in (S.STARTUP, S.ARM_BLOCKED, S.RECOVERY,
                          S.WAYPOINT_ARRIVED, S.MISSION_COMPLETE):
            self._pub(twist)
            return

        # ── Mission complete ──────────────────────────────────────────────
        if self.waypoint_idx >= len(WAYPOINTS):
            self._set_state(S.MISSION_COMPLETE)
            self._pub(twist)
            self.get_logger().info("All waypoints done — mission complete!")
            self.save_all_graphs()
            return

        # ── Waypoint arrival check ────────────────────────────────────────
        wp   = WAYPOINTS[self.waypoint_idx]
        dist = math.hypot(wp[0] - self.robot_x, wp[1] - self.robot_y)
        if dist < GOAL_TOLERANCE:
            self.get_logger().info(
                f"Arrived WP {self.waypoint_idx+1}/{len(WAYPOINTS)} "
                f"({wp[0]:.1f},{wp[1]:.1f})")
            self.waypoint_times.append(now)
            self._set_state(S.WAYPOINT_ARRIVED)
            self._pub(twist)
            # Save arrival frames with YOLO context
            self._log_detection_event(
                f"wp{self.waypoint_idx+1}_arrival",
                self.yolo_detections,
                save_frames=MAX_FRAMES_PER_EVENT)
            self._run_waypoint_sequence()
            return

        # ── YOLO hard stop — log and hold ────────────────────────────────
        if self.yolo_stop:
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            self._pub(twist)
            self._log_vel(twist)
            # Log detection event (throttled to once per 2s)
            if now - getattr(self, '_last_yolo_log', 0) > 2.0:
                self._log_detection_event(
                    "yolo_stop", self.yolo_detections, save_frames=2)
                self._last_yolo_log = now
            return

        # ── Emergency hard stop ───────────────────────────────────────────
        if min_all < EMERGENCY_DIST:
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            if self.state != S.BLOCKED:
                self.blocked_since = time.time()
                self._set_state(S.BLOCKED)
            # Deploy arm if stuck too long
            if (self.blocked_since is not None
                    and time.time() - self.blocked_since > BLOCKED_TIMEOUT
                    and not self._arm_busy):
                self.get_logger().warn(
                    f"Blocked >{BLOCKED_TIMEOUT:.0f}s — arm deploy")
                self._set_state(S.ARM_BLOCKED)
                self._run_blocked_sequence()
            self._log_vel(twist)
            self._pub(twist)
            return

        # Reset blocked timer when clear again
        if self.state == S.BLOCKED and min_all >= EMERGENCY_DIST:
            self.blocked_since = None
            self._set_state(S.NAVIGATING)

        # ── TURN_COMMIT ───────────────────────────────────────────────────
        if self.state == S.TURN_COMMIT:
            # BUG FIX: use ROS clock diff, compare against global constant
            elapsed = (self.get_clock().now() - self.turn_start_ros
                       ).nanoseconds / 1e9
            if elapsed < TURN_COMMIT_SECS and min_all > EMERGENCY_DIST + 0.1:
                twist.linear.x  = 0.0           # point turn — zero forward
                twist.angular.z = float(self.turn_direction)
                self._log_vel(twist)
                self._pub(twist)
                return
            else:
                self.blocked_since = None
                self._set_state(S.NAVIGATING)

        # ── NAVIGATING ────────────────────────────────────────────────────
        angle_to_wp = self._angle_to_waypoint()

        if min_all > OBSTACLE_CAUTION:
            # Clear path — head straight toward waypoint
            steer = float(np.clip(
                1.5 * angle_to_wp, -TURN_SPEED, TURN_SPEED))
            speed = FORWARD_SPEED_SLOW if self.yolo_slow else FORWARD_SPEED
            twist.linear.x  = float(speed)
            twist.angular.z = steer
            self._set_state(S.NAVIGATING)
            self.blocked_since = None

        elif (min_c <= OBSTACLE_CAUTION
              or min_r <= OBSTACLE_SIDE
              or min_l <= OBSTACLE_SIDE):
            # Obstacle — pick most open side, biased toward waypoint direction
            open_left  = min_l + min_far_l
            open_right = min_r + min_far_r
            # Waypoint direction bias
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
                f"Obstacle c={min_c:.2f}m → {side} "
                f"(L:{open_left:.2f} R:{open_right:.2f})")

            # Point turn — zero linear, full angular
            twist.linear.x  = 0.0
            twist.angular.z = float(self.turn_direction)
            # BUG FIX: store ROS Time, not wall clock
            self.turn_start_ros = self.get_clock().now()
            self._set_state(S.TURN_COMMIT)

            if self.blocked_since is None:
                self.blocked_since = time.time()

        self._log_vel(twist)
        self._pub(twist)

    # ─────────────────────────────────────────────────────────────────────────
    #  Nav helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _angle_to_waypoint(self):
        if self.waypoint_idx >= len(WAYPOINTS):
            return 0.0
        wp  = WAYPOINTS[self.waypoint_idx]
        dx  = wp[0] - self.robot_x
        dy  = wp[1] - self.robot_y
        yaw = math.atan2(dy, dx)
        a   = yaw - self.robot_yaw
        return float((a + math.pi) % (2 * math.pi) - math.pi)

    def _pub(self, twist):
        self.cmd_pub.publish(twist)

    def _log_vel(self, twist):
        self.vel_linear.append(twist.linear.x)
        self.vel_angular.append(twist.angular.z)

    def _set_state(self, new_state):
        if self.state != new_state:
            self.get_logger().info(f"  [{self.state}] → [{new_state}]")
            self.state = new_state

    # ─────────────────────────────────────────────────────────────────────────
    #  Arm trajectory infrastructure
    # ─────────────────────────────────────────────────────────────────────────
    def _send_trajectory(self, points, done_cb):
        """points = list of (joint_positions, time_sec)"""
        if not self.arm_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("Arm action server not available!")
            done_cb()
            return
        goal      = FollowJointTrajectory.Goal()
        traj      = JointTrajectory()
        traj.joint_names = self.JOINT_NAMES
        traj.points      = [_pt(pos, sec) for pos, sec in points]
        goal.trajectory  = traj
        self._arm_busy   = True
        fut = self.arm_client.send_goal_async(goal)
        fut.add_done_callback(lambda f: self._arm_accepted(f, done_cb))

    def _arm_accepted(self, future, done_cb):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("Arm goal rejected!")
            self._arm_busy = False
            done_cb()
            return
        handle.get_result_async().add_done_callback(
            lambda f: self._arm_done(f, done_cb))

    def _arm_done(self, future, done_cb):
        self._arm_busy = False
        self.get_logger().info("Arm trajectory complete")
        done_cb()

    # ─────────────────────────────────────────────────────────────────────────
    #  ARM SEQUENCE 1 — Grand startup salute
    # ─────────────────────────────────────────────────────────────────────────
    def _run_startup_sequence(self):
        self.get_logger().info("=== STARTUP — Grand Salute ===")
        pts = [
            (HOME_READY_POS,  2),
            (SALUTE_1_POS,    4),   # wind up
            (SALUTE_2_POS,    6),   # sweep right
            (SALUTE_3_POS,    8),   # sweep left
            (SALUTE_4_POS,   10),   # raise high
            (SALUTE_5_POS,   12),   # back to ready
            (POINT_FWD_POS,  14),   # "let's go"
            (STOW_POS,       17),   # stow — MUST happen before driving
        ]
        self._send_trajectory(pts, self._startup_done)

    def _startup_done(self):
        self.get_logger().info("Startup done — beginning navigation")
        self._set_state(S.NAVIGATING)

    # ─────────────────────────────────────────────────────────────────────────
    #  ARM SEQUENCE 2 — Waypoint arrived: area scan + gripper demo
    # ─────────────────────────────────────────────────────────────────────────
    def _run_waypoint_sequence(self):
        is_last = (self.waypoint_idx == len(WAYPOINTS) - 1)
        self.get_logger().info(
            f"=== WP {self.waypoint_idx+1} SEQUENCE "
            f"{'[FINAL GOAL]' if is_last else ''} ===")
        pts = [
            (HOME_READY_POS,  2),
            (LOOK_LEFT_POS,   4),
            (LOOK_RIGHT_POS,  7),
            (HOME_READY_POS,  9),
            (GRASP_OPEN_POS, 11),
            (GRASP_CLOSE_POS,13),
            (GRASP_OPEN_POS, 15),
            (GRASP_CLOSE_POS,17),
            (GRASP_OPEN_POS, 19),
            (SALUTE_4_POS if is_last else HOME_READY_POS, 21),
            (STOW_POS,       24),
        ]
        self._send_trajectory(pts,
            lambda: self._waypoint_done(is_last))

    def _waypoint_done(self, is_last):
        if is_last:
            self.get_logger().info("=== MISSION COMPLETE ===")
            self._set_state(S.MISSION_COMPLETE)
            self._save_vision_summary()
            self.save_all_graphs()
        else:
            self.waypoint_idx += 1
            self.blocked_since = None
            self.get_logger().info(
                f"Next WP {self.waypoint_idx+1}: "
                f"{WAYPOINTS[self.waypoint_idx]}")
            self._set_state(S.NAVIGATING)

    # ─────────────────────────────────────────────────────────────────────────
    #  ARM SEQUENCE 3 — Blocked: frustrated scan
    # ─────────────────────────────────────────────────────────────────────────
    def _run_blocked_sequence(self):
        self.get_logger().warn("=== BLOCKED SEQUENCE ===")
        pts = [
            (HOME_READY_POS, 2),
            (LOOK_LEFT_POS,  4),
            (LOOK_RIGHT_POS, 7),
            (HOME_READY_POS, 9),
            (STOW_POS,      11),
        ]
        self._send_trajectory(pts, self._blocked_done)

    def _blocked_done(self):
        self.get_logger().info("Blocked sequence done — recovery")
        self.blocked_since = None
        self._set_state(S.RECOVERY)
        self._start_recovery()

    # ─────────────────────────────────────────────────────────────────────────
    #  Recovery: reverse then rotate
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
        elif self._recovery_count <= 40:     # 1.5 s rotate
            twist.angular.z =  0.7
        else:
            self._recovery_timer.cancel()
            self._recovery_timer.destroy()
            self._recovery_timer = None
            self.blocked_since   = None
            self._set_state(S.NAVIGATING)
            self.get_logger().info("Recovery done — resuming navigation")
            return
        self.cmd_pub.publish(twist)

    # ─────────────────────────────────────────────────────────────────────────
    #  Vision summary — called at mission complete
    # ─────────────────────────────────────────────────────────────────────────
    def _save_vision_summary(self):
        if not self.detection_log:
            return

        # ── 1. Detection count per class bar chart ──────────────────────
        class_counts = {}
        for event in self.detection_log:
            for d in event['detections']:
                c = d['class']
                class_counts[c] = class_counts.get(c, 0) + 1

        if class_counts:
            fig, ax = plt.subplots(figsize=(9, 4))
            classes = sorted(class_counts, key=class_counts.get, reverse=True)
            counts  = [class_counts[c] for c in classes]
            bars    = ax.bar(classes, counts,
                             color=['#e74c3c','#3498db','#2ecc71',
                                    '#f39c12','#9b59b6'] * 10)
            ax.set_xlabel("Detected class")
            ax.set_ylabel("Count across all events")
            ax.set_title("YOLO Detection Counts — Full Mission")
            ax.grid(axis='y', alpha=0.3)
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.2,
                        str(count), ha='center', fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(self.log_dir,
                        "06_detection_counts.png"), dpi=130)
            plt.close(fig)

        # ── 2. Detection events timeline ────────────────────────────────
        if len(self.detection_log) > 1:
            fig, ax = plt.subplots(figsize=(11, 3))
            times  = [e['time']      for e in self.detection_log]
            counts = [e['detections'].__len__() for e in self.detection_log]
            ax.stem(times, counts, linefmt='C0-', markerfmt='C0o',
                    basefmt='gray')
            for t_wp in self.waypoint_times:
                ax.axvline(t_wp, color='green', ls=':', lw=1.5, alpha=0.8)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("# detections in event")
            ax.set_title("Detection Events Timeline (green lines = waypoint arrivals)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(self.log_dir,
                        "07_detection_timeline.png"), dpi=130)
            plt.close(fig)

        # ── 3. Contact sheet of saved frames ────────────────────────────
        if CV2_AVAILABLE:
            all_frames = []
            for event in self.detection_log:
                for fp in event.get('frames', []):
                    if os.path.exists(fp):
                        all_frames.append(fp)
            if all_frames:
                self._make_contact_sheet(
                    all_frames[:24],
                    os.path.join(self.log_dir, "08_vision_contact_sheet.jpg"))

        self.get_logger().info("Vision summary saved")

    def _make_contact_sheet(self, frame_paths, out_path, cols=4, thumb_w=320):
        """Tile saved frames into one contact sheet image."""
        thumbs = []
        for fp in frame_paths:
            img = cv2.imread(fp)
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = thumb_w / w
            thumb = cv2.resize(img, (thumb_w, int(h * scale)))
            thumbs.append(thumb)
        if not thumbs:
            return
        rows_needed = math.ceil(len(thumbs) / cols)
        th = thumbs[0].shape[0]
        sheet = np.zeros((rows_needed * th, cols * thumb_w, 3), dtype=np.uint8)
        for idx, thumb in enumerate(thumbs):
            r = idx // cols
            c = idx  % cols
            th_h = thumb.shape[0]
            sheet[r*th : r*th + th_h, c*thumb_w : (c+1)*thumb_w] = thumb
        cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.get_logger().info(f"Contact sheet → {out_path}")

    # ─────────────────────────────────────────────────────────────────────────
    #  Navigation graphs
    # ─────────────────────────────────────────────────────────────────────────
    def save_all_graphs(self):
        if len(self.timestamps) < 10:
            return
        self.get_logger().info(f"Saving graphs → {self.log_dir}")
        self._graph_distance()
        self._graph_state_timeline()
        self._graph_velocity()
        self._graph_sectors()
        self._graph_path()
        self.get_logger().info("Graphs saved.")

    def _graph_distance(self):
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(self.timestamps, self.distances,
                color='steelblue', lw=1.2, label='Min distance')
        ax.axhline(EMERGENCY_DIST,  color='red',    ls='--', lw=1,
                   label=f'Emergency ({EMERGENCY_DIST}m)')
        ax.axhline(OBSTACLE_CAUTION, color='orange', ls='--', lw=1,
                   label=f'Caution ({OBSTACLE_CAUTION}m)')
        for t in self.waypoint_times:
            ax.axvline(t, color='green', ls=':', lw=1.2, alpha=0.7)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance (m)")
        ax.set_title("Closest Obstacle Distance Over Time")
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(bottom=0); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "01_distance.png"), dpi=130)
        plt.close(fig)

    def _graph_state_timeline(self):
        if len(self.state_log) < 2:
            return
        fig, ax = plt.subplots(figsize=(11, 2.5))
        times  = [s[0] for s in self.state_log]
        states = [s[1] for s in self.state_log]
        for i in range(len(times)-1):
            ax.barh(0, times[i+1]-times[i], left=times[i],
                    color=S.COLORS[states[i]], height=0.6, align='center')
        labels  = list(S.INT.keys())
        patches = [mpatches.Patch(color=S.COLORS[i], label=labels[i])
                   for i in range(len(labels))]
        ax.legend(handles=patches, loc='upper right', fontsize=7, ncol=4)
        ax.set_xlabel("Time (s)"); ax.set_yticks([])
        ax.set_title("Robot State Timeline"); ax.grid(axis='x', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "02_state_timeline.png"), dpi=130)
        plt.close(fig)

    def _graph_velocity(self):
        n = min(len(self.vel_linear), len(self.timestamps))
        if n < 2:
            return
        t = self.timestamps[:n]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        a1.plot(t, self.vel_linear[:n],  color='#2980b9', lw=1.2)
        a1.set_ylabel("Linear (m/s)"); a1.axhline(0,color='gray',lw=0.8)
        a1.grid(True, alpha=0.3)
        a2.plot(t, self.vel_angular[:n], color='#e67e22', lw=1.2)
        a2.set_ylabel("Angular (rad/s)"); a2.set_xlabel("Time (s)")
        a2.axhline(0,color='gray',lw=0.8); a2.grid(True, alpha=0.3)
        fig.suptitle("Command Velocities Over Time"); fig.tight_layout()
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
        ax.axhline(EMERGENCY_DIST,   color='red',    ls=':', lw=1)
        ax.axhline(OBSTACLE_CAUTION, color='orange', ls=':', lw=1)
        for t_wp in self.waypoint_times:
            ax.axvline(t_wp, color='green', ls=':', lw=1.2, alpha=0.7)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance (m)")
        ax.set_title("LiDAR Sector Distances"); ax.legend(loc='upper right')
        ax.set_ylim(0, 5); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "04_sectors.png"), dpi=130)
        plt.close(fig)

    def _graph_path(self):
        if len(self.path_x) < 2:
            return
        fig, ax = plt.subplots(figsize=(7, 7))
        pts  = np.array([self.path_x, self.path_y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap='plasma', lw=1.5,
                              array=np.linspace(0, 1, len(segs)))
        ax.add_collection(lc)
        for i, (wx, wy) in enumerate(WAYPOINTS):
            ax.plot(wx, wy, 'g^', markersize=10)
            ax.annotate(f"WP{i+1}", (wx, wy),
                        textcoords="offset points",
                        xytext=(6, 4), fontsize=8)
        ax.plot(self.path_x[0],  self.path_y[0],
                'go', markersize=10, label='Start')
        ax.plot(self.path_x[-1], self.path_y[-1],
                'rs', markersize=10, label='End')
        all_x = self.path_x + [w[0] for w in WAYPOINTS]
        all_y = self.path_y + [w[1] for w in WAYPOINTS]
        m = 1.0
        ax.set_xlim(min(all_x)-m, max(all_x)+m)
        ax.set_ylim(min(all_y)-m, max(all_y)+m)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title("Robot Path with Waypoints")
        ax.legend(fontsize=8); ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.colorbar(lc, ax=ax, label='Time progression')
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "05_path.png"), dpi=130)
        plt.close(fig)

    def destroy_node(self):
        self.get_logger().info("Shutdown — saving final outputs...")
        self._save_vision_summary()
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
