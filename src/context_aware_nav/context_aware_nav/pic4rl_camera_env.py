# pic4rl_camera_env.py
# ~/clearpath_ws/src/context_aware_nav/context_aware_nav/pic4rl_camera_env.py
#
# Gymnasium environment wrapping the Clearpath A200 in Gazebo via ROS2 topics.
# Phase 1: LiDAR only (use_camera=False)
# Phase 2: LiDAR + camera (use_camera=True) — enable after Phase 1 converges

import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
import gymnasium as gym
from gymnasium import spaces

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty


# ── Constants ────────────────────────────────────────────────────────────────
LIDAR_RAYS      = 720       # adjust to match your actual scan ray count
LIDAR_MAX       = 10.0
COLLISION_DIST  = 0.35      # metres
GOAL_TOLERANCE  = 0.40
MAX_STEPS       = 500
RESET_TIMEOUT   = 5.0       # seconds to wait for reset service


class A200NavEnv(gym.Env):
    """
    Single-file Gymnasium environment for the Clearpath A200.
    Connects to the running Gazebo simulation via ROS2 topics.

    Observation (Phase 1 — lidar only):
        Box(LIDAR_RAYS + 2,) — normalised ranges + (dist_to_goal, angle_to_goal)

    Action:
        Box(2,) — [linear_vel, angular_vel]
    """

    metadata = {"render_modes": []}

    def __init__(self, use_camera=False, goal_position=(5.0, 0.0)):
        super().__init__()

        self.use_camera    = use_camera
        self.goal_pos      = np.array(goal_position, dtype=np.float32)
        self.step_count    = 0
        self.prev_dist     = None
        self._lidar_data   = np.ones(LIDAR_RAYS, dtype=np.float32) * LIDAR_MAX
        self._robot_pos    = np.zeros(2, dtype=np.float32)
        self._robot_yaw    = 0.0
        self._new_scan     = False

        # ── ROS2 init ────────────────────────────────────────────────────────
        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node('a200_nav_env')

        self._node.create_subscription(
            LaserScan,
            '/cpr_a200_0000/sensors/lidar2d_0/scan',
            self._scan_cb, 10)

        self._node.create_subscription(
            Odometry,
            '/cpr_a200_0000/platform/odom/filtered',
            self._odom_cb, 10)

        self._cmd_pub = self._node.create_publisher(
            Twist, '/cpr_a200_0000/cmd_vel', 10)

        self._reset_client = self._node.create_client(
            Empty, '/reset_simulation')

        # ── Spaces ───────────────────────────────────────────────────────────
        obs_dim = LIDAR_RAYS + 2   # ranges + dist_to_goal + angle_to_goal
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-0.3, -0.8], dtype=np.float32),
            high=np.array([ 0.5,  0.8], dtype=np.float32),
            dtype=np.float32)

        self._node.get_logger().info(
            f"A200NavEnv ready — goal={goal_position}, camera={use_camera}")

    # ── ROS callbacks ────────────────────────────────────────────────────────
    def _scan_cb(self, msg):
        raw = np.array(msg.ranges, dtype=np.float32)
        raw = np.where(np.isfinite(raw), raw, LIDAR_MAX)
        raw = np.clip(raw, 0.0, LIDAR_MAX)

        # Resize to fixed LIDAR_RAYS if sensor count differs
        if len(raw) != LIDAR_RAYS:
            raw = np.interp(
                np.linspace(0, len(raw)-1, LIDAR_RAYS),
                np.arange(len(raw)), raw).astype(np.float32)

        self._lidar_data = raw
        self._new_scan   = True

    def _odom_cb(self, msg):
        self._robot_pos[0] = msg.pose.pose.position.x
        self._robot_pos[1] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._robot_yaw = math.atan2(siny, cosy)

    # ── Gymnasium API ────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Stop robot
        self._publish_vel(0.0, 0.0)
        time.sleep(0.2)

        # Reset Gazebo if service is available
        if self._reset_client.wait_for_service(timeout_sec=1.0):
            self._reset_client.call_async(Empty.Request())
            time.sleep(1.0)

        # Optionally randomise goal within a 6m radius
        if options and options.get("random_goal", False):
            angle = self.np_random.uniform(0, 2 * math.pi)
            r     = self.np_random.uniform(2.0, 6.0)
            self.goal_pos = np.array(
                [r * math.cos(angle), r * math.sin(angle)], dtype=np.float32)

        self.step_count = 0
        self._new_scan  = False
        self._wait_for_scan()

        obs = self._get_obs()
        self.prev_dist = self._dist_to_goal()
        return obs, {}

    def step(self, action):
        lin_vel = float(np.clip(action[0], -0.3, 0.5))
        ang_vel = float(np.clip(action[1], -0.8, 0.8))
        self._publish_vel(lin_vel, ang_vel)

        # Wait for fresh scan
        self._new_scan = False
        self._wait_for_scan()

        self.step_count += 1
        obs             = self._get_obs()
        curr_dist       = self._dist_to_goal()
        reward, done    = self._compute_reward(curr_dist)
        self.prev_dist  = curr_dist

        truncated = (self.step_count >= MAX_STEPS)
        info      = {
            "dist_to_goal": curr_dist,
            "step": self.step_count,
            "robot_pos": self._robot_pos.tolist()
        }
        return obs, reward, done, truncated, info

    def close(self):
        self._publish_vel(0.0, 0.0)
        self._node.destroy_node()

    # ── Internal helpers ─────────────────────────────────────────────────────
    def _wait_for_scan(self, timeout=2.0):
        t0 = time.time()
        while not self._new_scan and (time.time() - t0) < timeout:
            rclpy.spin_once(self._node, timeout_sec=0.05)

    def _publish_vel(self, lin, ang):
        t = Twist()
        t.linear.x  = float(lin)
        t.angular.z = float(ang)
        self._cmd_pub.publish(t)
        rclpy.spin_once(self._node, timeout_sec=0.01)

    def _dist_to_goal(self):
        return float(np.linalg.norm(self.goal_pos - self._robot_pos))

    def _angle_to_goal(self):
        diff = self.goal_pos - self._robot_pos
        goal_yaw = math.atan2(diff[1], diff[0])
        angle = goal_yaw - self._robot_yaw
        # Normalise to [-pi, pi]
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        return float(angle)

    def _get_obs(self):
        lidar_norm = self._lidar_data / LIDAR_MAX          # 0..1
        dist_norm  = min(self._dist_to_goal() / 10.0, 1.0) # 0..1
        ang_norm   = self._angle_to_goal() / math.pi       # -1..1, remap →0..1
        ang_norm   = (ang_norm + 1.0) / 2.0
        return np.append(lidar_norm,
                         [dist_norm, ang_norm]).astype(np.float32)

    def _compute_reward(self, curr_dist):
        reward = 0.0

        # Collision
        if np.min(self._lidar_data) < COLLISION_DIST:
            return -50.0, True

        # Goal reached
        if curr_dist < GOAL_TOLERANCE:
            return +100.0, True

        # Progress towards goal
        reward += 5.0 * (self.prev_dist - curr_dist)

        # Keep moving — penalise standing still
        reward -= 0.05

        # Penalise being close to obstacles (soft penalty)
        min_scan = np.min(self._lidar_data)
        if min_scan < 1.0:
            reward -= 0.5 * (1.0 - min_scan)

        return reward, False
