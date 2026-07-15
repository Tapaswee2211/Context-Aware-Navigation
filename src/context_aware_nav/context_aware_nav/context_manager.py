# context_manager.py
# ~/clearpath_ws/src/context_aware_nav/context_aware_nav/context_manager.py
#
# Two jobs:
#   1. Adjust Nav2 local costmap inflation radius based on obstacle density
#      (only active if Nav2 is running — gracefully skips if not)
#   2. Publish current context on /smart_nav/context so smart_nav_node
#      can adjust speeds and commit times dynamically

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
import json
import math


# Context definitions — tune density thresholds for your world
CONTEXT_THRESHOLDS = {
    # (density_threshold, inflation_radius, cost_scaling, speed_limit, turn_commit_secs)
    "OPEN":    (0.50, 0.20, 2.0, 0.45, 2.0),
    "NARROW":  (0.75, 0.35, 5.0, 0.25, 2.8),
    "CROWDED": (1.00, 0.55, 7.0, 0.15, 3.5),
}


class ContextManager(Node):
    def __init__(self):
        super().__init__('context_manager')

        # ── Subscribe to correct LiDAR topic ────────────────────────────────
        self.create_subscription(
            LaserScan,
            '/cpr_a200_0000/sensors/lidar2d_0/scan',
            self._scan_cb, 10)

        # ── Publish context for smart_nav_node ───────────────────────────────
        self.context_pub = self.create_publisher(
            String, '/smart_nav/context', 10)

        # ── Nav2 costmap client (optional — skips if Nav2 not running) ───────
        self._nav2_available = False
        self.param_client = self.create_client(
            SetParameters,
            '/cpr_a200_0000/local_costmap/local_costmap/set_parameters')

        # Keep a reference to the timer so we can cancel it later
        self.nav_timer = self.create_timer(2.0, self._check_nav2_once)

        self.current_context = None
        self.get_logger().info("ContextManager started")

    def _check_nav2_once(self):
        """One-time check for Nav2 availability — non-blocking."""
        if self.param_client.service_is_ready():
            self._nav2_available = True
            self.get_logger().info("Nav2 costmap service found — will adjust inflation")
            # Cancel the timer so it stops spamming the terminal!
            self.nav_timer.cancel()
        else:
            self.get_logger().debug("Waiting for Nav2 costmap service...")

    def _scan_cb(self, msg):
        ranges = list(msg.ranges)
        n = len(ranges)
        
        # Mask the rear ~60 degrees to ignore the arm
        mask_n = int(n * 0.3) 
        for i in range(mask_n):
            ranges[i] = float('inf')
            ranges[n-1-i] = float('inf')

        # Define what an "obstacle" is. 
        # Exclude inf, nan, and max range (e.g., values near 10.0m or 30.0m depending on the LiDAR)
        # We assume anything closer than 8.0m is a real obstacle to consider for density.
        MAX_OBSTACLE_DIST = 8.0 
        
        valid_obstacles = [r for r in ranges
                           if not math.isinf(r) and not math.isnan(r) and 0.1 < r < MAX_OBSTACLE_DIST]

        if not ranges:
            return

        # Obstacle density: fraction of unmasked rays that return a hit
        active_rays = n - (2 * mask_n)
        if active_rays <= 0:
            return
            
        density = len(valid_obstacles) / active_rays

        # Classify context
        if density < 0.50:
            context = "OPEN"
        elif density < 0.75:
            context = "NARROW"
        else:
            context = "CROWDED"

        if context == self.current_context:
            return   # no change — don't spam

        self.current_context = context
        cfg = CONTEXT_THRESHOLDS[context]

        self.get_logger().info(
            f"Context → {context} "
            f"(density={density:.2f} "
            f"inflation={cfg[1]}m "
            f"speed_limit={cfg[3]}m/s)")

        # ── Publish to smart_nav_node ────────────────────────────────────────
        pub_msg = String()
        pub_msg.data = json.dumps({
            "context":          context,
            "density":          round(density, 3),
            "speed_limit":      cfg[3],
            "turn_commit_secs": cfg[4],
            "inflation_radius": cfg[1],
        })
        self.context_pub.publish(pub_msg)

        # ── Update Nav2 costmap if available ─────────────────────────────────
        if self._nav2_available:
            self._update_costmap(cfg[1], cfg[2])

    def _update_costmap(self, inflation, scaling):
        params = [
            Parameter(
                name='inflation_layer.inflation_radius',
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=inflation)),
            Parameter(
                name='inflation_layer.cost_scaling_factor',
                value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE,
                    double_value=scaling)),
        ]
        req = SetParameters.Request()
        req.parameters = params
        self.param_client.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = ContextManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
