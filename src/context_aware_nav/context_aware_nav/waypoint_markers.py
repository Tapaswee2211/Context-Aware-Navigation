# waypoint_markers.py
# ~/clearpath_ws/src/context_aware_nav/context_aware_nav/waypoint_markers.py
#
# Publishes RViz MarkerArray for all waypoints.
# Spawns visible SDF cylinders in Gazebo via /world/.../create service.
# Run alongside smart_nav_node — waypoints update colour as robot visits them.

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, String
import json
import math

# ── Import waypoints directly from your nav node ────────────────────────────
# Same list — edit once, both nodes stay in sync
WAYPOINTS = [
    ( 3.0,  0.0),
    ( 3.0,  3.0),
    ( 0.0,  3.0),
    ( 4.0,  2.0),
    ( 0.0,  0.0),
]
MARKER_HEIGHT   = 0.5    # cylinder height in metres
MARKER_RADIUS   = 0.15   # cylinder radius in metres
MARKER_Z        = 0.25   # z centre of marker (half height)


class WaypointMarkerNode(Node):
    def __init__(self):
        super().__init__('waypoint_marker_node')

        # RViz marker publisher
        self.marker_pub = self.create_publisher(
            MarkerArray, '/waypoint_markers', 10)

        # Subscribe to smart_nav status to colour visited waypoints
        # smart_nav_node publishes current waypoint index on this topic
        self.create_subscription(
            String, '/smart_nav/status', self._status_cb, 10)

        self.current_wp  = 0
        self.visited_wps = set()

        # Publish at 2 Hz — RViz needs periodic republish to keep markers alive
        self.create_timer(0.5, self._publish_markers)

        # Spawn Gazebo cylinders once at startup
        self._gazebo_spawned = False
        self.create_timer(3.0, self._spawn_gazebo_markers)

        self.get_logger().info(
            f"WaypointMarkerNode ready — {len(WAYPOINTS)} waypoints")

    def _status_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.current_wp  = data.get("waypoint_idx", 0)
            self.visited_wps = set(data.get("visited", []))
        except Exception:
            pass

    def _publish_markers(self):
        array = MarkerArray()

        for i, (wx, wy) in enumerate(WAYPOINTS):
            # ── Cylinder body ────────────────────────────────────────────
            cyl = Marker()
            cyl.header.frame_id = 'map'
            cyl.header.stamp    = self.get_clock().now().to_msg()
            cyl.ns     = 'waypoints'
            cyl.id     = i
            cyl.type   = Marker.CYLINDER
            cyl.action = Marker.ADD

            cyl.pose.position.x = wx
            cyl.pose.position.y = wy
            cyl.pose.position.z = MARKER_Z
            cyl.pose.orientation.w = 1.0

            cyl.scale.x = MARKER_RADIUS * 2
            cyl.scale.y = MARKER_RADIUS * 2
            cyl.scale.z = MARKER_HEIGHT

            # Colour logic:
            #   visited  → gray
            #   current  → bright yellow, pulsing
            #   future   → blue
            if i in self.visited_wps:
                cyl.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.7)
            elif i == self.current_wp:
                cyl.color = ColorRGBA(r=1.0, g=0.9, b=0.0, a=1.0)
            else:
                cyl.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.8)

            cyl.lifetime.sec = 1   # auto-expire if publisher dies
            array.markers.append(cyl)

            # ── Number label above cylinder ──────────────────────────────
            txt = Marker()
            txt.header.frame_id = 'map'
            txt.header.stamp    = self.get_clock().now().to_msg()
            txt.ns     = 'waypoint_labels'
            txt.id     = i + 100
            txt.type   = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD

            txt.pose.position.x = wx
            txt.pose.position.y = wy
            txt.pose.position.z = MARKER_HEIGHT + 0.3
            txt.pose.orientation.w = 1.0

            txt.scale.z = 0.35   # text height
            txt.color   = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            txt.text    = f"WP{i+1}\n({wx:.1f},{wy:.1f})"
            txt.lifetime.sec = 1
            array.markers.append(txt)

            # ── Connector line between consecutive waypoints ──────────────
            if i < len(WAYPOINTS) - 1:
                nx, ny = WAYPOINTS[i + 1]
                line = Marker()
                line.header.frame_id = 'map'
                line.header.stamp    = self.get_clock().now().to_msg()
                line.ns     = 'waypoint_path'
                line.id     = i + 200
                line.type   = Marker.LINE_STRIP
                line.action = Marker.ADD
                line.scale.x = 0.03   # line width
                line.color   = ColorRGBA(r=0.3, g=0.8, b=0.3, a=0.5)
                line.points  = [
                    Point(x=wx,  y=wy,  z=0.05),
                    Point(x=nx,  y=ny,  z=0.05),
                ]
                line.lifetime.sec = 1
                array.markers.append(line)

        self.marker_pub.publish(array)

    def _spawn_gazebo_markers(self):
        """Spawn coloured cylinders in Gazebo using ros2 service call."""
        if self._gazebo_spawned:
            return
        self._gazebo_spawned = True

        # We use subprocess to call the Gazebo spawn service once.
        # This avoids a heavy gz_msgs dependency in the Python node.
        import subprocess
        for i, (wx, wy) in enumerate(WAYPOINTS):
            color = "1 0.9 0 1" if i == 0 else "0 0.5 1 1"   # yellow or blue
            sdf = f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='waypoint_{i+1}'>
    <static>true</static>
    <link name='link'>
      <visual name='visual'>
        <geometry>
          <cylinder>
            <radius>{MARKER_RADIUS}</radius>
            <length>{MARKER_HEIGHT}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{color}</ambient>
          <diffuse>{color}</diffuse>
        </material>
      </visual>
    </link>
    <pose>{wx} {wy} {MARKER_Z} 0 0 0</pose>
  </model>
</sdf>"""
            cmd = [
                'gz', 'service', '-s', '/world/default/create',
                '--reqtype', 'gz.msgs.EntityFactory',
                '--reptype', 'gz.msgs.Boolean',
                '--timeout', '2000',
                '--req', f'sdf: "{sdf.strip()}"'
            ]
            try:
                subprocess.run(cmd, timeout=5, capture_output=True)
                self.get_logger().info(f"Spawned Gazebo marker WP{i+1}")
            except Exception as e:
                self.get_logger().warn(
                    f"Gazebo spawn failed for WP{i+1} (non-critical): {e}")


def main(args=None):
    rclpy.init(args=args)
    node = WaypointMarkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
