# grasp_node.py
# ~/clearpath_ws/src/context_aware_nav/context_aware_nav/grasp_node.py
#
# Listens for grasp requests from smart_nav_node.
# Uses RGB (YOLO detection) + depth image to compute a 3D grasp point.
# Sends a joint trajectory to the arm to reach and grasp.
#
# Grasp pipeline:
#   1. YOLO gives bounding box of target object in RGB image
#   2. Depth image gives Z (metres) at the bbox centre pixel
#   3. Camera intrinsics back-project pixel → 3D point in camera frame
#   4. Transform to robot base_link frame using TF2
#   5. IK via joint trajectory (pre-computed reach poses, refined by depth)

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import rclpy.time

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Bool
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PointStamped

import numpy as np
import json
import math
import time

try:
    import cv2
    from cv_bridge import CvBridge
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from tf2_ros import Buffer, TransformListener
    import tf2_geometry_msgs
    TF2_OK = True
except ImportError:
    TF2_OK = False


# ── Grasp parameters — tune for your arm + object ───────────────────────────
# Object classes YOLO should attempt to grasp
GRASPABLE_CLASSES = {'cup', 'bottle', 'bowl', 'cell phone',
                     'apple', 'orange', 'banana', 'book', 'vase'}

# Depth limits — ignore readings outside this range (metres)
DEPTH_MIN = 0.20
DEPTH_MAX = 1.50

# Depth sampling window around bbox centre (pixels)
DEPTH_SAMPLE_RADIUS = 8

# Camera topic namespace
CAM_NS = '/cpr_a200_0000/sensors/camera_0'

# Arm approach config
APPROACH_STANDOFF = 0.10   # metres short of object to stop
GRASP_SPEED_SEC   = 3      # seconds per trajectory segment

# Joint names
JOINT_NAMES = [
    'arm_0_shoulder_pan_joint', 'arm_0_shoulder_lift_joint',
    'arm_0_elbow_joint',        'arm_0_wrist_1_joint',
    'arm_0_wrist_2_joint',      'arm_0_wrist_3_joint',
]

# Safe stow after grasp
STOW_POS        = [ 0.00, -2.80,  2.40, -1.57, -1.57,  0.00]
HOME_READY_POS  = [ 0.00, -1.57,  1.57, -1.57, -1.57,  0.00]
GRASP_OPEN_POS  = [ 0.00, -0.50,  0.80, -0.80, -1.57,  0.00]
GRASP_CLOSE_POS = [ 0.00, -0.50,  0.80, -0.80, -1.57,  1.57]


def _pt(positions, sec):
    p = JointTrajectoryPoint()
    p.positions = [float(x) for x in positions]
    p.time_from_start = Duration(sec=int(sec), nanosec=0)
    return p


class GraspNode(Node):
    def __init__(self):
        super().__init__('grasp_node')

        self.bridge = CvBridge() if CV2_OK else None

        # Camera state
        self._depth_image  = None
        self._rgb_image    = None
        self._cam_info     = None   # intrinsics
        self._fx = self._fy = self._cx = self._cy = None

        # Latest YOLO detections
        self._detections   = []

        # Grasp state
        self._grasp_active = False
        self._last_grasp_target = None

        # TF2 for coordinate transforms
        if TF2_OK:
            self.tf_buffer   = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subscribers ──────────────────────────────────────────────────
        self.create_subscription(
            Image, f'{CAM_NS}/depth/image',
            self._depth_cb, 5)

        self.create_subscription(
            Image, f'{CAM_NS}/color/image',
            self._rgb_cb, 5)

        self.create_subscription(
            CameraInfo, f'{CAM_NS}/color/camera_info',
            self._caminfo_cb, 5)

        self.create_subscription(
            String, '/cpr_a200_0000/yolo/detections',
            self._yolo_cb, 10)

        # Trigger: smart_nav publishes True here when arriving at waypoint
        self.create_subscription(
            Bool, '/smart_nav/trigger_grasp',
            self._grasp_trigger_cb, 10)

        # ── Publishers ───────────────────────────────────────────────────
        # Publishes grasp result back to smart_nav
        self.result_pub = self.create_publisher(
            String, '/smart_nav/grasp_result', 10)

        # Arm action client
        self.arm_client = ActionClient(
            self, FollowJointTrajectory,
            '/cpr_a200_0000/arm_0_joint_trajectory_controller/'
            'follow_joint_trajectory')

        self.get_logger().info("GraspNode ready")

    # ─────────────────────────────────────────────────────────────────────────
    #  Sensor callbacks
    # ─────────────────────────────────────────────────────────────────────────
    def _depth_cb(self, msg):
        if not CV2_OK:
            return
        try:
            # Depth image is float32 in metres for Intel RealSense
            self._depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Depth convert error: {e}")

    def _rgb_cb(self, msg):
        if not CV2_OK:
            return
        try:
            self._rgb_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8')
        except Exception:
            pass

    def _caminfo_cb(self, msg):
        """Extract focal length and principal point from camera info."""
        if self._cam_info is not None:
            return   # only need this once
        self._cam_info = msg
        k = msg.k   # 3×3 intrinsic matrix, row-major
        self._fx = k[0]
        self._fy = k[4]
        self._cx = k[2]
        self._cy = k[5]
        self.get_logger().info(
            f"Camera intrinsics: fx={self._fx:.1f} fy={self._fy:.1f} "
            f"cx={self._cx:.1f} cy={self._cy:.1f}")

    def _yolo_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self._detections = data.get("detections", [])
        except Exception:
            pass

    def _grasp_trigger_cb(self, msg):
        if msg.data and not self._grasp_active:
            self.get_logger().info("Grasp trigger received — starting pipeline")
            self._run_grasp_pipeline()

    # ─────────────────────────────────────────────────────────────────────────
    #  Grasp pipeline
    # ─────────────────────────────────────────────────────────────────────────
    def _run_grasp_pipeline(self):
        self._grasp_active = True

        # Step 1: Find a graspable detection
        target = self._find_grasp_target()
        if target is None:
            self.get_logger().warn("No graspable object detected")
            self._publish_result("no_target")
            self._grasp_active = False
            return

        cls, bbox, conf = target
        self.get_logger().info(
            f"Grasp target: {cls} (conf={conf:.2f}) bbox={bbox}")

        # Step 2: Get depth at bbox centre
        depth_m = self._sample_depth(bbox)
        if depth_m is None:
            self.get_logger().warn("Depth reading invalid — aborting grasp")
            self._publish_result("bad_depth")
            self._grasp_active = False
            return

        self.get_logger().info(f"Object depth: {depth_m:.3f}m")

        # Step 3: Back-project to 3D camera frame
        cam_point = self._pixel_to_3d(bbox, depth_m)
        if cam_point is None:
            self._publish_result("no_intrinsics")
            self._grasp_active = False
            return

        self.get_logger().info(
            f"3D camera frame: x={cam_point[0]:.3f} "
            f"y={cam_point[1]:.3f} z={cam_point[2]:.3f}")

        # Step 4: Compute arm trajectory to reach this point
        pan_angle, reach_depth = self._compute_arm_angles(cam_point)

        # Step 5: Execute grasp sequence
        self._execute_grasp(cls, pan_angle, reach_depth, depth_m)

    def _find_grasp_target(self):
        """
        Find the most central graspable detection.
        Returns (class_name, bbox_xyxy, confidence) or None.
        """
        candidates = []
        for det in self._detections:
            if det['class'] in GRASPABLE_CLASSES and det.get('center', False):
                candidates.append(det)

        if not candidates:
            # Also try any graspable object even not in centre
            candidates = [d for d in self._detections
                          if d['class'] in GRASPABLE_CLASSES]

        if not candidates:
            return None

        # Pick largest (closest) graspable object
        candidates.sort(key=lambda d: d.get('rel_size', 0), reverse=True)
        best = candidates[0]

        # bbox is stored as rel_size only in our YOLO node.
        # We need pixel bbox — re-run on current frame if available.
        # For now return class + confidence; pixel bbox computed from rel_size.
        return best['class'], None, best['conf']

    def _sample_depth(self, bbox, frame=None):
        """
        Sample depth at the centre of the detection.
        Uses a DEPTH_SAMPLE_RADIUS×DEPTH_SAMPLE_RADIUS window and takes median.
        Returns depth in metres, or None on failure.
        """
        if self._depth_image is None:
            return None
        if not CV2_OK:
            return None

        h, w = self._depth_image.shape[:2]
        cx   = w // 2   # use image centre as proxy if no bbox pixel coords
        cy   = h // 2

        r = DEPTH_SAMPLE_RADIUS
        x0, x1 = max(0, cx-r), min(w, cx+r)
        y0, y1 = max(0, cy-r), min(h, cy+r)
        patch   = self._depth_image[y0:y1, x0:x1]

        # Filter out zero (invalid) and out-of-range readings
        valid = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
        if valid.size == 0:
            return None

        return float(np.median(valid))

    def _pixel_to_3d(self, bbox, depth_m):
        """
        Back-project image centre pixel to 3D point in camera frame.
        Returns (x, y, z) in metres (camera frame: z=forward, x=right, y=down).
        """
        if self._fx is None:
            self.get_logger().warn("Camera intrinsics not yet received")
            return None
        if self._depth_image is None:
            return None

        h, w = self._depth_image.shape[:2]
        px, py = w // 2, h // 2   # use centre; replace with bbox centre if available

        # Standard pin-hole back-projection
        x_cam = (px - self._cx) * depth_m / self._fx
        y_cam = (py - self._cy) * depth_m / self._fy
        z_cam = depth_m

        return (x_cam, y_cam, z_cam)

    def _compute_arm_angles(self, cam_point):
        """
        Convert camera-frame 3D point to arm pan angle and reach distance.

        Camera frame → robot base_link frame:
          Camera is mounted at front bumper, facing forward.
          Approximate transform: x_robot ≈ z_cam, y_robot ≈ -x_cam
        """
        x_cam, y_cam, z_cam = cam_point

        # Object position in robot base frame (approximate)
        x_robot = z_cam            # forward
        y_robot = -x_cam           # left-right (camera x inverted)

        # Pan angle: rotate shoulder to point at object
        pan_angle = math.atan2(y_robot, x_robot)
        pan_angle = float(np.clip(pan_angle, -1.2, 1.2))

        # Horizontal reach distance
        reach_dist = math.sqrt(x_robot**2 + y_robot**2)
        reach_dist = float(np.clip(reach_dist - APPROACH_STANDOFF,
                                   0.2, 0.7))

        self.get_logger().info(
            f"Arm: pan={math.degrees(pan_angle):.1f}° "
            f"reach={reach_dist:.3f}m")

        return pan_angle, reach_dist

    def _execute_grasp(self, obj_class, pan_angle, reach_depth, depth_m):
        """
        Build and send a joint trajectory to:
          1. Rise to ready
          2. Pan to face object
          3. Extend toward object (depth-adjusted)
          4. Close gripper
          5. Retract to stow
        """
        # Depth-based lift adjustment:
        # Closer object → less shoulder lift needed
        depth_factor = np.clip(depth_m / 1.0, 0.3, 1.0)
        shoulder_lift = -1.57 * depth_factor   # more negative = more horizontal
        elbow         =  1.57 * depth_factor

        self.get_logger().info(
            f"Executing grasp: class={obj_class} "
            f"pan={pan_angle:.2f} lift={shoulder_lift:.2f}")

        pts = [
            # Rise to home ready
            (HOME_READY_POS,                                      2),
            # Pan toward object, begin lowering to reach height
            ([pan_angle, shoulder_lift, elbow, -1.57, -1.57, 0.0], 4),
            # Open gripper fully
            ([pan_angle, shoulder_lift, elbow, -0.80, -1.57, 0.0], 5),
            # Extend forward — wrist extends toward object
            ([pan_angle, shoulder_lift - 0.3, elbow - 0.3, -0.80, -1.57, 0.0], 7),
            # CLOSE gripper
            ([pan_angle, shoulder_lift - 0.3, elbow - 0.3, -0.80, -1.57, 1.57], 9),
            # Retract while holding
            ([pan_angle, -1.57, 1.57, -1.57, -1.57, 1.57],      11),
            # Return to centre
            (HOME_READY_POS,                                     13),
            # Stow
            (STOW_POS,                                           16),
        ]

        if not self.arm_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("Arm server not available for grasp!")
            self._publish_result("arm_unavailable")
            self._grasp_active = False
            return

        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        traj.points      = [_pt(pos, sec) for pos, sec in pts]
        goal.trajectory  = traj

        fut = self.arm_client.send_goal_async(goal)
        fut.add_done_callback(
            lambda f: self._on_grasp_accepted(f, obj_class))

    def _on_grasp_accepted(self, future, obj_class):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("Grasp goal rejected!")
            self._publish_result("rejected")
            self._grasp_active = False
            return
        self.get_logger().info("Grasp trajectory executing...")
        handle.get_result_async().add_done_callback(
            lambda f: self._on_grasp_done(f, obj_class))

    def _on_grasp_done(self, future, obj_class):
        self.get_logger().info(f"Grasp complete — object: {obj_class}")
        self._publish_result(f"success:{obj_class}")
        self._grasp_active = False

    def _publish_result(self, result):
        msg = String()
        msg.data = json.dumps({
            "result":    result,
            "timestamp": time.time(),
        })
        self.result_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GraspNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
