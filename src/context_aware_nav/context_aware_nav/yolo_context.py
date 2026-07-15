# yolo_context.py
# ~/clearpath_ws/src/context_aware_nav/context_aware_nav/yolo_context.py
#
# Runs YOLOv8 on the robot camera and publishes:
#   /cpr_a200_0000/yolo/detections  (String — JSON summary)
#   /cpr_a200_0000/yolo/image       (Image  — annotated frame)
#
# The smart_demo node subscribes to /yolo/detections and adjusts
# behaviour based on what is detected in front of the robot.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import json
import numpy as np
from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Classes that should STOP the robot (treat as hard obstacles)
STOP_CLASSES = {
    'person', 'dog', 'cat', 'bird',          # living beings
    'stop sign', 'traffic light',             # traffic signals
}

# Classes that should make the robot slow down
SLOW_CLASSES = {
    'bicycle', 'motorcycle', 'car', 'truck',
    'bus', 'bench', 'chair', 'potted plant',
}


class YoloContextNode(Node):
    def __init__(self):
        super().__init__('yolo_context_node')

        self.bridge = CvBridge()

        # Load YOLOv8n (nano — fastest, good for real-time ROS use)
        # First run downloads weights automatically (~6MB)
        self.get_logger().info("Loading YOLOv8n...")
        self.model = YOLO('yolov8n.pt')
        self.model.fuse()   # fuse conv+bn layers for speed
        self.get_logger().info("YOLOv8n ready")

        # Subscribe to robot camera
        self.img_sub = self.create_subscription(
            Image,
            '/cpr_a200_0000/sensors/camera_0/color/image',
            self.image_callback, 5)   # queue 5 — drop old frames

        # Publish annotated image and detection summary
        self.det_pub = self.create_publisher(
            String, '/cpr_a200_0000/yolo/detections', 10)

        self.img_pub = self.create_publisher(
            Image, '/cpr_a200_0000/yolo/image', 5)

        # Latest context for other nodes to read
        self.latest_context = {
            "stop": False,
            "slow": False,
            "detections": [],
            "closest_person_dist": None,
        }

        self.get_logger().info("YoloContextNode started")

    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Resize for faster inference (320px wide)
        h, w = frame.shape[:2]
        scale = 320 / w
        small = cv2.resize(frame, (320, int(h * scale)))

        # Run YOLOv8 inference
        results = self.model(
            small,
            conf=0.40,       # confidence threshold
            iou=0.45,        # NMS IoU threshold
            verbose=False,
            device=device         # GPU; use 'cpu' if no GPU
        )[0]

        # Parse detections
        detections = []
        should_stop = False
        should_slow = False

        for box in results.boxes:
            cls_id   = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf     = float(box.conf[0])
            xyxy     = box.xyxy[0].tolist()   # [x1,y1,x2,y2] in small frame

            # Estimate relative size as proxy for distance
            # (larger box = closer object)
            box_area   = (xyxy[2]-xyxy[0]) * (xyxy[3]-xyxy[1])
            frame_area = small.shape[0] * small.shape[1]
            rel_size   = box_area / frame_area

            # Check if detection is in the central 40% of frame (forward path)
            frame_cx = small.shape[1] / 2
            box_cx   = (xyxy[0] + xyxy[2]) / 2
            in_center = abs(box_cx - frame_cx) < (small.shape[1] * 0.20)

            det = {
                "class":    cls_name,
                "conf":     round(conf, 2),
                "rel_size": round(rel_size, 3),
                "center":   in_center,
            }
            detections.append(det)

            # Only trigger navigation changes for central detections
            if in_center:
                if cls_name in STOP_CLASSES and rel_size > 0.03:
                    should_stop = True
                elif cls_name in SLOW_CLASSES and rel_size > 0.02:
                    should_slow = True

        self.latest_context = {
            "stop":        should_stop,
            "slow":        should_slow,
            "detections":  detections,
            "n_objects":   len(detections),
        }

        # Publish JSON summary
        msg_out = String()
        msg_out.data = json.dumps(self.latest_context)
        self.det_pub.publish(msg_out)

        # Draw and publish annotated image
        annotated = results.plot(img=small)
        # Draw a center-zone indicator
        cx = small.shape[1] // 2
        cv2.rectangle(annotated,
                      (cx - int(small.shape[1]*0.20), 0),
                      (cx + int(small.shape[1]*0.20), small.shape[0]),
                      (0, 255, 0), 1)
        status = "STOP" if should_stop else ("SLOW" if should_slow else "OK")
        color  = (0,0,255) if should_stop else ((0,165,255) if should_slow else (0,255,0))
        cv2.putText(annotated, status, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        img_msg.header = msg.header
        self.img_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloContextNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
