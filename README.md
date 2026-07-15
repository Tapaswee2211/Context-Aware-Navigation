# Context-Aware Planner: Dynamic Costmap Adaptation for Autonomous Mobile Robots

A ROS 2-based autonomous mobile robot system that dynamically adapts its navigation behavior and motion planning parameters in real time by fusing LiDAR-based spatial context classification with semantic object detection via YOLOv8. Validated on the Clearpath Husky A200 with 6-DOF robotic arm integration in Gazebo simulation.

## Overview

Autonomous mobile robots operating in unstructured or semi-structured environments face a fundamental challenge: traditional static costmap configurations in ROS 2 navigation stacks assign fixed inflation radii and cost scaling factors, which are inadequate for environments that transition between open corridors, narrow aisles, and crowded spaces. This mismatch causes inefficient navigation paths, higher collision risk, and poor performance in diverse spatial configurations.

This project presents a solution through the integration of three key components: a Context Manager that classifies environments in real time using LiDAR obstacle density; dynamic parameter reconfiguration of Nav2's local costmap; and a YOLO-based vision pipeline that adds semantic awareness to geometric obstacle detection. The system is demonstrated to successfully navigate complex multi-waypoint missions with active obstacle avoidance and deadlock recovery capabilities.

## Key Features

**Dynamic Context Classification**

- Real-time environment categorization into OPEN, NARROW, and CROWDED spatial contexts using LiDAR density metrics
- Automatic reconfiguration of Nav2 costmap parameters (inflation radius, cost scaling factor) based on context
- Waypoint-specific speed and turn-commitment parameters that adapt to environmental constraints

**Semantic Vision Integration**

- YOLOv8n object detection pipeline running at real-time frame rates
- Behavioral classification: STOP for persons and animals, SLOW for vehicles and obstacles
- Center-zone filtering to distinguish relevant forward-path objects from peripheral detections

**Structured Navigation State Machine**

- Eight-state FSM (STARTUP, NAVIGATING, TURN_COMMIT, BLOCKED, ARM_BLOCKED, RECOVERY, WAYPOINT_ARRIVED, MISSION_COMPLETE)
- Committed point-turn obstacle avoidance with directional bias toward goal waypoints
- Automatic deadlock detection and arm-assisted recovery mechanisms

**Mobile Manipulation**

- Autonomous robotic arm deployment for blocked-state recovery and environmental assessment
- Vision-guided grasp pipeline using RGB-D depth data, camera intrinsics, and 3D back-projection
- Joint-space trajectory control with depth-adaptive end-effector positioning

**Comprehensive Telemetry**

- Time-series logging of distances, velocities, sector readings, and state transitions
- Vision event logging with annotated detection frames saved at configurable intervals
- Automated graph generation for post-hoc mission analysis

## System Architecture

The system is structured as a loosely-coupled collection of ROS 2 nodes communicating through well-defined topic interfaces:

- **context_manager**: Classifies spatial environment from LiDAR density; publishes context updates and triggers Nav2 costmap reconfiguration
- **yolo_context_node**: Runs YOLOv8n inference on camera frames; publishes STOP/SLOW signals and detection JSON
- **smart_nav_node**: Central navigation FSM; processes sensor inputs and executes velocity commands and arm trajectories
- **grasp_node**: Computes 3D grasp points from depth data; executes object manipulation trajectories
- **waypoint_marker_node**: Publishes RViz markers and Gazebo cylinder visualizations for waypoint tracking

### Data Flow

```
LiDAR Scan → context_manager → Nav2 Costmap Update
         → smart_nav_node → cmd_vel (Robot Base)

RGB Camera → yolo_context_node → Detection JSON
           → smart_nav_node → Navigation Response

Depth Camera + CameraInfo → grasp_node → Arm Trajectory
```

## Requirements

**Hardware**

- Clearpath Husky A200 mobile base (or compatible differential-drive robot)
- 2D LiDAR sensor (Hokuyo UST-10LX or equivalent, 270° FoV, 30m range)
- Monocular RGB camera (30+ FPS, 640×480 minimum)
- Depth camera (Intel RealSense-style, float32 depth in meters)
- 6-DOF serial manipulator arm

**Software**

- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill (or Iron Irwini)
- Nav2 navigation stack
- Gazebo Fortress/Garden (for simulation)
- Python 3.10+
- YOLOv8 (via Ultralytics)
- OpenCV 4.5+

**Python Dependencies**

```
rclpy>=0.16
nav2-msgs
std-msgs
sensor-msgs
geometry-msgs
tf2-ros
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.20
matplotlib>=3.5
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Tapaswee2211/Context-Aware-Navigation.git
cd Context-Aware-Navigation
```

### 2. Create ROS 2 Workspace

```bash
mkdir -p ~/amr_ws/src
cd ~/amr_ws/src
ln -s /context-aware-amr .
cd ~/amr_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 3. Install Python Dependencies

```bash
pip install ultralytics opencv-python numpy matplotlib
```

### 4. Build the Workspace

```bash
colcon build --symlink-install
source install/setup.bash
```

### 5. Download YOLO Weights

The YOLOv8n model weights will be downloaded automatically on first run, or manually:

```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

## Usage

### Launch Gazebo Simulation

```bash
ros2 launch context_aware_amr gazebo_world.launch.py
```

### Run the Navigation System

In a separate terminal:

```bash
source ~/amr_ws/install/setup.bash
ros2 launch context_aware_amr navigation.launch.py
```

This will start all five nodes: context_manager, yolo_context_node, smart_nav_node, grasp_node, and waypoint_marker_node.

### Monitor in RViz

```bash
ros2 run rviz2 rviz2 -c src/Context-Aware-Navigation/config/rviz_config.rviz
```

### Publish Waypoints

Send the robot to a series of waypoints using the command line:

```bash
ros2 topic pub -1 /smart_nav/goal geometry_msgs/PoseStamped \
  "header: {stamp: now, frame_id: 'map'}, pose: {position: {x: 3.0, y: 0.0, z: 0.0}, orientation: {x: 0, y: 0, z: 0, w: 1}}"
```

Or use the waypoint_publisher utility:

```bash
ros2 run context_aware_amr waypoint_publisher --waypoints "[(3.0, 0.0), (3.0, 3.0), (0.0, 3.0), (0.0, 0.0)]"
```

## Validation Results

### Mission Overview

A five-waypoint navigation mission was conducted in a Gazebo simulation environment populated with static and dynamic obstacles:

- **Total Mission Duration**: ~330 seconds
- **Waypoints Completed**: 5/5 (100%)
- **Obstacle Encounters**: Multiple point-turn avoidance maneuvers
- **Deadlock Events**: 1 (successfully resolved via arm deployment)
- **YOLO Detections**: 44 person detections, 2 surfboard, plus bird, frisbee, banana, umbrella

### Performance Metrics

| Metric                        | Value                |
| ----------------------------- | -------------------- |
| Mission Completion Rate       | 100% (5/5 waypoints) |
| Minimum Obstacle Distance     | 0.4m                 |
| Emergency Threshold Crossings | 9                    |
| Maximum Linear Velocity       | 0.45 m/s             |
| Context Classification Events | 8 transitions        |
| ARM_BLOCKED Deployments       | 1                    |
| YOLO STOP Compliance          | 100%                 |

### Key Findings

1. **Dynamic Costmap Adaptation**: Switching inflation radius from 0.20m (OPEN) to 0.55m (CROWDED) enabled the robot to successfully navigate both spacious loading areas and constrained shelving aisles within the same mission.

2. **Vision-Guided Safety**: Person detection triggered immediate stops at 8 separate locations, with zero false positives, confirming the effectiveness of semantic-aware obstacle response.

3. **Deadlock Recovery**: When the robot became fully surrounded by obstacles after 75 seconds of navigation, the ARM_BLOCKED state triggered an autonomous arm sweep sequence that improved situational awareness, followed by a reverse-and-rotate recovery maneuver that successfully cleared the blockage.

4. **Velocity Adaptation**: Speed commands transitioned smoothly between 0.45 m/s (open areas) and 0.15 m/s (narrow/crowded contexts), demonstrating real-time responsiveness to environment changes.

## Results Visualization

### Obstacle Distance Over Time

![Closest Obstacle Distance](./graphs/10%20vision%20obstacle/01_distance.png)

The distance plot shows the robot's minimum approach distance to obstacles throughout the mission. Dips to the emergency threshold (0.45m, red dashed line) correspond to BLOCKED state activation and turn-commitment avoidance maneuvers.

### Robot State Timeline

![Robot State Timeline](./graphs/10%20vision%20obstacle/02_state_timeline.png)

The color-coded state timeline visualizes transitions between navigation states. NAVIGATING (green) dominates, with TURN_COMMIT (orange) and ARM_BLOCKED (purple) segments indicating obstacle avoidance and deadlock recovery events.

### Velocity Commands

![Command Velocities](./graphs/10%20vision%20obstacle/03_velocity.png)

Linear velocity (blue) shows context-driven speed selection, while angular velocity (orange) exhibits characteristic point-turn patterns during obstacle avoidance. Speed reduction is visible in narrow passages (t=50-100s).

### LiDAR Sector Distances

![LIDAR Sector Analysis](./graphs/10%20vision%20obstacle/04_sectors.png)

The three sector distances (center, left, right) reveal asymmetric obstacle distributions. Frequent separation of traces indicates the robot navigated through non-symmetric environments requiring directional selection during turns.

### Navigation Path with Waypoints

![Robot Path](./graphs/10%20vision%20obstacle/05_path.png)

The trajectory plot shows the robot's actual path (color-coded by time progression) through all five waypoints, with visible deviations at obstacles and smooth arrivals at waypoint locations.

### Object Detection Summary

![Detection Counts](./graphs/10%20vision%20obstacle/06_detection_counts.png)

YOLO detected 44 person instances, 2 surfboards, and single instances of bird, frisbee, banana, and umbrella. Person class dominance reflects the stationary human-shaped obstacles in the simulation environment.

### Detection Events Timeline

![Detection Timeline](./graphs/10%20vision%20obstacle/07_detection_timeline.png)

Detection events cluster around t=200-250s (WP3 to WP4 transit) and t=280-300s (WP4 to WP5 transit), corresponding to obstacle-dense regions of the path.

## Video Demonstration

[**Video Demonstrating Full Mission Execution**](./demo.webm)

This video shows:

- Initial robot startup and arm salute sequence
- Navigation through open areas with context classification as OPEN
- Entry into narrow shelving aisles with dynamic costmap reconfiguration to NARROW
- YOLO-driven stops in response to detected persons
- ARM_BLOCKED state activation and autonomous arm deployment during deadlock
- Recovery maneuver execution and return to navigation
- Successful waypoint arrivals and final mission completion

Recommended playback speed: 2x or 4x (original mission runtime 330 seconds).

---

## Code Snippets

### Context Manager - LiDAR Density Classification

The following snippet demonstrates how the context manager classifies environments in real time:

```python
def classify_context(self, ranges):
    """Classify environment based on LiDAR obstacle density."""
    valid_ranges = [r for r in ranges if 0.1 < r < float('inf')]
    density = len(valid_ranges) / len(ranges) if len(ranges) > 0 else 0.0

    if density < 0.50:
        return "OPEN", 0.20, 2.0, 0.45, 2.0
    elif density < 0.75:
        return "NARROW", 0.35, 5.0, 0.25, 2.8
    else:
        return "CROWDED", 0.55, 7.0, 0.15, 3.5
```

### YOLO Vision Node - Detection and Classification

The vision node processes camera frames and extracts STOP/SLOW signals:

```python
def process_detections(self, detections, frame_width):
    """Convert YOLO detections to navigation commands."""
    should_stop = False
    should_slow = False

    for detection in detections:
        class_name = detection['class']
        confidence = detection['confidence']
        bbox = detection['bbox']  # [x, y, w, h]

        # Relative bounding box area as distance proxy
        rel_size = (bbox['w'] * bbox['h']) / (frame_width ** 2)

        # Center-zone check: is object in forward path?
        bbox_center_x = bbox['x'] + bbox['w'] / 2
        in_center = abs(bbox_center_x - frame_width / 2) < (0.2 * frame_width)

        if class_name in ["person", "dog", "cat", "bird"] and in_center and rel_size > 0.03:
            should_stop = True
        elif class_name in ["car", "truck", "bicycle"] and in_center and rel_size > 0.02:
            should_slow = True

    return should_stop, should_slow
```

### Smart Navigation - State Machine Core

The navigation state machine demonstrates the point-turn logic:

```python
def navigate(self, scan_data):
    """Execute state machine navigation logic."""
    min_all = min(scan_data['center'], scan_data['left'], scan_data['right'])

    if self.state == "NAVIGATING":
        if min_all < 0.45:  # EMERGENCY_DIST
            self.state = "BLOCKED"
            self.blocked_since = time.time()
        else:
            angle_to_wp = self.compute_heading()
            if min_all > 2.0:  # OBSTACLE_CAUTION
                # Clear path
                self.cmd_vel = Twist(
                    linear=Velocity(x=self.speed_limit),
                    angular=Velocity(z=min(max(angle_to_wp * 1.5, -0.5), 0.5))
                )
            else:
                # Obstacle detected - select turn direction
                open_left = scan_data['left'] + scan_data['far_left']
                open_right = scan_data['right'] + scan_data['far_right']

                if angle_to_wp > 0:
                    open_left += 0.5  # Bias toward waypoint
                else:
                    open_right += 0.5

                turn_direction = 0.65 if open_left >= open_right else -0.65
                self.state = "TURN_COMMIT"
                self.turn_start = time.time()

                self.cmd_vel = Twist(
                    linear=Velocity(x=0.0),
                    angular=Velocity(z=turn_direction)
                )

    elif self.state == "BLOCKED":
        if min_all > 0.55:
            self.state = "NAVIGATING"
        elif time.time() - self.blocked_since > 3.0:
            self.state = "ARM_BLOCKED"
            self.deploy_arm_sequence("blocked_recovery")
```

### Grasp Node - 3D Back-Projection Pipeline

The depth-based grasp pipeline:

```python
def compute_grasp_point(self, detection, depth_image, camera_info):
    """Compute 3D grasp point from 2D detection and depth."""
    # Extract depth at bounding box center
    bbox = detection['bbox']
    bbox_center_x = int(bbox['x'] + bbox['w'] / 2)
    bbox_center_y = int(bbox['y'] + bbox['h'] / 2)

    # Sample depth with median filtering
    depth_patch = depth_image[
        max(0, bbox_center_y - 4):min(depth_image.shape[0], bbox_center_y + 4),
        max(0, bbox_center_x - 4):min(depth_image.shape[1], bbox_center_x + 4)
    ]
    depth_m = np.median(depth_patch[depth_patch > 0.2])

    if np.isnan(depth_m) or depth_m > 1.5:
        return None

    # Pinhole camera back-projection
    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]

    x_cam = (bbox_center_x - cx) * depth_m / fx
    y_cam = (bbox_center_y - cy) * depth_m / fy
    z_cam = depth_m

    # Transform to robot base_link frame
    x_robot = z_cam
    y_robot = -x_cam

    # Compute arm pan and reach
    pan_angle = math.atan2(y_robot, x_robot)
    reach_distance = math.sqrt(x_robot**2 + y_robot**2)

    return pan_angle, reach_distance, z_cam
```

---

## Project Status

**Current State**: Alpha (Active Development)

The system is fully functional in Gazebo simulation and successfully demonstrates all core concepts:

- Dynamic costmap adaptation works reliably
- YOLO integration provides semantic context
- State machine navigation handles multi-waypoint missions
- Mobile manipulation and deadlock recovery operational

**In Progress**:

- Real-world hardware deployment on physical Clearpath Husky A200
- Integration of 3D LiDAR for volumetric obstacle representation
- SLAM backend (RTAB-Map) for drift correction in real deployments
- Learning-based grasp planning using reinforcement learning

**Known Limitations**:

- Odometry drift accumulation in real hardware (no SLAM yet)
- Pre-computed arm trajectories limited to specific object classes and orientations
- 2D LiDAR cannot detect overhanging obstacles or negative obstacles (drop-offs)
- Binary STOP/SLOW vision response (no graduated stopping distances)

## Future Work

- **3D Point Clouds**: Integration of RGB-D or 3D LiDAR for voxel-based costmap representation
- **Real Hardware Deployment**: Sensor noise handling, SLAM integration, and URDF calibration for physical platform
- **Learning-Based Manipulation**: Reinforcement learning policies for adaptive grasping and novel object configurations
- **Multi-Robot Fleet Coordination**: Distributed context sharing and collision avoidance negotiation
- **Natural Language Task Specification**: LLM integration for language-conditioned navigation goals

## Contributing

For contributions, bug reports, or feature requests, please open an issue or submit a pull request.

## References

1. S. Thrun, W. Burgard, and D. Fox, _Probabilistic Robotics_. MIT Press, 2005.
2. S. Macenski et al., "Robot Operating System 2: Design, Architecture, and Uses in the Wild," _Science Robotics_, vol. 7, no. 66, 2022.
3. E. Marder-Eppstein et al., "The Office Marathon: Robust Navigation in an Indoor Office Environment," in _Proc. ICRA_, 2010.
4. D. Fox, W. Burgard, and S. Thrun, "The Dynamic Window Approach to Collision Avoidance," _IEEE RA Magazine_, vol. 4, no. 1, 1997.
5. J. Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection," in _Proc. CVPR_, 2016.
6. Ultralytics, "YOLOv8 Documentation," https://docs.ultralytics.com, 2023.
7. Clearpath Robotics, "Husky A200 Robot Platform," https://clearpathrobotics.com, 2023.
