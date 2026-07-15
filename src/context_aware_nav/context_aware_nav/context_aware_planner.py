import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

class ContextAwarePlanner(Node):
    def __init__(self):
        super().__init__('context_aware_costmap_node')
        
        # 1. Listen to the Clearpath LiDAR
        self.subscription = self.create_subscription(
            LaserScan,
            '/cpr_a200_0000/sensors/lidar2d_0/scan',
            self.scan_callback,
            10)
            
        # 2. Client to dynamically change Nav2 Local Costmap parameters
        self.param_client = self.create_client(
            SetParameters, 
            '/cpr_a200_0000/local_costmap/local_costmap/set_parameters')
            
        self.in_narrow_aisle = False
        self.get_logger().info("Context-Aware Costmap Planner Active.")

    def scan_callback(self, msg):
        # Focus on the laser beams directly in front of the robot (middle 40 beams)
        center_index = len(msg.ranges) // 2
        front_ranges = msg.ranges[center_index - 20 : center_index + 20]
        
        # Filter out 'inf' or 0.0 readings
        valid_ranges = [r for r in front_ranges if 0.1 < r < 10.0]
        if not valid_ranges:
            return
            
        min_front_dist = min(valid_ranges)

        # Context Logic: Adapt Costmap based on proximity to obstacles
        if min_front_dist < 1.2 and not self.in_narrow_aisle:
            self.get_logger().warn('CONTEXT CHANGE: Entering narrow space. Shrinking costmap inflation!')
            self.set_costmap_inflation(0.2)  # Allow robot to get closer to walls
            self.in_narrow_aisle = True
            
        elif min_front_dist >= 1.2 and self.in_narrow_aisle:
            self.get_logger().info('CONTEXT CHANGE: Entering open space. Restoring safe costmap.')
            self.set_costmap_inflation(0.55) # Default safer distance
            self.in_narrow_aisle = False

    def set_costmap_inflation(self, radius):
        while not self.param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for Nav2 local_costmap service...')
            
        req = SetParameters.Request()
        param = Parameter()
        param.name = 'inflation_layer.inflation_radius'
        param.value.type = ParameterType.PARAMETER_DOUBLE
        param.value.double_value = radius
        req.parameters.append(param)
        
        self.param_client.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = ContextAwarePlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
