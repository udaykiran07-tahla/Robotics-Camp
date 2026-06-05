#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class FlightCommander(Node):

    def __init__(self):
        super().__init__('flight_commander')

        self.publisher_ = self.create_publisher(
            Twist,
            'flight_commands',
            10
        )

        self.timer = self.create_timer(1.0, self.publish_command)

    def publish_command(self):
        msg = Twist()

        # Forward velocity
        msg.linear.x = 2.0

        # Turning velocity
        msg.angular.z = 0.5

        self.publisher_.publish(msg)

        self.get_logger().info(
            f'Publishing -> Linear: {msg.linear.x}, Angular: {msg.angular.z}'
        )


def main(args=None):
    rclpy.init(args=args)

    node = FlightCommander()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

