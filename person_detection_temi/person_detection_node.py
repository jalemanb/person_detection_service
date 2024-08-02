#!/usr/bin/env python3
import rclpy
from .submodules.person_detection import HumanPoseEstimationNode

def main(args=None):
    rclpy.init(args=args)
    node = HumanPoseEstimationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()