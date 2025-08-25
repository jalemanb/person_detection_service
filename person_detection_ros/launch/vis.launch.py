# person_detection_ros/launch/person_detection_launch.py
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    rviz_config_file = os.path.join(get_package_share_directory('person_detection_ros'), 'rviz', 'config.rviz')

    return LaunchDescription([
        # Launch RViZ with the proper configuration
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file]
        ),
    ])
