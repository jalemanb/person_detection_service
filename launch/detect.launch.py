# person_detection_temi/launch/person_detection_launch.py
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    rosbag_path = '/home/enrique/rosbags/test_ros2'  # Replace with the actual path to your rosbag file
    rviz_config_file = os.path.join(get_package_share_directory('person_detection_temi'), 'rviz', 'config.rviz')

    return LaunchDescription([
        # Launch the person_detection_node
        Node(
            package='person_detection_temi',
            executable='person_detection_node',
            name='person_detection_node',
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file]
        ),
        # Start playing the rosbag
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', rosbag_path, '--loop', '--rate', '0.5'],
            output='screen'
        ),
    ])
