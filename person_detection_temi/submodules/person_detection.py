# person_detection_temi/pose_estimation_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from realsense2_camera_msgs.msg import RGBD
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
import numpy as np
import cv2
import open3d as o3d
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pcl2
from std_msgs.msg import Header
import torch 
from person_detection_temi.submodules.SOD import SOD
import os
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation as R

class HumanPoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')

        # Create subscribers with message_filters
        self.rgbd_subscription = self.create_rgbd_subscription(RGBD,'/camera/camera/rgbd',self.image_callback,10)
        self.rgbd_subscription

        # Create publishers
        self.publisher_human_pose = self.create_publisher(PoseWithCovarianceStamped, '/human_pose', 10)
        self.publisher_pointcloud = self.create_publisher(PointCloud2, '/human_pointcloud', 10)
        self.publisher_debug_detection_image = self.create_publisher(CompressedImage, '/human_detection_debug/compressed/human_detected', 10)

        # Create a TransformBroadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Bridge to convert ROS messages to OpenCV
        self.cv_bridge = CvBridge()

        # Single Person Detection model
        # Setting up Available CUDA device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Setting up model paths (YOLO for object detection and segmentation, and orientation estimation model)
        pkg_shared_dir = get_package_share_directory('person_detection_temi')
        yolo_path = os.path.join(pkg_shared_dir, 'models', 'yolov8n-seg.pt')
        orientation_path = os.path.join(pkg_shared_dir, 'models', 'rgbd_resnet18.onnx')
        # Loading Template IMG
        template_img_path = os.path.join(pkg_shared_dir, 'template_imgs', 'template_rgb.png')
        self.template_img = cv2.imread(template_img_path)

        # Setting up Detection Pipeline
        self.model = SOD(yolo_path, orientation_path)
        self.model.to(device)
        self.get_logger().warning('Deep Learning Model Armed')

        # Warmup inference (GPU can be slow in the first inference)
        self.model.detect(np.ones((720, 1280, 3), dtype=np.uint8), np.ones((720, 1280), dtype=np.uint16), self.template_img, detection_thr = 0.3)
        self.get_logger().warning('Warmup Inference Executed')

        # Frame ID from where the human is being detected
        self.frame_id = None

        # Quaternion for 90 degrees rotation around the x-axis
        rot_x = R.from_euler('x', 90, degrees=True)

        # Quaternion for -90 degrees rotation around the z-axis
        rot_z = R.from_euler('z', -90, degrees=True)

        # Combine the rotations
        self.combined_rotation = rot_x * rot_z

    def image_callback(self, rgbd_msg):

        self.get_logger().warning('Ultimate TEMI Ready to Roll')

        return

        self.frame_id = rgbd_msg.camera_depth_optical_frame.header.frame_id

        # Convert ROS Image messages to OpenCV images
        depth_image = self.cv_bridge.imgmsg_to_cv2(rgbd_msg.depth, desired_encoding='passthrough')
        rgb_image = self.cv_bridge.imgmsg_to_cv2(rgbd_msg.rgb, desired_encoding='bgr8')

        # Check if RGB and depth images are the same size
        if depth_image.shape != rgb_image.shape[:2]:
            self.get_logger().warning('Depth and RGB images are not the same size. Skipping this pair.')
            return

        # Process the images and estimate pose
        self.process_images(rgb_image, depth_image)

    def process_images(self, rgb_image, depth_image):
        # Your pose estimation logic here
        # For demonstration, let's assume we get the pose from some function
        person_pose, person_orientation, bbox, pcd = self.model.detect(rgb_image, depth_image, self.template_img, detection_thr = 0.3)

        if person_pose is False:
            return
        
        else:

            #Publish Image with detection Bounding Box for Visualizing the proper detection of the desired target person
            if self.publisher_debug_detection_image.get_rgbd_subscription_count() > 0:
                self.get_logger().warning('Publishing Images with Detections for Debugging Purposes')
                self.publish_debug_img(rgb_image, bbox)

            # Generate and publish the point cloud
            if self.publisher_pointcloud.get_rgbd_subscription_count() > 0:
                self.get_logger().warning('Publishing Pointcloud that belongs to the desired Human')
                self.publish_pointcloud(np.asarray(pcd.points))

            # Generate and publish both the tf and the posewith covariance stamped for the desired detected person
            if self.publisher_human_pose.get_rgbd_subscription_count() > 0:
                self.get_logger().warning('Publishing Pose and yaw orientation that belongs to the desired Human')
                composed_orientation = self.combined_rotation * R.from_quat(person_orientation)
                self.publish_human_pose(person_pose, composed_orientation.as_quat())
                self.broadcast_human_pose(person_pose, composed_orientation.as_quat())


    def publish_debug_img(self, rgb_img, box):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.publisher_debug_detection_image.publish(self.bridge.cv2_to_compressed_imgmsg(rgb_img))

    def publish_human_pose(self, pose, orientation):
        # Publish the pose with covariance
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose.pose.position.x = pose[0]
        pose_msg.pose.pose.position.y = pose[1]
        pose_msg.pose.pose.position.z = pose[2]
        # Set the rotation using the composed quaternion
        pose_msg.pose.pose.orientation.x = orientation[0]
        pose_msg.pose.pose.orientation.y = orientation[1]
        pose_msg.pose.pose.orientation.z = orientation[2]
        pose_msg.pose.pose.orientation.w = orientation[3]
        self.publisher_human_pose.publish(pose_msg)

    def broadcast_human_pose(self, pose, orientation):
        # Broadcast the transform
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.frame_id
        transform.child_frame_id = "target_human"
        transform.transform.translation.x = pose[0]
        transform.transform.translation.y = pose[1]
        transform.transform.translation.z = pose[2]
        # Set the rotation using the composed quaternion
        transform.transform.rotation.x = orientation[0]
        transform.transform.rotation.y = orientation[1]
        transform.transform.rotation.z = orientation[2]
        transform.transform.rotation.w = orientation[3]        
        self.tf_broadcaster.sendTransform(transform)

    def publish_pointcloud(self, points):
        # Fill the Header
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        # Create PointCloud2 message
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        pointcloud = pcl2.create_cloud(header, fields, points)

        self.publisher_pointcloud.publish(pointcloud)