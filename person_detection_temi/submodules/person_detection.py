# person_detection_temi/pose_estimation_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from realsense2_camera_msgs.msg import RGBD
from person_detection_msgs.msg import BoundingBox
from person_detection_msgs.msg import BoundingBoxArray
from geometry_msgs.msg import Pose, TransformStamped, PoseArray
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch 
from person_detection_temi.submodules.SOD import SOD
import os
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation as R
import time

class HumanPoseEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_estimation_node')

        # Create subscribers with message_filters
        self.rgbd_subscription = self.create_subscription(RGBD,'/camera/camera/rgbd',self.image_callback,10)
        self.rgbd_subscription

        # Create publishers
        self.publisher_human_pose = self.create_publisher(PoseArray, '/detections', 10)
        self.publisher_debug_detection_image_compressed = self.create_publisher(CompressedImage, '/human_detection_debug/compressed/human_detected', 10)
        self.publisher_debug_detection_image = self.create_publisher(Image, '/human_detection_debug/human_detected', 10)
        self.publisher_debug_bounding_boxes = self.create_publisher(BoundingBoxArray, '/human_detection_debug/bounding_boxes', 10)

        # Create a TransformBroadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Bridge to convert ROS messages to OpenCV 
        self.cv_bridge = CvBridge()

        self.draw_box = False

        # Single Person Detection model
        # Setting up Available CUDA device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Setting up model paths (YOLO for object detection and segmentation, and orientation estimation model)
        pkg_shared_dir = get_package_share_directory('person_detection_temi')
        yolo_path = os.path.join(pkg_shared_dir, 'models', 'yolov8n-segpose.engine')
        tracker_path = os.path.join(pkg_shared_dir, 'models', 'bytetrack.yaml')
        resnet_path = os.path.join(pkg_shared_dir, 'models', 'osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth')
        
        # Loading Template IMG
        template_img_path = os.path.join(pkg_shared_dir, 'template_imgs', 'template_rgb_ultimate_2.png')
        # template_img_path = os.path.join(pkg_shared_dir, 'template_imgs', 'template_rgb_hallway_2.jpg')

        self.template_img = cv2.imread(template_img_path)

        # Setting up Detection Pipeline
        self.model = SOD(yolo_path, resnet_path, tracker_path)
        self.model.to(device)
        self.get_logger().warning('Deep Learning Model Armed')

        # Initialize the template
        self.model.template_update(self.template_img)

        # Warmup inference (GPU can be slow in the first inference)
        self.model.detect(img_rgb = np.ones((480, 640, 3), dtype=np.uint8), img_depth = np.ones((480, 640), dtype=np.uint16))
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

        self.frame_id = rgbd_msg.depth.header.frame_id

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

        start_time = time.time()
        ############################
        results = self.model.detect(rgb_image, depth_image)
        ############################
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
            
        self.get_logger().warning(f"Model Inference Time: {execution_time} ms")

        person_poses = []
        bbox = []
        kpts = []
        conf = []
        valid_idxs = []
        tracked_ids = []

        if results is not None:
            self.draw_box = True

            person_poses, bbox, kpts, tracked_ids, conf, valid_idxs = results
            self.get_logger().warning(f"CONFIDENCE: {conf} %")

            person_poses = person_poses[valid_idxs]

            person_poses = self.convert_to_frame(person_poses, self.frame_id, "base_link")

            print("person_poses", person_poses)

            # Generate and publish the point cloud
            if self.publisher_human_pose.get_subscription_count() > 0 and person_poses is not None:
                self.get_logger().warning('Publishing Person Pose that belongs to the desired Human')
                # self.broadcast_human_pose(person_poses, [0., 0., 0., 1.])
                self.publish_human_pose(person_poses, [0., 0., 0., 1.], "base_link")

        # Publish CompressedImage with detection Bounding Box for Visualizing the proper detection of the desired target person
        if self.publisher_debug_detection_image_compressed.get_subscription_count() > 0:
            self.get_logger().warning('Publishing Compressed Images with Detections for Debugging Purposes')
            self.publish_debug_img(rgb_image, bbox, kpts = kpts, valid_idxs = valid_idxs, compressed=True, conf = conf)

        #Publish Image with detection Bounding Box for Visualizing the proper detection of the desired target person
        if self.publisher_debug_detection_image.get_subscription_count() > 0:
            self.get_logger().warning('Publishing Images with Detections for Debugging Purposes')
            self.publish_debug_img(rgb_image, bbox, kpts = kpts, valid_idxs = valid_idxs, confidences = conf,  tracked_ids = tracked_ids, compressed=False, conf = conf)


    def publish_debug_img(self, rgb_img, boxes, kpts, valid_idxs, confidences, tracked_ids, compressed = True, conf = 0.5):
        color_kpts = (255, 0, 0) 
        radius_kpts = 10
        thickness = 2

        print("confidences", confidences)

        bounding_boxes_list = []

        if len(boxes) > 0 and len(kpts) > 0:

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box

                individual_bounding_box = BoundingBox()
                individual_bounding_box.x1 = int(x1)
                individual_bounding_box.y1 = int(y1)
                individual_bounding_box.x2 = int(x2)
                individual_bounding_box.y2 = int(y2)

                bounding_boxes_list.append(individual_bounding_box)

                # cv2.putText(rgb_img, f"{conf * 100:.2f}%" , (x1, y1 + int((y2-y1)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), thickness)

                # if i == target_idx and conf < 600.:
                # if target_idx[i]: #and conf < 0.8:
                if i in valid_idxs:
                    alpha = 0.2
                    overlay = rgb_img.copy()
                    cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, alpha, rgb_img, 1 - alpha, 0, rgb_img)
                    individual_bounding_box.tgt = True

                # Just for debugging
                cv2.putText(rgb_img, f"{confidences[i]:.2f}" , (x2-10, y2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(rgb_img, f"ID: {tracked_ids[i]}" , (x1, y1 + int((y2-y1)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            
                kpt = kpts[i]
                for j in range(kpt.shape[0]):
                    u = kpt[j, 0]
                    v = kpt[j, 1]
                    cv2.circle(rgb_img, (u, v), radius_kpts, color_kpts, thickness)



        if compressed:
            self.publisher_debug_detection_image_compressed.publish(self.cv_bridge.cv2_to_compressed_imgmsg(rgb_img))
        else:
            self.publisher_debug_detection_image.publish(self.cv_bridge.cv2_to_imgmsg(rgb_img, encoding='bgr8'))
        
        # Publishing the bounding boxes by frame
        bounding_boxes_msg = BoundingBoxArray()
        bounding_boxes_msg.boxes = bounding_boxes_list
        self.publisher_debug_bounding_boxes.publish(bounding_boxes_msg)


    def publish_human_pose(self, poses, orientation, frame_id):

        # Publish the pose with covariance
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()
        pose_array_msg.header.frame_id = frame_id

        for pose in poses:
            pose_msg = Pose()
            # Set the rotation using the composed quaternion
            pose_msg.position.x = pose[0]
            pose_msg.position.y = pose[1]
            pose_msg.position.z = 0.
            # Set the rotation using the composed quaternion
            pose_msg.orientation.x = 0.
            pose_msg.orientation.y = 0.
            pose_msg.orientation.z = 0.
            pose_msg.orientation.w = 1.
            # Create the pose Array
            pose_array_msg.poses.append(pose_msg)

        self.publisher_human_pose.publish(pose_array_msg)


    def convert_to_frame(self, poses, from_frame = 'source_frame', to_frame = 'base_link'):
        
        # Delete rows that contain -100 values in the columns (invalid pose)
        rows_to_delete = np.all(poses == -100, axis=1)
        poses = poses[~rows_to_delete]

        if len(poses) == 0:
            return None

        transform = self.tf_buffer.lookup_transform(to_frame, from_frame, rclpy.time.Time())

        # Extract translation and rotation
        translation = transform.transform.translation
        rotation = transform.transform.rotation

        # Convert quaternion to rotation matrix
        quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
        rotation_matrix = R.from_quat(quaternion).as_matrix()

        # Apply the rotation first
        poses_rotated = rotation_matrix.dot(poses.T)

        # Apply the translation (summation)
        poses_transformed = poses_rotated + np.array([[translation.x], [translation.y], [translation.z]])

        return poses_transformed.T.tolist()
    

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