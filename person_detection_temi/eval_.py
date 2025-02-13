#!/usr/bin/env python3
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

def transform_camera_to_ros(points):
    """
    Transform points from the camera optical frame to the ROS reference frame.

    Parameters:
    - points: np.ndarray of shape (n, 3), where each row is (x, y, z) in the camera frame.

    Returns:
    - transformed_points: np.ndarray of shape (n, 3), transformed to the ROS frame.
    """
    # Transformation matrix
    transform_matrix = np.array([
        [0,  0,  1],
        [-1, 0,  0],
        [0, -1,  0]
    ])
    
    # Apply the transformation
    transformed_points = points @ transform_matrix.T
    
    return transformed_points

def write_positions_to_file(positions_per_step, output_file, append=False):
    """
    Writes x and y positions to a file.

    Parameters:
    - positions_per_step: List of lists, where each inner list contains positions for a single step.
      Each position is in the format [x, y]. If the step or position is None, x and y will be set to -1.
    - output_file: Path to the output .txt file.
    - append: If True, appends to the file instead of overwriting it. Default is False.
    """
    mode = 'a' if append else 'w'
    rows = []

    for positions in positions_per_step:
        row = []  # Start with an empty row

        if positions is None:  # Handle None for the entire step
            row.extend([-1, -1])
        else:
            for pos in positions:
                if pos is None:
                    x, y = -1, -1
                else:
                    x, y = pos
                row.extend([x, y])  # Append x and y

        rows.append(row)

    # Convert to NumPy array
    rows_array = np.array(rows, dtype=object)

    # Save to file
    with open(output_file, mode) as f:
        for row in rows_array:
            f.write(','.join(map(str, row)) + '\n')

def write_bounding_boxes_to_file(bounding_boxes_per_step, output_file, append=False):
    """
    Writes bounding box coordinates to a file.

    Parameters:
    - bounding_boxes_per_step: List of lists, where each inner list contains bounding boxes for a single step.
      Each bounding box is in the format [x1, y1, x2, y2]. If the step is None, it will be skipped.
    - output_file: Path to the output .txt file.
    - append: If True, appends to the file instead of overwriting it. Default is False.
    """
    mode = 'a' if append else 'w'
    rows = []

    for seq_index, bounding_boxes in enumerate(bounding_boxes_per_step):
        row = [seq_index]  # Start with the sequence index

        if bounding_boxes is None:  # Handle None for the entire step
            row.extend([0, 0, 0, 0])
        else:
            for bbox in bounding_boxes:
                if bbox is None:
                    x, y, w, h = 0, 0, 0, 0
                else:
                    x1, y1, x2, y2 = bbox
                    x = x1
                    y = y1
                    w = x2 - x1
                    h = y2 - y1
                row.extend([x, y, w, h])  # Append converted bounding box

        rows.append(row)

    # Convert to NumPy array
    rows_array = np.array(rows, dtype=object)

    # Save to file
    with open(output_file, mode) as f:
        for row in rows_array:
            f.write(' '.join(map(str, row)) + '\n')

def denormalize_depth_image(normalized_depth, min_depth=0, max_depth=6000):
    """
    Denormalizes a depth image from the normalized range (0-254) to the actual depth range (in mm).

    Parameters:
    - normalized_depth: Normalized depth image (numpy array).
    - min_depth: Minimum depth value (default: 0 mm).
    - max_depth: Maximum depth value (default: 6000 mm).

    Returns:
    - Denormalized depth image as a numpy array.
    """
    return (normalized_depth / 254.0) * (max_depth - min_depth) + min_depth

def load_and_process_images(depth_dir, rgb_dir):
    """
    Loads, sorts, and processes corresponding depth and RGB images from specified directories.

    Parameters:
    - depth_dir: Path to the directory containing depth images.
    - rgb_dir: Path to the directory containing RGB images.

    Returns:
    - None. Processes each corresponding pair of images.
    """
    # Get sorted lists of depth and RGB images
    depth_images = sorted(
        [f for f in os.listdir(depth_dir) if f.startswith('depth') and f.endswith('.jpg')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
    rgb_images = sorted(
        [f for f in os.listdir(rgb_dir) if f.startswith('left') and f.endswith('.jpg')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    # Check if the number of images matches
    if len(depth_images) != len(rgb_images):
        print("Warning: Mismatched number of depth and RGB images.")

    # Iterate through corresponding pairs
    for depth_img_name, rgb_img_name in zip(depth_images, rgb_images):
        depth_img_path = os.path.join(depth_dir, depth_img_name)
        rgb_img_path = os.path.join(rgb_dir, rgb_img_name)

        # Load images
        depth_img = cv2.imread(depth_img_path, cv2.IMREAD_GRAYSCALE)  # Depth as grayscale
        rgb_img = cv2.imread(rgb_img_path, cv2.IMREAD_COLOR)  # RGB as BGR8

        if depth_img is None or rgb_img is None:
            print(f"Skipping pair: {depth_img_name}, {rgb_img_name} (Failed to load)")
            continue

        # Denormalize depth image
        denormalized_depth = denormalize_depth_image(depth_img)

        # Display or process images (for demonstration, we'll just show them)
        # cv2.imshow("Depth Image (Denormalized)", denormalized_depth / 6000)  # Normalize for display
        # cv2.imshow("RGB Image", rgb_img)

        # print(f"Processed pair: {depth_img_name}, {rgb_img_name}")

        # # Wait for a key press (comment out for automatic processing)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cv2.destroyAllWindows()

def main(args=None):

    print("Time to evaluate")

    DATASET = "corridor_corners"

    #################################################################################################################################
    #################################################################################################################################
    # Single Person Detection model
    # Setting up Available CUDA device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Setting up model paths (YOLO for object detection and segmentation, and orientation estimation model)
    pkg_shared_dir = get_package_share_directory('person_detection_temi')
    yolo_path = os.path.join(pkg_shared_dir, 'models', 'yolov8n-segpose.engine')
    tracker_path = os.path.join(pkg_shared_dir, 'models', 'bytetrack.yaml')
    resnet_path = os.path.join(pkg_shared_dir, 'models', 'osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth')
    
    # Loading Template IMG
    template_img_path = os.path.join(pkg_shared_dir, 'template_imgs', f'template_rgb_{DATASET}.jpg')
    template_img = cv2.imread(template_img_path)

    # Setting up Detection Pipeline
    model = SOD(yolo_path, resnet_path, tracker_path)
    model.to(device)

    # Initialize the template
    model.template_update(template_img)
    #################################################################################################################################
    #################################################################################################################################


    depth_dir = f"/media/enrique/Extreme SSD/jtl-stereo-tracking-dataset/icvs2017_dataset/zed/{DATASET}/depth"
    rgb_dir = f"/media/enrique/Extreme SSD/jtl-stereo-tracking-dataset/icvs2017_dataset/zed/{DATASET}/left"
    results_bboxes_file = f"/home/enrique/{DATASET}_results.txt"
    results_poses_file = f"/home/enrique/{DATASET}_measurements_results.txt"

    save_boxes = False
    save_poses = True

    # load_and_process_images(depth_directory, rgb_directory)

    depth_images = sorted(
        [f for f in os.listdir(depth_dir) if f.startswith('depth') and f.endswith('.jpg')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
    rgb_images = sorted(
        [f for f in os.listdir(rgb_dir) if f.startswith('left') and f.endswith('.jpg')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    # Check if the number of images matches
    if len(depth_images) != len(rgb_images):
        print("Warning: Mismatched number of depth and RGB images.")

    bboxes = []
    poses = []

    # Iterate through corresponding pairs
    for depth_img_name, rgb_img_name in zip(depth_images, rgb_images):
        depth_img_path = os.path.join(depth_dir, depth_img_name)
        rgb_img_path = os.path.join(rgb_dir, rgb_img_name)

        # Load images
        depth_img = cv2.imread(depth_img_path, cv2.IMREAD_GRAYSCALE)  # Depth as grayscale
        rgb_img = cv2.imread(rgb_img_path, cv2.IMREAD_COLOR)  # RGB as BGR8

        depth_img = denormalize_depth_image(depth_img)

        if depth_img is None or rgb_img is None:
            print(f"Skipping pair: {depth_img_name}, {rgb_img_name} (Failed to load)")
            continue

        ############################
        results = model.detect(rgb_img, depth_img)
        ############################

        if save_boxes:
            if results is None:
                bboxes.append(results)
            else:
                person_poses, bbox, kpts, tracked_ids, conf, valid_idxs = results

                if len(valid_idxs) == 0:
                    bboxes.append(None)
                else:
                    bboxes.append(bbox[valid_idxs].tolist())

        if save_poses:
            if results is None:
                poses.append(results)
            else:
                person_poses, bbox, kpts, tracked_ids, conf, valid_idxs = results

                if len(valid_idxs) == 0:
                    poses.append(None)
                else:
                    points_wrt_base = transform_camera_to_ros(person_poses[valid_idxs])
                    poses.append(points_wrt_base[:, :2].tolist())

    if save_boxes:
        write_bounding_boxes_to_file(bboxes, results_bboxes_file, append=False)

    if save_poses:
        write_positions_to_file(poses, results_poses_file, append=False)

    
if __name__ == '__main__':
    main()
