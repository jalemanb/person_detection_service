#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
from ament_index_python.packages import get_package_share_directory
from person_detection_temi.system.SOD import SOD
import time

save_boxes = False
save_bucket = False
save_time = False
show_img = True

def save_execution_times(times, filename):
    """
    Save a list of execution times to a file in 'idx execution_time' format.

    Parameters:
    - times: list or numpy array of execution times.
    - filename: output file path.
    """
    times = np.array(times)
    idx = np.arange(len(times))
    data = np.column_stack((idx, times))
    np.savetxt(filename, data, fmt='%d %.8f')

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

def evaluation(dataset, ocl_dataset_path, crowd_dataset_path, robot_dataset_path):

    print("TIME TO EVALUATE")

    ########################################################################################################
    # Initialize Person Detection Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model paths
    pkg_shared_dir = get_package_share_directory('person_detection_temi')

    yolo_models = ['yolo11n-pose.engine','yolo11n-pose.pt']
    yolo_model = None

    for model_file in yolo_models:
        file_path = os.path.join(pkg_shared_dir, "models", model_file)
        if os.path.exists(file_path):
            yolo_model = file_path
            break

    yolo_path = os.path.join(pkg_shared_dir, 'models', yolo_model)
    bytetrack_path = os.path.join(pkg_shared_dir, 'models', 'bytetrack.yaml')
    feature_extracture_cfg_path = os.path.join(pkg_shared_dir, 'models', 'kpr_market_test_in.yaml')
    feature_extracture_model_path = os.path.join(pkg_shared_dir, 'models', 'kpr_reid_in_shape_inferred.onnx')
    # feature_extracture_model_path = ""

    ### OCL Datasets ####################################################
    ocl_datasets = ["corridor1", "corridor2", "room", "lab_corridor", "ocl_demo", "ocl_demo2"]
    robot_datasets = ["corridor_corners", "hallway_2", "sidewalk", "walking_outdoor"]
    crowd_datasets = ["crowd1", "crowd2", "crowd3", "crowd4"]


    if dataset  in ocl_datasets:
        rgb_dir = f"{ocl_dataset_path}/{dataset}/"
        template_img_path = os.path.join(rgb_dir+f'template_{dataset}.png')
    ### OCL Datasets ####################################################


    ### Stotos Lab Datasets ####################################################
    # dataset = "walking_outdoor"
    if dataset in robot_datasets:
        rgb_dir = f"{robot_dataset_path}/{dataset}/left/"
        template_img_path = os.path.join(rgb_dir+f'template_{dataset}.jpg')
    ### Stotos Lab Datasets ####################################################


    ### Robocup Datasets ####################################################
    if dataset in crowd_datasets:
        rgb_dir = f"/home/enrique/Videos/crowds/{dataset}/"
        template_img_path = os.path.join(rgb_dir+f'template_{dataset}.png')
    ### Robocup Datasets ####################################################

    template_img = cv2.imread(template_img_path)


    # Setup Detection Pipeline
    model = SOD(
        yolo_model_path = yolo_path, 
        feature_extracture_cfg_path = feature_extracture_cfg_path, 
        feature_extracture_model_path = "",
        tracker_system_path = bytetrack_path,
    )
    model.to(device)

    model.target_id = 1

    # Initialize the template
    ########################################################################################################

    results_bboxes_file = rgb_dir + f"{dataset}_bboxes_results.txt"
    results_times_file = rgb_dir + f"{dataset}_times_results.txt"

    # Get all RGB images sorted by numeric order
    rgb_images = sorted(
        [f for f in os.listdir(rgb_dir) if (f.startswith('frame') or f.startswith('rgb') or f.startswith('left')) and (f.endswith('.png') or f.endswith('.jpg'))],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    bboxes = []
    times = []

    # Iterate through each RGB image
    for i, rgb_img_name in enumerate(rgb_images):

        rgb_img_path = os.path.join(rgb_dir, rgb_img_name)

        rgb_img = cv2.imread(rgb_img_path, cv2.IMREAD_COLOR)  # Read RGB as BGR8
        # rgb_img = cv2.resize(rgb_img, (640, 480), interpolation=cv2.INTER_LINEAR)

        height, width, _ = rgb_img.shape  # Get dimensions

        # Generate a random depth image (grayscale)
        depth_img = np.random.randint(0, 256, (height, width), dtype=np.uint8)

        # Perform Detection
        start_time = time.time()
        results = model.detect(rgb_img, depth_img, camera_params=[1.0, 1.0, 1.0, 1.0])
        end_time = time.time()

        # Compute execution time in milliseconds
        execution_time_s = (end_time - start_time) 

        if save_time:
            times.append(execution_time_s)
            
        # print(f"Execution Time: {execution_time_s:.3f} s")

        if results is not None:

            person_poses, bbox, _, tracked_ids, kpts = results
            

            for i in range(len(bbox)):

                x1, y1, x2, y2 = map(int, bbox[i])

                if show_img:
                    cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Green Box
                    cv2.putText(rgb_img, f"ID: {tracked_ids[i]}", (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            if model.target_id in tracked_ids:

                idx = np.where(tracked_ids == model.target_id)[0][0]

                # print("idx", idx, tracked_ids[idx])

                if save_boxes:
                    bboxes.append(bbox[idx].tolist())

                id = tracked_ids[idx]
                
                x1, y1, x2, y2 = map(int, bbox[idx])
                
                if show_img:
                    cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green Box

            else:
                if save_boxes:
                    bboxes.append(None)
        else:
            bboxes.append(None)

        # Show the image with bounding boxes
        if show_img:
            cv2.imshow("Detection", rgb_img)
            cv2.waitKey(67)  # Add small delay to update the window

    # Save Bounding Boxes to File
    if save_boxes:
        write_bounding_boxes_to_file(bboxes, results_bboxes_file, append=False)

    if save_time:
        save_execution_times(times, results_times_file)


    # Close OpenCV windows
    if show_img:
        cv2.destroyAllWindows()

def main():

    # datasets = ["corridor1", "corridor2", "room", "lab_corridor", "ocl_demo", "ocl_demo2"]
    # datasets = ["corridor_corners", "hallway_2", "sidewalk", "walking_outdoor"]
    # datasets = ["corridor1", "corridor2", "room", "lab_corridor", "corridor_corners", "hallway_2", "sidewalk", "walking_outdoor"]
    # datasets = ["corridor2"]
    # datasets = ["corridor1", "corridor2", "room", "lab_corridor"]
    datasets = ["ocl_demo2"]
    # datasets = ["room"]

    for dataset in datasets:
        evaluation(dataset, ocl_dataset_path = "/media/enrique/Extreme SSD/ocl", crowd_dataset_path = "/home/enrique/Videos/crowds", robot_dataset_path = "/media/enrique/Extreme SSD/jtl-stereo-tracking-dataset/icvs2017_dataset/zed")

if __name__ == '__main__':
    main()
