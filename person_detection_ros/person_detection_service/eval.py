#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
from ament_index_python.packages import get_package_share_directory
from person_detection_ros.system.SOD import SOD
import time

save_boxes = True
save_bucket = False
save_time = False
show_img = True
save_memory = False
save_atentions = False
use_depth = True

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
            x1, y1, x2, y2 = bounding_boxes
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
    pkg_shared_dir = get_package_share_directory('person_detection_ros')

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



    # Load focal lengths into dict
    focal_dict = {}
    with open(os.path.join(rgb_dir, "focal_lengths.txt"), "r") as f:
        for line in f:
            name, focal = line.strip().split()
            focal_dict[name] = float(focal)


    # Setup Detection Pipeline
    # model = SOD( # SORT
    #     yolo_model_path = yolo_path, 
    #     feature_extracture_cfg_path = feature_extracture_cfg_path, 
    #     tracker_system_path = bytetrack_path,
    #     use_experimental_tracker=True,
    #     use_mb=False,
    #     iou_threshold = 0.5, 
    #     kpr_kpt_conf = 0.3,
    #     reid_count_thr = 1,
    #     class_prediction_thr = 0.8,
    # )

    # model = SOD( # BYTETRACK - Done
    #     yolo_model_path = yolo_path, 
    #     feature_extracture_cfg_path = feature_extracture_cfg_path, 
    #     tracker_system_path = bytetrack_path,
    #     use_experimental_tracker=False,
    #     use_mb=False,
    #     iou_threshold = 0.4, 
    #     kpr_kpt_conf = 0.3,
    #     reid_count_thr = 1,
    #     class_prediction_thr = 0.8,
    # )

    if use_depth:
        print("USE SORT+DEPTH")
        model = SOD( # SORT+DEPTH
            yolo_model_path = yolo_path, 
            feature_extracture_cfg_path = feature_extracture_cfg_path, 
            tracker_system_path = bytetrack_path,
            use_experimental_tracker=True,
            use_mb=True,
            yolo_detection_thr = 0.5,
            min_hits = 1, 
            max_age = 1,
            iou_threshold = 0.5, 
            mb_threshold = 6.0, 
            kpr_kpt_conf = 0.3,
            reid_count_thr = 1,
            class_prediction_thr = 0.8,
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
    # for i, rgb_img_name in enumerate(rgb_images[3*(len(rgb_images)//8):]):
    for i, rgb_img_name in enumerate(rgb_images):

        # if i == len(rgb_images) //2:
        #     break

        rgb_img_path = os.path.join(rgb_dir, rgb_img_name)

        rgb_img = cv2.imread(rgb_img_path, cv2.IMREAD_COLOR)  # Read RGB as BGR8

        height, width, _ = rgb_img.shape  # Get dimensions

        # Generate a random depth image (grayscale)
        if use_depth:
            depth_path = os.path.join(rgb_dir, "depth", rgb_img_name[:-4]+".npy")
            depth_img = np.load(depth_path)
            fx = fy = focal_dict[rgb_img_name]
            camera_intrinsics = [fx, fy, width/2.0, height/2.0]

            # print("DEPTH DATA")
            # print("depth shape", depth_img.shape)
            # print("rgb shape", rgb_img.shape)
            # print("Max value", np.max(depth_img))
            # print("CAMERA INTRINSICS", camera_intrinsics)
            # exit()
        else:
            camera_intrinsics=[1.0, 1.0, 1.0, 1.0]
            depth_img = np.random.randint(0, 256, (height, width), dtype=np.uint8)

        # Perform Detection
        start_time = time.time()
        results = model.detect(rgb_img, depth_img, camera_params=camera_intrinsics)
        end_time = time.time()

        # Compute execution time in milliseconds
        execution_time_s = (end_time - start_time) 

        if save_time:
            times.append(execution_time_s)
            
        # print(f"Execution Time: {execution_time_s:.3f} s")

        if results is not None:

            person_poses, bbox, _, tracked_ids, kpts = results
                

            for j in range(len(bbox)):

                x1, y1, x2, y2 = map(int, bbox[j])

                if show_img:
                    cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Green Box
                    cv2.putText(rgb_img, f"ID: {tracked_ids[j]}", (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2),
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
            # cv2.waitKey(0)  # Add small delay to update the window
            # cv2.waitKey(67)
            cv2.waitKey(3)

    # Save Bounding Boxes to File
    if save_boxes:
        write_bounding_boxes_to_file(bboxes, results_bboxes_file, append=False)

    if save_time:
        save_execution_times(times, results_times_file)

    # Close OpenCV windows
    if show_img:
        cv2.destroyAllWindows()

    if save_memory:
        print("The Memmory Bucket was succesfully saved")
        model.memory.save("/home/enrique/reid_research/memory_buffer.npz")

    if save_atentions:
        print("Saving attention maps and logits...")

        # Get positive and negative sample counts
        n_pos = model.memory.n_pos
        n_neg = model.memory.n_neg

        # Get all positive and negative features + visibilities
        pos_feats, pos_vis = model.memory.pos_feats[:n_pos], model.memory.pos_vis[:n_pos]
        neg_feats, neg_vis = model.memory.neg_feats[:n_neg], model.memory.neg_vis[:n_neg]

        print(pos_vis)

        print(neg_vis)

        # Concatenate features and visibilities
        feats = torch.cat([neg_feats, pos_feats], dim=0)  # shape [N, 6, 512]
        vis = torch.cat([neg_vis, pos_vis], dim=0)        # shape [N, 6]

        # Create corresponding labels: 0 = negative, 1 = positive
        neg_labels = torch.zeros(n_neg, 1)
        pos_labels = torch.ones(n_pos, 1)
        labels = torch.cat([neg_labels, pos_labels], dim=0)  # shape [N, 1]

        # Move to GPU
        feats = feats.to(model.device)
        vis = vis.to(model.device)
        labels = labels.to(model.device)

        # Inference
        model.transformer_classifier.eval()
        with torch.no_grad():
            logits, attention_maps = model.transformer_classifier(feats, vis, return_mask=True)

        # Move to CPU
        logits_np = logits.cpu().numpy()                    # [N, 1]
        labels_np = labels.cpu().numpy()                    # [N, 1]
        attn_np = np.stack([a.cpu().numpy() for a in attention_maps], axis=0)  # [L, N, H, 7, 7]

        # Save everything
        save_path = "/home/enrique/reid_research/attentions_and_logits.npz"
        np.savez_compressed(save_path,
            attention_maps=attn_np,  # [L, N, H, 7, 7]
            logits=logits_np,        # [N, 1]
            labels=labels_np         # [N, 1]
        )

        print(f"✅ Attention maps and logits saved to {save_path}")

def main():

    # datasets = ["corridor1", "corridor2", "room", "lab_corridor", "ocl_demo", "ocl_demo2"]
    datasets = ["corridor_corners", "hallway_2", "sidewalk", "walking_outdoor"]
    # datasets = ["corridor1", "corridor2", "room", "lab_corridor", "corridor_corners", "hallway_2", "sidewalk", "walking_outdoor"]
    # datasets = ["corridor2"]
    # datasets = ["corridor1", "corridor2", "room", "lab_corridor"]
    # datasets = ["lab_corridor"]

    # datasets = ["sidewalk"]
    # datasets = ["ocl_demo2"]
    # datasets = ["sidewalk", "walking_outdoor"]
    datasets = ["lab_corridor"]

    for dataset in datasets:
        evaluation(dataset, ocl_dataset_path = "/media/enrique/Extreme SSD/ocl", crowd_dataset_path = "/home/enrique/Videos/crowds", robot_dataset_path = "/media/enrique/Extreme SSD/jtl-stereo-tracking-dataset/icvs2017_dataset/zed")

if __name__ == '__main__':
    main()