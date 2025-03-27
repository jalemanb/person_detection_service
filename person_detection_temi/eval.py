#!/usr/bin/env python3
import os
import cv2
import numpy as np
import torch
from ament_index_python.packages import get_package_share_directory
from person_detection_temi.submodules.SOD import SOD
import time


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

def evaluation(dataset):

    print("Time to evaluate")

    ########################################################################################################
    # Initialize Person Detection Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model paths
    pkg_shared_dir = get_package_share_directory('person_detection_temi')
    yolo_path = os.path.join(pkg_shared_dir, 'models', 'yolo11n-pose.pt')
    feature_extracture_model_path = os.path.join(pkg_shared_dir, 'models', 'kpr_reid.onnx')
    feature_extracture_cfg_path = os.path.join(pkg_shared_dir, 'models', 'kpr_market_test_in.yaml')

    ### OCL Datasets ####################################################
    ocl_datasets = ["corridor1", "corridor2", "room", "lab_corridor", "ocl_demo", "ocl_demo2"]
    robot_datasets = ["corridor_corners", "hallway_2", "sidewalk", "walking_outdoor"]

    # dataset = "room"
    # dataset = "ocl_demo2"
    # dataset = "corridor2"

    if dataset  in ocl_datasets:
        rgb_dir = f"/media/enrique/Extreme SSD/ocl/{dataset}/"
        template_img_path = os.path.join(rgb_dir+f'template_{dataset}.png')
    ### OCL Datasets ####################################################


    ### Stotos Lab Datasets ####################################################
    # dataset = "walking_outdoor"
    if dataset in robot_datasets:
        rgb_dir = f"/media/enrique/Extreme SSD/jtl-stereo-tracking-dataset/icvs2017_dataset/zed/{dataset}/left/"
        template_img_path = os.path.join(rgb_dir+f'template_{dataset}.jpg')
    ### Stotos Lab Datasets ####################################################


    ### Robocup Datasets ####################################################
    # dataset = "crowd3"

    # rgb_dir = f"/home/enrique/Videos/crowds/{dataset}/"
    # template_img_path = os.path.join(rgb_dir+f'template_{dataset}.png')
    ### Robocup Datasets ####################################################


    # Load Template Image
    # template_img_path = os.path.join(pkg_shared_dir, 'template_imgs', 'crowd2.png')

    # template_img_path = os.path.join("/media/enrique/Extreme SSD/ocl_demo/ocl_template.png")
    template_img = cv2.imread(template_img_path)

    # Setup Detection Pipeline
    model = SOD(yolo_path, feature_extracture_model_path, feature_extracture_cfg_path)
    model.to(device)

    # Initialize the template
    model.template_update(template_img)
    ########################################################################################################

    # Paths
    # rgb_dir = "/media/enrique/Extreme SSD/ocl_demo2/"
    # rgb_dir = "/media/enrique/Extreme SSD/ocl_demo/"
    # rgb_dir = "/home/enrique/Videos/crowds/crowd3/"

    results_bboxes_file = rgb_dir + f"{dataset}_bboxes_results.txt"

    save_boxes = True
    save_bucket = False

    # Get all RGB images sorted by numeric order
    rgb_images = sorted(
        [f for f in os.listdir(rgb_dir) if (f.startswith('frame') or f.startswith('rgb') or f.startswith('left')) and (f.endswith('.png') or f.endswith('.jpg'))],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )

    bboxes = []

    print(rgb_images)

    # Iterate through each RGB image
    for i, rgb_img_name in enumerate(rgb_images):

        rgb_img_path = os.path.join(rgb_dir, rgb_img_name)

        print(rgb_img_path)

        rgb_img = cv2.imread(rgb_img_path, cv2.IMREAD_COLOR)  # Read RGB as BGR8
        rgb_img = cv2.resize(rgb_img, (640, 480), interpolation=cv2.INTER_LINEAR)

        height, width, _ = rgb_img.shape  # Get dimensions

        # Generate a random depth image (grayscale)
        depth_img = np.random.randint(0, 256, (height, width), dtype=np.uint8)

        # Perform Detection
        start_time = time.time()
        results = model.detect(rgb_img, depth_img)
        end_time = time.time()

        # Compute execution time in milliseconds
        execution_time_s = (end_time - start_time) 

        print(f"Execution Time: {execution_time_s:.3f} s")

        if results is not None:

            person_poses, bbox, kpts, tracked_ids, conf, valid_idxs = results

            print("BOXES", bbox)

            if len(valid_idxs) != 0 :
                detected_bboxes = bbox[valid_idxs].tolist()

                # **Draw Bounding Boxes on the Image**
                for i, box in enumerate(detected_bboxes):
                    x1, y1, x2, y2 = map(int, box)

                    # Draw rectangle on the image
                    cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green Box

                    # Put confidence score (if available)
                    if conf is not None and len(conf) > i:
                        cv2.putText(rgb_img, f"{conf[valid_idxs[i]]:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the image with bounding boxes
        cv2.imshow("Detection", rgb_img)
        cv2.waitKey(3)  # Add small delay to update the window

    # Save Bounding Boxes to File
    if save_boxes:
        write_bounding_boxes_to_file(bboxes, results_bboxes_file, append=False)

    if save_bucket: 
        model.memory_bucket.save("/home/enrique/bucket.npz")

    # Close OpenCV windows
    cv2.destroyAllWindows()

def main():

    datasets = ["corridor1", "corridor2", "room", "lab_corridor", "ocl_demo", "ocl_demo2"]
    datasets = ["corridor_corners", "hallway_2", "sidewalk", "walking_outdoor"]
    for dataset in datasets:
        evaluation(dataset)



if __name__ == '__main__':

        main()
