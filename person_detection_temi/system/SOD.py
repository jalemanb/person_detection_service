import logging

from ultralytics import settings

settings.ONLINE = False

from ultralytics import YOLO
import torch.nn.functional as F
import torch
import cv2
import time
import numpy as np
import threading


from person_detection_temi.system.kpr_reid import KPR as KPR_torch
from person_detection_temi.system.kpr_reid_onnx import KPR as KPR_onnx

from person_detection_temi.system.utils import kp_img_to_kp_bbox, rescale_keypoints, iou_vectorized, compute_center_distances

class SOD:
    def __init__(self, 
                 yolo_model_path, 
                 feature_extracture_cfg_path, 
                 feature_extracture_model_path = "",
                 tracker_system_path = "",
                 logger_level=logging.DEBUG,
            ) -> None:
        
        self.logger = logging.getLogger("SOD")
        formatter = logging.Formatter("{levelname} - {message}", style="{")
        self.logger.setLevel(logger_level)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tracker_file = tracker_system_path

        # Detection Model
        self.yolo = YOLO(
            yolo_model_path
        )  # load a pretrained model (recommended for training)

        if feature_extracture_model_path != "":
            # ReID System ONNX based for Acceleration
            self.kpr_reid = KPR_onnx(
                feature_extracture_cfg_path, 
                feature_extracture_model_path, 
                kpt_conf=0., 
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            # Based on TORCH for testing
            # ReID System
            self.kpr_reid = KPR_torch(
                feature_extracture_cfg_path, 
                kpt_conf=0., 
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

        # Reidentifiation utilities
        self.target_id = None
        self.closest_person_id = None
        self.reid_lock = threading.Lock()
        self.target_feats = None
        self.start = True
        ############################

        # Intel Realsense Values
        self.fx, self.fy, self.cx, self.cy = None, None, None, None

        self.man_kf = None
        self.cov_kf = None

        self.logger.info("Tracker Armed")

    def to(self, device):
        # Set the device to use for the reidenticifaction system (cpu - gpu)
        self.device = device

    def set_target_id(self):

        self.target_id = self.closest_person_id

    
    def unset_target_id(self):
        self.target_id = None

    def detect(
            self, 
            img_rgb, 
            img_depth,
            camera_params=[1.0, 1.0, 1.0, 1.0], 
            detection_class=0
    ):
        # Get Image Dimensions (Assumes noisy message wth varying image size) 
        img_h = img_rgb.shape[0]
        img_w = img_rgb.shape[1]    

        # Update Camera parameters
        self.fx, self.fy, self.cx, self.cy = camera_params

        with torch.inference_mode():
                
            total_execution_time = 0  # To accumulate total time

            # Measure time for `masked_detections`
            start_time = time.time()
            detections = self.masked_detections(
                img_rgb, 
                img_depth, 
                detection_class=detection_class, 
                track = True, 
                detection_thr=0.0
            )
            end_time = time.time()
            masked_detections_time = (end_time - start_time) * 1000  # Convert to milliseconds
            # print(f"masked_detections execution time: {masked_detections_time:.2f} ms")
            total_execution_time += masked_detections_time

            if len(detections) < 1:
                return None
            
            # YOLO Detection Results
            detections_imgs, detection_kpts, bboxes, poses, track_ids = detections
            # self.logger.debug(f"DETECTIONS: {len(detections_imgs)}")

            # If no detection (No human) then stay on reid mode and return Nothing
            if self.target_id is not None and self.target_id not in track_ids:
                threading.Thread(
                        target=self.reidentification,
                        args=(detections_imgs, detection_kpts, track_ids),
                        daemon=True
                    ).start()
            elif self.target_id is not None and self.target_id in track_ids and self.start:
                # Collect samples for online training the feature extractor in another train
                threading.Thread(
                        target=self.updating_reid,
                        args=(detections_imgs, detection_kpts, track_ids),
                        daemon=True
                    ).start()
                self.start = False

            # Return results
            return (poses, bboxes, detection_kpts, track_ids)


    def masked_detections(
            self, 
            img_rgb, 
            img_depth = None, 
            detection_class = 0, 
            size = (128, 384), 
            track = False, 
            detection_thr = 0.5):

        results = self.detect_mot(img_rgb, detection_class=detection_class, track = track, detection_thr = detection_thr)  

        if not (len(results[0].boxes) > 0):
            return []

        subimages = []
        total_keypoints = []
        poses = []
        bboxes = []
        track_ids = []

        for result in results: # Need to iterate because if batch is longer than one it should iterate more than once
            boxes = result.boxes  # Boxes object
            keypoints = result.keypoints

            for i ,(box, kpts) in enumerate(zip(boxes, keypoints.data)):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = -1 if box.id is None else box.id.int().cpu().item()

                # Crop the Image
                subimage = cv2.resize(img_rgb[y1:y2, x1:x2], size)
                # Getting the Person Central Pose (Based on Torso Keypoints)
                pose = self.get_person_pose([x1, y1, x2, y2], img_depth)
                # Store all the bounding box detections and subimages in a tensor
                subimages.append(
                    torch.tensor(subimage, dtype=torch.float16)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )
                bboxes.append((x1, y1, x2, y2))
                track_ids.append(track_id)
                poses.append(pose)
                # Scaling points to be with respect to the bounding box 
                kpts_box = kp_img_to_kp_bbox(kpts, (x1, y1, x2, y2))
                kpts_scaled = rescale_keypoints(
                    kpts_box, 
                    (x2 - x1, y2 - y1), 
                    (size[0], size[1])
                )
                total_keypoints.append(kpts_scaled)

        poses = np.array(poses)
        bboxes = np.array(bboxes)
        track_ids = np.array(track_ids)
        batched_imgs = torch.cat(subimages).to(device=self.device)
        batched_kpts = torch.stack(total_keypoints, dim=0).to(device=self.device)

        # The closes person to the Camera is updated in case the ID is to be manually selected

        self.closest_person_id = track_ids[np.argmin(poses[:, 2])]

        return [batched_imgs, batched_kpts, bboxes, poses, track_ids]

    def detect_mot(self, img, detection_class, track = False, detection_thr = 0.5):
        # Run multiple object detection with a given desired class
        if track:
            return self.yolo.track(
                img, 
                persist=True, 
                classes = detection_class, 
                tracker=self.tracker_file, 
                iou = 0.2, 
                conf = detection_thr,
                verbose = False,
            )
        else:
            return self.yolo(
                img, 
                classes = detection_class, 
                conf = detection_thr,
                verbose = False,
            )

    def updating_reid(self, detections_imgs, detection_kpts, tracked_ids):
        if not self.reid_lock.acquire(blocking=False):
            return  # Skip if already running
        try:
            self.target_feats = self.feature_extraction(detections_imgs, detection_kpts)

        finally:
            self.reid_lock.release()

    def reidentification(self, detections_imgs, detection_kpts, tracked_ids):
        if not self.reid_lock.acquire(blocking=False):
            return  # Skip if already running

        try:
            if self.target_feats is not None:
                # Here is the reidentification procedure
                f_, v_ = self.feature_extraction(detections_imgs, detection_kpts)

                result = self.kpr_reid.compare(f_, self.target_feats[0], v_, self.target_feats[1])
                # self.features torch.Size([1, 6, 512])
                # self.visibilities torch.Size([1, 6])
                # RESULT
                # torch.Size([1, 1])
                # torch.Size([6, 1, 1])

                min_idx = np.argmin(result[0])
                min_dist = result[0][min_idx, :]
                print(min_dist)

                if min_dist < 0.7:
                    self.target_id = tracked_ids[min_idx]
        finally:
            self.reid_lock.release()



    def feature_extraction(self, detections_imgs, detection_kpts):
        # Extract features for similarity check
        fg_, vg_ = self.kpr_reid.extract(
                    detections_imgs, 
                    detection_kpts, 
                    return_heatmaps=False
                )
        
        return (fg_, vg_)
        

    def get_person_pose(self, bbox, depth_img):  
        """
        Estimate the 3D position of a person using the mode of depth values inside the bounding box.
        The output is the 3D point (x, y, z) in the camera optical frame.
        """

        if depth_img is None:
            return None

        x_min, y_min, x_max, y_max = bbox

        # Clip bounds to image dimensions
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(depth_img.shape[1] - 1, x_max)
        y_max = min(depth_img.shape[0] - 1, y_max)

        # Extract depth values inside the bounding box
        depth_patch = depth_img[y_min:y_max + 1, x_min:x_max + 1].astype(np.float32)
        valid_depths = depth_patch[(depth_patch > 0) & (depth_patch < 10000)]  # Filter reasonable depths (0 < d < 10m)

        if valid_depths.size == 0:
            return [-100., -100., -100.]

        # Compute mode using np.unique
        values, counts = np.unique(valid_depths, return_counts=True)
        mode_depth = values[np.argmax(counts)]
        z = mode_depth / 1000.0  # Convert from mm to meters

        # Compute 2D center of the bounding box
        u = (x_min + x_max) / 2.0
        v = (y_min + y_max) / 2.0

        # Back-project to 3D using pinhole model
        x = z * (u - self.cx) / self.fx
        y = -z * (v - self.cy) / self.fy  # Negative because of camera optical frame convention

        return [float(x), float(y), float(z)]