import logging

from ultralytics import settings

settings.ONLINE = False

from ultralytics import YOLO
import torch.nn.functional as F
import torch, cv2
import time
import numpy as np

from person_detection_temi.system.kpr_reid import KPR as KPR_torch
from person_detection_temi.system.memory import Bucket
from person_detection_temi.system.iknn import iknn
from person_detection_temi.system.utils import kp_img_to_kp_bbox, rescale_keypoints, iou_vectorized, compute_center_distances

class SOD:

    def __init__(self, 
                 yolo_model_path, 
                 feature_extracture_cfg_path, 
                 tracker_system_path = "",
                 logger_level=logging.DEBUG,
            ) -> None:
        
        self.iknn_threshold = 0.9
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


        # ReID System
        self.kpr_reid = KPR_torch(
            feature_extracture_cfg_path, 
            kpt_conf=0., 
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.template = None
        self.template_features = None

        # Intel Realsense Values
        self.fx, self.fy, self.cx, self.cy = None, None, None, None

        self.man_kf = None
        self.cov_kf = None

        self.memory_bucket = Bucket(
            max_identities = 10, 
            samples_per_identity = 20, 
            thr = 0.5
        )

        # The following flags are use for debugging and visualization
        self.debug = False # Flag to enable saving features and images for inspection and visualization
        self.bucket_counter = 0 # Flag to keep track of which features belong to which image

        # Always aim to store features for the iknn 
        self.store_features = True

        # Initialize the Re-id system mode
        self.reid_mode = True

        # Initialize the lists for tracking and for avoiding reidentifying
        self.whitelist = []
        self.blacklist = [-1]

        self.logger.info("Tracker Armed")

    def to(self, device):
        # Set the device to use for the reidenticifaction system (cpu - gpu)
        self.device = device

    def set_track_id(self, id):
        # Function to manually set an id to be tracked
        self.whitelist.append(id)
        self.reid_mode = False

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
        person_kpts = []
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
                # Getting Eyes+Torso+knees Keypoints for pose estimation
                torso_kpts = kpts[:, :2].cpu().numpy()[[1, 2, 5, 6, 11, 12, 13, 14], :]
                torso_kpts = torso_kpts[~np.all(torso_kpts == 0, axis=1)].astype(np.int32) - 1 # Rest one to avoid incorrect pixel corrdinates
                # Getting the Person Central Pose (Based on Torso Keypoints)
                pose = self.get_person_pose(torso_kpts, img_depth)
                # Store all the bounding box detections and subimages in a tensor
                subimages.append(
                    torch.tensor(subimage, dtype=torch.float16)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )
                bboxes.append((x1, y1, x2, y2))
                track_ids.append(track_id)
                person_kpts.append(torso_kpts)
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
        return [batched_imgs, batched_kpts, bboxes, person_kpts, poses, track_ids]

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

        if self.debug:
            self.bucket_counter += 1

        with torch.inference_mode():
                
            # If there is not template initialization then dont return anything
            if self.template is None:
                self.logger.warning("No template provided")
                return None

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

            # If no detection (No human) then stay on reid mode and return Nothing
            if not (len(detections) > 0):
                self.reid_mode = True
                return None
            
            # YOLO Detection Results
            detections_imgs, detection_kpts, bboxes, person_kpts, poses, track_ids = detections
            self.logger.debug(f"DETECTIONS: {len(detections_imgs)}")

            # Up to This Point There are Only Yolo Detections #####################################

            if self.reid_mode: # ReId mode

                self.logger.debug("REID MODE")
                tracked_ids = track_ids.tolist()
                # Check which boxes are worth trying to Re-ID
                to_test = [tracked_ids.index(i) for i in tracked_ids if i not in self.blacklist]

                # print("self.blacklist", self.blacklist)
                # print("to_test", to_test)

                if len(to_test) > 0:
                    # Measure time for `feature_extraction` - Extract features to all subimages
                    start_time = time.time()
                    detections_features = self.feature_extraction(
                        detections_imgs=detections_imgs[to_test], 
                        detection_kpts=detection_kpts[to_test],
                    )
                    end_time = time.time()
                    feature_extraction_time = (end_time - start_time) * 1000  # Convert to milliseconds
                    # print(f"feature_extraction execution time: {feature_extraction_time:.2f} ms")
                    total_execution_time += feature_extraction_time

                    # Measure time for `iknn_time` - Classify the features with KNN
                    start_time = time.time()
                    memory_f, memory_v, memory_l = self.memory_bucket.get()
                    memmory_num_samples = self.memory_bucket.get_samples_num()
                    classification = iknn(
                        detections_features[0], 
                        detections_features[1], 
                        memory_f, 
                        memory_v, 
                        memory_l, 
                        memmory_num_samples, 
                        threshold=self.iknn_threshold)
                    end_time = time.time()
                    iknn_time = (end_time - start_time) * 1000  # Convert to milliseconds
                    # print(f"iknn_time execution time: {iknn_time:.2f} ms")
                    total_execution_time += iknn_time
                    # self.logger.debug(f"CLASSIFICATION: {classification}")

                    # print("Visibilitieis", detections_features[1].T)

                    # When Relying only on visual features, it is necessary to reidentfy consdiering all body parts to make a decision
                    # knn_gate = (torch.sum(classification[1:] & detections_features[1].T[1:], dim=0) >= torch.sum(detections_features[1].T[1:], dim=0)).cpu().numpy()
                    # print("torch.sum(classification[1:], dim=0)", torch.sum(classification[1:], dim=0))
                    # print("torch.sum(detections_features[1].T[1:], dim=0)", torch.sum(detections_features[1].T[1:], dim=0))
                    # knn_gate = (torch.sum(classification[1:], dim=0) >= torch.sum(detections_features[1].T[1:], dim=0)).cpu().numpy()

                    sum1 = torch.sum(classification[1:], dim=0)
                    sum2 = torch.sum(detections_features[1].T[1:], dim=0)
                    positive_mask = (sum1 > 0) & (sum2 > 0)
                    knn_gate = ((sum1 >= sum2) & positive_mask).cpu().numpy()

                    gate = knn_gate

                    # Get All indices belonging to valid Detections
                    best_idx = np.argwhere(gate == 1).flatten().tolist()

                    valid_idxs = [to_test[i] for i in best_idx]

                    self.whitelist = []
                    for i in valid_idxs:
                        self.whitelist.append(track_ids[i]) 
                
                    # If there is not a valid detection but a box is being tracked (keep prediction until box is out of fov)
                    if np.sum(gate) == 0:
                        self.reid_mode = True
                        valid_idxs = []
                        similarity = [0 for i in range(len(tracked_ids))]

                    # If there is only valid detection 
                    elif np.sum(gate) == 1:
                        self.reid_mode = False
                        similarity = [0 for i in range(len(tracked_ids))]
                        similarity[valid_idxs[0]] = 1

                    # If there are multiple valid detections 
                    elif np.sum(gate) > 1:
                        self.reid_mode = True
                        similarity = [0 for i in range(len(tracked_ids))]
                        for idx in valid_idxs:
                            similarity[idx] = (1/len(valid_idxs))

                else:
                    valid_idxs = []
                    similarity = [0 for i in range(len(tracked_ids))]


            else: # Tracking mode
                self.store_features = True
                valid_idxs = []
                tracked_ids = track_ids.tolist()

                for target_id in self.whitelist:
                    if target_id in track_ids:
                        valid_idxs.append(tracked_ids.index(target_id))
                        # When Tracking the bounding box without overlap the assumption is that the bounding box is well identified
                        similarity = [0 for i in range(len(tracked_ids))]
                        similarity[valid_idxs[0]] = 1
                    else:
                        self.whitelist = []
                        self.reid_mode = True
                        self.store_features = False
                        # When Tracking the bounding box without overlap the assumption is that the bounding box is well identified
                        similarity = [0 for i in range(len(tracked_ids))]

                #### IF SPATIAL AMBIGUITY IS PRESENT GO BACK TO ADD APPEARANCE INFORMATION FOR ASSOCIATION ###############
                second_best_idx = 0


                if len(valid_idxs) > 0: 
                    # Activate Reid mode when target is reaching the edges of the image
                    x1, y1, x2, y2 = bboxes[valid_idxs[0]]

                    if x1 < 20 or x2 > img_w - 20:


                        self.reid_mode = True
                        self.store_features = False
                        # print("x1", x1, "x2", x2)


                # Check if bounding boxes are too close to the target
                # if so return nothing and change to reid_mode
                if len(valid_idxs) > 0  and len(tracked_ids) > 1:

                    # distractor_bbox = np.delete(bboxes, valid_idxs[0], axis=0)
                    ious_to_target = iou_vectorized(bboxes[valid_idxs[0]],  bboxes)
                    ious_to_target = np.where(ious_to_target == 1, -1, ious_to_target)

                    distances = compute_center_distances(bboxes[valid_idxs[0]],  bboxes)
                    distances = np.where(distances > 0, distances, np.inf)
                    second_best_idx = np.argmin(distances)
                    
                    if  np.any(ious_to_target > 0.):
                        self.store_features = False

                    if  np.any(ious_to_target > 0.2):

                        self.whitelist = []

                        for i, iou in enumerate(ious_to_target):

                            if iou > 0.2 and tracked_ids[i] in self.blacklist:

                                self.blacklist.pop(self.blacklist.index(tracked_ids[i]))

                        # Enable the closest box to the target to be reidentified
                        # if tracked_ids[second_best_idx] in self.blacklist:
                        #     self.blacklist.pop(self.blacklist.index(tracked_ids[second_best_idx]))

                        self.reid_mode = True

                

                if not self.reid_mode:
                    for target_id in self.whitelist:
                        # Blacklisting tracks belonging to distractors
                        for id in tracked_ids:
                            if id == target_id:
                                continue
                            elif id not in self.blacklist:
                                self.blacklist.append(id)
            
                ###############################################################################################################
                
                # Store features only when the target person is clearly identified and not occluded by any other distractor box

                if self.store_features:
                    # Incremental Learning

                    # Measure time for `feature_extraction and feature storing` - Extract features to all subimages
                    start_time = time.time()

                    if  len(tracked_ids) < 2:
                        latest_features = self.feature_extraction(
                            detections_imgs=detections_imgs[valid_idxs], 
                            detection_kpts=detection_kpts[valid_idxs]
                        )
                        self.memory_bucket.store_feats(
                            latest_features[0], 
                            latest_features[1], 
                            counter = self.bucket_counter, 
                            img_patch = detections_imgs[valid_idxs].cpu().numpy(), 
                            debug = self.debug)
                    else:
                        valid_idxs.append(second_best_idx)
                        latest_features = self.feature_extraction(
                            detections_imgs=detections_imgs[valid_idxs], 
                            detection_kpts=detection_kpts[valid_idxs])
                        
                        self.memory_bucket.store_feats(
                            latest_features[0][[0]], 
                            latest_features[1][[0]], 
                            debug = self.debug)
                        
                        self.memory_bucket.store_distractor_feats(
                            latest_features[0][[1]], 
                            latest_features[1][[1]])
                        valid_idxs = [valid_idxs[0]]

                    end_time = time.time()
                    incremental_time = (end_time - start_time) * 1000  # Convert to milliseconds
                    # print(f"INCREMENTAL execution time: {incremental_time:.2f} ms")

            # Return results
            return (poses, bboxes, person_kpts, track_ids, similarity, valid_idxs)

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
            
    def template_update(self, template):

        detections = self.masked_detections(template, track=False, detection_thr = 0.8)

        if len(detections):
            self.template = detections[0]
            self.template_kpts = detections[1]

        self.template_features = self.extract_features(
            self.template, 
            self.template_kpts
        )

        # Store First Initial Features on Galery
        self.memory_bucket.store_feats(
            self.template_features[0], 
            self.template_features[1], 
            img_patch = self.template.cpu().numpy(), 
            debug = False
        )

    def feature_extraction(self, detections_imgs, detection_kpts):
        # Extract features for similarity check
        return self.extract_features(
            detections_imgs, 
            detection_kpts
        )

    def extract_features(self, image, kpt):

        fg_, vg_ = self.kpr_reid.extract(
            image, 
            kpt, 
            return_heatmaps=False
        )
        return (fg_, vg_)

    def get_person_pose(self, kpts, depth_img): # 3d person pose estimation wrt the camera reference frame
        # Wrap the person around a 3D bounding box

        if depth_img is None:
            return None

        if kpts.shape[0] < 2: # Not Enough Detected Keypoints proceed to compute the human pose
            return [-100., -100., -100.]
        
        u = kpts[:, 0]

        v = kpts[:, 1]
        
        z = depth_img[v, u]/1000. # get depth for specific keypoints and convert into m

        x = z*(u - self.cx)/self.fx

        y = -z*(v - self.cy)/self.fy

        return [x.mean(), y.mean(), z.mean()]