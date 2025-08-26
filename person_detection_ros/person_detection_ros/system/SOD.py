from ultralytics import YOLO
from torch.optim import AdamW
import torch
import cv2
import numpy as np
import threading

from person_detection_ros.system.kpr_reid import KPR as KPR_torch
from person_detection_ros.system.utils import kp_img_to_kp_bbox, rescale_keypoints
from person_detection_ros.system.memory_manager import MemoryManager
from person_detection_ros.system.sort import Sort
from person_detection_ros.system.tinyTransformer import TinyTransformer, nn


class SOD:
    def __init__(self, 
                 # Main Elements Params (Detector, Tracker, Feature Extractor, Logger) 
                 yolo_model_path, 
                 feature_extracture_cfg_path, 
                 tracker_system_path = "",
                 use_experimental_tracker = True,
                 yolo_detection_thr = 0.0,
                 # REID Params
                 kpr_kpt_conf = 0.3,
                 reid_count_thr = 3,
                 class_prediction_thr = 0.8,
                 # Custom SORT
                 max_age = 3,
                 min_hits = 3, 
                 iou_threshold = 0.2, 
                 mb_threshold = 6.0, 
                 use_mb = True,
                 # Memory Manager Params
                 max_samples = 100,
                 sim_thresh = 0.75, 
                 feature_dim = 512, 
                 num_parts = 6,
                 pseudo_std = 0.001,
                 beta = 0.1,
            ) -> None:
        


        self.kpr_kpt_conf = kpr_kpt_conf
        self.yolo_detection_thr = yolo_detection_thr

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tracker_file = tracker_system_path

        self.experimental_tracker = Sort(max_age = max_age, 
                                         min_hits = min_hits, 
                                         iou_threshold = iou_threshold, 
                                         mb_threshold = mb_threshold, 
                                         use_mb = use_mb)
        # Detection Model
        self.yolo = YOLO(
            yolo_model_path
        )  # load a pretrained model (recommended for training)

        # Based on TORCH for testing
        # ReID System
        self.kpr_reid = KPR_torch(
            feature_extracture_cfg_path, 
            kpt_conf=self.kpr_kpt_conf, 
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Reidentifiation utilities
        self.target_id = None
        self.closest_person_id = None
        self.reid_lock = threading.Lock()
        self.target_feats = None
        self.reid_counter = 0
        self.reid_count_thr = reid_count_thr
        self.blacklist = set()
        self.use_experimental_tracker = use_experimental_tracker
        ############################

        # ONLINE training ############################
        self.transformer_classifier = TinyTransformer()
        self.transformer_classifier.to(self.device)
        self.classifier_optimizer = AdamW(self.transformer_classifier.parameters(), lr=1e-5, weight_decay=1e-5)
        self.classifier_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([5.0]).to(self.device))
        self.class_prediction_thr = class_prediction_thr
        self.reid_stream = torch.cuda.Stream()        
        self.yolo_stream = torch.cuda.Stream()        
        self.extract_feats_target = True
        self.distractors = []
        ###############################################
        # MEMORY MANAGER ##############################
        self.memory = MemoryManager(max_samples = max_samples, 
                                    sim_thresh = sim_thresh, # If Small aways preserve the newest features
                                    feature_dim = feature_dim, 
                                    num_parts = num_parts,
                                    pseudo_std = pseudo_std,
                                    beta=beta)
        ##############################################

        # Intel Realsense Values
        self.fx, self.fy, self.cx, self.cy = None, None, None, None

    def to(self, device):
        # Set the device to use for the reidenticifaction system (cpu - gpu)
        self.device = device

    def set_target_id(self):
        self.target_id = self.closest_person_id
        self.blacklist.clear()
        self.memory.reset()

        self.experimental_tracker.reset_counter()

        # Reset all the reidentifying mechanism weights to train from scratch whn apropiate
        for layer in self.transformer_classifier.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def unset_target_id(self):
        # Set the target ID to none to avoid Re-identifying when Unnecesary
        self.target_id = None
        self.blacklist.clear()
        self.memory.reset()

        self.experimental_tracker.reset_counter()

        # Reset all the reidentifying mechanism weights to train from scratch whn apropiate
        for layer in self.transformer_classifier.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


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
            # start_time = time.time()
            with torch.cuda.stream(self.yolo_stream):
                detections = self.masked_detections(
                    img_rgb, 
                    img_depth, 
                    detection_class=detection_class, 
                    track = not self.use_experimental_tracker, 
                    detection_thr=self.yolo_detection_thr
                )
            self.yolo_stream.synchronize()  # Wait for GPU ops to finish
            # end_time = time.time()
            # masked_detections_time = (end_time - start_time)  # Convert to milliseconds

            if len(detections) < 1:
                # Call the sort even with empty detections
                if self.use_experimental_tracker:
                    dets = np.empty((0, 6))
                    detections_imgs = torch.empty((0, 3, 384, 128), dtype=None, device=None, requires_grad=False)
                    detection_kpts = torch.empty((0, 17, 3), dtype=None, device=None, requires_grad=False)
                    original_kpts = np.empty((0, 17, 3))
                    self.experimental_tracker.update(dets, detections_imgs, detection_kpts, original_kpts)
                return None

            # YOLO Detection Results
            if self.use_experimental_tracker:
                # print("USING SORT IN STEREOIDs")

                detections_imgs, detection_kpts, bboxes, poses, _, original_kpts = detections

                dets = np.concatenate([bboxes, poses[:, [0, 2]]], axis=1)

                # start_time = time.time()

                sort_results = self.experimental_tracker.update(dets, detections_imgs, detection_kpts, original_kpts)

                # end_time = time.time()
                # tracking_time = (end_time - start_time)  # Convert to milliseconds

                if sort_results is None:
                    return None
                else:
                    tracks, detections_imgs, detection_kpts, original_kpts = sort_results
                    # Reformatting Results
                    bboxes = tracks[:, :4]
                    poses = np.hstack((tracks[:, [4]], np.zeros((tracks.shape[0], 1)), tracks[:, [5]]))
                    track_ids = tracks[:, -1]  # same length as det_idxs
                    detections_imgs = detections_imgs.to(device=self.device)
                    detection_kpts = detection_kpts.to(device=self.device)
            else:
                # print("USING BYTETRACK")

                detections_imgs, detection_kpts, bboxes, poses, track_ids, original_kpts = detections
                detections_imgs = detections_imgs.to(device=self.device)
                detection_kpts = detection_kpts.to(device=self.device)

            # Always update the ID of the person that is closest to the camera for manual ReID
            self.closest_person_id = track_ids[np.argmin(poses[:, 2])]

            if self.target_id is not None and self.target_id in track_ids:
                self.blacklist.update(tid for tid in track_ids if tid != self.target_id)
                self.blacklist.intersection_update(track_ids)

            # If no detection (No human) then stay on reid mode and return Nothing
            if self.target_id is not None and self.target_id not in track_ids:
                threading.Thread(
                        target=self.reidentification,
                        args=(detections_imgs, detection_kpts, track_ids),
                        daemon=True
                    ).start()
            elif self.target_id is not None and self.target_id in track_ids:
                # Collect samples for online training the feature extractor in another train
                threading.Thread(
                        target=self.updating_reid,
                        args=(detections_imgs, detection_kpts, track_ids),
                        daemon=True
                    ).start()

            # Return results
            return (poses, bboxes, detection_kpts, track_ids, original_kpts)

    def masked_detections(
            self, 
            img_rgb, 
            img_depth = None, 
            detection_class = 0, 
            size = (128, 384), 
            track = False, 
            detection_thr = 0.75):

        results = self.detect_mot(img_rgb, detection_class = detection_class, track = track, detection_thr = detection_thr)  

        if not (len(results[0].boxes) > 0):
            return []

        subimages = []
        original_keypoints = [] # Keypoints with respect the original image dimensions, mainly used for visualization
        scaled_keypoints = [] # Kepints in the patch reference frame (small image) not original image
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
                pose = self.get_person_pose([x1, y1, x2, y2], kpts, img_depth)
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
                # Appending keypoints with resepct the original image size
                original_keypoints.append(kpts.detach().cpu().numpy())
                # Append keypoints with respect the bounding box patch size
                scaled_keypoints.append(kpts_scaled)

        poses = np.array(poses)
        bboxes = np.array(bboxes)
        track_ids = np.array(track_ids)
        batched_imgs = torch.cat(subimages)
        batched_scaled_kpts = torch.stack(scaled_keypoints, dim=0)
        original_keypoints = np.array(original_keypoints)

        return [batched_imgs, batched_scaled_kpts, bboxes, poses, track_ids, original_keypoints]

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
        
    def grab_single_sample(self, target_id_idx, tracked_ids):

        idx_to_extract = None

        distractor_id_ = None

        number_of_tracks = len(tracked_ids)

        if number_of_tracks == 1: 
            self.extract_feats_target = True

        if self.extract_feats_target:
            # Extract Only Features of target
            # This flag is temporal until the memory manager is built
            idx_to_extract = target_id_idx

            # If there are more bounding boxes then lets extract features from the distractors
            if number_of_tracks > 1: 
                # This one is used to toggle between selecting a distractor sample
                # and selecting the real target ID
                self.extract_feats_target = False

        else:
            # This code section is to grab a nonn repeating distractor sample, one by one to avoid memory overhead
            ######################################################################################################
            self.distractors = [d for d in self.distractors if d in tracked_ids]

            for id_ in tracked_ids:
                if id_ not in self.distractors and id_ != self.target_id:
                    self.distractors.append(id_)
                    distractor_id_ = id_
                    break
            
            ######################################################################################################
            ######################################################################################################

            if distractor_id_ is None: # In this case it has already been through allt he distractor images go an provide a positive sample
                self.distractors = []
                label = torch.ones(1, 1).to(self.device).reshape(1, -1)  # shape [B, 1], all positive class
                idx_to_extract = target_id_idx
                self.extract_feats_target = False
            else:  # In this case there are still distractor bounding boxes that can be used as negative class samples           
                # Getting one of the distractor bounding boxes
                idx_to_extract = np.where(tracked_ids == distractor_id_)[0]

                # This one is used to toggle between selecting a distractor sample
                # and selecting the real target ID
                self.extract_feats_target = True

        is_positive = idx_to_extract == target_id_idx

        text = "TARGET" if is_positive else f"DISTRACTOR id: {distractor_id_}" 

        return idx_to_extract, is_positive

    def updating_reid(self, detections_imgs, detection_kpts, tracked_ids):
        if not self.reid_lock.acquire(blocking=False):
            return  # Skip if already running
        try:
            # start_time = time.time()

            with torch.cuda.stream(self.reid_stream):

                # Get the index belonging to the target id
                target_id_idx = np.where(tracked_ids == self.target_id)[0]

                # From who to extract features?
                # This approach constraints the feature extraction to be of Batch one, avoiding memory overflow or high computation code
                # Extract features from 5 people in the image? 5 continuous frames needed
                idx_to_extract, is_positive = self.grab_single_sample(target_id_idx, tracked_ids)

                # Extract the features of the chosen index
                feats = self.feature_extraction(detections_imgs[[idx_to_extract]], detection_kpts[[idx_to_extract]])

                # Save Latest Extracted feature into memory
                if is_positive:
                    self.memory.insert_positive(feats)
                else:
                    self.memory.insert_negative(feats)

                # Get a sample based on the memory manager policy
                mem_feats, mem_vis, label, is_pseudo = self.memory.get_sample(use_pseudo=False)

                # Move everything into device
                mem_feats = mem_feats.to(self.device)
                mem_vis = mem_vis.to(self.device)
                label = label.to(self.device)

                #############################################################################################################
                # Online Continual Learning #################################################################################

                self.transformer_classifier.train()

                logits = self.transformer_classifier(mem_feats, mem_vis)
                
                loss = self.classifier_criterion(logits.flatten(), label.flatten())
                self.classifier_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transformer_classifier.parameters(), max_norm=0.5)
                self.classifier_optimizer.step()  
                # Online Continual Learning #################################################################################
                #############################################################################################################

            self.reid_stream.synchronize()

            # end_time = time.time()
            # train_time = (end_time - start_time)  # Convert to milliseconds

        finally:
            self.reid_lock.release()

    def reidentification(self, detections_imgs, detection_kpts, tracked_ids):
        if not self.reid_lock.acquire(blocking=False):
            return  # Skip if already running

        try:
            # start_time = time.time()

            with torch.cuda.stream(self.reid_stream):
                # This is the reidentification procedure

                with torch.inference_mode():

                    # Filter out blacklisted IDs
                    valid_indices = [i for i, tid in enumerate(tracked_ids) if tid not in self.blacklist]

                    if not valid_indices:
                        print("No valid IDs to reidentify (all blacklisted).")
                        return
                    
                    tracked_ids = np.array(tracked_ids)[valid_indices].tolist()
                
                    # Consider trying to reid one at a time

                    self.transformer_classifier.eval()

                    f_, v_ = self.feature_extraction(detections_imgs[valid_indices], detection_kpts[valid_indices])

                    logits = self.transformer_classifier(f_, v_)

                    result = torch.sigmoid(logits).detach().cpu().numpy()

                    class_idx = np.argmax(result[:, 0])

                    if result[class_idx, 0] > np.floor(self.class_prediction_thr*10)/10:

                        self.reid_counter += 1

                        if self.reid_counter > self.reid_count_thr:

                            self.target_id = tracked_ids[class_idx]
                            self.reid_counter = 0
                    else:
                        self.reid_counter = 0
   
            self.reid_stream.synchronize()

            # end_time = time.time()
            # reid_time = (end_time - start_time)  # Convert to milliseconds

        finally:
            self.reid_lock.release()

    def feature_extraction(self, detections_imgs, detection_kpts):
        # Extract features for similarity check
        fg_, vg_ = self.kpr_reid.extract(
                    detections_imgs, 
                    detection_kpts, 
                    return_heatmaps=False)
        
        return (fg_, vg_)
        
    def get_person_pose(self, bbox, kpts, depth_img,
                                max_depth_m: float = 6.0,
                                edge_shrink: float = 0.15,
                                floor_deemph: float = 0.6,
                                band_m: float = 0.25,
                                huber_sigma_m: float = 0.10,
                                min_pix: int = 50):
        """
        Robust 3D pose estimate [x,y,z] from a depth image and a person bbox only.

        Args (same call signature as before):
            bbox: [x_min, y_min, x_max, y_max] in pixels
            kpts: unused (kept for compatibility)
            depth_img: HxW depth in millimeters
        Tunables:
            max_depth_m: reject anything beyond this (meters)
            edge_shrink: shrink bbox on each side to avoid background bleed
            floor_deemph: downweight bottom rows inside the bbox (0..1)
            band_m: +/- depth band around the foreground estimate used to mask (meters)
            huber_sigma_m: Huber-like depth weighting scale (meters)
            min_pix: minimum valid pixels needed

        Returns:
            [x, y, z] in meters in the camera optical frame (y positive down).
            Returns [-100., -100., -100.] if robust estimate not possible.
        """
        if depth_img is None:
            return [-100., -100., -100.]

        H, W = depth_img.shape
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return [-100., -100., -100.]

        # 1) Shrink bbox to reduce background at the edges.
        bw = x2 - x1
        bh = y2 - y1
        sx = int(round(edge_shrink * bw))
        sy = int(round(edge_shrink * bh))
        xi1 = max(0, x1 + sx)
        yi1 = max(0, y1 + sy)
        xi2 = min(W, x2 - sx)
        yi2 = min(H, y2 - sy)
        if xi2 <= xi1 or yi2 <= yi1:
            return [-100., -100., -100.]

        # Crop depth region
        roi = depth_img[yi1:yi2, xi1:xi2]

        # 2) Valid depth mask (0 < d <= 6m)
        max_depth_mm = int(max_depth_m * 1000.0)
        valid = (roi > 0) & (roi <= max_depth_mm)
        if not np.any(valid):
            return [-100., -100., -100.]

        # Pixel coordinate grids (global image coords)
        ys, xs = np.mgrid[yi1:yi2, xi1:xi2]
        dvals_mm = roi[valid].astype(np.float32)
        xs_v = xs[valid].astype(np.float32)
        ys_v = ys[valid].astype(np.float32)

        # 3) Get a robust *foreground* depth seed z0 using lower-percentile stats.
        # People are typically *closer* than the background inside their bbox.
        # Take the median of the closest 35% valid depths.
        if dvals_mm.size < min_pix:
            # not enough depth points in shrunken box; expand to original box as fallback
            roi_fallback = depth_img[y1:y2, x1:x2]
            ys_f, xs_f = np.mgrid[y1:y2, x1:x2]
            valid_fb = (roi_fallback > 0) & (roi_fallback <= max_depth_mm)
            if not np.any(valid_fb):
                return [-100., -100., -100.]
            dvals_mm = roi_fallback[valid_fb].astype(np.float32)
            xs_v = xs_f[valid_fb].astype(np.float32)
            ys_v = ys_f[valid_fb].astype(np.float32)

        q35 = np.percentile(dvals_mm, 35.0)
        fg_pool = dvals_mm[dvals_mm <= q35]
        if fg_pool.size < max(min_pix, 20):
            # Fallback: just use overall median if not enough "close" points
            z0_mm = float(np.median(dvals_mm))
        else:
            z0_mm = float(np.median(fg_pool))

        # 4) Build a depth mask around z0 within +/- band_m (meters).
        band_mm = band_m * 1000.0
        depth_mask = (dvals_mm >= (z0_mm - band_mm)) & (dvals_mm <= (z0_mm + band_mm))
        if not np.any(depth_mask):
            return [-100., -100., -100.]

        xs_m = xs_v[depth_mask]
        ys_m = ys_v[depth_mask]
        dsel_mm = dvals_mm[depth_mask]

        # 5) Weighting:
        #    - Center prior (closer to bbox center -> higher weight)
        #    - De-emphasize bottom rows (floor) via linear taper
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        # Normalize distances within bbox size
        dx = (xs_m - cx) / (0.5 * max(bw, 1))
        dy = (ys_m - cy) / (0.5 * max(bh, 1))
        center_w = 1.0 / (1.0 + 2.0 * (dx * dx + dy * dy))  # peak=1 at center, decays outward

        # Floor de-emphasis: rows near the bottom get reduced weight
        # y increases downward; map y in [y1,y2] -> t in [0,1]
        t_row = (ys_m - y1) / max(bh, 1)
        floor_w = 1.0 - floor_deemph * t_row
        floor_w = np.clip(floor_w, 0.2, 1.0)

        # Huber-like depth consistency weight around z0
        sigma_mm = huber_sigma_m * 1000.0
        depth_res = (dsel_mm - z0_mm) / max(sigma_mm, 1e-6)
        huber_w = 1.0 / (1.0 + (depth_res * depth_res))

        w = center_w * floor_w * huber_w
        if np.sum(w) < 1e-3:
            return [-100., -100., -100.]

        # 6) Weighted estimates
        z_m = float(np.sum(w * (dsel_mm / 1000.0)) / np.sum(w))
        u_mean = float(np.sum(w * xs_m) / np.sum(w))
        v_mean = float(np.sum(w * ys_m) / np.sum(w))

        # 7) Back-project to 3D (camera intrinsics on self: fx, fy, cx, cy)
        # x right, y down, z forward. (+y down per your note)
        x = z_m * (u_mean - self.cx) / self.fx
        y = z_m * (v_mean - self.cy) / self.fy   # <-- positive down
        return [float(x), float(y), float(z_m)]