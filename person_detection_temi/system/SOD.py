import logging

from ultralytics import YOLO
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
import torch.nn as nn
import torch
import cv2
import time
import numpy as np
import threading

from person_detection_temi.system.kpr_reid import KPR as KPR_torch
from person_detection_temi.system.kpr_reid_onnx import KPR as KPR_onnx
from person_detection_temi.system.utils import kp_img_to_kp_bbox, rescale_keypoints, iou_vectorized, compute_center_distances
from person_detection_temi.system.memory_manager import MemoryManager
from lion_pytorch import Lion
from torchvision.ops import sigmoid_focal_loss

def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoidal positional encoding: shape (1, n_position, d_hid)"""
    position = torch.arange(n_position, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2, dtype=torch.float32) * -(np.log(10000.0) / d_hid))
    
    pe = torch.zeros(n_position, d_hid)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # shape: (1, n_position, d_hid)

# ---- Model ----
class TinyTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=2, num_layers=2, seq_len=7, num_classes=1, dropout = 0.0):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # CLS token (1, 1, 512)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional embeddings for CLS + 6 tokens → total 7
        self.register_buffer("pos_embed", get_sinusoid_encoding(seq_len + 1, d_model))  # [1, 7, 512]

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = dropout

        # MLP classifier
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, attention_mask=None):  # x: (B, 6, 512)
        B = x.size(0)

        # Prepend CLS token
        cls_token = self.cls_token.expand(B, 1, self.d_model)  # (B, 1, 512)
        x = torch.cat([cls_token, x], dim=1)  # (B, 7, 512)

        # Add positional embeddings
        x = x + self.pos_embed[:, :x.size(1), :]

        # If attention_mask is provided, adjust it
        if attention_mask is not None:
            # attention_mask is visibility: shape (B, 6), dtype=bool, True=visible
            # CLS token is always visible
            cls_mask = torch.ones((B, 1), dtype=torch.bool, device=attention_mask.device)
            extended_mask = torch.cat([cls_mask, attention_mask], dim=1)  # (B, 7)
            # Transformer expects True for *masked* tokens
            key_padding_mask = ~extended_mask  # (B, 7)
        else:
            key_padding_mask = None

        # Transformer
        x = self.transformer(x, src_key_padding_mask = key_padding_mask)  # (B, 7, 512)
        cls_output = x[:, 0]     # Take CLS token output

        return self.mlp_head(cls_output)  # (B, num_classes), (B, 512)

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
        self.reid_counter = 0
        self.reid_count_thr = 3
        ############################

        # ONLINE training ############################
        self.transformer_classifier = TinyTransformer()
        self.transformer_classifier.to(self.device)
        self.classifier_optimizer = AdamW(self.transformer_classifier.parameters(), lr=1e-5, weight_decay=1e-5)
        # self.classifier_optimizer = SGD(self.transformer_classifier.parameters(), lr=1e-2, weight_decay=1e-5)

        # self.classifier_optimizer = Lion(
        #     self.transformer_classifier.parameters(),
        #     lr=1e-4,
        #     weight_decay=1e-5
        # )
        self.classifier_criterion = nn.BCEWithLogitsLoss()
        # self.classifier_criterion = nn.BCEWithLogitsLoss(weight=torch.Tensor([5.0]).to(self.device))
        # self.classifier_criterion = nn.CrossEntropyLoss()
        # self.classifier_criterion = lambda logits, targets: sigmoid_focal_loss(
        #     inputs=logits,
        #     targets=targets,
        #     alpha=0.8,   # equivalent to weighting positives more heavily
        #     gamma=2.0,
        #     reduction='mean'
        # )
        self.class_prediction_thr = 0.8
        self.reid_stream = torch.cuda.Stream()        
        self.yolo_stream = torch.cuda.Stream()        
        self.extract_feats_target = True
        self.distractors = []
        self.memory = MemoryManager(max_samples=100, 
                                    alpha=0.5, 
                                    sim_thresh=0.75, # If Small aways preserve the newest features
                                    feature_dim=512, 
                                    num_parts=6,
                                    pseudo_std = 0.001)
        ##############################################

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
        self.class_prediction_thr = 0.8

    def unset_target_id(self):
        # Set the target ID to none to avoid Re-identifying when Unnecesary
        self.target_id = None
        self.class_prediction_thr = 0.8

        # Reset all the reidentifying mechanism weights to train from scratch whn apropiate
        for layer in self.transformer_classifier.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


    def compute_iou_with_flag_numpy(self, bboxes, ref_box):
        """
        Compute IoU of a reference box against all other boxes in a NumPy array.

        Args:
            bboxes (np.ndarray): shape [N, 4] with [x1, y1, x2, y2]
            ref_box (np.ndarray): shape [1, 4], extracted as bboxes[[idx]]

        Returns:
            ious (np.ndarray): IoU values, shape [N-1]
            has_intersection (bool): True if any IoU > 0
        """
        assert ref_box.shape == (1, 4), f"Expected ref_box shape [1, 4], got {ref_box.shape}"

        # Convert [1, 4] → [4]
        ref_box = ref_box[0]  # Now shape [4]

        # Mask out the reference box
        other_boxes = bboxes[~np.all(bboxes == ref_box, axis=1)]  # [N-1, 4]

        # Intersection coordinates
        x1 = np.maximum(ref_box[0], other_boxes[:, 0])
        y1 = np.maximum(ref_box[1], other_boxes[:, 1])
        x2 = np.minimum(ref_box[2], other_boxes[:, 2])
        y2 = np.minimum(ref_box[3], other_boxes[:, 3])

        # Compute intersection area
        inter_w = np.clip(x2 - x1, a_min=0, a_max=None)
        inter_h = np.clip(y2 - y1, a_min=0, a_max=None)
        inter_area = inter_w * inter_h

        # Areas
        ref_area = (ref_box[2] - ref_box[0]) * (ref_box[3] - ref_box[1])
        other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])

        # Union and IoU
        union_area = ref_area + other_areas - inter_area
        ious = inter_area / np.clip(union_area, a_min=1e-6, a_max=None)

        has_intersection = np.any(ious > 0.0)

        return ious, has_intersection


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
            with torch.cuda.stream(self.yolo_stream):
                detections = self.masked_detections(
                    img_rgb, 
                    img_depth, 
                    detection_class=detection_class, 
                    track = True, 
                    detection_thr=0.0
                )
            self.yolo_stream.synchronize()  # Wait for GPU ops to finish
            end_time = time.time()
            masked_detections_time = (end_time - start_time) * 1000  # Convert to milliseconds
            # print(f"masked_detections execution time: {masked_detections_time:.2f} ms")
            total_execution_time += masked_detections_time

            if len(detections) < 1:
                return None

            # YOLO Detection Results
            detections_imgs, detection_kpts, bboxes, poses, track_ids, original_kpts = detections

            # Deactivate learning when intersection between bounding boxes to 
            # Have god qualit samples
            iou_intersect = False
            # if self.target_id in track_ids:
            #     if bboxes.shape[0] > 1:
            #         target_id_idx = np.where(track_ids == self.target_id)[0]
            #         _, iou_intersect = self.compute_iou_with_flag_numpy(bboxes, bboxes[target_id_idx])

            # If no detection (No human) then stay on reid mode and return Nothing
            if self.target_id is not None and self.target_id not in track_ids:
                threading.Thread(
                        target=self.reidentification,
                        args=(detections_imgs, detection_kpts, track_ids),
                        daemon=True
                    ).start()
            elif self.target_id is not None and self.target_id in track_ids and not iou_intersect:
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
            detection_thr = 0.5):

        results = self.detect_mot(img_rgb, detection_class=detection_class, track = track, detection_thr = detection_thr)  

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
        batched_imgs = torch.cat(subimages).to(device=self.device)
        batched_scaled_kpts = torch.stack(scaled_keypoints, dim=0).to(device=self.device)

        # The closes person to the Camera is updated in case the ID is to be manually selected

        self.closest_person_id = track_ids[np.argmin(poses[:, 2])]

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

        print("EXTRACTING FEATURES FROM: ", text)

        return idx_to_extract, is_positive


    def updating_reid(self, detections_imgs, detection_kpts, tracked_ids):
        if not self.reid_lock.acquire(blocking=False):
            return  # Skip if already running
        try:
            with torch.cuda.stream(self.reid_stream):

                # Get the index belonging to the target id
                target_id_idx = np.where(tracked_ids == self.target_id)[0]

                # From who to extract features?
                # This approach constraints the feature extraction to be of Batch one, avoiding memory overflow or high computation code
                # Extract features from 5 people in the image? 5 continuous frames needed
                idx_to_extract, is_positive = self.grab_single_sample(target_id_idx, tracked_ids)

                # Extract the features of the chosen index
                feats = self.feature_extraction(detections_imgs[[idx_to_extract]], detection_kpts[[idx_to_extract]])
                print("VISIBILITIES")
                print(feats[1])

                print("WOW", target_id_idx == idx_to_extract, is_positive)

                # Save Latest Extracted feature into memory
                if is_positive:
                    self.memory.insert_positive(feats)
                else:
                    self.memory.insert_negative(feats)

                # Get a sample based on the memory manager policy
                mem_feats, mem_vis, label = self.memory.get_sample()

                # Move everything into device
                mem_feats = mem_feats.to(self.device)
                mem_vis = mem_vis.to(self.device)
                label = label.to(self.device)

                # print("TRACKS", tracked_ids)
                # print("LOGITS:", logits.shape, logits.T)
                # print("LABELS", label.shape, label.T)

                ########################################################################
                # This part is about Retraining 
                # No Changes in this area for now
                # Probably fix the o find a better way to select a threshold
                ########################################################################

                #############################################################################################################
                # Online Continual Learning #################################################################################

                self.transformer_classifier.train()

                logits = self.transformer_classifier(mem_feats, mem_vis)

                print("Prediction_thr:", self.class_prediction_thr)
                print("Probs:", torch.sigmoid(logits).detach().cpu().numpy().T)
                
                best_prediction = torch.sigmoid(logits)[label.bool()].item()
                # alpha = 0.5
                # if best_prediction >= self.class_prediction_thr:
                #     self.class_prediction_thr = np.clip(alpha*self.class_prediction_thr + (1 - alpha)*best_prediction, a_min=0.4, a_max=0.8)

                loss = self.classifier_criterion(logits.flatten(), label.flatten())
                self.classifier_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transformer_classifier.parameters(), max_norm=0.5)
                self.classifier_optimizer.step()  
                # Online Continual Learning #################################################################################
                #############################################################################################################

            self.reid_stream.synchronize()

        finally:
            self.reid_lock.release()

    def reidentification(self, detections_imgs, detection_kpts, tracked_ids):
        if not self.reid_lock.acquire(blocking=False):
            return  # Skip if already running

        try:

            with torch.cuda.stream(self.reid_stream):
                # This is the reidentification procedure

                with torch.inference_mode():

                    # Consider trying to reid one at a time

                    self.transformer_classifier.eval()

                    f_, v_ = self.feature_extraction(detections_imgs, detection_kpts)

                    logits = self.transformer_classifier(f_, v_)

                    result = torch.sigmoid(logits).detach().cpu().numpy()

                    class_idx = np.argmax(result[:, 0])

                    print("THR:", self.class_prediction_thr)
                    print("BEST:", result[class_idx, 0])

                    if result[class_idx, 0] > np.floor(self.class_prediction_thr*10)/10:

                        self.reid_counter += 1

                        if self.reid_counter > self.reid_count_thr:

                            self.target_id = tracked_ids[class_idx]
                            self.reid_counter = 0

                    else:
                        self.reid_counter = 0
   
            self.reid_stream.synchronize()

        finally:

            self.reid_lock.release()

    def feature_extraction(self, detections_imgs, detection_kpts):
        # Extract features for similarity check
        fg_, vg_ = self.kpr_reid.extract(
                    detections_imgs, 
                    detection_kpts, 
                    return_heatmaps=False)
        
        return (fg_, vg_)
    
    def get_person_pose(self, bbox, kpts, depth_img, win_size=1):
        """
        Estimate 3D pose from keypoints and depth image using the mode of the depth
        values in a window around confident keypoints. Vectorized implementation.

        Args:
            bbox: [x_min, y_min, x_max, y_max] (not used)
            kpts: numpy array of shape (17, 3) where [:, 0] = u, [:, 1] = v, [:, 2] = conf
            depth_img: numpy array HxW with depth values in mm
            win_size: int, half-size of the window around each keypoint

        Returns:
            [x, y, z] in meters in the camera optical frame
        """
        if depth_img is None or kpts is None or len(kpts) == 0:
            return [-100., -100., -100.]

        H, W = depth_img.shape

        kpts = kpts.cpu().numpy()

        # Filter keypoints with confidence > 0.5
        valid_kpts = kpts[kpts[:, 2] > 0.5]
        if len(valid_kpts) == 0:
            return [-100., -100., -100.]

        u_coords = valid_kpts[:, 0].astype(np.int32)
        v_coords = valid_kpts[:, 1].astype(np.int32)

        # Create window offsets
        offset_range = np.arange(-win_size, win_size + 1)
        du, dv = np.meshgrid(offset_range, offset_range)
        du = du.flatten()
        dv = dv.flatten()

        # Broadcast offsets to all keypoints
        all_u = (u_coords[:, None] + du[None, :]).reshape(-1)
        all_v = (v_coords[:, None] + dv[None, :]).reshape(-1)

        # Filter out-of-bounds pixel indices
        valid_mask = (all_u >= 0) & (all_u < W) & (all_v >= 0) & (all_v < H)
        all_u = all_u[valid_mask]
        all_v = all_v[valid_mask]

        # Sample depth values
        depth_values = depth_img[all_v, all_u]
        valid_depths = depth_values[(depth_values > 0) & (depth_values < 10000)]

        if valid_depths.size == 0:
            return [-100., -100., -100.]

        # Mode depth estimation
        values, counts = np.unique(valid_depths, return_counts=True)
        mode_depth = values[np.argmax(counts)]
        z = mode_depth / 1000.0  # mm → meters

        # Use the average of all valid keypoints for projection
        u_mean = np.mean(valid_kpts[:, 0])
        v_mean = np.mean(valid_kpts[:, 1])

        # Back-project to 3D
        x = z * (u_mean - self.cx) / self.fx
        y = -z * (v_mean - self.cy) / self.fy

        return [float(x), float(y), float(z)]