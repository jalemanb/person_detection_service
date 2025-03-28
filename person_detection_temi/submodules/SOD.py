from ultralytics import YOLO
import torch.nn.functional as F
import torch, cv2
import time
import numpy as np

from person_detection_temi.submodules.kpr_onnx import KPR as KPR_onnx
from person_detection_temi.submodules.kpr_reid import KPR as KPR_torch
from person_detection_temi.submodules.bbox_kalman_filter import BboxKalmanFilter, chi2inv95
from person_detection_temi.submodules.memory import Bucket
from person_detection_temi.submodules.utils import kp_img_to_kp_bbox, rescale_keypoints, iou_vectorized, compute_center_distances, bbox_to_xyah, xyah_to_bbox

class SOD:

    def __init__(self, yolo_model_path, feature_extracture_model_path, feature_extracture_cfg_path, tracker_system_path = "") -> None:

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tracker_file = tracker_system_path

        # Detection Model
        self.yolo = YOLO(yolo_model_path)  # load a pretrained model (recommended for training)

        # ReID System
        # self.kpr_reid = KPR_onnx(feature_extracture_cfg_path, feature_extracture_model_path, kpt_conf=0.8, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.kpr_reid = KPR_torch(feature_extracture_cfg_path, kpt_conf=0., device='cuda' if torch.cuda.is_available() else 'cpu')

        self.template = None
        self.template_features = None

        # Intel Realsense Values
        self.fx, self.fy, self.cx, self.cy = 620.8472290039062, 621.053466796875, 325.1631164550781, 237.45947265625  # RGB Camera Intrinsics
        self.fx, self.fy, self.cx, self.cy = 618.119349, 615.823749, 318.472087, 231.353083

        # ZED 1.0 Values
        self.fx, self.fy, self.cx, self.cy = 338.5, 338.5, 342.6, 176.2

        self.kk = [[self.fx, 0., self.cx],
                   [0., self.fy, self.cy],
                   [0., 0., 1.]]

        self.erosion_kernel = np.ones((9, 9), np.uint8)  # A 3x3 kernel, you can change the size

        self.tracker = BboxKalmanFilter()
        self.man_kf = None
        self.cov_kf = None

        self.reid_thr = 0.9

        self.memory_bucket = Bucket(max_identities = 10, samples_per_identity = 20, thr = 0.5)
        self.debug = False
        self.bucket_counter = 0
        self.store_features = True

        self.start = True

        self.reid_mode = True
        self.is_tracking = False

        self.whitelist = []
        self.blacklist = [-1]

    def to(self, device):
        self.device = device


    def set_track_id(self, id):
        self.whitelist.append(id)
        self.reid_mode = False


    def iknn(self, feats, feats_vis, metric="euclidean", threshold=0.8):
        """
        Compare tensors A[N, 6, 512] and B[batch, 6, 512] part-by-part with visibility filtering.
        Retrieve k smallest distances, classify based on nearest neighbors' labels, and apply a threshold.

        Args:
            A (torch.Tensor): Feature tensor of shape [N, 6, 512]
            B (torch.Tensor): Feature tensor of shape [batch, 6, 512]
            visibility_A (torch.Tensor): Visibility mask for A [N, 6] (bool)
            visibility_B (torch.Tensor): Visibility mask for B [batch, 6] (bool)
            labels_A (torch.Tensor): Boolean labels of shape [N] corresponding to A
            metric (str): Distance metric, "euclidean" or "cosine"
            k (int): Number of smallest distances to retrieve per part.
            threshold (float): Values greater than this threshold will be masked.

        Returns:
            classification (torch.Tensor): Classification tensor of shape [batch, 6] (boolean values)
            binary_mask (torch.Tensor): Binary mask of shape [k, batch, 6] indicating which top-k distances are within threshold
        """

        A, visibility_A, labels_A = self.memory_bucket.get()

        B = feats
        visibility_B = feats_vis

        print("visibility_B")
        print(visibility_B)
        # k = int(np.minimum(self.memory_bucket.get_samples_num(), np.sqrt(self.memory_bucket.get_max_samples())))
        k = int(np.minimum(self.memory_bucket.get_samples_num(), 10))

        N, parts, dim = A.shape
        batch = B.shape[0]

        # Expand A and B to match dimensions for pairwise comparison
        A_expanded = A.unsqueeze(1).expand(N, batch, parts, dim)  # [N, batch, 6, 512]
        B_expanded = B.unsqueeze(0).expand(N, batch, parts, dim)  # [N, batch, 6, 512]

        # Compute similarity/distance based on the selected metric
        if metric == "euclidean":
            distance = torch.norm(A_expanded - B_expanded, p=2, dim=-1)  # Euclidean distance
        elif metric == "cosine":
            distance = 1 - F.cosine_similarity(A_expanded, B_expanded, dim=-1)  # Cosine distance
        else:
            raise ValueError("Unsupported metric. Choose 'euclidean' or 'cosine'.")

        # Expand visibility masks for proper masking
        vis_A_expanded = visibility_A.unsqueeze(1).expand(N, batch, parts)  # [N, batch, 6]
        vis_B_expanded = visibility_B.unsqueeze(0).expand(N, batch, parts)  # [N, batch, 6]

        # Apply visibility mask: Only compare if both A and B parts are visible
        valid_mask = vis_A_expanded & vis_B_expanded  # Boolean mask
        distance[~valid_mask] = float("inf")  # Ignore invalid comparisons

        # Retrieve the k smallest distances along dim=0 (N dimension)
        top_k_values, top_k_indices = torch.topk(distance, k, dim=0, largest=False)  # [k, batch, 6]

        # Retrieve the corresponding labels for the k nearest neighbors This is the knn-prediction
        top_k_labels = labels_A[top_k_indices]  # Shape [k, batch, 6], labels for nearest N indices

        print("top_k_values")
        print(top_k_values)


        # Create binary mask based on threshold
        binary_mask = top_k_values <= threshold  # [k, batch, 6]
        #binary_mask = (top_k_values <= threshold) | (top_k_values > 10)

        # Apply threshold influence: Set labels to zero where distances exceed the threshold
        valid_labels = top_k_labels * binary_mask  # Zero out labels where threshold is exceeded


        # Perform classification by majority vote (sum up valid labels and classify based on majority vote)
        classification = (valid_labels.sum(dim=0) > (k // 2)).to(torch.bool)  # Shape [batch, 6]

        return classification.T


    def masked_detections(self, img_rgb, img_depth = None, detection_class = 0, size = (128, 384), track = False, detection_thr = 0.5):

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

                print("BOX", x1, x2, y1, y2, track_id)
                # Crop the Image
                subimage = cv2.resize(img_rgb[y1:y2, x1:x2], size)
                # Getting Eyes+Torso+knees Keypoints for pose estimation
                torso_kpts = kpts[:, :2].cpu().numpy()[[1, 2, 5, 6, 11, 12, 13, 14], :]
                torso_kpts = torso_kpts[~np.all(torso_kpts == 0, axis=1)].astype(np.int32) - 1 # Rest one to avoid incorrect pixel corrdinates
                # Getting the Person Central Pose (Based on Torso Keypoints)
                pose = self.get_person_pose(torso_kpts, img_depth)
                # Store all the bounding box detections and subimages in a tensor
                subimages.append(torch.tensor(subimage, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0))

                print("SUBIMAGE")
                print(subimage.shape)
                print(len(subimages))
                bboxes.append((x1, y1, x2, y2))
                track_ids.append(track_id)
                person_kpts.append(torso_kpts)
                poses.append(pose)
                # Scaling points to be with respect to the bounding box 
                kpts_box = kp_img_to_kp_bbox(kpts, (x1, y1, x2, y2))
                kpts_scaled = rescale_keypoints(kpts_box, (x2 - x1, y2 - y1), (size[0], size[1]))
                total_keypoints.append(kpts_scaled)

        poses = np.array(poses)
        bboxes = np.array(bboxes)
        track_ids = np.array(track_ids)
        batched_imgs = torch.cat(subimages).to(device=self.device)
        batched_kpts = torch.stack(total_keypoints, dim=0).to(device=self.device)
        return [batched_imgs, batched_kpts, bboxes, person_kpts, poses, track_ids]

    def detect(self, img_rgb, img_depth, detection_class=0):

        # Get Image Dimensions (Assumes noisy message wth varying image size) 
        img_h = img_rgb.shape[0]
        img_w = img_rgb.shape[1]

        self.bucket_counter += 1

        with torch.inference_mode():
                
            # If there is not template initialization then dont return anything
            if self.template is None:
                return None

            total_execution_time = 0  # To accumulate total time

            # Measure time for `masked_detections`
            start_time = time.time()
            detections = self.masked_detections(img_rgb, img_depth, detection_class=detection_class, track = True, detection_thr=0.0)
            end_time = time.time()
            masked_detections_time = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"masked_detections execution time: {masked_detections_time:.2f} ms")
            total_execution_time += masked_detections_time

            # If no detection (No human) then stay on reid mode and return Nothing
            if not (len(detections) > 0):
                self.reid_mode = True
                return None
            
            # YOLO Detection Results
            detections_imgs, detection_kpts, bboxes, person_kpts, poses, track_ids = detections

            # Up to This Point There are Only Yolo Detections #####################################

            if self.reid_mode: # ReId mode

            #     print("REID MODE")

            #     if self.is_tracking:

            #         print("TRACKING")
            #         print("USING SOME BBOXES")

            #         self.mean_kf, self.cov_kf = self.tracker.predict(self.mean_kf, self.cov_kf)
            #         mb_dist = self.tracker.gating_distance(self.mean_kf, self.cov_kf, bbox_to_xyah(bboxes))

            #         within_mb = np.argwhere(mb_dist < chi2inv95[4]).flatten().tolist()
            #         print("within_mb", within_mb)

            #         if len(within_mb) < 1:
            #             similarity = np.random.uniform(0, 1,  poses.shape[0]).tolist()
            #             return (poses, bboxes, person_kpts, track_ids, similarity, [])

            #         # Prune Bounding Boxes to only evaluate the closest ones to the target that might cause occlusion
            #         detections_imgs = detections_imgs[within_mb]
            #         detection_kpts = detection_kpts[within_mb]
            #         bboxes = bboxes[within_mb]
            #         person_kpts = [person_kpts[i] for i in within_mb]
            #         poses = poses[within_mb]
            #         track_ids = track_ids[within_mb]

            #         # Measure time for `feature_extraction` - Extract features to all subimages
            #         start_time = time.time()
            #         detections_features = self.feature_extraction(detections_imgs=detections_imgs, detection_kpts=detection_kpts)
            #         end_time = time.time()
            #         feature_extraction_time = (end_time - start_time) * 1000  # Convert to milliseconds
            #         print(f"feature_extraction execution time: {feature_extraction_time:.2f} ms")
            #         total_execution_time += feature_extraction_time

            #         # Measure time for `iknn_time` - Classify the features with KNN
            #         start_time = time.time()
            #         classification = self.iknn(detections_features[0], detections_features[1], threshold=self.reid_thr)
            #         end_time = time.time()
            #         iknn_time = (end_time - start_time) * 1000  # Convert to milliseconds
            #         print(f"iknn_time execution time: {iknn_time:.2f} ms")
            #         total_execution_time += iknn_time
            #         print("CLASSIFICATION", classification)

            #         if len(within_mb) > 1:

            #             part_gates = torch.sum(classification & detections_features[1].T, dim=0)
            #             max_part_gate = torch.max(part_gates)

            #             knn_gate = (part_gates == max_part_gate).cpu().numpy()
            #         else:
            #             knn_gate = (torch.sum(classification & detections_features[1].T, dim=0) >= torch.sum(detections_features[1].T, dim=0) - 1).cpu().numpy()


            #         mb_dist = np.array(mb_dist)[within_mb]
            #         mb_gate = mb_dist < chi2inv95[4]

            #         gate = knn_gate*mb_gate

            #         # Get All indices belonging to valid Detections
            #         best_idx = np.argwhere(gate == 1).flatten().tolist()

            #         if np.sum(gate) == 1:
            #             self.mean_kf, self.cov_kf = self.tracker.update(self.mean_kf, self.cov_kf, bbox_to_xyah(bboxes)[best_idx[0]])

            #             # latest_features = detections_features[0][best_idx]
            #             # latest_visibilities = detections_features[1][best_idx]
            #             ## For debugging Purposes #####################
            #             # latest_template = detections_imgs[best_idx]
            #             ###############################################
            #             # self.memory_bucket.store_feats(latest_features, latest_visibilities, counter = self.bucket_counter, img_patch=latest_template.cpu().numpy(), debug = self.debug)
              
            #     else:

                print("NO TRACKING")
                print("USING NON BLACKLISTED BBOXES")
                tracked_ids = track_ids.tolist()

                to_test = [tracked_ids.index(i) for i in tracked_ids if i not in self.blacklist]

                print("self.blacklist", self.blacklist)

                print("To_TEST_LIST", to_test)

                if len(to_test) > 0:
                    # Measure time for `feature_extraction` - Extract features to all subimages
                    start_time = time.time()
                    detections_features = self.feature_extraction(detections_imgs=detections_imgs[to_test], detection_kpts=detection_kpts[to_test])
                    end_time = time.time()
                    feature_extraction_time = (end_time - start_time) * 1000  # Convert to milliseconds
                    print(f"feature_extraction execution time: {feature_extraction_time:.2f} ms")
                    total_execution_time += feature_extraction_time

                    # Measure time for `iknn_time` - Classify the features with KNN
                    start_time = time.time()
                    classification = self.iknn(detections_features[0], detections_features[1], threshold=self.reid_thr)
                    end_time = time.time()
                    iknn_time = (end_time - start_time) * 1000  # Convert to milliseconds
                    print(f"iknn_time execution time: {iknn_time:.2f} ms")
                    total_execution_time += iknn_time
                    print("CLASSIFICATION", classification)

                    # When Relying only on visual features, it is necessary to reidentfy consdiering all body parts to make a decision
                    knn_gate = (torch.sum(classification & detections_features[1].T, dim=0) >= torch.sum(detections_features[1].T, dim=0) - 1 ).cpu().numpy()

                    gate = knn_gate

                    # Get All indices belonging to valid Detections
                    best_idx = np.argwhere(gate == 1).flatten().tolist()

                    valid_idxs = [to_test[i] for i in best_idx]

                    self.whitelist = []
                    for i in valid_idxs:
                        self.whitelist.append(track_ids[i]) 

                    print("self.whitelist", self.whitelist)
                
                    # If there is not a valid detection but a box is being tracked (keep prediction until box is out of fov)
                    if not np.sum(gate) and self.is_tracking:
                        self.reid_mode = True
                        valid_idxs = []
                    
                    # If there are no valid detection and no box is being tracked
                    elif not np.sum(gate) and not self.is_tracking:
                        self.reid_mode = True
                        valid_idxs = []
                        
                    # If there is only valid detection 
                    elif np.sum(gate) == 1 and not self.is_tracking: 
                        self.reid_mode = False

                    elif np.sum(gate) == 1 and self.is_tracking: 
                        # Extra conditions
                        self.reid_mode = False
                else:
                    valid_idxs = []

            else: # Tracking mode
                self.store_features = True
                valid_idxs = []
                tracked_ids = track_ids.tolist()

                for target_id in self.whitelist:
                    if target_id in track_ids:
                        valid_idxs.append(tracked_ids.index(target_id))

                        # Clearing blacklisted ids if they are not in the image anymore
                        # for i in range(len(self.blacklist)):
                        #     if self.blacklist[i] not in tracked_ids:
                        #         self.blacklist.pop(i)
                    else:
                        self.whitelist = []
                        self.reid_mode = True
                        self.store_features = False

                #### IF SPATIAL AMBIGUITY IS PRESENT GO BACK TO ADD APPEARANCE INFORMATION FOR ASSOCIATION ###############

                second_best_idx = 0

                # Check if bounding boxes are too close to the target
                # if so return nothing and change to reid_mode
                if len(valid_idxs) > 0  and len(tracked_ids) > 1:
                    # See one time steps into the futre if there will be an intersection
                    # fut_mean, fut_cov = self.tracker.predict(self.mean_kf, self.cov_kf)
                    # fut_mean, fut_cov = self.tracker.predict(fut_mean, fut_cov)
                    # fut_target_bbox = xyah_to_bbox(fut_mean[:4])[0]

                    distractor_bbox = np.delete(bboxes, valid_idxs[0], axis=0)
                    ious_to_target = iou_vectorized(bboxes[valid_idxs[0]],  distractor_bbox)
                    distances = compute_center_distances(bboxes[valid_idxs[0]],  bboxes)
                    distances = np.where(distances > 0, distances, np.inf)
                    second_best_idx = np.argmin(distances)

                    print("ious_to_target", ious_to_target)

                    if  np.any(ious_to_target > 0.):
                        self.store_features = False

                    if  np.any(ious_to_target > 0.):
                        print("Too close to obstacle")
                        self.whitelist = []

                        if tracked_ids[second_best_idx] in self.blacklist:
                            self.blacklist.pop(self.blacklist.index(tracked_ids[second_best_idx]))

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
                        latest_features = self.feature_extraction(detections_imgs=detections_imgs[valid_idxs], detection_kpts=detection_kpts[valid_idxs])
                        self.memory_bucket.store_feats(latest_features[0], latest_features[1], counter = self.bucket_counter, img_patch = detections_imgs[valid_idxs].cpu().numpy(), debug = self.debug)
                    else:
                        valid_idxs.append(second_best_idx)
                        print("POS NEG FEATURE EXTRACT", valid_idxs)
                        latest_features = self.feature_extraction(detections_imgs=detections_imgs[valid_idxs], detection_kpts=detection_kpts[valid_idxs])
                        
                        self.memory_bucket.store_feats(latest_features[0][[0]], latest_features[1][[0]], debug = False)
                        self.memory_bucket.store_distractor_feats(latest_features[0][[1]], latest_features[1][[1]])
                        valid_idxs = [valid_idxs[0]]

                    end_time = time.time()
                    incremental_time = (end_time - start_time) * 1000  # Convert to milliseconds
                    print(f"INCREMENTAL execution time: {incremental_time:.2f} ms")

            #To erase #####################################################
            similarity = np.random.uniform(0, 1,  poses.shape[0]).tolist()

            # Return results
            return (poses, bboxes, person_kpts, track_ids, similarity, valid_idxs)

    def similarity_check(self, template_features, detections_features, similarity_thr):

        fq_, vq_ = template_features
        fg_, vg_ = detections_features

        dist, part_dist = self.kpr_reid.compare(fq_, fg_, vq_, vg_)

        return dist, part_dist
    
    def detect_mot(self, img, detection_class, track = False, detection_thr = 0.5):
        # Run multiple object detection with a given desired class
        if track:
            return self.yolo.track(img, persist=True, classes = detection_class, tracker=self.tracker_file, iou = 0.2, conf = detection_thr)
        else:
            return self.yolo(img, classes = detection_class, conf = detection_thr)
            
    def template_update(self, template):

        detections = self.masked_detections(template, track=False, detection_thr = 0.8)

        if len(detections):
            self.template = detections[0]
            self.template_kpts = detections[1]

        self.template_features = self.extract_features(self.template, self.template_kpts)

        # Store First Initial Features on Galery
        self.memory_bucket.store_feats(self.template_features[0], self.template_features[1], img_patch = self.template.cpu().numpy(), debug = False)
        
    def feature_extraction(self, detections_imgs, detection_kpts):
        # Extract features for similarity check
        return self.extract_features(detections_imgs, detection_kpts)

    def get_target_rgb_and_depth(self, rgb_img, depth_img, bbox, seg_mask):
        # Get the desired person subimage both in rgb and depth
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])

        # Ensure the mask is binary (0 or 255)
        binary_mask = (seg_mask > 0).astype(np.uint8) * 255

        # Create an output image that shows only the highlighted pixels
        masked_rgb_img = cv2.bitwise_and(rgb_img, rgb_img, mask=binary_mask)
        masked_depth_img = cv2.bitwise_and(depth_img, depth_img, mask=binary_mask)

        # Return Target Images With no background of the target person for orientation estimation
        return masked_depth_img, masked_rgb_img

    def feature_distance(self, template_features, detections_features, mode = 'cosine'):

        # Compute Similarity Check
        if mode == 'cosine':
            L = F.cosine_similarity(template_features, detections_features, dim=1)
        elif mode == 'eucledian':
            L = torch.cdist(template_features.to(torch.float32), detections_features.to(torch.float32), p=2)

        # Return most similar index
        return L
        
    def get_template_results(self, detections, most_similar_idx, img_size):
        # Get the segmentation mask
        segmentation_mask = detections[0].masks.data[most_similar_idx].to('cpu').numpy()
        # Resize the mask to match the image size
        segmentation_mask = cv2.resize(segmentation_mask, img_size, interpolation=cv2.INTER_NEAREST)
        # Get the corresponding bounding box
        bbox = detections[0].boxes[most_similar_idx].to('cpu')
        return bbox, segmentation_mask

    def extract_subimages(self, image, results, size=(224, 224)):
        subimages = []
        for result in results:
            boxes = result.boxes  # Boxes object
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                subimage = image[y1:y2, x1:x2]
                subimage = cv2.resize(subimage, size)
                subimages.append(subimage)
        batched_tensor = torch.stack([torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) for img in subimages]) 
        return batched_tensor

    def extract_features(self, image, kpt):

        fg_, vg_ = self.kpr_reid.extract(image, kpt, return_heatmaps=False)
        
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
        
    def yaw_to_quaternion(self, yaw):
        """
        Convert a yaw angle (in radians) to a quaternion.
        Parameters:
        yaw (float): The yaw angle in radians.
        Returns:
        np.ndarray: The quaternion [w, x, y, z].
        """
        half_yaw = yaw / 2.0
        qw = np.cos(half_yaw)
        qx = 0.0
        qy = 0.0
        qz = np.sin(half_yaw)
        
        return (qx, qy, qz, qw)