from ultralytics import YOLO
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import torch, cv2
import os, time
from person_detection_temi.submodules.utils.preprocessing import preprocess_rgb, preprocess_depth
import numpy as np
import torchvision.transforms as transforms

from person_detection_temi.submodules.super_reid.keypoint_promptable_reidentification.torchreid.scripts.builder import build_config
from person_detection_temi.submodules.super_reid.kpr_reid import KPR

from person_detection_temi.submodules.OCL import MultiPartClassifier, MemoryManager

def kp_img_to_kp_bbox(kp_xyc_img, bbox_xyxy):
    """
    Convert keypoints in image coordinates to bounding box coordinates and filter out keypoints 
    that are outside the bounding box.
    
    Args:
        kp_xyc_img (np.ndarray): Keypoints in image coordinates, shape (K, 3) where columns are (x, y, c).
        bbox_xyxy (np.ndarray): Bounding box, shape (4,) as [x1, y1, x2, y2].
    
    Returns:
        np.ndarray: Keypoints in bounding box coordinates, shape (K, 3), with invalid keypoints set to 0.
    """
    # Unpack the bounding box coordinates
    x1, y1, x2, y2 = bbox_xyxy
    
    # Calculate the width and height of the bounding box
    w = x2 - x1
    h = y2 - y1
    
    # Create a copy of the keypoints to work on
    kp_xyc_bbox = kp_xyc_img.clone()
    
    # Transform keypoints to bounding box-relative coordinates
    kp_xyc_bbox[..., 0] = kp_xyc_img[..., 0] - x1
    kp_xyc_bbox[..., 1] = kp_xyc_img[..., 1] - y1
    
    # Filter out keypoints outside the bounding box or with confidence == 0
    invalid_mask = (
        (kp_xyc_img[..., 2] == 0) |    # Keypoint has 0 confidence
        (kp_xyc_bbox[..., 0] < 0) |   # X-coordinate is out of bounds (left)
        (kp_xyc_bbox[..., 0] >= w) |  # X-coordinate is out of bounds (right)
        (kp_xyc_bbox[..., 1] < 0) |   # Y-coordinate is out of bounds (top)
        (kp_xyc_bbox[..., 1] >= h)    # Y-coordinate is out of bounds (bottom)
    )
    
    # Set invalid keypoints to zero
    kp_xyc_bbox[invalid_mask] = 0
    
    return kp_xyc_bbox

def rescale_keypoints(rf_keypoints, size, new_size):
    """
    Rescale keypoints to new size.
    Args:
        rf_keypoints (np.ndarray): keypoints in relative coordinates, shape (K, 2)
        size (tuple): original size, (w, h)
        new_size (tuple): new size, (w, h)
    Returns:
        rf_keypoints (np.ndarray): rescaled keypoints in relative coordinates, shape (K, 2)
    """
    w, h = size
    new_w, new_h = new_size
    rf_keypoints = rf_keypoints.clone()
    rf_keypoints[..., 0] = rf_keypoints[..., 0] * new_w / w
    rf_keypoints[..., 1] = rf_keypoints[..., 1] * new_h / h

    assert ((rf_keypoints[..., 0] >= 0) & (rf_keypoints[..., 0] <= new_w)).all()
    assert ((rf_keypoints[..., 1] >= 0) & (rf_keypoints[..., 1] <= new_h)).all()

    return rf_keypoints

def get_indices_and_values_as_lists_torch(tensor, threshold, less_than = True):
    """
    Get the flattened indices and values of elements in a tensor smaller than a given threshold as Python lists.
    
    Args:
        tensor (torch.Tensor): The input tensor of any shape.
        threshold (float): The threshold value.
    
    Returns:
        list, list: Flattened indices and values as Python lists.
    """
    if less_than:
        # Create a boolean mask for elements smaller than the threshold
        mask = tensor < threshold
    else:
        mask = tensor > threshold
    
    # Get the flattened indices of the elements that match the condition
    valid_indices = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
    
    # Extract the corresponding values
    values = tensor[mask]
    
    # Convert indices and values to Python lists
    indices_list = valid_indices.tolist()
    values_list = values.tolist()
    
    return indices_list, values_list

def get_indices_and_values_as_lists_np(array, threshold, less_than=True):
    """
    Get the flattened indices and values of elements in a NumPy array smaller (or greater) than a given threshold as Python lists.
    
    Args:
        array (np.ndarray): The input array of any shape.
        threshold (float): The threshold value.
        less_than (bool): If True, look for values less than the threshold. Otherwise, greater than the threshold.
    
    Returns:
        list, list: Flattened indices and values as Python lists.
    """
    if less_than:
        # Create a boolean mask for elements smaller than the threshold
        mask = array < threshold
    else:
        mask = array > threshold
    
    # Get the flattened indices of the elements that match the condition
    valid_indices = np.flatnonzero(mask)
    
    # Extract the corresponding values
    values = array[mask]
    
    # Convert indices and values to Python lists
    indices_list = valid_indices.tolist()
    values_list = values.tolist()
    
    return indices_list, values_list

class SOD:

    def __init__(self, yolo_model_path, feature_extracture_model_path, tracker_system_path) -> None:

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tracker_file = tracker_system_path

        # Detection Model
        self.yolo = YOLO(yolo_model_path, task="segment_pose")  # load a pretrained model (recommended for training)

        # ReID System
        kpr_cfg = build_config(config_path="/home/enrique/vision_ws/src/person_detection_temi/models/kpr_market_test.yaml", display_diff=True)
        self.kpr_reid = KPR(cfg=kpr_cfg, kpt_conf=0.8, device='cuda' if torch.cuda.is_available() else 'cpu')

        self.cls = MultiPartClassifier(6, 512, 'svm')
        self.mem = MemoryManager(100, 6, 512)

        self.template = None
        self.template_features = None

        self.fx, self.fy, self.cx, self.cy = 620.8472290039062, 621.053466796875, 325.1631164550781, 237.45947265625  # RGB Camera Intrinsics

        self.fx, self.fy, self.cx, self.cy = 618.119349, 615.823749, 318.472087, 231.353083

        self.kk = [[self.fx, 0., self.cx],
                   [0., self.fy, self.cy],
                   [0., 0., 1.]]

        self.erosion_kernel = np.ones((9, 9), np.uint8)  # A 3x3 kernel, you can change the size

    def to(self, device):
        self.device = device

    def masked_detections(self, img_rgb, img_depth = None, detection_class = 0, size = (128, 384), track = False):

        results = self.detect_mot(img_rgb, detection_class=detection_class, track = track)  

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
            masks = result.masks
            keypoints = result.keypoints

            for i ,(box, mask, kpts) in enumerate(zip(boxes, masks, keypoints.data)):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = -1 if box.id is None else box.id.int().cpu().item()
                # Masking the original RGB image
                contour = mask.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                b_mask = np.zeros(img_rgb.shape[:2], np.uint8)
                b_mask = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                # Define the kernel for erosion (you can adjust the size for stronger/weaker erosion)
                b_mask = cv2.erode(b_mask, self.erosion_kernel, iterations=2)  # Apply erosion to the binary mask
                mask3ch = cv2.bitwise_and(img_rgb, img_rgb, mask=b_mask)
                # Crop the Image
                subimage = cv2.resize(img_rgb[y1:y2, x1:x2], size)
                # Getting Eyes+Torso+knees Keypoints for pose estimation
                torso_kpts = kpts[:, :2].cpu().numpy()[[1, 2, 5, 6, 11, 12, 13, 14], :]
                torso_kpts = torso_kpts[~np.all(torso_kpts == 0, axis=1)].astype(np.int32) - 1 # Rest one to avoid incorrect pixel corrdinates
                torso_kpts = np.array([kp for kp in torso_kpts if b_mask[kp[1], kp[0]] > 0])
                # Getting the Person Central Pose (Based on Torso Keypoints)
                pose = self.get_person_pose(torso_kpts, img_depth)
                # Store all the bounding box detections and subimages in a tensor
                subimages.append(torch.tensor(subimage, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0))
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
        batched_tensor = torch.cat(subimages).to(device=self.device)
        batched_kpts = torch.stack(total_keypoints, dim=0).to(device=self.device)
        return [batched_tensor, batched_kpts, bboxes, person_kpts, poses, track_ids]

    def detect(self, img_rgb, img_depth, detection_thr=0.7, detection_class=0):

        if self.template is None:
            return None

        total_execution_time = 0  # To accumulate total time

        # Measure time for `masked_detections`
        start_time = time.time()
        detections = self.masked_detections(img_rgb, img_depth, detection_class=detection_class, track = True)
        end_time = time.time()
        masked_detections_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"masked_detections execution time: {masked_detections_time:.2f} ms")
        total_execution_time += masked_detections_time

        if not (len(detections) > 0):
            return None

        detections_imgs, detection_kpts, bboxes, person_kpts, poses, track_ids = detections

        # Measure time for `feature_extraction`
        start_time = time.time()
        detections_features = self.feature_extraction(detections_imgs=detections_imgs, detection_kpts=detection_kpts)
        end_time = time.time()
        feature_extraction_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"feature_extraction execution time: {feature_extraction_time:.2f} ms")
        total_execution_time += feature_extraction_time

        # Measure time for `similarity_check`
        start_time = time.time()
        similarity_check = self.similarity_check(self.template_features, detections_features, 0.75)
        end_time = time.time()
        similarity_check_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"similarity_check execution time: {similarity_check_time:.2f} ms")
        total_execution_time += similarity_check_time

        if similarity_check is None:
            return None
        else:
            valid_idxs, similarity = similarity_check

        print("valid_idxs", valid_idxs)
        print("similarity", similarity)

        # Return results
        #return (poses[valid_idxs], bboxes[valid_idxs], person_kpts, track_ids, similarity, valid_idxs)
        return (poses, bboxes, person_kpts, track_ids, similarity, valid_idxs)

    def similarity_check(self, template_features, detections_features, similarity_thr):

        fq_, vq_ = template_features
        fg_, vg_ = detections_features

        if self.cls.is_trained():
            predictions = self.cls.predict(fg_.detach().cpu().numpy(), vg_.detach().cpu().numpy())
            print("predictions", predictions.shape, predictions)

            # Implement this to make decisions based on an svm or logistic 
            if self.cls.is_svm:
                best_idx,  decision_v = get_indices_and_values_as_lists_np(predictions, 0, less_than=False)

                if (len(best_idx) > 1 and np.max(decision_v) > 3) :
                    labels = np.zeros(fg_.shape[0]); labels[np.argmax(decision_v)] = 1
                    self.cls.train(fg_.detach().cpu().numpy().astype(np.float64), vg_.detach().cpu().numpy(), labels)

            elif self.cls.is_logistic:
                best_idx,  decision_v = get_indices_and_values_as_lists_np(predictions, 0.5, less_than=False)

            decision_vals = predictions.tolist()
        
        else:
            dist, part_dist = self.kpr_reid.compare(fq_, fg_, vq_, vg_)
            print("dist", dist.shape, dist)

            best_idx,  decision_v = get_indices_and_values_as_lists_torch(dist, similarity_thr)

            decision_vals = dist[0].tolist()

            print("decision_v", len(decision_v), decision_v)

            # training a more robust decision maker 
            if (len(best_idx) > 0 and np.min(decision_v) < 0.6 and not self.cls.is_trained()) :

                labels = np.zeros(fg_.shape[0]); labels[np.argmin(decision_v)] = 1

                if self.mem.positive_count() < 15 or self.mem.negative_count() < 15:

                    self.mem.collect(fg_.detach().cpu().numpy(), vg_.detach().cpu().numpy(), labels)
                else:
                    # train for the first time
                    samples_train, vis_train, labels_train = self.mem.get_samples()
                    self.cls.train(samples_train, vis_train, labels_train)


        print("Number of samples: ", self.mem.total_count())
        print("best_idx", len(best_idx), best_idx)
        print("decision_v", len(decision_v), decision_v)
        print("decision_vals", len(decision_vals), decision_vals)

        return (best_idx, decision_vals)
    
    def detect_mot(self, img, detection_class, track = False):
        # Run multiple object detection with a given desired class
        if track:
            return self.yolo.track(img, persist=True, classes = detection_class, tracker=self.tracker_file, iou = 0.2, conf = 0.8)
        else:
            return self.yolo(img, classes = detection_class, conf = 0.8)
            
    def template_update(self, template):

        detections = self.masked_detections(template, track=False)

        if len(detections):
            self.template = detections[0]
            self.template_kpts = detections[1]

        self.template_features = self.extract_features(self.template, self.template_kpts)
    
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
    
