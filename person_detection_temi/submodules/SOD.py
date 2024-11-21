from ultralytics import YOLO
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import torch, cv2
import os, time
from person_detection_temi.submodules.utils.preprocessing import preprocess_rgb, preprocess_depth
import numpy as np
import torchvision.transforms as transforms
import torchreid
from monoloco.network import Loco

class SOD:

    def __init__(self, yolo_model_path, feature_extracture_model_path) -> None:

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Detection Model
        self.yolo = YOLO(yolo_model_path, task="segment_pose")  # load a pretrained model (recommended for training)

        self.features = torchreid.models.build_model(
            name='osnet_x0_25',
            num_classes=1,
            pretrained=True  # Set to False if you want an untrained model
        )

        torchreid.utils.load_pretrained_weights(self.features, feature_extracture_model_path) 
        self.features.eval()
        self.features.half()
        self.features.to(self.device)

        self.template = None
        self.template_features = None
        
        self.img_transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225])
        ])

        self.fx, self.fy, self.cx, self.cy = 620.8472290039062, 621.053466796875, 325.1631164550781, 237.45947265625  # RGB Camera Intrinsics

        self.fx, self.fy, self.cx, self.cy = 618.119349, 615.823749, 318.472087, 231.353083

        self.kk = [[self.fx, 0., self.cx],
                   [0., self.fy, self.cy],
                   [0., 0., 1.]]
        
        width, height = 640, 480

        self.erosion_kernel = np.ones((9, 9), np.uint8)  # A 3x3 kernel, you can change the size

    def to(self, device):
        self.device = device

    def masked_detections(self, img_rgb, img_depth = None, detection_class = 0, size = (256, 128)):

        results = self.detect_mot(img_rgb, detection_class=detection_class)  

        if not (len(results[0].boxes) > 0):
            return []

        subimages = []
        person_kpts = []
        poses = []
        bboxes = []
        for result in results: # Need to iterate because if batch is longer than one it should iterate more than once
            boxes = result.boxes  # Boxes object
            masks = result.masks
            keypoints = result.keypoints

            for i ,(box, mask, kpts) in enumerate(zip(boxes, masks, keypoints.xy)):
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Masking the original RGB image
                contour = mask.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                b_mask = np.zeros(img_rgb.shape[:2], np.uint8)
                b_mask = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                # Define the kernel for erosion (you can adjust the size for stronger/weaker erosion)
                b_mask = cv2.erode(b_mask, self.erosion_kernel, iterations=2)  # Apply erosion to the binary mask
                mask3ch = cv2.bitwise_and(img_rgb, img_rgb, mask=b_mask)
                # Crop the Image
                subimage = cv2.resize(mask3ch[y1:y2, x1:x2], size)

                # Getting Eyes+Torso+knees Keypoints for pose estimation
                torso_kpts = kpts.cpu().numpy()[[1, 2, 5, 6, 11, 12, 13, 14], :]
                torso_kpts = torso_kpts[~np.all(torso_kpts == 0, axis=1)].astype(np.int32) - 1 # Rest one to avoid incorrect pixel corrdinates
                torso_kpts = np.array([kp for kp in torso_kpts if b_mask[kp[1], kp[0]] > 0])

                # Getting the Person Central Pose (Based on Torso Keypoints)
                pose = self.get_person_pose(torso_kpts, img_depth)

                # Store all the bounding box detections and subimages in a tensor
                subimages.append(torch.tensor(subimage, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0))
                bboxes.append((x1, y1, x2, y2))
                person_kpts.append(torso_kpts)
                poses.append(pose)

        poses = np.array(poses)
        bboxes = np.array(bboxes)
        batched_tensor = torch.cat(subimages).to(device=self.device)
        return [batched_tensor, bboxes, person_kpts, poses]

    def detect(self, img_rgb, img_depth, detection_thr=0.7, detection_class=0):

        if self.template is None:
            return None

        total_execution_time = 0  # To accumulate total time

        # Measure time for `masked_detections`
        start_time = time.time()
        detections = self.masked_detections(img_rgb, img_depth, detection_class=detection_class)
        end_time = time.time()
        masked_detections_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"masked_detections execution time: {masked_detections_time:.2f} ms")
        total_execution_time += masked_detections_time

        if not (len(detections) > 0):
            return None

        detections_imgs, bboxes, kpts, poses = detections

        # Measure time for `feature_extraction`
        start_time = time.time()
        detections_features = self.feature_extraction(detections_imgs=detections_imgs)
        end_time = time.time()
        feature_extraction_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"feature_extraction execution time: {feature_extraction_time:.2f} ms")
        total_execution_time += feature_extraction_time

        # Measure time for `similarity_check`
        start_time = time.time()
        similarity_check = self.similarity_check(self.template_features, detections_features, 0.8, 500, 0.4)
        end_time = time.time()
        similarity_check_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"similarity_check execution time: {similarity_check_time:.2f} ms")
        total_execution_time += similarity_check_time

        if similarity_check is None:
            return None
        else:
            valid_idxs, similarity = similarity_check

        # Return results
        return (poses[valid_idxs], bboxes[valid_idxs], kpts, similarity, valid_idxs[0])
    
    def similarity_check(self, template_features, detections_features, similarity_thr, eucledian_thr, ratio_thr):

        # In case of Just one person being detected
        if detections_features.shape[0] == 1:
            similarity = self.feature_distance(template_features, detections_features, mode='eucledian')
            
            print("Single Target Detected Waht to do:")
            print( self.feature_distance(template_features, detections_features, mode='eucledian'))

            similarity = similarity.item()

            if similarity > eucledian_thr:
                return None
            else:
                valid_indices = [0]

        # In case of multiple detections
        elif detections_features.shape[0] > 1:

            # Just cionsider features that have a cosine similarity score greater than 0.8
            similarities = self.feature_distance(template_features, detections_features, mode='cosine')

            filtered_detection_features = detections_features[similarities > similarity_thr]
        
            if filtered_detection_features.shape[0] < 1:
                return None

            distances = self.feature_distance(template_features, filtered_detection_features, mode='eucledian')

            # Set Lowe's ratio threshold
            k = min(5, distances.shape[1])  # Number of top smallest distances to consider

            # Get the top k smallest distances and their indices
            top_k_values, top_k_indices = torch.topk(distances, k, largest=False)
            ratios = 1 - (top_k_values[0, 0] / top_k_values )

            # Filter indices where the ratio is below the threshold, excluding the smallest itself
            valid_indices = top_k_indices[ratios < ratio_thr].tolist()
            filtered_distances = top_k_values[ratios < ratio_thr]
            
            if filtered_distances.shape[0] == 1:
                similarity = filtered_distances
            elif filtered_distances.shape[0] > 1:
                similarity = filtered_distances[1]

            print("filtered_distances", filtered_distances)

        return (valid_indices, similarity)
    
    def detect_mot(self, img, detection_class):
        # Run multiple object detection with a given desired class
        return self.yolo(img, classes = detection_class)
    
    def template_update(self, template):

        detections = self.masked_detections(template)

        if len(detections):
            self.template = detections[0]

        self.template_features = self.extract_features(self.template)
    
    def feature_extraction(self, detections_imgs):
        # Extract features for similarity check
        return self.extract_features(detections_imgs)
    
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

    def extract_features(self, image):
        img = self.img_transform(image).half()
        # Extract a 512 feature vector from the image using a pretrained RESNET18 model
        features = self.features(img)
        
        return features

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