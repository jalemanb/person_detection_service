from ultralytics import YOLO
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import torch, cv2
import os
from person_detection_temi.submodules.utils.preprocessing import preprocess_rgb, preprocess_depth
import numpy as np
import open3d as o3d
import torchvision.transforms as transforms

import onnx
import onnxruntime as ort

class SOD:

    def __init__(self, yolo_model_path, feature_extracture_model_path) -> None:

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.onnx_provider = 'CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'


        # Detection Model
        self.yolo = YOLO(yolo_model_path)  # load a pretrained model (recommended for training)
        
        # Feature Extraction Model
        opt = ort.SessionOptions()
        opt.enable_profiling = False
        onnx_model = onnx.load(feature_extracture_model_path)
        self.resnet = ort.InferenceSession(
        onnx_model.SerializeToString(),
        providers=[self.onnx_provider], 
        sess_options=opt)

        self.template = None
        self.template_features = None
        
        self.resnet_transform = transforms.Compose([
            transforms.Resize((256, 128)),
        ])

        fx, fy, cx, cy = 620.8472290039062, 621.053466796875, 325.1631164550781, 237.45947265625  # RGB Camera Intrinsics
        # fx, fy, cx, cy = 384.20147705078125, 384.20147705078125, 324.1680908203125, 245.62278747558594  # Depth Camera Intrinsics

        width, height = 640, 480

        self.intrinsics = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsics.set_intrinsics(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

        self.person_pcd = o3d.geometry.PointCloud()
        
    def to(self, device):
        self.device = device
        # self.resnet.to(device)

    def detect(self, img_rgb, img_depth, detection_thr = 0.7, detection_class = 0):

        if self.template is None:
            return False, False, False, False

        # Run Object Detection
        detections = self.detect_mot(img_rgb, detection_class=detection_class)  

        # If not detections then return None
        if not (len(detections[0].boxes) > 0):
            return False, False, False, False
        
        # Run Object Detection

        # Obtain Detection Subimages
        detections_imgs = self.extract_subimages(img_rgb, detections)

        # Move search region img and template into selected device
        detections_imgs = detections_imgs.to(self.device)

        # Extract features from both the template image and each resulting detection (subimages taking considering the bounding boxes)
        detections_features = self.feature_extraction(detections_imgs=detections_imgs)

        # Apply similarity check between the template image features and the features from all the other candidate images to find the closest one
        most_similar_idx, max_similarity = self.similarity_check(self.template_features, detections_features, detection_thr) 

        # If the similarity score doesnt pass a threshold value, then return no detection
        if most_similar_idx is None:
            return False, False, False, False

        # Get the bounding box and mask corresponding to the candidate most similar to the tempate img
        bbox, mask = self.get_template_results(detections, most_similar_idx, (img_rgb.shape[1], img_rgb.shape[0]))
        
        # Given the desired person was detected, get RGB+D patches (subimages) to find the person orientation
        # Also get the masked depth image for later 3D pose estimation
        masked_depth_img, masked_rgb_img = self.get_target_rgb_and_depth(img_rgb, img_depth, bbox, mask)

        # Compute the person pointloud fromt the given depth image and intrinsic camera parameters
        self.compute_point_cloud(masked_depth_img)

        # Get the person pose from the center of fitting a 3D bounding box around the person points
        person_pose = self.get_person_pose(self.person_pcd)

        # Return Corresponding bounding box for visualization
        return person_pose, bbox, self.person_pcd, max_similarity
    
    def detect_mot(self, img, detection_class):
        # Run multiple object detection with a given desired class
        return self.yolo(img, classes = detection_class)
    
    def template_update(self, template):
        self.template = torch.from_numpy(template).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
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

    def similarity_check(self, template_features, detections_features, detection_thr):
        # Compute Similarity Check
        similarities = F.cosine_similarity(template_features, detections_features, dim=1)
        # similarities = torch.norm(template_features - detections_features, 2)
        print(similarities)

        # FInd most similar image
        most_similar_idx = torch.argmax(similarities).item()
        max_similarity = torch.max(similarities).item()

        # Return most similar index
        return most_similar_idx, max_similarity
        
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

        # Extract a 512 feature vector from the image using a pretrained RESNET18 model
        features = self.resnet.run(None, {'input': self.resnet_transform(image).cpu().numpy()})
        t_feat = torch.Tensor(features[0])
        print(t_feat.shape)
        return t_feat

    def get_person_pose(self, pcd): # 3d person pose estimation wrt the camera reference frame
        # Wrap the person around a 3D bounding box
        box = pcd.get_oriented_bounding_box()

        # Get the entroid of the bounding box wrappig the person
        return box.get_center()

    # Function to compute point cloud from depth image
    def compute_point_cloud(self, depth_image):
        # Converting depth image into o3d image format for pointcloud omputation
        depth_o3d = o3d.geometry.Image(depth_image)

        # Create a point cloud from the depth image
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, self.intrinsics)

        # Remove person pcl outliers 
        self.person_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.1)