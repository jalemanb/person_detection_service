from ultralytics import YOLO
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import torch, cv2
import os, time
from person_detection_temi.submodules.utils.preprocessing import preprocess_rgb, preprocess_depth
import numpy as np
import open3d as o3d
import torchvision.transforms as transforms
import torchreid

class SOD:

    def __init__(self, yolo_model_path, feature_extracture_model_path) -> None:

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.onnx_provider = 'CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider'

        # Detection Model
        self.yolo = YOLO(yolo_model_path, task="segment_pose")  # load a pretrained model (recommended for training)

        self.features = torchreid.models.build_model(
            name='osnet_ain_x1_0',
            num_classes=1000,  # You can set num_classes to your number of classes, if using for training.
            pretrained=True  # Set to False if you want an untrained model
        )

        self.features.eval()
        self.features.eval().half()
        self.features.to(self.device)

        self.template = None
        self.template_features = None
        
        self.img_transform = transforms.Compose([
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0., 0., 0.],  std=[1, 1, 1])
        ])

        self.fx, self.fy, self.cx, self.cy = 620.8472290039062, 621.053466796875, 325.1631164550781, 237.45947265625  # RGB Camera Intrinsics
        # fx, fy, cx, cy = 384.20147705078125, 384.20147705078125, 324.1680908203125, 245.62278747558594  # Depth Camera Intrinsics

        width, height = 640, 480

        self.intrinsics = o3d.camera.PinholeCameraIntrinsic()


        self.erosion_kernel = np.ones((9, 9), np.uint8)  # A 3x3 kernel, you can change the size

        
    def to(self, device):
        self.device = device
        # self.resnet.to(device)

    def masked_detections(self, img_rgb, detection_class = 0, size = (256, 256)):

        results = self.detect_mot(img_rgb, detection_class=detection_class)  

        if not (len(results[0].boxes) > 0):
            return []

        subimages = []
        person_kpts = []
        bboxes = []
        for result in results: # Need to iterate because if batch is longer than one it should iterate more than once
            boxes = result.boxes  # Boxes object
            masks = result.masks
            keypoints = result.keypoints
            for box, mask, kpts in zip(boxes, masks, keypoints.xy):
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

                # Store all the bounding box detections and subimages in a tensor
                subimages.append(torch.tensor(subimage, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0))
                bboxes.append((x1, y1, x2, y2))
                person_kpts.append(torso_kpts)

        batched_tensor = torch.cat(subimages).to(device=self.device)
        return [batched_tensor, bboxes, person_kpts]


    def detect(self, img_rgb, img_depth, detection_thr=0.7, detection_class=0):
        if self.template is None:
            return False, False, False, False

        total_execution_time = 0  # To accumulate total time

        # Measure time for `masked_detections`
        start_time = time.time()
        detections = self.masked_detections(img_rgb, detection_class=detection_class)
        end_time = time.time()
        masked_detections_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"masked_detections execution time: {masked_detections_time:.2f} ms")
        total_execution_time += masked_detections_time

        if not (len(detections) > 0):
            return False, False, False, False

        detections_imgs, bboxes, kpts = detections

        # Measure time for `feature_extraction`
        start_time = time.time()
        detections_features = self.feature_extraction(detections_imgs=detections_imgs)
        end_time = time.time()
        feature_extraction_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"feature_extraction execution time: {feature_extraction_time:.2f} ms")
        total_execution_time += feature_extraction_time

        # Measure time for `similarity_check`
        start_time = time.time()
        most_similar_idx, max_similarity = self.similarity_check(self.template_features, detections_features, detection_thr)
        end_time = time.time()
        similarity_check_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"similarity_check execution time: {similarity_check_time:.2f} ms")
        total_execution_time += similarity_check_time

        if most_similar_idx is None:
            return False, False, False, False

        bbox = bboxes[most_similar_idx]


        # Measure time for `get_person_pose`

        start_time = time.time()
        person_pose = self.get_person_pose(kpts[most_similar_idx], img_depth)
        end_time = time.time()
        get_person_pose_time = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"get_person_pose execution time: {get_person_pose_time:.2f} ms")
        total_execution_time += get_person_pose_time

        # Print total execution time
        print(f"Total execution time: {total_execution_time:.2f} ms")

        # Return results
        return person_pose, bbox, kpts[most_similar_idx], max_similarity
    
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

    def similarity_check(self, template_features, detections_features, detection_thr):

        # Compute Similarity Check
        similarities = F.cosine_similarity(template_features, detections_features, dim=1)

        # FInd most similar image
        max_similarity, most_similar_idx = torch.max(similarities, dim=0)

        # Return most similar index
        return most_similar_idx.item(), max_similarity.item()
        
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

        if kpts.shape[0] < 2: # Not Enough Detected Keypoints proceed to compute the human pose
            return False
        
        u = kpts[:, 0]

        v = kpts[:, 1]
        
        z = depth_img[v, u]/1000. # get depth for specific keypoints and convert into m

        x = z*(u - self.cx)/self.fx

        y = -z*(u - self.cy)/self.fy

        return (x.mean(), y.mean(), z.mean())
        