import torch
from torch.nn import functional as F
import onnxruntime as ort
import numpy as np
from person_detection_temi.submodules.keypoint_promptable_reidentification.torchreid.scripts.builder import build_config
from person_detection_temi.submodules.keypoint_promptable_reidentification.torchreid.metrics.distance import compute_distance_matrix_using_bp_features
from person_detection_temi.submodules.keypoint_promptable_reidentification.torchreid.data.datasets.keypoints_to_masks import KeypointsToMasks
from person_detection_temi.submodules.keypoint_promptable_reidentification.torchreid.data.transforms import build_transforms
from person_detection_temi.submodules.keypoint_promptable_reidentification.torchreid.data import ImageDataset
from person_detection_temi.submodules.keypoint_promptable_reidentification.torchreid.utils.constants import *

class KPR_onnx_wrapper:
    def __init__(self, onnx_model_path):
        self.onnx_model_path = onnx_model_path
        # self.providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        self.providers = ["CUDAExecutionProvider"]
        self.providers_options = [{}]

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.model =  ort.InferenceSession(onnx_model_path, 
                                           sess_options=session_options, 
                                           providers=self.providers,
                                           provider_options=self.providers_options)

        self.input_names = [inp.name for inp in self.model.get_inputs()]
        print("ONNX Model Inputs:", self.input_names)

        self.output_names = [inp.name for inp in self.model.get_outputs()]
        print("ONNX Model Outputs:", self.output_names)

    def __call__(self, images, prompt_masks):

        images_input = images.astype(np.float32)
        prompt_input = prompt_masks.astype(np.float32)

        outputs = self.model.run(None, {self.input_names[0]: images_input, self.input_names[1]: prompt_input})

        for i in range(len(outputs)):
            outputs[i] = torch.Tensor(outputs[i]).cuda()


        return ({'globl': outputs[0], 'backg': outputs[1], 'foreg': outputs[2], 'conct': outputs[3], 'parts': outputs[4], 'bn_globl': outputs[5], 'bn_backg': outputs[6], 'bn_foreg': outputs[7], 'bn_conct': outputs[8], 'bn_parts': outputs[9]},
                {'globl': outputs[10], 'backg': outputs[11], 'foreg': outputs[12], 'conct': outputs[13], 'parts': outputs[14]},
                {'globl': outputs[15], 'backg': outputs[16], 'foreg': outputs[17], 'conct': outputs[18], 'parts': outputs[19]},
                outputs[20],
                outputs[21],
                {'globl': outputs[22], 'backg': outputs[23], 'foreg': outputs[24], 'conct': outputs[25], 'parts': outputs[26]})



class KPR(object):
    def __init__(self, cfg_path , model_path, kpt_conf = 0.8, device = 'cpu') -> None:

        self.kpt_conf = kpt_conf
        self.device = device

        self.model = KPR_onnx_wrapper(model_path)
        cfg = build_config(config_path=cfg_path, display_diff=True)
        self.cfg = cfg

        _, self.preprocess, self.target_preprocess, self.prompt_preprocess = build_transforms(
                                                                                cfg.data.height,
                                                                                cfg.data.width,
                                                                                cfg,
                                                                                transforms=None,
                                                                                norm_mean=cfg.data.norm_mean,
                                                                                norm_std=cfg.data.norm_std,
                                                                                masks_preprocess=cfg.model.kpr.masks.preprocess,
                                                                                softmax_weight=cfg.model.kpr.masks.softmax_weight,
                                                                                background_computation_strategy=cfg.model.kpr.masks.background_computation_strategy,
                                                                                mask_filtering_threshold=cfg.model.kpr.masks.mask_filtering_threshold,
                                                                            )
        
        self.keypoints_to_prompt_masks = KeypointsToMasks(mode=cfg.model.kpr.keypoints.prompt_masks,
                                                                vis_thresh=kpt_conf,
                                                                vis_continous=cfg.model.kpr.keypoints.vis_continous,
                                                                )
        
        self.keypoints_to_target_masks = KeypointsToMasks(mode=cfg.model.kpr.keypoints.target_masks,
                                                            vis_thresh=kpt_conf,
                                                            vis_continous=False,
                                                            )   


    def extract_test_embeddings(self, model_output, test_embeddings):
        embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, parts_masks = model_output
        embeddings_list = []
        visibility_scores_list = []
        embeddings_masks_list = []

        for test_emb in test_embeddings:
            embds = embeddings[test_emb]
            embeddings_list.append(embds if len(embds.shape) == 3 else embds.unsqueeze(1))
            if test_emb in bn_correspondants:
                test_emb = bn_correspondants[test_emb]
            vis_scores = visibility_scores[test_emb]
            visibility_scores_list.append(vis_scores if len(vis_scores.shape) == 2 else vis_scores.unsqueeze(1))
            pt_masks = parts_masks[test_emb]
            embeddings_masks_list.append(pt_masks if len(pt_masks.shape) == 4 else pt_masks.unsqueeze(1))

        assert len(embeddings) != 0

        embeddings = torch.cat(embeddings_list, dim=1)  # [N, P+2, D]
        visibility_scores = torch.cat(visibility_scores_list, dim=1)  # [N, P+2]
        embeddings_masks = torch.cat(embeddings_masks_list, dim=1)  # [N, P+2, Hf, Wf]

        return embeddings, visibility_scores, embeddings_masks, pixels_cls_scores
    
    def normalize(self, features):
        return F.normalize(features, p=2, dim=-1)
    
    
    def extract(self, imgs, kpts, return_heatmaps = False): 
        # Input imgs are tensors of shape [Batch, C, W, H]
        # Input kpts are tensors of shape [Batch, 17, 3]

        imgs_list = []
        prompts_list = []
        for i in range(imgs.shape[0]):
            sample = {"image":imgs[i, :, :, :].permute(1, 2, 0).cpu().numpy(), "keypoints_xyc":kpts[i, :, :].cpu().numpy(), "negative_kps":[]}
            preprocessed_sample = ImageDataset.getitem(
                            sample,
                            self.cfg,
                            self.keypoints_to_prompt_masks,
                            self.prompt_preprocess,
                            self.keypoints_to_target_masks,
                            self.target_preprocess,
                            self.preprocess,
                            load_masks=True,
                        )
            imgs_list.append(preprocessed_sample["image"])
            prompts_list.append(preprocessed_sample["prompt_masks"])
        
        # Preprocessed images and Keypoint Prompts
        ready_imgs = np.stack(imgs_list, axis = 0)
        ready_prompts = np.stack(prompts_list, axis = 0)

        # print("ready_imgs", ready_imgs.shape)
        # print("ready_prompts", ready_prompts.shape)


        output = self.model(images = ready_imgs, prompt_masks = ready_prompts)
        
        features = self.extract_test_embeddings(output,  self.cfg.model.kpr.test_embeddings)

        # The first Feature is the foreground and the rest are the K parts
        # For this inference model K = 5 which consist on 
        # k_five = {
        #     'head': ['nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear'],
        #     'torso': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
        #     'arms': ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'],
        #     'legs': ['left_knee', 'right_knee'],
        #     'feet': ['left_ankle', 'right_ankle'],
        # }

        f_, v_, _, _ = features


        if self.cfg.test.normalize_feature:
            f_ = self.normalize(f_)

        if return_heatmaps:
            return f_, v_, ready_prompts

        return f_, v_ > 0.1

    def compare(self, fq, fg, vq, vg): 
        # Comparing Query Feature (Target Person) against Gallery features (Detected People)
        return compute_distance_matrix_using_bp_features(fq,
                                                         fg,
                                                         vq,
                                                         vg,
                                                         self.cfg.test.part_based.dist_combine_strat,
                                                         self.cfg.test.batch_size_pairwise_dist_matrix,
                                                         use_gpu = self.cfg.use_gpu,
                                                         metric = self.cfg.test.dist_metric,)
    