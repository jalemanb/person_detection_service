import torch
import numpy as np
import torch.nn.functional as F
import os

class Bucket:
    def __init__(self, max_identities = 5, samples_per_identity = 20, thr = 0.8) -> None:

        # Incremental KNN Utils #########################
        self.max_samples = max_identities * samples_per_identity
        self.max_identities = max_identities
        self.samples_per_id = samples_per_identity
        self.distance_thr = thr

        self.part_num = 6
        self.feats_num = 512
        self.mean_prototypes_feats = torch.zeros((self.max_identities, self.part_num, self.feats_num)).cuda()
        self.mean_prototypes_vis = torch.zeros((self.max_identities, self.part_num)).cuda()
        self.samples_per_id_num = [0 for i in range(self.max_identities)]
        self.active_prototypes_num = 0
        self.empty = True

        self.gallery_feats = torch.zeros((self.max_identities, self.samples_per_id, self.part_num, self.feats_num)).cuda()
        self.gallery_vis = torch.zeros((self.max_identities, self.samples_per_id, self.part_num)).to(torch.bool).cuda()

        self.distractor_feats = torch.zeros((self.max_samples, self.part_num, self.feats_num)).cuda()
        self.distractor_vis = torch.zeros((self.max_samples, self.part_num)).to(torch.bool).cuda()
        self.distractors_num = 0

        self.current_feats = None
        self.current_vis = None
        self.labels_pos = None

        ##########################################################################
        # For Later Dbugging #####################################################
        self.templates_prototypes = np.zeros((self.max_identities, 3, 384, 128))
        self.prototype_means_feats_debug = []
        self.prototype_means_vis_debug = []
        self.feats_debug = []
        self.vis_debug = []
        self.existing_frames = []
        self.img_patches = []
        # For Later Dbugging #####################################################
        ##########################################################################

        #################################################


    def gaussian_mixture_pdf(N, sigma1=2.0, sigma2=2.0):
        x = np.arange(N)
        mu1 = 0.25 * (N - 1)
        mu2 = 0.75 * (N - 1)

        g1 = np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
        g2 = np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
        pdf = g1 + g2
        pdf /= pdf.sum()  # Normalize to make it a valid probability distribution

        return pdf

    def store_distractor_feats(self, feats, vis):
            """
            Stores feature vectors, visibility masks, and labels into a fixed-size buffer.
            Uses `torch.roll` to implement a circular buffer. If the batch size is larger than `max_samples`,
            it discards excess samples.

            Args:
                feats (torch.Tensor): Feature tensor of shape [batch, 6, 512]
                vis (torch.Tensor): Visibility tensor of shape [batch, 6] (bool)
                label (torch.Tensor): Labels tensor of shape [batch] (bool)
            """
            new_feats_num = feats.shape[0]

            # If batch is larger than max_samples, keep only the most recent samples
            if new_feats_num > self.max_samples:
                feats = feats[-self.max_samples:]  # Keep only last `max_samples` samples
                vis = vis[-self.max_samples:]
                new_feats_num = self.max_samples  # Adjust count

            if self.distractors_num < self.max_samples:
                # Append new samples normally
                available_space = self.max_samples - self.distractors_num
                num_to_store = min(new_feats_num, available_space)

                self.distractor_feats[self.distractors_num:self.distractors_num + num_to_store] = feats[:num_to_store]
                self.distractor_vis[self.distractors_num:self.distractors_num + num_to_store] = vis[:num_to_store]

                self.distractors_num += num_to_store

            else:
                # Use torch.roll to shift old data and insert new samples at the beginning
                self.distractor_feats = torch.roll(self.distractor_feats, shifts=-new_feats_num, dims=0)
                self.distractor_vis = torch.roll(self.distractor_vis, shifts=-new_feats_num, dims=0)

                # Overwrite the first `new_feats_num` positions with new data
                self.distractor_feats[-new_feats_num:] = feats
                self.distractor_vis[-new_feats_num:] = vis


    def store_feats(self, feats, vis, counter = 0, img_patch = None, debug = False):
        """
        Stores feature vectors and visibility masks into a fixed-size buffer.
        Uses `torch.roll` to implement a circular buffer. If the batch size is larger than `max_samples`,
        it discards excess samples.

        Args:
            Batch is 1 (only one feature set of features from the target fromframe to frame)
            feats (torch.Tensor): Feature tensor of shape [batch, 6, 512]
            vis (torch.Tensor): Visibility tensor of shape [batch, 6] (bool)
        """
        new_feats_num = feats.shape[0]

        # Bucket Initialization
        if self.empty:
            self.mean_prototypes_feats[0] = feats
            self.mean_prototypes_vis[0] = vis
            self.gallery_feats[0, 0] = feats
            self.gallery_vis[0, 0] = vis
            self.samples_per_id_num[0] = 1 
            self.active_prototypes_num = 1
            self.templates_prototypes[0] = img_patch
            self.empty = False
        else:


            memership = self.compute_mean_distance(feats, 
                                                self.mean_prototypes_feats[:self.active_prototypes_num], 
                                                vis, 
                                                self.mean_prototypes_vis[:self.active_prototypes_num], 
                                                distance_type="euclidean")
            
            # print("Membership", memership)
                        
            min_dist, min_idx = torch.min(memership, dim=1)

            min_dist, prototype_id = min_dist.item(), min_idx.item()

            if min_dist < self.distance_thr:
                # If the minimal distance to the mean prototypes is lower than a threshold then add a sample to the prototype wth the smallest distance
                # If the samples per identity are full then it is necessary to use round robin to the closest one or randmly remove one and add the new one
                
                if self.samples_per_id_num[prototype_id] < self.samples_per_id:
                    self.gallery_feats[prototype_id, self.samples_per_id_num[prototype_id]:self.samples_per_id_num[prototype_id] + 1] = feats
                    self.gallery_vis[prototype_id, self.samples_per_id_num[prototype_id]:self.samples_per_id_num[prototype_id] + 1] = vis
                    self.samples_per_id_num[prototype_id] += 1 

                else:
                    # Here do something when the memory for storing features belonging to a given prototype is full
                    self.gallery_feats[prototype_id] = torch.roll(self.gallery_feats[prototype_id], shifts=-1, dims=0)
                    self.gallery_vis[prototype_id] = torch.roll(self.gallery_vis[prototype_id], shifts=-1, dims=0)
                    self.gallery_feats[prototype_id, -1:] = feats
                    self.gallery_vis[prototype_id, -1:] = vis

                self.templates_prototypes[prototype_id] = img_patch

            else:
                # If the smallest distance to all the existing prototypes is larger than a given threshold then add a new mean prototype
                # If full then reset the mean rpotype withtthe least features or randomly choose one to be deleted 
                if self.active_prototypes_num < self.max_identities:

                    id_with_least = np.argmin(self.samples_per_id_num[:self.active_prototypes_num])

                    # Change so that it selects the one with the smallest number 
                    if self.samples_per_id_num[id_with_least] < self.samples_per_id:
                        # if a new prototype is gonna take place but the previous prototype is not full, 
                        # then it mightjust be noise or a very short time appearance occurence, in such case substitue that occurrence with the new prorotype
                        self.samples_per_id_num[id_with_least] = 1 
                        self.mean_prototypes_feats[id_with_least] = feats
                        self.mean_prototypes_vis[id_with_least] = vis
                        self.gallery_feats[id_with_least, 0] = feats
                        self.gallery_vis[id_with_least, 0] = vis
                        if debug:
                            self.templates_prototypes[id_with_least] = img_patch
                    else: 
                        # Adding a new mean prototype
                        self.samples_per_id_num[self.active_prototypes_num] += 1 
                        self.mean_prototypes_feats[self.active_prototypes_num] = feats
                        self.mean_prototypes_vis[self.active_prototypes_num] = vis
                        self.gallery_feats[self.active_prototypes_num, 0] = feats
                        self.gallery_vis[self.active_prototypes_num, 0] = vis
                        if debug:
                            self.templates_prototypes[self.active_prototypes_num] = img_patch
                        self.active_prototypes_num += 1
                else:
                    id_with_least = np.argmin(self.samples_per_id_num[:self.active_prototypes_num])

                    # Change so that it selects the one with the smallest number 
                    if self.samples_per_id_num[id_with_least] < self.samples_per_id:
                        # if a new prototype is gonna take place but the previous prototype is not full, 
                        # then it mightjust be noise or a very short time appearance occurence, in such case substitue that occurrence with the new prorotype
                        self.samples_per_id_num[id_with_least] = 1 
                        self.mean_prototypes_feats[id_with_least] = feats
                        self.mean_prototypes_vis[id_with_least] = vis
                        self.gallery_feats[id_with_least, 0] = feats
                        self.gallery_vis[id_with_least, 0] = vis
                        if debug:
                            self.templates_prototypes[id_with_least] = img_patch
                    else:
                        # Replace the id prototypes, its features, visibilities, etc. based on how many nn have 
                        # Highly isolated prototypes are less likely to be replaced
                        distances_matrix = self.compute_mean_distance(self.mean_prototypes_feats, self.mean_prototypes_feats, self.mean_prototypes_vis, self.mean_prototypes_vis, "euclidean")
                        mask = (distances_matrix < self.distance_thr)
                        mask.fill_diagonal_(False)  # Ignore self-distances
                        ids_nn_count = mask.sum(dim=1).to(torch.float32)
                        replacement_prob = F.softmax(ids_nn_count, dim=0).cpu().numpy()
                        replacement_idx = np.random.choice(len(replacement_prob), p=replacement_prob)

                        self.samples_per_id_num[replacement_idx] = 1 
                        self.mean_prototypes_feats[replacement_idx] = feats
                        self.mean_prototypes_vis[replacement_idx] = vis
                        self.gallery_feats[replacement_idx, 0] = feats
                        self.gallery_vis[replacement_idx, 0] = vis

            # Update Mean prototype representation
            # Update visibility representation for each mean prototype
            self.mean_prototypes_feats[:self.active_prototypes_num], self.mean_prototypes_vis[:self.active_prototypes_num] = self.compute_avg_features(self.gallery_feats, self.gallery_vis, self.active_prototypes_num, self.samples_per_id_num)

        if debug:
            prototype_means_feats_np = self.mean_prototypes_feats[:self.active_prototypes_num].cpu().numpy()
            prototype_means_vis_np = self.mean_prototypes_vis[:self.active_prototypes_num].cpu().numpy()
            prototype_templates_vis_np = self.templates_prototypes[:self.active_prototypes_num].copy()
            feats_np = self.current_feats.cpu().numpy()
            vis_np = self.current_vis.cpu().numpy()
            self.prototype_means_feats_debug.append(prototype_means_feats_np)
            self.prototype_means_vis_debug.append(prototype_means_vis_np)
            self.img_patches.append(prototype_templates_vis_np)
            self.feats_debug.append(feats_np)
            self.vis_debug.append(vis_np)
            self.existing_frames.append(counter)


    def get(self):
        #  This code is to retrieve all the valid existing feature sets and visibility scores consdiering the availavbility of these
        valid_samples_feats = [self.gallery_feats[i, :self.samples_per_id_num[i]] for i in range(self.active_prototypes_num)]
        valid_samples_vis = [self.gallery_vis[i, :self.samples_per_id_num[i]] for i in range(self.active_prototypes_num)]
        pos_feats = torch.cat(valid_samples_feats, dim=0)
        pos_vis = torch.cat(valid_samples_vis, dim=0)
        pos_labels = torch.ones(pos_vis.shape[0]).cuda()

        neg_feats = self.distractor_feats[:self.distractors_num]
        neg_vis = self.distractor_vis[:self.distractors_num]
        neg_labels = torch.zeros(self.distractors_num).cuda()

        total_feats = torch.cat((pos_feats, neg_feats), dim=0)
        total_vis = torch.cat((pos_vis, neg_vis), dim=0)
        total_labels = torch.cat((pos_labels, neg_labels))

        return total_feats, total_vis, total_labels

    
    def get_samples_num(self):
        return np.sum(self.samples_per_id_num).item()
    
    def get_max_samples(self):
        return self.max_samples
    

    def save(self, file_path):
        self.save_frames_data(file_path)


    def compute_mean_distance(self, tensor1, tensor2, visibility1, visibility2, distance_type="cosine"):
        """
        Generalized pairwise distance computation between tensor1 and tensor2, handling different batch sizes.

        Args:
            tensor1 (Tensor): Shape [B1, P, F]
            tensor2 (Tensor): Shape [B2, P, F]
            visibility1 (Tensor): Shape [B1, P]
            visibility2 (Tensor): Shape [B2, P]
            distance_type (str): 'cosine' or 'euclidean'

        Returns:
            Tensor: Shape [B1, B2] mean distances between all tensor1[i] and tensor2[j]
        """
        B1, P, F = tensor1.shape
        B2 = tensor2.shape[0]

        # Expand tensors for broadcasting
        t1 = tensor1.unsqueeze(1).expand(B1, B2, P, F)       # [B1, B2, P, F]
        t2 = tensor2.unsqueeze(0).expand(B1, B2, P, F)       # [B1, B2, P, F]
        
        vis1 = visibility1.unsqueeze(1).expand(B1, B2, P).to(torch.bool)    # [B1, B2, P]
        vis2 = visibility2.unsqueeze(0).expand(B1, B2, P).to(torch.bool)    # [B1, B2, P]

        valid_mask = vis1 & vis2                             # [B1, B2, P]

        if distance_type == "cosine":
            t1_norm = F.normalize(t1, p=2, dim=-1)
            t2_norm = F.normalize(t2, p=2, dim=-1)
            cos_sim = torch.sum(t1_norm * t2_norm, dim=-1)   # [B1, B2, P]
            distances = 1 - cos_sim                          # [B1, B2, P]

        elif distance_type == "euclidean":
            distances = torch.norm(t1 - t2, p=2, dim=-1)      # [B1, B2, P]

        else:
            raise ValueError("Invalid distance_type. Choose 'cosine' or 'euclidean'.")

        distances = distances * valid_mask                   # zero out invalid parts

        valid_counts = valid_mask.sum(dim=-1).float()        # [B1, B2]
        valid_counts = torch.where(valid_counts > 0, valid_counts, torch.tensor(1.0, device=valid_counts.device))

        mean_distances = distances.sum(dim=-1) / valid_counts  # [B1, B2]
        
        return mean_distances  # shape: [B1, B2]

    def compute_avg_features(self, features, visibilities, current_valid_ids, num_samples_per_id):
        """
        Computes the average feature vector per ID considering only visible features.
        
        Args:
            features (torch.Tensor): Shape [max_ids, max_samples, 6, 512]
            visibilities (torch.Tensor): Shape [max_ids, max_samples, 6]
            current_valid_ids (int): Number of valid identities (e.g., 4)
            num_samples_per_id (list): List of valid sample counts per ID (e.g., [3, 5, 2, 0, 0])

        Returns:
            avg_features (torch.Tensor): Shape [current_valid_ids, 6, 512]
            fused_visibilities (torch.Tensor): Shape [current_valid_ids, 6] (binary OR over samples)
        """
        # Select valid identities
        valid_features = features[:current_valid_ids]  # Shape: [current_valid_ids, max_samples, 6, 512]
        valid_visibilities = visibilities[:current_valid_ids]  # Shape: [current_valid_ids, max_samples, 6]

        # Create mask based on num_samples_per_id
        mask = torch.arange(valid_features.shape[1], device=features.device).unsqueeze(0) < torch.tensor(num_samples_per_id[:current_valid_ids], device=features.device).unsqueeze(1)

        # Expand mask to match feature shape
        mask = mask.unsqueeze(-1).unsqueeze(-1)  # Shape: [current_valid_ids, max_samples, 1, 1]

        # Apply mask to features and visibilities
        masked_features = valid_features * mask  # Shape: [current_valid_ids, max_samples, 6, 512]
        masked_visibilities = valid_visibilities * mask.squeeze(-1)  # Shape: [current_valid_ids, max_samples, 6]

        # Compute per-ID fused visibility (binary OR along the sample dimension)
        fused_visibilities = masked_visibilities.any(dim=1)  # Shape: [current_valid_ids, 6]

        # Expand visibility mask to match feature shape
        visibility_mask = masked_visibilities.unsqueeze(-1)  # Shape: [current_valid_ids, max_samples, 6, 1]

        # Zero out non-visible features
        masked_features = masked_features * visibility_mask  # Shape: [current_valid_ids, max_samples, 6, 512]

        # Compute sum of visible features and count valid ones
        sum_features = masked_features.sum(dim=1)  # Sum along sample dimension -> [current_valid_ids, 6, 512]
        count_visible = visibility_mask.sum(dim=1)  # Count visible features per ID -> [current_valid_ids, 6, 1]

        # Avoid division by zero (replace 0s with 1s in count_visible)
        count_visible = count_visible.clamp(min=1)

        # Compute average feature per ID
        avg_features = sum_features / count_visible  # Shape: [current_valid_ids, 6, 512]

        return avg_features, fused_visibilities

    def save_frames_data(self, file_path):

        # Save updated data
        np.savez_compressed(file_path, 
                            frames=np.array(self.existing_frames),
                            prototype_means_feats=np.array(self.prototype_means_feats_debug, dtype=object),
                            prototype_means_vis=np.array(self.prototype_means_vis_debug, dtype=object),
                            feats=np.array(self.feats_debug, dtype=object),
                            vis=np.array(self.vis_debug, dtype=object),
                            templates = np.array(self.img_patches, dtype=object))

        print(f"✅ Saved frame to {file_path}")