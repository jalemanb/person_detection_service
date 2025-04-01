import torch
import numpy as np
import torch.nn.functional as F

def iknn(feats, feats_vis, memory_f, memory_v, memory_l, num_samples = 0, k =10,  metric = "euclidean", threshold=0.8):
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

    A, visibility_A, labels_A = memory_f, memory_v, memory_l

    B = feats
    visibility_B = feats_vis

    # k = int(np.minimum(self.memory_bucket.get_samples_num(), np.sqrt(self.memory_bucket.get_max_samples())))
    k = int(np.minimum(num_samples, k))

    N, parts, dim = A.shape
    batch = B.shape[0]

    # Expand A and B to match dimensions for pairwise comparison
    A_expanded = A.unsqueeze(1).expand(N, batch, parts, dim)  # [N, batch, 6, 512]
    B_expanded = B.unsqueeze(0).expand(N, batch, parts, dim)  # [N, batch, 6, 512]

    # Compute similarity/distance based on the selected metric
    if metric == "euclidean":
        distance = torch.norm(
            A_expanded - B_expanded, 
            p=2, 
            dim=-1
    )  # Euclidean distance
    elif metric == "cosine":
        distance = 1 - F.cosine_similarity(
            A_expanded, 
            B_expanded, 
            dim=-1
    )  # Cosine distance
    else:
        raise ValueError("Unsupported metric. Choose 'euclidean' or 'cosine'.")

    # Expand visibility masks for proper masking
    vis_A_expanded = visibility_A.unsqueeze(1).expand(N, batch, parts)  # [N, batch, 6]
    vis_B_expanded = visibility_B.unsqueeze(0).expand(N, batch, parts)  # [N, batch, 6]

    # Apply visibility mask: Only compare if both A and B parts are visible
    valid_mask = vis_A_expanded & vis_B_expanded  # Boolean mask
    distance[~valid_mask] = float("inf")  # Ignore invalid comparisons

    # Retrieve the k smallest distances along dim=0 (N dimension)
    top_k_values, top_k_indices = torch.topk(
        distance, 
        k, 
        dim=0, 
        largest=False
    )  # [k, batch, 6]

    # print("top_k_values")
    # print(top_k_values)

    # Retrieve the corresponding labels for the k nearest neighbors This is the knn-prediction
    top_k_labels = labels_A[top_k_indices]  # Shape [k, batch, 6], labels for nearest N indices

    # Create binary mask based on threshold
    binary_mask = top_k_values <= threshold  # [k, batch, 6]

    # Apply threshold influence: Set labels to zero where distances exceed the threshold
    valid_labels = top_k_labels * binary_mask  # Zero out labels where threshold is exceeded
    # print("valid_labels")
    # print(valid_labels)

    # Perform classification by majority vote (sum up valid labels and classify based on majority vote)
    classification = (valid_labels.sum(dim=0) > (k // 2)).to(torch.bool)  # Shape [batch, 6]

    return classification.T
