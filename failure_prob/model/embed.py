from typing import Optional
import warnings

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseModel

from failure_prob.conf import Config

def get_model(cfg: Config, input_dim: int) -> BaseModel:
    return EmbedModel(cfg, input_dim)


def compute_mahala_dist(
    x: torch.Tensor, 
    mean: torch.Tensor, 
    cov: torch.Tensor,
    chunk_size: int = 16384,
):
    '''
    Compute the Mahalanobis distance between x and the distribution with given mean and covariance,
    processing the data in chunks.
    
    Args:
        x: torch.Tensor of shape (N, D)
        mean: torch.Tensor of shape (D,)
        cov: torch.Tensor of shape (D, D)
        chunk_size: int, size of each chunk (must be less than N)
        
    Returns:
        torch.Tensor of shape (N,)
    '''
    N, D = x.shape
    
    # Compute the inverse of the covariance matrix once.
    inv_cov = torch.pinverse(cov)  # Shape: (D, D)
    
    # List to store the results for each chunk.
    m_dists = []
    
    # Process the data in chunks.
    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        # Chunk of data (chunk_size, D)
        x_chunk = x[start:end]
        
        # Compute the difference between each data point in the chunk and the mean.
        diff = x_chunk - mean  # Shape: (chunk_size, D)
        
        # Compute the squared Mahalanobis distance: diff @ inv_cov @ diff^T (for each row).
        # The computation `diff @ inv_cov` results in a (chunk_size, D) tensor.
        # Then, multiplying element-wise by `diff` and summing along dim=1 gives a (chunk_size,) tensor.
        m_dist_squared = torch.sum(diff @ inv_cov * diff, dim=1)  # Shape: (chunk_size,)
        
        # Take the square root to obtain the Mahalanobis distance.
        m_chunk = torch.sqrt(m_dist_squared)  # Shape: (chunk_size,)
        m_dists.append(m_chunk)
    
    # Concatenate all chunk results to get the final (N,) tensor.
    m_dist = torch.cat(m_dists, dim=0)
    
    # Check for any NaN values in the computed distances.
    if m_dist.isnan().any():
        raise ValueError("Mahalanobis distance has NaN. Check the covariance matrix.")
    
    return m_dist
    
    
def compute_cosine_dist(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    '''
    Compute the pairwise cosine distance between x and y.

    Args:
        x: torch.Tensor of shape (N, D)
        y: torch.Tensor of shape (M, D)

    Returns:
        torch.Tensor of shape (N, M) where each entry [i, j] is the cosine distance 
        between x[i] and y[j].
    '''
    # Normalize x and y to unit vectors.
    # Adding a small epsilon to avoid division by zero.
    eps = 1e-8
    x_norm = x / (x.norm(dim=1, keepdim=True) + eps)
    y_norm = y / (y.norm(dim=1, keepdim=True) + eps)
    
    # Compute the cosine similarity matrix: (N, D) @ (D, M) -> (N, M)
    cosine_similarity = torch.matmul(x_norm, y_norm.t())
    
    # Cosine distance is defined as 1 - cosine similarity.
    cosine_distance = 1 - cosine_similarity
    return cosine_distance
    

def compute_euclid_dist(
    x: torch.Tensor,
    y: torch.Tensor,  
) -> torch.Tensor:
    '''
    Compute the pairwise Euclidean distance between x and y.
    
    Args:
        x: torch.Tensor of shape (N, D)
        y: torch.Tensor of shape (M, D)
        
    Returns:
        torch.Tensor of shape (N, M) where each entry [i, j] is the Euclidean 
        distance between x[i] and y[j].
    '''
    # Compute squared norms of each row in x and y.
    x_sq = torch.sum(x ** 2, dim=1, keepdim=True)  # Shape: (N, 1)
    y_sq = torch.sum(y ** 2, dim=1, keepdim=True).t()  # Shape: (1, M)
    
    # Compute the squared Euclidean distance:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * (x . y)
    dist_sq = x_sq + y_sq - 2 * torch.matmul(x, y.t())
    
    # Clamp any negative values to 0 (numerical stability)
    dist_sq = torch.clamp(dist_sq, min=0.0)
    
    # Return the square root of the squared distances
    return torch.sqrt(dist_sq)
    
    
def compute_dist_closest_k_by_chunk(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    dist_func: callable,
    chunk_size: Optional[int] = 128,
) -> torch.Tensor:
    '''
    Compute the distances to the k closest neighbors in y for each point in x,
    by chunking the distance computation to reduce memory usage.

    Args:
        x: torch.Tensor of shape (N, D)
        y: torch.Tensor of shape (M, D)
        k: int, the number of nearest neighbors to consider
        dist_func: a callable that computes pairwise distances
                   (e.g., compute_cosine_dist or compute_euclid_dist)
        chunk_size: int, the size of each chunk to process at once

    Returns:
        torch.Tensor of shape (N, k), where row i contains the distances
        to the k closest points in y for x[i].
    '''
    # Prepare to store the top-k distances for all points in x
    N, D = x.shape
    top_k_list = []
    
    # Process x in chunks to avoid OOM
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        x_chunk = x[start_idx:end_idx]  # shape: (chunk_size, D)

        # Compute distances for this chunk: (chunk_size, M)
        dist_matrix_chunk = dist_func(x_chunk, y)

        # Get the k nearest distances (smallest k values) along dim=1
        # topk() returns (values, indices); we only need the values here
        top_k_chunk = dist_matrix_chunk.topk(k, largest=False, dim=1).values  # shape: (chunk_size, k)

        top_k_list.append(top_k_chunk)

    # Concatenate the chunk results back along the first dimension
    # final shape: (N, k)
    return torch.cat(top_k_list, dim=0)



class EmbedModel(BaseModel):
    '''
    This model stores the features from training set and compare test feature with the training distribution
    '''
    
    def __init__(self, cfg, input_dim):
        super().__init__(cfg, input_dim)
        self.distance = cfg.model.distance
        self.topk = cfg.model.topk
        self.use_success_only = cfg.model.use_success_only
        self.reset()


    def reset(self):
        self.trained = False
        self.feats_succ = None
        self.feats_fail = None
        self.feats_succ_mean, self.feats_succ_cov = None, None
        self.feats_fail_mean, self.feats_fail_cov = None, None
        

    def forward(
        self, 
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = batch["features"]
        assert x.dim() == 3, f"Input tensor must have 3 dimensions, got {x.dim()}"
        B, T, D = x.shape
        
        # Return all zeros if the model is not trained
        if not self.trained:
            warnings.warn("Model is not trained. Returning zeros.")
            return x.new_zeros(B, T, 1)
        
        x = x.reshape(-1, D)  # (B*T, D)

        # *Distance to the success distribution* positively correlates with *failure score*
        # *............... failure ............* negatively ............... *failure score*
        if self.distance == "mahala":
            scores = compute_mahala_dist(x, self.feats_succ_mean, self.feats_succ_cov)  # (B*T,)
            if not self.use_success_only:
                dist_to_fail = compute_mahala_dist(x, self.feats_fail_mean, self.feats_fail_cov)  # (B*T,)
                scores = scores - dist_to_fail
        elif self.distance == "cosine":
            dist_to_succ = compute_dist_closest_k_by_chunk(x, self.feats_succ, self.topk, compute_cosine_dist)  # (B*T, k)
            scores = dist_to_succ.mean(dim=-1)  # (B*T,)
            if not self.use_success_only:
                dist_to_fail = compute_dist_closest_k_by_chunk(x, self.feats_fail, self.topk, compute_cosine_dist)  # (B*T, k)
                scores = scores - dist_to_fail.mean(dim=-1)  # (B*T,)
        elif self.distance == "euclid":
            dist_to_succ = compute_dist_closest_k_by_chunk(x, self.feats_succ, self.topk, compute_euclid_dist)  # (B*T, k)
            scores = dist_to_succ.mean(dim=-1)  # (B*T,)
            if not self.use_success_only:
                dist_to_fail = compute_dist_closest_k_by_chunk(x, self.feats_fail, self.topk, compute_euclid_dist)
                scores = scores - dist_to_fail.mean(dim=-1)
        elif self.distance == "pca_kmeans":
            x_pca = self.pca_succ.transform(x.cpu().numpy())
            x_pca = torch.tensor(x_pca).to(self.feats_succ)
            dist_to_succ = compute_dist_closest_k_by_chunk(x_pca, self.centroids_succ, 1, compute_euclid_dist)
            scores = dist_to_succ.mean(dim=-1)  # (B*T,)
            if not self.use_success_only:
                dist_to_fail = compute_dist_closest_k_by_chunk(x_pca, self.centroids_fail, 1, compute_euclid_dist)
                scores = scores - dist_to_fail.mean(dim=-1)  # (B*T,)
        else:
            raise ValueError(f"Distance metric {self.distance} is not supported.")
        
        scores = scores.reshape(B, T, 1) # (B, T, 1)
        
        if self.cfg.model.cumsum:
            scores = torch.cumsum(scores, dim=-2) # (B, T, 1)
            
            # # Compute running mean 
            # scores = scores / torch.arange(1, T + 1, device=scores.device).view(1, T, 1) # (B, T, 1)
            
        if scores.isnan().any():
            raise ValueError("NaN values found in the scores. Check the input data.")
        
        return scores
                
    
    def forward_compute_loss(self, features, valid_masks, labels, weights = None):
        '''
        This model is not trained through gradient descent. 
        '''
        return 0.0, {}
    
        
    def train_epoch(
        self, 
        optimizer: torch.optim.Optimizer, 
        dataloader: DataLoader,
        force_retrain: bool = False,
    ) -> float:
        if self.trained:
            if force_retrain: self.reset()
            else: return 0.0
    
        device = self.get_device()
        features = dataloader.dataset.get_features()
        valid_masks = dataloader.dataset.get_valid_masks()
        labels = dataloader.dataset.get_labels()

        B, T, D = features.shape
        
        # Store the features
        labels = labels.unsqueeze(1)  # (B, 1)
        succ_mask = (labels == 1) & valid_masks.bool()  # (B, T)
        fail_mask = (labels == 0) & valid_masks.bool()  # (B, T)
        self.feats_succ = features[succ_mask].to(device)  # (N, D)
        self.feats_fail = features[fail_mask].to(device)  # (N, D)
        
        
        if self.distance in ["cosine", "euclid"]:
            assert len(self.feats_succ) > self.topk, f"Number of success samples {len(self.feats_succ)} is less than topk {self.topk}"
            assert len(self.feats_fail) > self.topk, f"Number of failure samples {len(self.feats_fail)} is less than topk {self.topk}"
        elif self.distance in ["mahala"]:
            self.feats_succ_mean = self.feats_succ.mean(0)  # (D,)
            self.feats_fail_mean = self.feats_fail.mean(0) # (D,)
            
            # to ensure that the convariance matrix is PSD
            epsilon = 1e-4 * torch.eye(D).to(self.feats_succ)
            self.feats_succ_cov = torch.cov(self.feats_succ.T) + epsilon # (D, D)
            self.feats_fail_cov = torch.cov(self.feats_fail.T) + epsilon # (D, D)
        elif self.distance == "pca_kmeans":
            # Implement the PCA-KMeans distance metric from https://arxiv.org/pdf/2410.22689
            self.pca_succ = PCA(n_components=self.cfg.model.pca_dim)
            self.pca_fail = PCA(n_components=self.cfg.model.pca_dim)
            self.feats_succ_pca = self.pca_succ.fit_transform(self.feats_succ.cpu().numpy())
            self.feats_fail_pca = self.pca_fail.fit_transform(self.feats_fail.cpu().numpy())
            
            self.kmeans_succ = KMeans(n_clusters=self.cfg.model.n_clusters).fit(self.feats_succ_pca)
            self.kmeans_fail = KMeans(n_clusters=self.cfg.model.n_clusters).fit(self.feats_fail_pca)
            self.centroids_succ = torch.tensor(self.kmeans_succ.cluster_centers_).to(device)
            self.centroids_fail = torch.tensor(self.kmeans_fail.cluster_centers_).to(device)

            self.feats_succ_pca = torch.tensor(self.feats_succ_pca).to(device)
            self.feats_fail_pca = torch.tensor(self.feats_fail_pca).to(device)

        self.trained = True
        
        return 0.0

        
    def compute_regularization_loss(self):
        return 0.0
    

    def get_optimizer(self):
        return None, None