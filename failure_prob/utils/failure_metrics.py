import numpy as np
import scipy
import torch
import torch.nn.functional as F

from sklearn.cluster import AgglomerativeClustering

def rbf_kernel(x, y, beta):
    """
    Compute the RBF kernel value between x and y.

    Parameters:
        x (torch.Tensor): A tensor of shape (..., d).
        y (torch.Tensor): A tensor of shape (..., d).
        beta (float): Bandwidth of the RBF kernel.

    Returns:
        torch.Tensor: The RBF kernel values, shape (...,).
    """
    return torch.exp(-torch.norm(x - y, dim=-1) ** 2 / beta)


def compute_mmd(X, Y, beta):
    """
    Compute the Maximum Mean Discrepancy (MMD) with RBF kernels for each timestep.

    Parameters:
        X (torch.Tensor): A tensor of shape (n_timesteps, batch_size, h, A) representing action sequences over time.
        Y (torch.Tensor): A tensor of shape (n_timesteps, batch_size, h, A) representing action sequences over time.
        beta (float): Bandwidth of the RBF kernel.

    Returns:
        torch.Tensor: A tensor of shape (n_timesteps,) with the MMD value for each timestep.
    """
    T, b, h, A = X.shape

    X_flat = X.view(T, b, -1) # (T, b, h * A)
    Y_flat = Y.view(T, b, -1) # (T, b, h * A)

    # Compute pairwise RBF kernels
    XX_kernel = rbf_kernel(X_flat[:, :, None, :], X_flat[:, None, :, :], beta) # (T, b, b)
    YY_kernel = rbf_kernel(Y_flat[:, :, None, :], Y_flat[:, None, :, :], beta) # (T, b, b)
    XY_kernel = rbf_kernel(X_flat[:, :, None, :], Y_flat[:, None, :, :], beta) # (T, b, b)

    # Compute expectations
    expectation_1 = XX_kernel.mean(dim=(1, 2))
    expectation_2 = YY_kernel.mean(dim=(1, 2))
    expectation_3 = XY_kernel.mean(dim=(1, 2))

    # Compute MMD for each timestep
    mmd_values = expectation_1 + expectation_2 - 2 * expectation_3

    return mmd_values
    

def compute_stac_metrics(
    sampled_actions: torch.Tensor,
    exec_horizon: int = 1, # Octo uses execution horizon of 1
    metric: str = "mmd",
    rbf_beta: float = 1.0,
) -> torch.Tensor:
    '''
    Compute the STAC metric as mentioned https://arxiv.org/pdf/2405.12213
    
    Args:
        sampled_actions (torch.Tensor): shape (ts_timesteps, b_samples, pred_horizon, A)
        exec_horizon (int): execution horizon, how many actions are executed at each timestep
        metric (str): metric to use, currently only "mmd" is supported
        
    Returns:
        torch.Tensor: shape (ts_timesteps,) with the STAC metric value for each timestep
    '''
    pred_horizon = sampled_actions.shape[2]
    
    assert pred_horizon >= exec_horizon, f"pred_horizon should be greater than exec_horizon, got {pred_horizon} and {exec_horizon}"
    
    actions_pred_old = sampled_actions[:-1, :, exec_horizon: , :] # (ts-1, b, t-exec_horizon, A)
    actions_pred_new = sampled_actions[1:,  :, :-exec_horizon, :] # (ts-1, b, t-exec_horizon, A)
    
    if metric == "mmd":
        metric_values = compute_mmd(actions_pred_old, actions_pred_new, beta=rbf_beta) # (ts-1,)
    else:
        raise NotImplementedError(f"Metric {metric} not implemented")
    
    # Prepend a 0 for the first timestep
    metric_values = torch.cat([metric_values.new_zeros(1), metric_values]) # (ts,)
    
    return metric_values


def compute_stac_metrics_single(
    actions: torch.Tensor,
    exec_horizon: int = 1, # Octo uses execution horizon of 1
) -> torch.Tensor:
    '''
    Compute the STAC scores based on a single sampled action - difference between the overlapping part of consecutive actions
    
    Args:
        actions (torch.Tensor): shape (T, pred_horizon, A)
        exec_horizon (int): execution horizon, how many actions are executed at each timestep
        
    Returns:
        torch.Tensor: shape (T,) with the STAC metric value for each timestep
    '''
    pred_horizon = actions.shape[1]
    assert pred_horizon > exec_horizon, f"pred_horizon should be greater than exec_horizon, got {pred_horizon} and {exec_horizon}"
    
    actions_pred_old = actions[:-1, exec_horizon:, :] # (T-1, t-exec_horizon, A)
    actions_pred_new = actions[1:, :-exec_horizon, :] # (T-1, t-exec_horizon, A)
    
    actions_pred_diff = actions_pred_old - actions_pred_new # (T-1, t-exec_horizon, A)
    stac_single = actions_pred_diff.norm(dim=-1).mean(dim=-1) # (T-1,)
    
    # Prepend a 0 for the first timestep
    stac_single = torch.cat([stac_single.new_zeros(1), stac_single]) # (T,)

    return stac_single
    


def trace_3d(X: torch.Tensor) -> torch.Tensor:
    return X.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def compute_sample_unc_metrics(
    sampled_actions: torch.Tensor,
):
    '''
    Compute the uncertainty metrics based on sample variance
    
    Args:
        sampled_actions (torch.Tensor): shape (ts_timesteps, b_samples, pred_horizon, A)
        
    Returns:
        dict: dictionary containing the variance metrics
    '''
    T, b, t, A = sampled_actions.shape
    X = sampled_actions.reshape(T, b, -1) # (T, b, t * A)
    
    # Compute the covariance matrix for each timestep. The resulting matrix should be of shape (T, t*A, t*A)
    X_centered = X - X.mean(dim=1, keepdim=True)  # (T, b, t * A)
    X_cov = torch.einsum('tbi,tbj->tij', X_centered, X_centered) / (b - 1) # (T, t * A, t * A)
    
    total_var = trace_3d(X_cov) # (T,)
    general_var = X_cov.det() # (T,)
    pos_var = trace_3d(X_cov[:, :3, :3]) # (T,)
    rot_var = trace_3d(X_cov[:, 3:6, 3:6]) # (T,)
    gripper_var = trace_3d(X_cov[:, 6:, 6:]) # (T,)
    
    metrics = {
        "total_var": total_var,
        "general_var": general_var,
        "pos_var": pos_var,
        "rot_var": rot_var,
        "gripper_var": gripper_var,
    }
    
    # Perform linkage clustering for computing entropy metrics
    for threshold in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        key_name = f"entropy_linkage{threshold}"
        metrics[key_name] = []
        for actions in X: # (T, t * A)
            linkage = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)
            cluster_labels = linkage.fit_predict(actions.cpu().numpy())
            _, counts = np.unique(cluster_labels, return_counts=True)
            entropy = scipy.stats.entropy(counts)
            metrics[key_name].append(entropy)
        metrics[key_name] = torch.tensor(metrics[key_name])
    
    return metrics
    

def compute_token_metrics(
    token_logits: torch.Tensor,
) -> dict:
    '''
    Compute the metrics based on token probabilities on one generated sentence

    Args:
        token_logits (torch.Tensor): shape (n_tokens, vocab_size)
    '''
    token_prob = F.softmax(token_logits, dim=-1) # (n_actions, n_bins)
    selected_token_prob = token_prob.max(dim=-1).values # (n_actions)
    token_entr = logits2entropy(token_logits) # (n_actions)
    
    metrics = {}
    metrics["avg_token_entropy"] = token_entr.mean().item()
    metrics["max_token_entropy"] = token_entr.max().item()
    # metrics["neg_mean_token_prob"] = 1 - selected_token_prob.mean().item()
    # metrics["neg_max_token_prob"] = 1 - selected_token_prob.max().item()
    metrics['max_token_prob'] = (- torch.log(selected_token_prob)).max().item()
    metrics['avg_token_prob'] = (- torch.log(selected_token_prob)).mean().item()

    return metrics

    
def logits2entropy(logits, dim=-1):
    """
    Computes the entropy of a set of logits.

    Args:
        logits (torch.Tensor): The input logits tensor.
        dim (int, optional): The dimension along which softmax and log-softmax are computed. 
                             Default is -1.
    Returns:
        torch.Tensor: The entropy of the input logits.
    """
    p = F.softmax(logits, dim=dim)
    logp = F.log_softmax(logits, dim=dim)
    return - torch.sum(p * logp, dim=dim)
