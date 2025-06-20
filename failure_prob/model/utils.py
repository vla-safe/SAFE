import torch

def cumsum_stopgrad(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute a “stopped‐gradient” cumulative sum along `dim`, 
    so that each output y[..., i, ...] only backprops to x[..., i, ...].

    Args:
        x (Tensor): input tensor.
        dim (int): dimension along which to do the cumsum.

    Returns:
        Tensor: same shape as `x`, where
            y.select(dim, i) = sum_{j=0..i} x.select(dim, j),
            but gradients for y[..., i, ...] flow only to x[..., i, ...].
    """
    # handle negative dims
    if dim < 0:
        dim = x.dim() + dim
    # prepare output
    y = x.new_empty(x.shape)
    # running sum has shape of x with dim removed
    running = torch.zeros_like(x.select(dim, 0))

    # build a mutable index template
    idx = [slice(None)] * x.dim()
    for i in range(x.size(dim)):
        # select x[..., i, ...] along dim
        idx[dim] = i
        current = x[tuple(idx)]
        # detach previous running before adding
        running = running.detach().add(current)
        # write into output
        y[tuple(idx)] = running

    return y


def get_time_weight(use_weighting, valid_masks):
    '''
    Design the weights based on time.
    
    Args:
        use_weighting: bool, whether to use time weighting
        valid_masks: torch.Tensor of shape (B, T), the valid masks
        
    Returns:
        time_weights: torch.Tensor of shape (B, T), the time weights
    '''
    B, T = valid_masks.shape
    seq_lengths = valid_masks.sum(dim=1).long()  # (B,)

    if use_weighting:
        time_weights = torch.arange(T).to(valid_masks)  # (T,)
        time_weights = time_weights.unsqueeze(0).expand(B, -1)  # (B, T)
        time_weights = time_weights / seq_lengths.unsqueeze(1)  # (B, T)
        time_weights = 5 * torch.exp(- 3 * time_weights) + 1  # Exponential weights, (B, T)
        time_weights = time_weights * valid_masks  # (B, T)
        weights_normalizer = time_weights.sum(-1) / seq_lengths # (B,)
        time_weights = time_weights / weights_normalizer.unsqueeze(1)  # (B, T)
    else:
        time_weights = torch.ones(B, T).to(valid_masks) # (B, T)
        time_weights = time_weights * valid_masks  # (B, T)
        
    return time_weights


def aggregate_monitor_loss(
    losses: torch.Tensor,
    valid_masks: torch.Tensor,
    labels: torch.Tensor,
    weights: list[float],
    one_loss_per_seq: bool = False,
):
    '''
    Aggregate per-seq per-step losses to per-seq losses and then to monitor loss.
    
    Args:
        losses: torch.Tensor of shape (B, T), the per-seq per-step losses
        valid_masks: torch.Tensor of shape (B, T), the valid masks
        labels: torch.Tensor of shape (B,), the labels
        weights: list[float], the weights for each class
        one_loss_per_seq: bool. If set, sample only one sample per sequence to apply the loss.
        
    Returns:
        monitor_loss: torch.Tensor, the monitor loss
        avg_fail_loss: torch.Tensor, the average failure loss
        avg_success_loss: torch.Tensor, the average success loss
    '''
    B = losses.shape[0]
    
    # Seq-level aggregation
    fail_mask = labels == 0  # (B,)
    success_mask = labels == 1  # (B,)
    if one_loss_per_seq:
        # On each sample in the batch dimension (B), randomly sample one along the time dimension (T)
        # The sample should be within the valid mask. Only use that sample to compute the loss.
        sampled_indices = torch.multinomial(valid_masks.float(), num_samples=1).squeeze(-1)  # (B,)
        seq_loss = losses[torch.arange(B), sampled_indices]  # (B,)
    else:
        seq_loss = (losses * valid_masks).sum(-1) / valid_masks.sum(-1)  # (B,)
        
    success_loss = (success_mask * seq_loss).sum() # scalar
    fail_loss = (fail_mask * seq_loss).sum() # scalar
    
    # Weight losses according to the input weight and take mean over batch
    monitor_loss = weights[0] * fail_loss + weights[1] * success_loss # scalar
    monitor_loss = monitor_loss / B # scalar

    avg_fail_loss = (fail_loss / fail_mask.sum()) # scalar
    avg_success_loss = (success_loss / success_mask.sum()) # scalar
    
    return monitor_loss, avg_success_loss, avg_fail_loss


def hard_negative_loss(preds, labels, valid_mask, alpha, beta=None):
    """
    Computes a hard negative loss given:
      - preds: Tensor of shape (B, T) with prediction scores in [0, 1].
      - labels: Tensor of shape (B,) with binary labels (0 or 1).
      - valid_mask: Boolean tensor of shape (B, T). True indicates the prediction is valid.
      - alpha: Margin hyperparameter.
      - beta: If provided, used for a soft maximum approximation.
    """
    # Replace invalid predictions with -infinity so they don't affect the maximum.
    masked_preds = torch.where(valid_mask > 0.5, preds, torch.tensor(-float('inf'), device=preds.device)) # (B, T)
    
    if beta is None:
        # Use the hard max over the valid predictions.
        s = masked_preds.max(dim=1).values  # (B,)
    else:
        # Use a soft maximum approximation via logsumexp.
        s = (1.0 / beta) * torch.logsumexp(beta * masked_preds, dim=1) # (B,)
    
    # For positive samples (labels==1): we want s >= 0.5 + alpha.
    pos_loss = torch.clamp((0.5 + alpha) - s, min=0) # (B,)
    # For negative samples (labels==0): we want s <= 0.5 - alpha.
    neg_loss = torch.clamp(s - (0.5 - alpha), min=0) # (B,)
    
    # Combine the losses: square the hinge terms to penalize larger violations.
    loss = labels * (pos_loss ** 2) + (1 - labels) * (neg_loss ** 2) # (B,)
    
    return loss.mean()