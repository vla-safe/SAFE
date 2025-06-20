import torch
import torch.nn as nn

from .base import BaseModel
from .utils import get_time_weight, aggregate_monitor_loss

from failure_prob.conf import Config


def get_model(cfg: Config, input_dim: int) -> BaseModel:
    return IndepModel(cfg, input_dim)


class IndepModel(BaseModel):
    '''
    In this model, we are treating the model features at each timestep independently.
    Each feature is projected to a single scalar value and accumulated throughout rollout
    '''
    def __init__(self, cfg: Config, input_dim: int):
        super().__init__(cfg, input_dim)
        
        self.total_input_dim = input_dim * cfg.model.n_history_steps
        self.hidden_dim = cfg.model.hidden_dim
        
        # Build up the model
        projector = []
        if cfg.model.n_layers == 1:
            projector.append(nn.Linear(self.total_input_dim, 1))
        else:
            projector.append(nn.Linear(self.total_input_dim, self.hidden_dim))
            projector.append(nn.ReLU())
            for _ in range(cfg.model.n_layers - 2):
                projector.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                projector.append(nn.ReLU())
            projector.append(nn.Linear(self.hidden_dim, 1))
            
        if cfg.model.final_act_layer == "sigmoid":
            projector.append(nn.Sigmoid())
        elif cfg.model.final_act_layer == "relu":
            projector.append(nn.ReLU())
        elif cfg.model.final_act_layer == "none":
            pass
        else:
            raise ValueError(f"Unknown final activation: {cfg.model.final_act_layer}")
            
        self.projector = nn.Sequential(*projector)

        
    def forward(
        self, 
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = batch["features"]
        assert x.ndim == 3, f"Input dim mismatch: {x.ndim} != 3"
        assert x.shape[-1] == self.input_dim, f"Input dim mismatch: {x.shape[-1]} != {self.input_dim}"

        x = self.projector(x) # (batch_size, seq_len, 1)
        
        # assert not (self.cfg.model.cumsum and self.cfg.model.rmean), "Cannot use both cumsum and rmean at the same time"

        if self.cfg.model.cumsum or self.cfg.model.rmean:
            # Accumulate the scores over the time dimension
            x = torch.cumsum(x, dim=-2) # (batch_size, seq_len, 1)
            
            # rmean will overwrite the cumsum
            if self.cfg.model.rmean:
                x = x / torch.arange(1, x.shape[1] + 1, device=x.device).view(1, -1, 1) # (batch_size, seq_len, 1)

        return x
        
    
    def forward_compute_loss(
        self, 
        batch: dict[str, torch.Tensor],
        weights: list[float] = None, 
    ) -> tuple[torch.Tensor, dict[str, float]]:
        features = batch["features"]
        valid_masks = batch["valid_masks"]
        labels = batch["success_labels"]
        B, T, D = features.shape
        
        scores = self(batch)  # (B, T, 1)
        scores = scores.squeeze(-1)  # (B, T)
        
        # Design the weights based on time
        time_weights = get_time_weight(self.cfg.model.use_time_weighting, valid_masks) # (B, T)
        time_weights = time_weights.to(scores) # (B, T)
        
        # Compute the loss as if each sequence is successful or failure, then aggregate back to (B, T)
        higher_thresh = self.cfg.model.threshold
        lower_thresh = 0
        seq_loss_success = torch.relu(scores - lower_thresh)  # (B, T)
        if self.cfg.model.use_threshold:
            seq_loss_fail = time_weights * torch.relu(higher_thresh - scores)
        else:
            seq_loss_fail = time_weights * (- scores)
            
        losses = (labels == 1).float()[:, None] * seq_loss_success + \
            (labels == 0).float()[:, None] * seq_loss_fail  # (B, T)
        
        monitor_loss, success_loss, fail_loss = aggregate_monitor_loss(losses, valid_masks, labels, weights)

        # Log the losses
        logs = {
            "monitor_loss": monitor_loss.item(),
            "success_loss": success_loss.item(),
            "fail_loss": fail_loss.item(),
        }
        
        return monitor_loss, logs