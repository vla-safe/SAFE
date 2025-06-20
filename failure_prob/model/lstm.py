import torch
import torch.nn as nn

from .base import BaseModel
from .utils import get_time_weight, aggregate_monitor_loss, hard_negative_loss, cumsum_stopgrad

from failure_prob.conf import Config


def get_model(cfg, input_dim):
    return LstmModel(cfg, input_dim)


class LstmModel(BaseModel):
    def __init__(self, cfg: Config, input_dim: int):
        super().__init__(cfg, input_dim)
        self.hidden_dim = cfg.model.hidden_dim
        self.n_layers = cfg.model.n_layers
        self.lstm = nn.LSTM(
            input_dim, 
            self.hidden_dim, 
            self.n_layers,
            batch_first=True,
            dropout=cfg.model.dropout,
        )
        self.fc = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(cfg.model.dropout)
        self.n_history_steps = cfg.model.n_history_steps
        
        self._scale_weights(self.cfg.model.init_weight_scale)
    

    def forward(
        self, 
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = batch["features"]
        B, T, D = x.shape
        n = self.n_history_steps

        assert x.ndim == 3, f"Input dim mismatch: {x.ndim} != 3"
        assert D == self.input_dim, f"Input dim mismatch: {D} != {self.input_dim}"

        if n < 0:
            out, _ = self.lstm(x)  # (B, T, hidden_dim)
        else:
            # Prepare sliding windows: for each timestep t, extract [t-n, ..., t-1]
            x_padded = torch.nn.functional.pad(x, (0, 0, n, 0), mode="constant", value=0)  # (B, T+n, D)

            x_windows = []
            for t in range(T):
                x_window = x_padded[:, t:t+n, :]  # (B, n, D)
                x_windows.append(x_window)

            x_seq = torch.stack(x_windows, dim=1)  # (B, T, n, D)
            x_seq = x_seq.reshape(B * T, n, D)     # (B*T, n, D)

            out, _ = self.lstm(x_seq)              # (B*T, n, hidden_dim)
            out = out[:, -1, :]                    # (B*T, hidden_dim)
            out = out.view(B, T, -1)               # (B, T, hidden_dim)

        out = self.dropout(out)                 # (B, T, hidden_dim)
        p_seq = torch.sigmoid(self.fc(out))    # (B, T, 1)

        if self.cfg.model.cumsum:
            # p_seq = p_seq.cumsum(dim=1)
            p_seq = cumsum_stopgrad(p_seq, dim=1)  # (B, T, 1)
            if self.cfg.model.rmean:
                normalizer = p_seq.new_ones(p_seq.shape).cumsum(dim=1)
                p_seq = p_seq / normalizer

        return p_seq


    def forward_compute_loss(
        self, 
        batch: dict[str, torch.Tensor],
        weights: list[float] = None, 
    ) -> tuple[torch.Tensor, dict[str, float]]:
        valid_masks = batch["valid_masks"]
        success_labels = batch["success_labels"]
        B, T, D = batch["features"].shape
        
        scores = self(batch)  # (B, T, 1)
        scores = scores.squeeze(-1)  # (B, T)
        
        # Design the weights based on time
        time_weights = get_time_weight(self.cfg.model.use_time_weighting, valid_masks)  # (B, T)
        time_weights = time_weights.to(scores) # (B, T)
        
        if self.cfg.model.cumsum:
            # Compute the loss as if each sequence is successful or failure, then aggregate back to (B, T)
            lower_thresh = 0
            seq_loss_success = torch.relu(scores - lower_thresh)  # (B, T)
            seq_loss_fail = time_weights * (- scores)
                
            losses = (success_labels == 1).float()[:, None] * seq_loss_success + \
                (success_labels == 0).float()[:, None] * seq_loss_fail  # (B, T)
        else:
            # Compute BCE loss on scores at all timesteps
            criterion = nn.BCELoss(reduction="none")
            # Failure is the positive class
            if scores.isnan().any():
                import pdb; pdb.set_trace()
            losses = criterion(scores, 1 - success_labels.unsqueeze(-1).expand_as(scores)) # (B, T)
            
            # Apply the time weights only on the failure samples
            losses[success_labels == 0] *= time_weights[success_labels == 0] # (B, T)
        
        monitor_loss, success_loss, fail_loss = aggregate_monitor_loss(
            losses, valid_masks, success_labels, weights,
            self.cfg.model.one_loss_per_seq,
        )

        # Now that we want to do classification based on the max scores before termination
        # Therefore add hard nagative mining loss
        hard_neg_loss = torch.tensor(0.0).to(scores)
        if self.cfg.model.lambda_hard_heg > 0:
            # Note that in success_labels==0 means failure
            hard_neg_loss = hard_negative_loss(
                scores, 1-success_labels, valid_masks, 
                self.cfg.model.hard_neg_margin, 
                self.cfg.model.hard_neg_beta
            )
            hard_neg_loss = self.cfg.model.lambda_hard_heg * hard_neg_loss
        
        monitor_loss += hard_neg_loss

        # Log the losses
        logs = {
            "monitor_loss": monitor_loss.item(),
            "success_loss": success_loss.item(),
            "fail_loss": fail_loss.item(),
            "hard_neg_loss": hard_neg_loss.item(),
        }
        
        return monitor_loss, logs