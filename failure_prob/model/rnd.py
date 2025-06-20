import torch
import torch.nn as nn
import sys

from .diffusion.conditional_unet1d import ConditionalUnet1D

from .base import BaseModel
from failure_prob.conf import Config

def get_model(cfg: Config, input_dim: int) -> BaseModel:
    return RNDPolicy(cfg, input_dim)

def get_unet(input_dim, global_cond_dim):
    return ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=None,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True
    )


def adjust_xshape(x, in_dim):
    total_dim = x.shape[1]
    # Calculate the padding needed to make total_dim a multiple of in_dim
    remain_dim = total_dim % in_dim
    if remain_dim > 0:
        pad = in_dim - remain_dim
        total_dim += pad
        x = torch.cat([x, torch.zeros(x.shape[0], pad, device=x.device)], dim=1)
    # Calculate the padding needed to make (total_dim // in_dim) a multiple of 4
    reshaped_dim = total_dim // in_dim
    if reshaped_dim % 4 != 0:
        extra_pad = (4 - (reshaped_dim % 4)) * in_dim
        x = torch.cat([x, torch.zeros(x.shape[0], extra_pad, device=x.device)], dim=1)
    return x.reshape(x.shape[0], -1, in_dim)


class RNDPolicy(BaseModel):
    def __init__(self, cfg: Config, input_dim: int):
        super().__init__(cfg, input_dim)
        
        self.Df = input_dim
        self.Da = cfg.dataset.dim_action

        self.target = get_unet(self.Da, self.Df)
        
        # The better predictor_success matches to the target, the lower the failure score
        self.predictor_success = get_unet(self.Da, self.Df)
        
        if not cfg.model.use_success_only:
            # The better the predictor_failure matches to the target, the higher the failure score
            self.predictor_failure = get_unet(self.Da, self.Df)

        for param in self.target.parameters():
            param.requires_grad = False
            
        self.dist = nn.PairwiseDistance(p=2)

    def forward(
        self, 
        batch: dict[str, torch.Tensor],
        return_sep_scores: bool = False,
    ) -> torch.Tensor:
        action = batch["action_vectors"] # (B, T, H*Da)
        observation = batch["features"] # (B, T, Do)
        
        B, T, H_x_Da = action.shape
        assert H_x_Da % self.Da == 0
        _, _, Do = observation.shape
        
        # "Flatten" the batch and time dimensions
        action = action.reshape(B*T, H_x_Da) # (B*T, H*Da)

        # The UNet require H to be a multiple of 4 (becomes H')
        action = adjust_xshape(action, self.Da) # (B*T, H', Da)
        observation = observation.reshape(-1, Do) # (B*T, Do)
        t = torch.zeros(len(action)).to(action.device) # (B*T)
        
        tgt_feat = self.target(action, t, global_cond=observation)
        
        pred_feat_succ = self.predictor_success(action, t, global_cond=observation)
        pred_err_succ = self.dist(pred_feat_succ, tgt_feat)
        
        if self.cfg.model.use_success_only:
            pred_err_fail = torch.zeros_like(pred_err_succ)
        else:
            pred_feat_fail = self.predictor_failure(action, t, global_cond=observation)
            pred_err_fail = self.dist(pred_feat_fail, tgt_feat)
            
        # Take the mean for each rollout time, and reshape back to (B, T, 1)
        pred_err_succ = pred_err_succ.mean(-1).reshape(B, T, 1) # (B, T, 1)
        pred_err_fail = pred_err_fail.mean(-1).reshape(B, T, 1) # (B, T, 1)
        
        if return_sep_scores:
            return pred_err_succ, pred_err_fail
        else:
            # pred_err_succ positively correlates with failure score
            # pred_err_fail negatively correlates with failure score
            pred_err = pred_err_succ - pred_err_fail
            return pred_err
    
    
    def forward_compute_loss(
        self, 
        batch: dict[str, torch.Tensor],
        weights: list[float] = None, 
    ) -> tuple[torch.Tensor, dict[str, float]]:
        valid_masks = batch["valid_masks"] # (B, T)
        success_labels = batch["success_labels"] # (B,)
        B, T, D = batch["features"].shape # (B, T, D)
        
        pred_err_succ, pred_err_fail = self(batch, return_sep_scores=True)
        pred_err_succ = pred_err_succ.squeeze(-1) # (B, T)
        pred_err_fail = pred_err_fail.squeeze(-1) # (B, T)
        
        success_loss = (pred_err_succ * valid_masks)[success_labels == 1].mean()

        if self.cfg.model.use_success_only:
            fail_loss = torch.zeros_like(success_loss)
        else:
            fail_loss = (pred_err_fail * valid_masks)[success_labels == 0].mean()
        
        monitor_loss = success_loss + fail_loss
        
        logs = {
            "monitor_loss": monitor_loss.item(),
            "success_loss": success_loss.item(),
            "fail_loss": fail_loss.item(),
        }
        
        return monitor_loss, logs