import torch
import torch.nn as nn
import sys

from .diffusion.conditional_unet1d import ConditionalUnet1D

from .base import BaseModel
from failure_prob.conf import Config

def get_model(cfg: Config, input_dim: int) -> BaseModel:
    return LogpZOModel(cfg, input_dim)


def get_unet(input_dim):
    return ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=False
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


class LogpZOModel(BaseModel):
    def __init__(self, cfg: Config, input_dim: int):
        super().__init__(cfg, input_dim)
        
        # actual dimension of input features
        self.input_dim_total = input_dim
        # dimension used for adjust_xshape
        self.input_dim = cfg.model.in_dim
        
        self.net_succ = get_unet(self.input_dim)
        
        if not self.cfg.model.use_success_only:
            self.net_fail = get_unet(self.input_dim)
        
    
    def forward(
        self, 
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        observation = batch["features"] # (B, T, Do)
        B, T, Do = observation.shape
        
        observation = observation.reshape(B*T, Do) # (B*T, Do)
        observation = adjust_xshape(observation, self.input_dim) # (B*T, H', self.input_dim)
        timesteps = torch.zeros(observation.shape[0], device=observation.device) # (B*T,)
        
        # Compute distance to the success distribution
        if self.cfg.model.forward_chunk_size:
            pred_v_succ_all = []
            for start in range(0, observation.shape[0], self.cfg.model.forward_chunk_size):
                end = min(start + self.cfg.model.forward_chunk_size, observation.shape[0])
                pred_v_succ = self.net_succ(observation[start:end], timesteps[start:end])
                pred_v_succ_all.append(pred_v_succ)
            pred_v_succ = torch.cat(pred_v_succ_all, dim=0)
        else:
            pred_v_succ = self.net_succ(observation, timesteps)
        scores_succ = (observation + pred_v_succ).reshape(len(observation), -1).pow(2).sum(dim=-1) # (B*T,)
        
        if self.cfg.model.use_success_only:
            scores = scores_succ
        else:
            if self.cfg.model.forward_chunk_size:
                pred_v_fail_all = []
                for start in range(0, observation.shape[0], self.cfg.model.forward_chunk_size):
                    end = min(start + self.cfg.model.forward_chunk_size, observation.shape[0])
                    pred_v_fail = self.net_fail(observation[start:end], timesteps[start:end])
                    pred_v_fail_all.append(pred_v_fail)
                pred_v_fail = torch.cat(pred_v_fail_all, dim=0)
            else:
                pred_v_fail = self.net_fail(observation, timesteps)    
            scores_fail = (observation + pred_v_fail).reshape(len(observation), -1).pow(2).sum(dim=-1) # (B*T,)
            scores = scores_succ - scores_fail
        
        scores = scores.reshape(B, T, 1) # (B, T, 1)
        
        return scores
        
    
    def compute_flow_matching_loss(
        self,
        net: nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x0, x1 = x, torch.randn_like(x)
        vtrue = x1 - x0 # (B*T, H', Di)
        cont_t = torch.rand(len(x1),).to(x) # (B*T,)
        cont_t = cont_t.view(-1, *[1 for _ in range(len(x.shape)-1)]) # (B*T, 1, 1)
        xnow = x0 + cont_t * vtrue # (B*T, H', Di)
        time_scale = 100 # In UNet, which takes discrete time steps
        vhat = net(xnow, (cont_t.view(-1)*time_scale).long()) # (B*T, H', Di)
        loss = (vhat - vtrue).pow(2).mean()
        
        return loss
    
    
    def forward_compute_loss(
        self, 
        batch: dict[str, torch.Tensor],
        weights: list[float] = None, 
    ) -> tuple[torch.Tensor, dict[str, float]]:
        valid_masks = batch["valid_masks"] # (B, T)
        success_labels = batch["success_labels"] # (B,)
        
        x = batch["features"] # (B, T, Do)
        B, T, Do = x.shape
        
        # For the success rollouts
        loss_succ = torch.zeros(1, device=x.device)
        if success_labels.sum() > 0:
            x_succ = x[success_labels == 1] # (B1, T, Do)
            valid_mask_succ = valid_masks[success_labels == 1] # (B1, T)
            x_succ = x_succ.reshape(-1, Do) # (B1*T, Do)
            valid_mask_succ = valid_mask_succ.reshape(-1) # (B1*T,)
            x_succ = x_succ[valid_mask_succ.bool()] # (N1, Do)

            if self.cfg.model.forward_chunk_size:
                indices = torch.randperm(x_succ.shape[0])[:self.cfg.model.forward_chunk_size]
                x_succ =  x_succ[indices] # (N1', Do)
                
            x_succ = adjust_xshape(x_succ, self.input_dim) # (N1, H', Di)
            loss_succ = self.compute_flow_matching_loss(self.net_succ, x_succ) # scalar
        
        # For the failure rollouts
        loss_fail = torch.zeros_like(loss_succ)
        if (not self.cfg.model.use_success_only) and (success_labels == 0).sum() > 0:
            # For the Failure rollouts
            x_fail = x[success_labels == 0] # (B0, T, Do)
            valid_mask_fail = valid_masks[success_labels == 0] # (B0, T)
            x_fail = x_fail.reshape(-1, Do) # (B0*T, Do)
            valid_mask_fail = valid_mask_fail.reshape(-1) # (B0*T,)
            x_fail = x_fail[valid_mask_fail.bool()] # (N0, Do)

            if self.cfg.model.forward_chunk_size:
                indices = torch.randperm(x_fail.shape[0])[:self.cfg.model.forward_chunk_size]
                x_fail =  x_fail[indices]
            
            x_fail = adjust_xshape(x_fail, self.input_dim) # (N0, H', Di)
            loss_fail = self.compute_flow_matching_loss(self.net_fail, x_fail) # scalar
        
        monitor_loss = loss_succ + loss_fail
        
        logs = {
            "monitor_loss": monitor_loss.item(),
            "success_loss": loss_succ.item(),
            "fail_loss": loss_fail.item(),
        }
        
        return monitor_loss, logs