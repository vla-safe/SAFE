import abc
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, StepLR, SequentialLR

import wandb

from failure_prob.data.utils import Rollout
from failure_prob.conf import Config
from failure_prob.utils.torch import move_to_device


class BaseModel(nn.Module):
    '''
    A sequential failure detection model based on a sequence of features.
    Given a sequence of feature vectors collected at different timesteps of a robot rollout, 
    it return a score indicating the likelihood of failure, uncertainty or other meaningful metric.
    
    The model is causal - it only uses past or current information to make predictions.
    '''
    def __init__(self, cfg: Config, input_dim: int):
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self._device = "cpu"
    
    @abc.abstractmethod
    def forward(
        self, 
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        '''
        Forward pass of the failure detection model. 
        
        Args:
            batch: dict[str, torch.Tensor], a dictionary containing at least
                - features: torch.Tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor of shape (batch_size, seq_len, 1)
        '''
        raise NotImplementedError
    
    @abc.abstractmethod
    def forward_compute_loss(
        self, 
        batch: dict[str, torch.Tensor],
        weights: list[float] = None, 
    ) -> tuple[torch.Tensor, dict[str, float]]:
        '''
        Forward pass of the failure detection model and compute the losses.
        
        Args:
            batch: dict[str, torch.Tensor], a dictionary containing at least
                - features: torch.Tensor of shape (batch_size, seq_len, input_dim)
                - valid_masks: torch.Tensor of shape (batch_size, seq_len)
                - labels: torch.Tensor of shape (batch_size,)
            weights: list[float], the weights for each class
            
        Returns:
            monitor_loss: torch.Tensor, the monitor loss
            logs: dict[str, float], a dictionary of logs to be used by the logger
        '''
        raise NotImplementedError
    
    
    def _scale_weights(self, scale_factor: float):
        """Scales all weights in the given module by the scale_factor,
        leaving biases unchanged."""
        with torch.no_grad():
            for name, param in self.named_parameters():
                if 'weight' in name and param is not None:
                    param.mul_(scale_factor)
    
    
    def train_epoch(
        self, 
        optimizer: torch.optim.Optimizer, 
        dataloader: DataLoader,
    ) -> float:
        device = self.get_device()
        
        total_losses = []
        
        weights = dataloader.dataset.get_class_weights()
        
        def add_log_prefix(logs: dict) -> dict:
            return {f"train_loss/{k}": v for k, v in logs.items()}
        
        for batch in dataloader:
            batch = move_to_device(batch, device)

            # Model forward and compute losses
            training_logs = {}
            loss, logs = self.forward_compute_loss(batch, weights)
            training_logs.update(add_log_prefix(logs))

            reg_loss, logs = self.compute_regularization_loss(self.cfg.model.lambda_reg)
            training_logs.update(add_log_prefix(logs))

            total_loss = loss + reg_loss

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()

            # Add gradient clipping here:
            if self.cfg.model.grad_max_norm is not None:
                clip_grad_norm_(self.parameters(), max_norm=self.cfg.model.grad_max_norm)

            optimizer.step()
            
            training_logs['train_loss/total_loss'] = total_loss.item()
            wandb.log(training_logs)
            total_losses.append(total_loss.item())
            
        avg_loss = sum(total_losses) / len(total_losses)
        return avg_loss
    
    
    def compute_regularization_loss(self, lambda_reg: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        '''
        Compute the regularization loss for the model.
        
        Args:
            lambda_reg: float, the regularization coefficient
            
        Returns:
            reg_loss: torch.Tensor, the regularization loss
            logs: Dict[str, float], a dictionary of logs to be used by the logger
        '''
        if lambda_reg == 0:
            return 0.0, {}
        
        reg_loss = 0.0
        for name, param in self.named_parameters():
            if "bias" not in name:
                reg_loss += torch.sum(param ** 2)
        
        reg_loss = lambda_reg * reg_loss        
        logs = {
            "reg_loss": reg_loss.item(),
        }
        return reg_loss, logs
    
    
    def to(self, device):
        self._device = device
        return super().to(device)
    

    def get_device(self):
        return self._device
    
    
    def get_optimizer(self):
        if self.cfg.model.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.model.lr)
        elif self.cfg.model.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.model.lr)
        elif self.cfg.model.optimizer == "sgdm":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.cfg.model.lr, momentum=0.9
            )
        elif self.cfg.model.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.cfg.model.lr,
                weight_decay=self.cfg.model.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.model.optimizer}")
        
        step_scheduler = StepLR(
            optimizer,
            step_size=self.cfg.model.lr_step_size,
            gamma=self.cfg.model.lr_gamma
        )

        
        warmup_steps = self.cfg.model.warmup_steps
        if warmup_steps > 0:
            # linearly increase from 0 → 1 over warmup_steps
            lr_lambda = lambda step: min((step + 1) / warmup_steps, 1.0)
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

            # SequentialLR will use warmup_scheduler for the first warmup_steps,
            # then automatically switch over to step_scheduler.
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, step_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = step_scheduler
        
        return optimizer, scheduler