import os
import random
import hydra
import imageio
from omegaconf import OmegaConf
from tqdm import tqdm, trange

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import wandb

from failure_prob.data import load_rollouts, split_rollouts
from failure_prob.data.utils import Rollout, RolloutDataset, normalize_rollouts_hidden_states
from failure_prob.model import get_model
from failure_prob.model.base import BaseModel
from failure_prob.utils.constants import MANUAL_METRICS, EVAL_TIME_QUANTILES
from failure_prob.utils.timer import Timer
from failure_prob.utils.routines import (
    eval_model_and_log,
    eval_metrics_and_log,
    eval_save_timing_plots,
)
from failure_prob.utils.video import eval_save_videos, eval_save_videos_functional_cp
from failure_prob.utils.random import seed_everything

from failure_prob.conf import Config, process_cfg


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config) -> None:
    cfg = process_cfg(cfg)
    print(OmegaConf.to_yaml(cfg))
    
    if (
        cfg.train.eval_save_logs 
        or cfg.train.eval_save_video 
        or cfg.train.eval_save_ckpt 
        or cfg.train.eval_save_video_functional
        or cfg.train.eval_save_timing_plots
    ):
        os.makedirs(cfg.train.logs_save_path, exist_ok=True)
        # Save the config
        with open(os.path.join(cfg.train.logs_save_path, "config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
            
    # Set seed for randomness in loading rollouts
    seed_everything(0)
            
    # Load and preprocess the rollout data
    with Timer("Loading rollouts"):
        all_rollouts = load_rollouts(cfg)
        print(f"Loaded {len(all_rollouts)} rollouts")
        if cfg.dataset.load_to_cuda:
            all_rollouts = [r.to("cuda") for r in all_rollouts]

    if len(all_rollouts) == 0:
        raise ValueError(f"No rollouts loaded from {cfg.dataset.data_path}")
    
    if cfg.dataset.normalize_hidden_states:
        # Normalize the hidden states to zero mean and unit variance
        all_rollouts = normalize_rollouts_hidden_states(all_rollouts)
    
    # Seed again for splitting rollouts
    if isinstance(cfg.train.seed, int):
        seeds = [cfg.train.seed]
    elif cfg.train.seed.isnumeric() and "-" not in cfg.train.seed:
        seeds = [int(cfg.train.seed)]
    else:
        # assert there are only integer seeds separated by "-" in cfg.train.seed
        assert all(s.isdigit() for s in cfg.train.seed.split("-")), "All seeds must be integers separated by '_'"

        # Run different seeds in the same call, to speed up 
        seeds = [int(s) for s in cfg.train.seed.split("-")]
        
    for seed in seeds:
        print(f"Running seed {seed}")
        cfg.train.seed = seed
        seed_everything(seed)
        
        wandb.init(
            project = cfg.train.wandb_project, 
            dir = cfg.train.wandb_dir,
            name = cfg.train.exp_name,
            group = cfg.train.wandb_group_name,
            config = OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )
        
        rollouts_by_split_name = split_rollouts(cfg, all_rollouts)
        
        train_rollouts = rollouts_by_split_name["train"]
        input_dim = train_rollouts[0].hidden_states.shape[-1]
        print(f"hidden_feature: {train_rollouts[0].hidden_states.shape} {train_rollouts[0].hidden_states.dtype}")

        # Construct datasets and dataloaders from the rollouts
        dataset_by_split_name = {
            k: RolloutDataset(cfg, v) 
            for k, v in rollouts_by_split_name.items()
        }
        dataloader_by_split_name = {
            k: DataLoader(
                v, 
                batch_size=cfg.model.batch_size, 
                shuffle="train" in k, 
                num_workers=0)
            for k, v in dataset_by_split_name.items()
        }

        # Plot and log the precomputed metrics
        if cfg.train.log_precomputed or cfg.train.log_precomputed_only:
            to_be_logged = eval_metrics_and_log(
                cfg,
                rollouts_by_split_name, 
                MANUAL_METRICS[cfg.dataset.name],
                EVAL_TIME_QUANTILES[cfg.dataset.name],
            )
            to_be_logged['epoch'] = 0
            wandb.log(to_be_logged)
        
        if cfg.train.log_precomputed_only:
            wandb.finish(quiet=True)
            continue
        
        model: BaseModel = get_model(cfg, input_dim)
        print(model)
        model.to("cuda")
        
        optimizer, lr_scheduler = model.get_optimizer()

        n_epochs = cfg.model.n_epochs
        epoch_losses = []
        pbar = trange(n_epochs)
        for epoch in pbar:
            to_be_logged = {"epoch": epoch + 1}
            
            # Training
            model.train()
            avg_loss = model.train_epoch(optimizer, dataloader_by_split_name["train"])
            epoch_losses.append(avg_loss)
            pbar.set_description(f"Avg Loss: {avg_loss:.4f}")
            to_be_logged["train_loss"] = avg_loss

            if lr_scheduler is not None:
                lr_scheduler.step()
                to_be_logged["learning_rate"] = optimizer.param_groups[0]['lr']
            
            # Evaluation
            model.eval()
            if epoch % cfg.train.roc_every == 0 or epoch == n_epochs - 1:
                eval_logs = eval_model_and_log(
                    cfg,
                    model, 
                    rollouts_by_split_name, 
                    dataloader_by_split_name,
                    EVAL_TIME_QUANTILES[cfg.dataset.name], 
                    plot_score_curves = epoch == n_epochs - 1,
                    plot_auc_curves = epoch == n_epochs - 1,
                    log_classification_metrics = epoch == n_epochs - 1,
                )
                to_be_logged.update(eval_logs)
                
            wandb.log(to_be_logged)
        
        # Final evaluation
        if cfg.train.eval_save_video:
            for split, dataloader in dataloader_by_split_name.items():
                if split == "train":
                    continue
                video_save_folder = os.path.join(cfg.train.logs_save_path, f"videos_{split}")
                print("Saving videos to", os.path.abspath(video_save_folder))
                os.makedirs(video_save_folder, exist_ok=True)
                eval_save_videos(dataloader, model, cfg, video_save_folder)
                
        if cfg.train.eval_save_video_functional:
            video_save_folder = os.path.join(cfg.train.logs_save_path, f"videos_functional")
            os.makedirs(video_save_folder, exist_ok=True)
            eval_save_videos_functional_cp(
                cfg, model, 
                rollouts_by_split_name,
                dataloader_by_split_name,
                video_save_folder,
                alpha = cfg.train.eval_cp_alpha,
            )
            
        if cfg.train.eval_save_timing_plots:
            plot_save_folder = os.path.join(cfg.train.logs_save_path, f"timing_plots")
            os.makedirs(plot_save_folder, exist_ok=True)
            eval_save_timing_plots(
                cfg, model, 
                rollouts_by_split_name,
                dataloader_by_split_name,
                plot_save_folder,
                alpha= cfg.train.eval_cp_alpha,
            )
            
                
        if cfg.train.eval_save_ckpt:
            os.makedirs(cfg.train.logs_save_path, exist_ok=True)
            ckpt_save_path = os.path.join(cfg.train.logs_save_path, "model_final.ckpt")
            print("Saving model checkpoint to", os.path.abspath(ckpt_save_path))
            torch.save(model.state_dict(), ckpt_save_path)
        
        wandb.finish(quiet=True)
        

if __name__ == "__main__":
    main()
