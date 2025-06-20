import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from failure_prob.conf import Config
from failure_prob.utils.failure_metrics import (
    compute_stac_metrics, 
    compute_sample_unc_metrics, 
    compute_stac_metrics_single,
)
from failure_prob.data.openvla import split_rollouts as split_rollouts_openvla

from .utils import Rollout, set_task_min_step, process_tensor_idx_rel


def compute_hand_crafted_metrics(
    cfg: Config,
    sampled_actions: torch.Tensor,
):
    '''
    Compute handcrafted metrics for the open-pi-zero model
    
    Args:
        cfg (Config): Configuration object
        sampled_actions (torch.Tensor): Tensor of sampled actions with shape (T, N, H, A)
    '''
    selected_actions = sampled_actions[:, -1, :, :] # (T, N, A)
    
    # Compute the per-step metrics
    metrics_raw = {}
    with torch.no_grad():
        # Compute the STAC metric
        sampled_actions = sampled_actions.cuda()
        stac_mmd = compute_stac_metrics(
            sampled_actions, 
            exec_horizon=cfg.dataset.exec_horizon, 
            metric="mmd", 
            rbf_beta=cfg.dataset.rbf_beta
        ).cpu()
        metrics_raw['stac_mmd'] = stac_mmd
        metrics_raw['neg_stac_mmd'] = - stac_mmd
        
        # Compute the SATC-Single metric
        stac_single = compute_stac_metrics_single(
            selected_actions,
            cfg.dataset.exec_horizon,
        )
        metrics_raw['stac_single'] = stac_single.cpu()
        metrics_raw['neg_stac_single'] = - stac_single.cpu()
        
        # Compute the sample-based uncertainty metrics
        cov_metrics = compute_sample_unc_metrics(sampled_actions)
        metrics_raw.update(cov_metrics)
        
    metrics_raw = {k: v.cpu() for k, v in metrics_raw.items()}
    
    # Also compute the cumulative version of the metrics (running mean)
    metrics = {}
    for k, v in metrics_raw.items():
        metrics[k] = np.asarray(v) # (n_steps, )
        metrics[f"{k}_rmean"] = np.cumsum(np.asarray(v)) / (np.arange(len(v)) + 1) # (n_steps, )
    
    return pd.DataFrame(metrics)


def load_rollouts(cfg: Config) -> list[Rollout]:
    load_root = cfg.dataset.data_path
    assert os.path.exists(load_root), f"Path {load_root} does not exist"
    assert os.path.isdir(load_root), f"Path {load_root} is not a directory"
    
    all_task_names = os.listdir(load_root)
    
    keep_tasks = [
        "widowx_carrot_on_plate",
        "widowx_put_eggplant_in_basket",
        "widowx_spoon_on_towel",
        "widowx_stack_cube",
        "google_robot_move_near_v0",
        "google_robot_open_drawer",
        "google_robot_close_drawer",
        "google_robot_place_apple_in_closed_top_drawer",
    ]
    
    task_names = [t for t in keep_tasks if t in all_task_names]
    
    print(f"Keeping {len(task_names)} tasks: {task_names}")
    
    all_rollouts = []
    
    for task_id, task_name in enumerate(task_names):
        task_folder = os.path.join(load_root, task_name)
        meta_info_paths = glob.glob(f"{task_folder}/*_meta.json")
        for meta_info_path in tqdm(meta_info_paths):
            pkl_path = meta_info_path.replace("_meta.json", ".pkl")
            mp4_path = meta_info_path.replace("_meta.json", ".mp4")
            
            # Load the raw rollout data
            with open(meta_info_path, "r") as f:
                meta_info = json.load(f)
            with open(pkl_path, "rb") as f:
                rollout_raw = pickle.load(f)

            # Extract the hidden states and sampled actions
            hidden_states = []
            sampled_actions = []
            for r in rollout_raw:
                # Pi-zero generates actions by diffusions. 
                # N: number of action sampled at each inference time step
                # T: number of diffusion steps
                # H: number of prediction horizon in action chunking
                # A: dimensions of action space
                # E: dimensions of action embedding
                action_embeds = r[cfg.dataset.feat_name] # (N, T, H, E)
                
                # handle the horizon dimension
                # (N, T, H, E) -> (N, T, E)
                action_embeds = process_tensor_idx_rel(action_embeds, cfg.dataset.horizon_idx_rel)
                
                # handle the diff_steps dimension
                # (N, T, E) -> (N, E)
                action_embeds = process_tensor_idx_rel(action_embeds, cfg.dataset.diff_idx_rel)
                
                # Only keep embeddings from the last sampled action (used for the actual execution)
                embed = action_embeds[-1].reshape(-1) 
                
                hidden_states.append(embed)
                
                sampled_actions_t = r['sampled_actions'] # (N, H, A)
                sampled_actions.append(sampled_actions_t)
                
            # the loaded action embeddings are already torch tensors (bf16 or float32)
            hidden_states = torch.stack(hidden_states).float() # (T, E)
            
            sampled_actions = np.stack(sampled_actions) # (T, N, H, A)
            sampled_actions = torch.from_numpy(sampled_actions).float() # (T, N, H, A)
            
            # Handle the information needed for chen's method
            cfg.dataset.dim_features = hidden_states.shape[1]
            cfg.dataset.dim_action = sampled_actions.shape[-1]
            action_vectors = sampled_actions[:, -1, :, :] # (T, N, A)
            action_vectors = action_vectors.reshape(action_vectors.shape[0], -1) # (T, N * A)
            
            # Compute hand-crafted metrics
            hand_crafted_metrics = None
            if cfg.train.log_precomputed or cfg.train.log_precomputed_only:
                hand_crafted_metrics = compute_hand_crafted_metrics(
                    cfg,
                    sampled_actions,
                )
            
            rollout = Rollout(
                hidden_states=hidden_states,
                task_suite_name="simplerenv",
                task_id=task_id,
                task_description=meta_info['task'],
                episode_idx=meta_info['episode_id'],
                episode_success=meta_info['success'],
                mp4_path=mp4_path,
                logs=hand_crafted_metrics,
                action_vectors = action_vectors,
            )
            
            all_rollouts.append(rollout)

    all_rollouts = set_task_min_step(all_rollouts)

    return all_rollouts
                
                
def split_rollouts(cfg: Config, all_rollouts):
    return split_rollouts_openvla(cfg, all_rollouts)