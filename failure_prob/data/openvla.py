import glob
import json
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from failure_prob.conf import Config

from .utils import Rollout, set_task_min_step, split_rollouts_by_seen_unseen, process_tensor_idx_rel

def compute_hand_crafted_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    metrics_raw = {}
    
    # Compute the token-level uncertainty metrics
    if 'action/token_0_prob' in df.columns:
        token_prob = df[[
            'action/token_0_prob', 'action/token_1_prob', 'action/token_2_prob', 'action/token_3_prob',
            'action/token_4_prob', 'action/token_5_prob', 'action/token_6_prob'
        ]].values # (n_step, n_token)
        token_entropy = df[[
            'action/token_0_entropy', 'action/token_1_entropy', 'action/token_2_entropy', 'action/token_3_entropy',
            'action/token_4_entropy', 'action/token_5_entropy', 'action/token_6_entropy'
        ]].values # (n_step, n_token)
        max_token_prob = (- np.log(token_prob)).max(axis=-1) # (n_step, )
        avg_token_prob = (- np.log(token_prob)).mean(axis=-1) # (n_step, )
        max_token_entropy = token_entropy.max(axis=-1) # (n_step, )
        avg_token_entropy = token_entropy.mean(axis=-1) # (n_step, )
        metrics_raw.update({
            "max_token_prob": max_token_prob,
            "avg_token_prob": avg_token_prob,
            "max_token_entropy": max_token_entropy,
            "avg_token_entropy": avg_token_entropy,
        })
    
    # Extract the sample-level uncertianty metrics
    for k in [
        "total_var", "general_var", "pos_var", "rot_var", "gripper_var", "entropy_linkage.01", "entropy_linkage.05"
    ]:
        if f"action/{k}" in df.columns:
            values = df[f"action/{k}"].values # (n_step, )
            metrics_raw[k] = np.asarray(values) # (n_steps, )

    # Compute the cumulative version of the metrics (running mean)
    metrics = {}
    for k, v in metrics_raw.items():
        metrics[k] = np.asarray(v) # (n_steps, )
        metrics[f"{k}_rmean"] = np.cumsum(np.asarray(v)) / (np.arange(len(v)) + 1) # (n_steps, )

        # # In LIBERO, the rollouts terminates early when the task is successfully completed. 
        # # Cumsum metrics gives advantages for failure detection to cheat and thus not proper to use. 
        # metrics[f"{k}_csum"] = np.cumsum(np.asarray(v)) # (n_steps, )
        
    df_metrics = pd.DataFrame(metrics)
    
    return df_metrics


def extract_info_from_path(filename):
    # Define the regex pattern
    pattern = r"task(\d+)--ep(\d+)--succ(\d+)\.csv"
    
    # Match the pattern
    match = re.match(pattern, filename)
    
    if match:
        # Extract and convert values
        task_id = int(match.group(1))
        episode_id = int(match.group(2))
        success = bool(int(match.group(3)))  # Convert 1/0 to True/False
        return task_id, episode_id, success
    else:
        raise ValueError("Filename format is incorrect")
    


def load_rollouts(cfg: Config) -> list[Rollout]:
    # Load data
    all_csv = glob.glob(f"{cfg.dataset.data_path}*.csv")
    all_rollouts = []
    
    for csv_path in tqdm(all_csv, desc="Loading rollouts"):
        # Load the hand-crafted metrics from the csv
        with open(csv_path, "r") as f:
            df = pd.read_csv(f)
            
        df_metrics = None
        if cfg.train.log_precomputed or cfg.train.log_precomputed_only:
            df_metrics = compute_hand_crafted_metrics(df)
        
        # Extract action vectors from the logged df
        action_vectors = df[[
            "action/dx", "action/dy", "action/dz", 
            "action/droll", "action/dpitch", "action/dyaw", "action/dgripper"
        ]].values # (n_step, d_action)
        action_vectors = torch.tensor(action_vectors, dtype=torch.float32) # (n_step, d_action)
        cfg.dataset.dim_action = action_vectors.shape[-1]
        
        # Load the meta information and saved features
        pkl_path = csv_path.replace(".csv", ".pkl")
        mp4_path = pkl_path.replace(".pkl", ".mp4")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                rollout = pickle.load(f)
            
            # Fix a typo in the dataset
            if "eposide_idx" in rollout:
                rollout['episode_idx'] = rollout['eposide_idx']
                del rollout['eposide_idx']
            
            # Process the hidden states
            # (n_step, n_token, d) -> (n_step, d')
            hidden_states = rollout["hidden_states"].float() # (n_step, n_token, d)
            hidden_states = process_tensor_idx_rel(hidden_states, cfg.dataset.token_idx_rel) # (n_step, d')
        else:
            file_name = os.path.basename(csv_path)
            task_id, episode_id, success = extract_info_from_path(file_name)
            rollout = {
                "task_suite_name": "openvla",
                "task_id": task_id,
                "task_description": f"Task {str(task_id)}",
                "episode_idx": episode_id,
                "episode_success": int(success),
            }
            rollout_length = len(df_metrics)
            hidden_states = torch.zeros((rollout_length, 1))
            action_vectors = None
        
        r = Rollout(
            hidden_states=hidden_states,
            task_suite_name=rollout["task_suite_name"],
            task_id=rollout["task_id"],
            task_description=rollout["task_description"],
            episode_idx=rollout["episode_idx"],
            episode_success=rollout["episode_success"],
            mp4_path=mp4_path,
            logs=df_metrics,
            action_vectors=action_vectors,
        )
            
        r.mp4_path = mp4_path
        all_rollouts.append(r)

    print(f"Loaded {len(all_rollouts)} rollouts")
    
    all_rollouts = set_task_min_step(all_rollouts)
        
    return all_rollouts


def split_rollouts(cfg: Config, all_rollouts: list[Rollout]) -> dict[str, list[Rollout]]:
    # Split rollouts into seen and unseen tasks
    task_ids = list(set([r.task_id for r in all_rollouts]))
    n_unseen = round(cfg.dataset.unseen_task_ratio * len(task_ids))
    n_seen = len(task_ids) - n_unseen
    
    np.random.shuffle(task_ids)
    seen_task_ids = task_ids[:n_seen]
    unseen_task_ids = task_ids[n_seen:]
    
    rollouts_by_split_name = split_rollouts_by_seen_unseen(
        cfg, all_rollouts, seen_task_ids, unseen_task_ids
    )

    return rollouts_by_split_name