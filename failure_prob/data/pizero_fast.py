'''
Dataloader for rollout datasets generated by pi0 with FAST Tokenizer. 
Note that for pi0-fast, action prediction is mostly treated as a sentence prediction tasks as well. 

pi0 is from the following repo:
https://github.com/Physical-Intelligence/openpi
'''

from collections import defaultdict
import glob
import pickle
from pathlib import Path

import cv2
import imageio
import natsort
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# import matplotlib
# matplotlib.use('TkAgg') # Use an interactive backend for plotting
# from matplotlib import pyplot as plt

from failure_prob.conf import Config
from failure_prob.utils.failure_metrics import compute_token_metrics, compute_sample_unc_metrics, compute_stac_metrics_single, compute_stac_metrics
from .utils import Rollout, set_task_min_step, split_rollouts_by_seen_unseen, process_tensor_idx_rel
from .openvla import split_rollouts as split_rollouts_openvla


def compute_hand_crafted_metrics(
    cfg: Config,
    policy_records: list[dict],
    exec_horizon: int,
) -> pd.DataFrame:
    '''
    Compute the hand crafted metrics for the pi0-fast rollouts
    '''
    metrics_raw = defaultdict(list)
    sampled_actions = []
    for i in range(len(policy_records)):
        this_record = policy_records[i]
        
        # metrics based on token logits
        if "logits" in this_record:
            logits = torch.from_numpy(this_record["logits"].astype(np.float32)) # (n_tokens, n_vocab)
            token_metrics = compute_token_metrics(logits)
            for k, v in token_metrics.items():
                metrics_raw[k].append(v)
            
        sampled_actions.append(torch.tensor(this_record["actions"]))
        
    # (n_steps, pred_horizon, action_dim) or (n_steps, n_sampled, pred_horizon, action_dim)
    sampled_actions = torch.stack(sampled_actions, dim=0) 
    
    if sampled_actions.ndim == 3:
        # Only one sampled action
        stac_single = compute_stac_metrics_single(sampled_actions, exec_horizon)
        metrics_raw['stac_single'] = stac_single.cpu()
    elif sampled_actions.ndim == 4:
        # Multiple sampled actions
        selected_actions = sampled_actions[:, 0, :, :] # (T, N, A)
        
        stac_single = compute_stac_metrics_single(selected_actions, exec_horizon)
        metrics_raw['stac_single'] = stac_single.cpu()
        
        cov_metrics = compute_sample_unc_metrics(sampled_actions)
        metrics_raw.update(cov_metrics)
        
        sampled_actions = sampled_actions.cuda()
        stac_mmd = compute_stac_metrics(
            sampled_actions, 
            exec_horizon=exec_horizon, 
            metric="mmd", 
            rbf_beta=cfg.dataset.rbf_beta
        )
        metrics_raw['stac_mmd'] = stac_mmd.cpu()
    else:
        raise ValueError(f"Unsupported sampled actions shape: {sampled_actions.shape}")

    metrics = {}
    for k, v in metrics_raw.items():
        metrics[k] = np.asarray(v) # (n_steps, )
        metrics[f"{k}_rmean"] = np.cumsum(np.asarray(v)) / (np.arange(len(v)) + 1) # (n_steps, )

        ## In LIBERO, the rollouts terminates early when the task is successfully completed. 
        ## Cumsum metrics gives advantages for failure detection to cheat and thus not proper to use. 
        # metrics[f"{k}_csum"] = np.cumsum(np.asarray(v)) # (n_steps, )

    return pd.DataFrame(metrics)


def load_rollouts_from_root(load_root: Path, cfg: Config) -> list[Rollout]:
    '''
    Saved dict
    observation/state <class 'numpy.ndarray'> (8,) float64
    prompt <class 'str'> put both the alphabet soup and the tomato sauce in the basket
    actions <class 'numpy.ndarray'> (10, 7) float64
    decode_step <class 'numpy.ndarray'> () int32
    encoded <class 'numpy.ndarray'> (17, 2048) bfloat16
    logits <class 'numpy.ndarray'> (17, 2048) bfloat16
    pre_logits <class 'numpy.ndarray'> (17, 2048) bfloat16
    raw_actions <class 'numpy.ndarray'> (17,) float32
    state <class 'numpy.ndarray'> (8,) float32
    action_start_index_in_vocab <class 'int'> 254976
    action_end_index_in_vocab <class 'int'> 257024
    '''
    env_records_folder = load_root / "env_records"
    policy_records_folder = load_root / "policy_records"
    
    assert env_records_folder.exists(), f"Path {env_records_folder} does not exist"
    assert policy_records_folder.exists(), f"Path {policy_records_folder} does not exist"
    
    env_record_paths = glob.glob(str(env_records_folder / "*.pkl"))
    policy_record_paths = glob.glob(str(policy_records_folder / "*meta.pkl"))
    
    assert len(env_record_paths) > 0, f"No env records found in {env_records_folder}"
    assert len(policy_record_paths) > 0, f"No policy records found in {policy_records_folder}"
    
    env_record_paths = natsort.natsorted(env_record_paths)
    policy_record_paths = natsort.natsorted(policy_record_paths)
    
    all_rollouts = []
    
    policy_step = 0

    for env_record_path in tqdm(env_record_paths, desc="Loading rollouts"):
        # Load the meta data from the env record
        env_record = pickle.load(open(env_record_path, "rb"))
        mp4_path = env_record_path.replace(".pkl", ".mp4")

        # Fix a typo in the dataset. 
        if "eposide_idx" in env_record:
            env_record['episode_idx'] = env_record['eposide_idx']
            del env_record['eposide_idx']
            
        # Load hidden features from corresponding policy records
        model_infer_times = env_record["model_infer_times"]
        policy_records = []
        for i in range(model_infer_times):
            policy_record_path = policy_record_paths[policy_step]
            policy_records.append(pickle.load(open(policy_record_path, "rb")))
            policy_step += 1

        # Extract hidden states and actions from policy records
        hidden_states = []
        action_vectors = []
        for policy_record in policy_records:
            hidden_state = policy_record[cfg.dataset.feat_name] # (n_tokens, dim_feat)
            
            # handle the token dimension
            # (n_tokens, dim_feat) -> (dim_feat)
            hidden_state = process_tensor_idx_rel(hidden_state, cfg.dataset.token_idx_rel)

            hidden_states.append(hidden_state)
            
            pred_horizon = policy_record["actions"].shape[0]
            dim_action = policy_record["actions"].shape[1]
            action = policy_record["actions"].reshape(-1) # (pred_horizon*action_dim)
            action_vectors.append(action)
        hidden_states = np.stack(hidden_states, axis=0).astype(np.float32)
        hidden_states = torch.from_numpy(hidden_states) # (n_steps, hidden_dim)
        action_vectors = np.stack(action_vectors, axis=0).astype(np.float32)
        action_vectors = torch.from_numpy(action_vectors) # (n_steps, pred_horizon*action_dim)
        
        cfg.dataset.dim_features = hidden_states.shape[-1]
        cfg.dataset.dim_action = dim_action
        cfg.dataset.pred_horizon = pred_horizon
        cfg.dataset.exec_horizon = env_record['replan_steps']

        # Compute hand-crafted metrics
        hand_crafted_metrics = None
        if cfg.train.log_precomputed or cfg.train.log_precomputed_only:
            hand_crafted_metrics = compute_hand_crafted_metrics(
                cfg=cfg,
                policy_records=policy_records,
                exec_horizon=env_record['replan_steps'],
            )
            if cfg.train.log_precomputed_only: 
                hidden_states = torch.zeros((hidden_states.shape[0], 1))
                action_vectors = None
        
        rollout = Rollout(
            hidden_states=hidden_states,
            task_suite_name=env_record["task_suite_name"],
            task_id=env_record["task_id"],
            task_description=env_record["task_description"],
            episode_idx=env_record["episode_idx"],
            episode_success=env_record["episode_success"],
            mp4_path=mp4_path,
            logs=hand_crafted_metrics,
            exec_horizon=env_record['replan_steps'],
            action_vectors=action_vectors,
        )
        all_rollouts.append(rollout)
    
    all_rollouts = set_task_min_step(all_rollouts)
    
    return all_rollouts


def load_rollouts(cfg: Config) -> list[Rollout]:
    load_root = Path(cfg.dataset.data_path)
    all_rollouts = load_rollouts_from_root(load_root, cfg)

    if cfg.dataset.data_path_unseen is not None:
        seen_rollouts = all_rollouts
        seen_task_ids = set([r.task_id for r in seen_rollouts])
        n_seen_tasks = len(seen_task_ids)
        print(f"Seen tasks: {seen_task_ids}")
        print(f"Number of seen tasks: {n_seen_tasks}")
        
        unseen_root = Path(cfg.dataset.data_path_unseen)
        unseen_rollouts = load_rollouts_from_root(unseen_root, cfg)
        
        for r in unseen_rollouts:
            r.task_id = r.task_id + n_seen_tasks

        unseen_task_ids = set([r.task_id for r in unseen_rollouts])
        n_unseen_tasks = len(unseen_task_ids)
        print(f"Unseen tasks: {unseen_task_ids}")
        print(f"Number of unseen tasks: {n_unseen_tasks}")
        
        # Overwrite the unseen_task_ratio in the config
        cfg.dataset.unseen_task_ratio = n_unseen_tasks / (n_seen_tasks + n_unseen_tasks)
        print(f"Overwrite unseen_task_ratio: {cfg.dataset.unseen_task_ratio}")
        
        all_rollouts = seen_rollouts + unseen_rollouts
    
    return all_rollouts


def split_rollouts(cfg: Config, all_rollouts: list[Rollout]) -> tuple[list[Rollout], list[Rollout]]:
    if cfg.dataset.data_path_unseen is None:
        rollouts_by_split_name = split_rollouts_openvla(cfg, all_rollouts)
    else:
        task_ids = list(set([r.task_id for r in all_rollouts]))
        task_ids = sorted(task_ids)
        n_unseen = round(cfg.dataset.unseen_task_ratio * len(task_ids))
        n_seen = len(task_ids) - n_unseen
        seen_task_ids = task_ids[:n_seen]
        unseen_task_ids = task_ids[n_seen:]
        
        rollouts_by_split_name = split_rollouts_by_seen_unseen(
            cfg, all_rollouts, seen_task_ids, unseen_task_ids
        )
        
    return rollouts_by_split_name