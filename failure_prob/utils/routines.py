import json
import os
from typing import Optional
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
import wandb

from failure_prob.conf import Config
from failure_prob.model.base import BaseModel

# Use a non-interactive backend for multiprocessing (if needed)
import matplotlib

from failure_prob.utils.torch import move_to_device
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from failure_prob.data.utils import Rollout
import cv2
from .metrics import (
    eval_fixed_threshold,
    eval_scores_roc_prc, 
    get_metrics_curve, 
    eval_split_conformal,
    eval_functional_conformal,
    eval_det_time_vs_classification,
)


def model_forward_dataloader(
    model: BaseModel,
    loader: DataLoader,
):
    device = model.get_device()
    
    scores = []
    valid_masks = []
    labels = []
    
    for batch in loader:
        batch = move_to_device(batch, device)
        scores.append(model(batch))
        valid_masks.append(batch["valid_masks"])
        labels.append(batch["success_labels"])
        
    scores = torch.cat(scores, dim=0).squeeze(-1)
    valid_masks = torch.cat(valid_masks, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return scores, valid_masks, labels


def eval_metrics_and_log(
    cfg: Config,
    rollouts_by_split_name: dict[str, list[Rollout]],
    metric_keys: Optional[list[str]] = None, 
    time_quantiles: list[str] = [1.0]
):
    to_be_logged = {}
    classification_logs = []
    
    if metric_keys is None:
        metric_keys = rollouts_by_split_name['train'][0].logs.columns
        
    for metric_key in metric_keys:
        if metric_key not in rollouts_by_split_name['train'][0].logs.columns:
            print(f"Skipping {metric_key}")
            continue
        
        metric_name = metric_key.split("/")[-1]
        # scores_by_split_name will be a dict: split name -> list of np arrays
        scores_by_split_name = {
            k: get_metrics_curve(v, metric_key) 
            for k, v in rollouts_by_split_name.items()
        }
        
        #### Evaluate ROC and PRC metrics at certain timesteps ####
        metrics_logs = eval_scores_roc_prc(
            rollouts_by_split_name, 
            scores_by_split_name, 
            metric_name, 
            time_quantiles,
            plot_auc_curves=True,
            plot_score_curves=True,
        )
        to_be_logged.update(metrics_logs)
        
        #### Evaluate the classification performance using different thresholding methods ####
        # Split Conformal Prediction: val_seen for calibration, val_unseen for testing
        # Here we only use val_seen for calibration, to make it comparable to learned methods
        split_cp_logs = eval_split_conformal(
            rollouts_by_split_name, scores_by_split_name, metric_name,
            calib_split_names=["val_seen"], test_split_names=["val_unseen"]
        )
        split_cp_logs = pd.DataFrame(split_cp_logs)
        to_be_logged[f"classify_cp_maxsofar/{metric_name}"] = wandb.Table(dataframe=split_cp_logs)
        
        # Functional Conformal Prediction
        df, cp_bands_by_alpha = eval_functional_conformal(
            rollouts_by_split_name, scores_by_split_name, metric_name,
            calib_split_names=["val_seen"], test_split_names=["val_unseen"]
        )
        to_be_logged[f"classify_cp_functional/{metric_name}"] = wandb.Table(dataframe=df)
        
        # Compute the classification metrics vs. detection time
        df, logs = eval_perf_det_time_curves(
            rollouts_by_split_name, scores_by_split_name, metric_name
        )
        to_be_logged.update(logs)
        
        if cfg.train.eval_save_logs:
            os.makedirs(cfg.train.logs_save_path, exist_ok=True)
            df.to_csv(f"{cfg.train.logs_save_path}/{metric_name}_perf_vs_det.csv", index=False)
        
    # Convert the classification logs to a wandb table
    classification_logs = pd.DataFrame(classification_logs)
    to_be_logged["classify/metrics"] = wandb.Table(dataframe=classification_logs)
    
    return to_be_logged


def eval_model_and_log(
    cfg: Config,
    model: BaseModel, 
    rollouts_by_split_name: dict[str, list[Rollout]],
    dataloader_by_split_name: dict[str, DataLoader],
    eval_time_quantiles: list[float], 
    plot_auc_curves: bool = True,
    plot_score_curves: bool = True,
    log_classification_metrics: bool = True,
):
    to_be_logged = {}
    method_name = "model"
    
    #### Forward the model and compute the scores ####
    scores_by_split_name = {}
    for split, dataloader in dataloader_by_split_name.items():
        # Re-create a dataset to disable shuffling
        dataloader = DataLoader(dataloader.dataset, batch_size=cfg.model.batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            scores, valid_masks, _ = model_forward_dataloader(model, dataloader)
        scores = scores.detach().cpu().numpy()
        seq_lengths = valid_masks.sum(dim=-1).cpu().numpy() # (B,)
        scores_by_split_name[split] = [scores[i, :int(seq_lengths[i])] for i in range(len(seq_lengths))]

    #### Evaluate ROC and PRC metrics at certain timesteps ####
    roc_rpc_logs = eval_scores_roc_prc(
        rollouts_by_split_name, 
        scores_by_split_name, 
        method_name,
        eval_time_quantiles,
        plot_auc_curves, 
        plot_score_curves
    )
    to_be_logged.update(roc_rpc_logs)
    
    #### Evaluate the classification performance using different thresholding methods ####
    if log_classification_metrics:
        # Compute the metrics at fixed thresholds
        df = eval_fixed_threshold(
            rollouts_by_split_name, scores_by_split_name, method_name,
        )
        to_be_logged[f"classify_fixed_thresh/{method_name}"] = wandb.Table(dataframe=df)

        # Split Conformal Prediction: val_seen for calibration, val_unseen for testing
        split_cp_logs = eval_split_conformal(
            rollouts_by_split_name, scores_by_split_name, method_name,
            calib_split_names=["val_seen"], test_split_names=["val_unseen"]
        )
        split_cp_logs = pd.DataFrame(split_cp_logs)
        to_be_logged[f"classify_cp_maxsofar/{method_name}"] = wandb.Table(dataframe=split_cp_logs)
        
        # Functional Conformal Prediction
        df, cp_bands_by_alpha = eval_functional_conformal(
            rollouts_by_split_name, scores_by_split_name, method_name,
            calib_split_names=["val_seen"], test_split_names=["val_unseen"]
        )
        to_be_logged[f"classify_cp_functional/{method_name}"] = wandb.Table(dataframe=df)
        
        # Compute the classification metrics vs. detection time, by varying thresholds
        df, logs = eval_perf_det_time_curves(
            rollouts_by_split_name, scores_by_split_name, method_name
        )
        to_be_logged.update(logs)

        if cfg.train.eval_save_logs:
            os.makedirs(cfg.train.logs_save_path, exist_ok=True)
            df.to_csv(f"{cfg.train.logs_save_path}/{method_name}_perf_vs_det.csv", index=False)
    
    return to_be_logged


def eval_perf_det_time_curves(
    rollouts_by_split_name: dict[str, list[Rollout]],
    scores_by_split_name: dict[str, list[np.ndarray]],
    method_name: str,
):
    # Compute the classification metrics vs. detection time
    dfs = []
    logs = {}
    fig, ax = plt.subplots()
    y_key = "bal_acc"
    
    for split_name in rollouts_by_split_name:
        rollouts = rollouts_by_split_name[split_name]
        scores_all = scores_by_split_name[split_name]
        labels = np.asarray([1-r.episode_success for r in rollouts])
        results = eval_det_time_vs_classification(
            rollouts, scores_all, labels
        )
        df = pd.DataFrame(results)
        df['method_name'] = method_name
        df['split_name'] = split_name
        dfs.append(df)
        
        df.plot(x="avg_det_time", y=y_key, ax=ax, label=f"{split_name}")

    ax.set_xlabel("Mean detection time of GT failure")
    ax.set_ylabel(y_key)
    fig.tight_layout()
    logs[f"perf_vs_det/{method_name}_{y_key}_vs_Tdet"] = fig
    plt.close(fig)

    dfs = pd.concat(dfs)
    return dfs, logs


def eval_save_timing_plots(
    cfg: Config, 
    model: BaseModel, 
    rollouts_by_split_name: dict[str, list[Rollout]],
    dataloader_by_split_name: dict[str, DataLoader],
    save_folder: str,
    alpha: float = 0.2,
    calib_split_names=["val_seen"], 
    test_split_names=["val_unseen"],
):
    # Load the failure timestep annotations
    label_path = str(cfg.dataset.failure_time_label_path)
    assert os.path.exists(label_path), f"Label file not found: {label_path}"
    labels = json.load(open(label_path, "r"))
    
    # Convert the labels to a dict, indexed by task id, episode_id
    labels_dict = {
        (r['task_id'], r['episode_id']): r for r in labels
    }
    
    #### Forward the model and compute the scores ####
    scores_by_split_name = {}
    for split, dataloader in dataloader_by_split_name.items():
        # Re-create a dataset to disable shuffling.
        dataloader = DataLoader(
            dataloader.dataset, 
            batch_size=cfg.model.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        with torch.no_grad():
            scores, valid_masks, _ = model_forward_dataloader(model, dataloader)
        scores = scores.detach().cpu().numpy()
        seq_lengths = valid_masks.sum(dim=-1).cpu().numpy()  # (B,)
        scores_by_split_name[split] = [
            scores[i, :int(seq_lengths[i])] for i in range(len(seq_lengths))
        ]
        
    df, cp_bands_by_alpha = eval_functional_conformal(
        rollouts_by_split_name, scores_by_split_name, "model",
        calib_split_names=calib_split_names, test_split_names=test_split_names
    )
    
    # Retrieve the CP band for the given alpha.
    cp_band = cp_bands_by_alpha[alpha][0]  # Shape: (T,)
    
    # Gather test rollouts and their corresponding scores.
    test_rollouts = sum([rollouts_by_split_name[k] for k in test_split_names], [])
    test_scores = sum([scores_by_split_name[k] for k in test_split_names], [])
    
    # Determine detection times for each rollout
    records = []
    for i, r in enumerate(test_rollouts):
        score = test_scores[i]
        task_id, episode_idx = r.task_id, r.episode_idx

        # Whether the rollouts is a failure or not
        gt_fail_flag = not bool(r.episode_success)

        detection_mask = score > cp_band[:len(score)]
        # detection_mask = score[:r.task_min_step] >= cp_band[:r.task_min_step]
        
        pred_fail_flag = detection_mask.any() if len(detection_mask) > 0 else False
        
        # predicted failure time
        pred_fail_time = np.argmax(detection_mask) if pred_fail_flag else 2 * len(score)
        pred_fail_time_rel = pred_fail_time / len(score)

        if (task_id, episode_idx) not in labels_dict:
            if gt_fail_flag:
                print(f"Label not found for task_id {task_id}, episode_idx {episode_idx}")
            gt_fail_frame = 0
        else:
            label = labels_dict[(task_id, episode_idx)]
            gt_fail_frame = label["frame"]
        
        # Get the total number of frames from mp4_path
        cap = cv2.VideoCapture(r.mp4_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        gt_fail_time_rel = gt_fail_frame / total_frames
        
        records.append({
            "task_id": task_id,
            "episode_idx": episode_idx,
            "gt_fail_flag": gt_fail_flag,
            "pred_fail_flag": pred_fail_flag,
            "gt_fail_time_rel": gt_fail_time_rel,
            "pred_fail_time_rel": pred_fail_time_rel,
        })
        
    df = pd.DataFrame(records)
    
    # Save the dataframe to a CSV file
    save_path = os.path.join(save_folder, f"timing_data_alpha_{alpha}.csv")
    os.makedirs(save_folder, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved timing data to {save_path}")
    
    # Plot the scatter of predicted vs GT failure time
    fig, ax = plt.subplots(figsize=(3,3), dpi=300)
    df_gt_fail = df[df["gt_fail_flag"]]
    df_tp = df_gt_fail[df_gt_fail["pred_fail_flag"]]
    df_fn = df_gt_fail[~df_gt_fail["pred_fail_flag"]]
    ax.scatter(
        df_tp["gt_fail_time_rel"], 
        df_tp["pred_fail_time_rel"], 
        s=9,
        alpha=0.8, 
        label="TP",
        color="red",
    )
    ax.scatter(
        df_fn["gt_fail_time_rel"], 
        [1.0] * len(df_fn), 
        s=9,
        alpha=0.8, 
        marker="x",
        label="FN",
    )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("GT failure time (relative)")
    ax.set_ylabel("Detected failure time\n(relative)")
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, f"timing_plot_alpha_{alpha}.pdf")
    os.makedirs(save_folder, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved timing plot to {save_path}")
    
    # Plot the cumulative GT and detected failure over time
    gt_fail_time_rel = df_gt_fail["gt_fail_time_rel"].values
    pred_fail_time_rel = df_gt_fail["pred_fail_time_rel"].values
    fig, ax = plt.subplots(figsize=(4.5,3), dpi=300)
    
    x = np.linspace(0, 1, 100)
    y_gt = np.array([np.mean(gt_fail_time_rel <= t) for t in x])
    y_pred = np.array([np.mean(pred_fail_time_rel <= t) for t in x])
    ax.plot(x, y_pred, label="TP", color="red")
    ax.plot(x, y_gt, label="GT Positives", color="blue")
    ax.set_xlabel("Time (relative)")
    ax.set_ylabel("Cumulative failures\n(proportion)")
    ax.legend()
    plt.tight_layout()
    
    save_path = os.path.join(save_folder, f"cumulative_failures_alpha_{alpha}.pdf")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved cumulative failure plot to {save_path}")