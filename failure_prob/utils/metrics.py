from collections import defaultdict
from typing import Optional
import warnings
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score

import wandb

from failure_prob.data.utils import Rollout
from .vis import plot_roc_curves, plot_prc_curves, plot_scores_by_splits
from .conformal.split_cp import split_conformal_binary
from .conformal.functional_predictor import (
    RegressionType,
    ModulationType,
    FunctionalPredictor
)


EVAL_TIMES = [
    "at earliest stop",
    "by earliest stop",
    "by final end",
]

# Doing failure detection, 1 means failure, 0 means success
def compute_roc(success_scores, fail_scores):
    y_true = [1] * len(fail_scores) + [0] * len(success_scores)
    y_score = fail_scores + success_scores
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def compute_prc(success_scores, fail_scores):
    y_true = [1] * len(fail_scores) + [0] * len(success_scores)
    y_score = fail_scores + success_scores
    
    pre, rec, thresholds = precision_recall_curve(y_true, y_score)
    prc_auc = auc(rec, pre)

    return pre, rec, prc_auc


def compute_roc_by_quantiles(
    scores_full: list, 
    rollouts: list[Rollout], 
    time_quantiles: list[float] = [1.0], 
):
    fpr_by_time = {}
    tpr_by_time = {}
    auc_by_time = {}
    for q in time_quantiles:
        success_scores = [
            s[ round((r.task_min_step - 1) * q) ] 
            for s, r in zip(scores_full, rollouts)
            if r.episode_success == 1
        ]
        fail_scores = [
            s[ round((r.task_min_step - 1) * q) ] 
            for s, r in zip(scores_full, rollouts)
            if r.episode_success == 0
        ]
        
        fpr, tpr, roc_auc = compute_roc(success_scores, fail_scores)
        
        fpr_by_time[q] = fpr
        tpr_by_time[q] = tpr
        auc_by_time[q] = roc_auc
    
    return auc_by_time, fpr_by_time, tpr_by_time


def compute_prc_by_quantiles(
    scores_full: list, 
    rollouts: list[Rollout], 
    time_quantiles: list[float] = [1.0], 
):
    pre_by_time = {}
    rec_by_time = {}
    auc_by_time = {}
    for q in time_quantiles:
        success_scores = [
            s[ round((r.task_min_step - 1) * q) ] 
            for s, r in zip(scores_full, rollouts)
            if r.episode_success == 1
        ]
        fail_scores = [
            s[ round((r.task_min_step - 1) * q) ] 
            for s, r in zip(scores_full, rollouts)
            if r.episode_success == 0
        ]
        
        pre, rec, prc_auc = compute_prc(success_scores, fail_scores)
        
        pre_by_time[q] = pre
        rec_by_time[q] = rec
        auc_by_time[q] = prc_auc
    
    return auc_by_time, pre_by_time, rec_by_time


def get_metrics_curve(rollouts, key) -> list[np.ndarray]:
    return [r.logs[key].values for r in rollouts]


def eval_scores_roc_prc(
    rollouts_by_split_name: dict[str, list[Rollout]],
    scores_by_split_name: dict[str, list[np.ndarray]],
    method_name: str,
    time_quantiles: list[float],
    plot_auc_curves: bool = True,
    plot_score_curves: bool = True,
) -> list:
    to_be_logged = {}
    
    # Log the failure scores by plot all the curves
    if plot_score_curves:
        # Plot the score curves
        fig, axes = plot_scores_by_splits(scores_by_split_name, rollouts_by_split_name, individual=True)
        fig.suptitle(method_name)
        to_be_logged[f"failure_scores/{method_name}_indiv"] = fig
        plt.close(fig)
        
        fig, axes = plot_scores_by_splits(scores_by_split_name, rollouts_by_split_name, individual=False)
        fig.suptitle(method_name)
        to_be_logged[f"failure_scores/{method_name}_agg"] = wandb.Image(fig)
        plt.close(fig)
        
        
    # Evaluate ROC and PRC at different timesteps
    roc_curves_data = []
    prc_curves_data = []
    for split, rollouts in rollouts_by_split_name.items():
        scores = scores_by_split_name[split]
        # Compute ROC curves and AUC
        auc_by_time, fpr_by_time, tpr_by_time = compute_roc_by_quantiles(scores, rollouts, time_quantiles)
        for q in time_quantiles:
            roc_auc, fpr, tpr = auc_by_time[q], fpr_by_time[q], tpr_by_time[q]
            to_be_logged[f"roc_auc/{method_name}_{split}_tq{q}"] = roc_auc
        roc_curves_data.append((fpr, tpr, split, roc_auc))
        
        # Compute PRC curves and AUC
        auc_by_time, pre_by_time, rec_by_time = compute_prc_by_quantiles(scores, rollouts, time_quantiles)
        for q in time_quantiles:
            prc_auc, pre, rec = auc_by_time[q], pre_by_time[q], rec_by_time[q]
            to_be_logged[f"prc_auc/{method_name}_{split}_tq{q}"] = prc_auc
        prc_curves_data.append((rec, pre, split, prc_auc))
        
    # Plot the curves
    if plot_auc_curves:
        fig = plot_roc_curves(roc_curves_data, method_name)
        to_be_logged[f"roc_curve/{method_name}"] = fig
        plt.close(fig)
        
        fig = plot_prc_curves(prc_curves_data, method_name)
        to_be_logged[f"prc_curve/{method_name}"] = fig
        plt.close(fig)
        

    # Evaluate the ROC and PRC based on highest score so far
    roc_curves_data_early = []
    prc_curves_data_early = []
    roc_curves_data_end = []
    prc_curves_data_end = []
    for split, rollouts_split in rollouts_by_split_name.items():
        scores_split = scores_by_split_name[split]
        task_ids = sorted(list(set([rollout.task_id for rollout in rollouts_split])))
        
        task_metrics = defaultdict(list)
        
        for task_id in task_ids + ["all"]:
            with warnings.catch_warnings():
                if task_id != "all":
                    indices_task = [i for i, r in enumerate(rollouts_split) if r.task_id == task_id]
                    # It's possible that some task only have success or failure
                    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)
                else:
                    indices_task = range(len(rollouts_split))
                    
                rollouts_task = [rollouts_split[i] for i in indices_task]
                scores_task = [scores_split[i] for i in indices_task]
                labels_task = [1-r.episode_success for r in rollouts_task]
            
                # Compute the failure detection performance at earliest stop
                scores = [s[:r.task_min_step].max() for s, r in zip(scores_task, rollouts_task)]

                fpr, tpr, thresholds = roc_curve(labels_task, scores)
                roc_auc = auc(fpr, tpr)
                
                if task_id == "all":
                    to_be_logged[f'falert_early_roc_auc/{method_name}_{split}'] = roc_auc
                    roc_curves_data_early.append((fpr, tpr, split, roc_auc))
                else:
                    to_be_logged[f'falert_early_roc_auc_taskwise/{method_name}_{split}_{task_id}'] = roc_auc
                    task_metrics["falert_early_roc_auc_taskwise"].append(roc_auc)
                
                pre, rec, thresholds = precision_recall_curve(labels_task, scores)
                prc_auc = auc(rec, pre)
                
                if task_id == "all":
                    to_be_logged[f'falert_early_prc_auc/{method_name}_{split}'] = prc_auc
                    prc_curves_data_early.append((rec, pre, split, prc_auc))
                else:
                    to_be_logged[f'falert_early_prc_auc_taskwise/{method_name}_{split}_{task_id}'] = prc_auc
                    task_metrics["falert_early_prc_auc_taskwise"].append(prc_auc)
                
                # Compute the failure detection performance until the end
                scores = [s[:len(r.hidden_states)].max() for s, r in zip(scores_task, rollouts_task)]
                fpr, tpr, thresholds = roc_curve(labels_task, scores)
                roc_auc = auc(fpr, tpr)
                
                if task_id == "all":
                    to_be_logged[f'falert_end_roc_auc/{method_name}_{split}'] = roc_auc
                    roc_curves_data_end.append((fpr, tpr, split, roc_auc))
                else:
                    to_be_logged[f'falert_end_roc_auc_taskwise/{method_name}_{split}_{task_id}'] = roc_auc
                    task_metrics["falert_end_roc_auc_taskwise"].append(roc_auc)
                
                pre, rec, thresholds = precision_recall_curve(labels_task, scores)
                prc_auc = auc(rec, pre)
                
                if task_id == "all":
                    to_be_logged[f'falert_end_prc_auc/{method_name}_{split}'] = prc_auc
                    prc_curves_data_end.append((rec, pre, split, prc_auc))
                else:
                    to_be_logged[f'falert_end_prc_auc_taskwise/{method_name}_{split}_{task_id}'] = prc_auc
                    task_metrics["falert_end_prc_auc_taskwise"].append(prc_auc)
                
        for key, values in task_metrics.items():
            to_be_logged[f"{key}/{method_name}_{split}"] = np.mean(values)
        
        
    # Plot the curves
    if plot_auc_curves:
        fig = plot_roc_curves(roc_curves_data_early, method_name)
        to_be_logged[f"falert_early_roc_curve/{method_name}"] = fig
        plt.close(fig)
        
        fig = plot_prc_curves(prc_curves_data_early, method_name)
        to_be_logged[f"falert_early_prc_curve/{method_name}"] = fig
        plt.close(fig)
        
        fig = plot_roc_curves(roc_curves_data_end, method_name)
        to_be_logged[f"falert_end_roc_curve/{method_name}"] = fig
        plt.close(fig)

        fig = plot_prc_curves(prc_curves_data_end, method_name)
        to_be_logged[f"falert_end_prc_curve/{method_name}"] = fig
        plt.close(fig)
        
        
    return to_be_logged


def eval_binary_classification(
    scores: np.ndarray | list,
    labels: np.ndarray | list,
    threshold: float,
) -> dict[str, float]:
    '''
    Compute the metrics for a binary classification task.
    Compute TPR, TNR, Accuracy, F1 Score based on the given threshold.
    Also compute the ROC AUC and PRC AUC, which are agnostic to the threshold.
    Properly handle the case where there is only one class in the labels.
    
    Args:
        scores: classifier scores, shape (n_samples,), higher score means more likely to be positive.
        labels: GT labels, shape (n_samples,), 1 means positive, 0 means negative.
        threshold: The threshold for the binary classification.
    
    Returns:
        dict: A dictionary of the computed metrics, with keys {tpr, tnr, accuracy, f1, roc_auc, prc_auc}.
    '''
    if isinstance(scores, list):
        scores = np.array(scores)
    if isinstance(labels, list):
        labels = np.array(labels)
        
    pos_freq = np.sum(labels) / len(labels)
    neg_freq = 1 - pos_freq

    # Generate binary predictions using the threshold.
    preds = (scores >= threshold).astype(int)
    
    # Calculate confusion matrix components.
    TP = np.sum((preds == 1) & (labels == 1))
    FP = np.sum((preds == 1) & (labels == 0))
    TN = np.sum((preds == 0) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))
    
    # Compute TPR (Recall) and TNR.
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    
    # Compute Accuracy.
    acc = (TP + TN) / len(labels) if len(labels) > 0 else 0.0
    bal_acc = (tpr + tnr) / 2
    weighted_acc = (tpr * neg_freq + tnr * pos_freq) # Weighted by the inverse class frequency
    
    # Compute Precision.
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # Compute F1 Score.
    f1 = (2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0
    
    # Compute ROC AUC and PRC AUC, handling the case of a single class.
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        roc_auc = float('nan')
        prc_auc = float('nan')
    else:
        roc_auc = roc_auc_score(labels, scores)
        prc_auc = average_precision_score(labels, scores)
    
    # Return the computed metrics.
    return {
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "acc": acc,
        "bal_acc": bal_acc,
        "f1": f1,
        "weighted-acc": weighted_acc,
        "roc_auc": roc_auc,
        "prc_auc": prc_auc,
    }

    
def eval_detection_time(
    scores: list[np.ndarray],
    labels: np.ndarray,
    threshold: float,
) -> float:
    '''
    Evaluate the earliest detection time, which is the earliest timestep that a score exceeds the threshold.
    Each time series in scores is labelled 1 or 0. A time series is classified as positive if its score at any time
    exceeds the threshold. This function returns the average detection time for the true positive time series.

    Args:
        scores: List of classifier scores (length n_samples), each a numpy array of shape (n_timesteps,).
        labels: Ground truth labels, numpy array of shape (n_samples,), where 1 indicates positive.
        threshold: The threshold above which a time point is considered a detection.

    Returns:
        A dictionary with a single key "avg_det_time" whose value is the average detection time
        (i.e., the earliest timestep index at which the score exceeds the threshold) for all
        time series that are labeled positive and for which a detection occurs.
        If no positive series are detected, returns NaN.
    '''
    detection_times = []

    # Loop over each time series and its corresponding ground truth label.
    for score, label in zip(scores, labels):
        if label == 1:
            # Find indices where score exceeds (or equals) the threshold.
            detection_indices = np.where(score >= threshold)[0]
            if detection_indices.size > 0:
                # Record the first occurrence (earliest detection time).
                detection_times.append(detection_indices[0] / len(score))
            else:
                # If not detected, record the maximum time (1.0).
                detection_times.append(1.0)

    # Calculate average detection time if there are any detections.
    if detection_times:
        avg_det_time = sum(detection_times) / len(detection_times)
    else:
        avg_det_time = float('nan')  # No detections for positive samples.

    return avg_det_time
    

def eval_fixed_threshold(
    rollouts_by_split_name: dict[str, list[Rollout]],
    scores_by_split_name: dict[str, list[np.ndarray]],
    method_name: str,
    thresholds: list[float] = [0.5],
) -> pd.DataFrame:
    classification_logs = []
    
    # Compute the metrics given a fixed threshold
    for split_name in rollouts_by_split_name:
        rollouts = rollouts_by_split_name[split_name]
        scores_all = scores_by_split_name[split_name]
        labels = [1-r.episode_success for r in rollouts]
        
        for eval_time in EVAL_TIMES:
            if eval_time == "at earliest stop":
                scores = [s[r.task_min_step - 1] for s, r in zip(scores_all, rollouts)]
            elif eval_time == "by earliest stop":
                scores = [s[:r.task_min_step].max() for s, r in zip(scores_all, rollouts)]
            elif eval_time == "by final end":
                scores = [s[:len(r.hidden_states)].max() for s, r in zip(scores_all, rollouts)]
            else:
                raise ValueError(f"Unknown eval_time: {eval_time}")

            for thresh in thresholds:
                avg_det_time = eval_detection_time(scores_all, labels, thresh)
                result = eval_binary_classification(scores, labels, thresh)
                classification_logs.append({
                    "detect_method": method_name,
                    "split": split_name,
                    "task": "all",
                    "thresh_method": "fixed",
                    "time": eval_time,
                    "threshold": thresh,
                    "avg_det_time": avg_det_time,
                    **result,
                })

    df = pd.DataFrame(classification_logs)
    return df


def eval_split_conformal(
    rollouts_by_split_name: dict[str, list[Rollout]],
    scores_by_split_name: dict[str, list[np.ndarray]],
    method_name: str,
    alphas: Optional[list[float]] = None,
    calib_split_names: list[str] = ["val_seen"],
    test_split_names: list[str] = ["val_unseen"],
) -> list[dict]:
    if alphas is None:
        alphas = [0.02] + [0.05 * i for i in range(1, 10)] + [0.5, 0.6, 0.7, 0.8, 0.9]
        
    classification_logs = []
    # Construct data for calibration and test sets
    cal_rollouts, cal_scores_all = [], []
    for split_name in calib_split_names:
        cal_rollouts.extend(rollouts_by_split_name[split_name])
        cal_scores_all.extend(scores_by_split_name[split_name])
    cal_labels = [1-r.episode_success for r in cal_rollouts]

    test_rollouts, test_scores_all = [], []
    for split_name in test_split_names:
        test_rollouts.extend(rollouts_by_split_name[split_name])
        test_scores_all.extend(scores_by_split_name[split_name])
    test_labels = [1-r.episode_success for r in test_rollouts]
    
    for eval_time in EVAL_TIMES:
        if eval_time == "at earliest stop":
            cal_scores = [s[r.task_min_step - 1] for s, r in zip(cal_scores_all, cal_rollouts)]
            test_scores = [s[r.task_min_step - 1] for s, r in zip(test_scores_all, test_rollouts)]
        elif eval_time == "by earliest stop":
            cal_scores = [s[:r.task_min_step].max() for s, r in zip(cal_scores_all, cal_rollouts)]
            test_scores = [s[:r.task_min_step].max() for s, r in zip(test_scores_all, test_rollouts)]
        elif eval_time == "by final end":
            cal_scores = [s[:len(r.hidden_states)].max() for s, r in zip(cal_scores_all, cal_rollouts)]
            test_scores = [s[:len(r.hidden_states)].max() for s, r in zip(test_scores_all, test_rollouts)]
        else:
            raise ValueError(f"Unknown eval_time: {eval_time}")
            
        for alpha in alphas:
            test_pred_sets, thresholds = split_conformal_binary(cal_scores, cal_labels, test_scores, alpha)
            for calib_label in ['pos', 'neg']:
                if calib_label == 'pos':
                    thresh_pos = 1 - thresholds[1]
                else:
                    thresh_pos = thresholds[0]

                result = eval_binary_classification(test_scores, test_labels, thresh_pos)
                result['avg_det_time'] = eval_detection_time(test_scores_all, test_labels, thresh_pos)
                classification_logs.append({
                    "detect_method": method_name,
                    "cal split": f"{'+'.join(calib_split_names)}",
                    "test split": f"{'+'.join(test_split_names)}",
                    "calib on": calib_label,
                    "task": "all",
                    "thresh_method": f"split CP, cal on {'+'.join(calib_split_names)}",
                    "alpha": alpha,
                    "time": eval_time,
                    **result,
                    "threshold": thresh_pos,
                })
        
    return classification_logs


def eval_functional_conformal(
    rollouts_by_split_name: dict[str, list[Rollout]],
    scores_by_split_name: dict[str, list[np.ndarray]],
    method_name: str,
    alphas: Optional[list[float]] = None,
    calib_split_names: list[str] = ["val_seen"],
    test_split_names: list[str] = ["val_unseen"],
    align_method: str = "extend",
) -> tuple[pd.DataFrame, dict]:
    if alphas is None:
        # alphas = [0.02] + [0.05 * i for i in range(1, 10)] + [0.5, 0.6, 0.7, 0.8, 0.9]
        alphas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]
    classification_logs = []
    
    cal_rollouts, cal_scores_all = [], []
    for split_name in calib_split_names:
        cal_rollouts.extend(rollouts_by_split_name[split_name])
        cal_scores_all.extend(scores_by_split_name[split_name])
    cal_labels_all = np.asarray([1-r.episode_success for r in cal_rollouts])

    test_rollouts, test_scores_all = [], []
    for split_name in test_split_names:
        test_rollouts.extend(rollouts_by_split_name[split_name])
        test_scores_all.extend(scores_by_split_name[split_name])
    test_labels_all = np.asarray([1-r.episode_success for r in test_rollouts])
    
    # Handle the different lengths of predicted scores. 
    test_earliest_stop = np.array([r.task_min_step for r in test_rollouts]) # (N,)
    if align_method == "extend":
        # Extend the early-stoping scores with the last value
        max_length = max(len(s) for s in cal_scores_all + test_scores_all)
        for i, s in enumerate(cal_scores_all):
            cal_scores_all[i] = np.pad(s, (0, max_length - len(s)), mode='edge')
        for i, s in enumerate(test_scores_all):
            test_scores_all[i] = np.pad(s, (0, max_length - len(s)), mode='edge')
    elif align_method == "truncate":
        # Truncate the early-stoping scores to the same length
        raise NotImplementedError("Truncate alignment is not implemented yet.")
    else:
        raise ValueError(f"Unknown align_method: {align_method}")
    
    for calib_on in ['neg']:
        if calib_on == "neg": # Calibration on the successful rollouts
            lower_bound = False
            cal_scores_used = [s for s, r in zip(cal_scores_all, cal_rollouts) if r.episode_success == 1]
        else: # Calibration on the failed rollouts
            lower_bound = True
            cal_scores_used = [s for s, r in zip(cal_scores_all, cal_rollouts) if r.episode_success == 0]
            raise NotImplementedError("Functional CP calibrated on failures does not make sense. .")
        
        # Split the cal scores evenly randomly into two set (for regression and modulation respectively)
        cal_scores_used = np.array(cal_scores_used)
        if len(cal_scores_used) == 1:
            cal_scores_1 = cal_scores_used
            cal_scores_2 = cal_scores_used
        else:
            np.random.shuffle(cal_scores_used)
            n_cal_1 = int(len(cal_scores_used) * 0.3) # 30% according to Chen's implementation
            cal_scores_1 = cal_scores_used[:n_cal_1]
            cal_scores_2 = cal_scores_used[n_cal_1:]
        
        # Compute the conformal prediction band
        test_scores_all = np.array(test_scores_all) # (N, T)
        n_test_samples = len(test_scores_all)
        
        cp_bands_by_alpha = {}
        
        for eval_time in ['by final end', 'by earliest stop']:
            for alpha in alphas:
                predictor = FunctionalPredictor(ModulationType.Tfunc, RegressionType.Mean)
                cp_band = predictor.get_one_sided_prediction_band(
                    cal_scores_1, cal_scores_2, alpha, lower_bound=lower_bound)
                
                cp_bands_by_alpha[alpha] = cp_band
            
                # Flag is raised if the test score is out of the band. 
                if lower_bound: detection_mask = test_scores_all <= cp_band # (N, T)
                else:           detection_mask = test_scores_all >= cp_band # (N, T)
            
                # Handle different evaluation time modes    
                if eval_time == "by final end":
                    lengths = test_scores_all.shape[1] # scalar, T
                elif eval_time == "by earliest stop":
                    lengths = test_earliest_stop # (N,)
                    # After the earliest stop, no more detection is possible. 
                    for i in range(len(test_scores_all)):
                        detection_mask[i, lengths[i]:] = False
                else:
                    raise ValueError(f"Unknown eval_time: {eval_time}")
            
                has_detection = np.any(detection_mask, axis=1) # (N,)
                first_detection = np.argmax(detection_mask, axis=1) # (N,)
                detection_times = np.where(has_detection, first_detection, lengths) # (N,)
                relative_detection_times = detection_times / lengths # (N,)

                # Compute detection time and classification metrics
                pos_mask = test_labels_all == 1 # (N,)
                avg_det_time = np.mean(relative_detection_times[pos_mask])
                predicted = has_detection # (N,)
                tp = (predicted & pos_mask).sum()
                fn = (~predicted & pos_mask).sum()
                fp = (predicted & ~pos_mask).sum()
                tn = (~predicted & ~pos_mask).sum()
                
                # Safe division for metrics
                with np.errstate(divide='ignore', invalid='ignore'):
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                    acc = (tp + tn) / n_test_samples
                    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
                    bal_acc = (tpr + tnr) / 2
                    
                classification_logs.append({
                    "detect_method": method_name,
                    "cal split": f"{'+'.join(calib_split_names)}",
                    "test split": f"{'+'.join(test_split_names)}",
                    "calib on": calib_on,
                    "task": "all",
                    "thresh_method": "functional CP",
                    "alpha": alpha,
                    "time": eval_time,
                    "avg_det_time": avg_det_time,
                    "tpr": tpr,
                    "tnr": tnr,
                    "fpr": fpr,
                    "fnr": fnr,
                    "acc": acc,
                    "bal_acc": bal_acc,
                    "f1": f1,
                })
        
    df = pd.DataFrame(classification_logs)
    
    return df, cp_bands_by_alpha
    
    


def eval_det_time_vs_classification(rollouts: list, scores: list[np.ndarray], labels: np.ndarray) -> list[dict]:
    '''
    We consider a time series as positive if its score exceeds a threshold at any time,
    and negative if its scores remain below the threshold.
    By varying the threshold, the detection time and classification performance (accuracy, F1 score, etc)
    change. This function computes the curve of detection time vs classification performance.
    A good detection method should retain an early detection time while maintaining high classification performance.
    
    Args:
        rollouts: List of rollouts, length n_samples.
        scores: List of classifier scores (length n_samples), each a numpy array of shape (n_timesteps,).
        labels: Ground truth labels, numpy array of shape (n_samples,), where 1 indicates positive.
        
    Returns:
        A list of dictionaries, each containing the detection time and classification performance at a threshold.
        Example 
            [{
                "threshold": float,
                "avg_det_time": float,   # relative detection time in [0,1] computed over positive samples only
                "tpr": float,
                "tnr": float,
                "acc": float,
                "f1": float,
            },
            ...
            ]
    '''
    # Normalize all scores to [0, 1] using the global min and max.
    all_scores = np.concatenate(scores)
    min_score, max_score = all_scores.min(), all_scores.max()
    normalized_scores = [(s - min_score) / (max_score - min_score + 1e-5) for s in scores]
    
    # --- Construct thresholds ---
    threshold_set = set()
    
    # 1. Quantile-based thresholds: 101 quantiles (including 0th and 100th percentiles)
    quantile_levels = np.linspace(0, 100, 101)
    quantile_thresholds = np.percentile(np.concatenate(normalized_scores), quantile_levels)
    threshold_set.update(quantile_thresholds.tolist())
    
    # 2. Explicitly include endpoints 0 and 1.
    threshold_set.add(-0.01)
    threshold_set.add(1.01)
    
    # 3. Include the midpoint of each time series ((min + max) / 2)
    for s in normalized_scores:
        mid = (s.min() + s.max()) / 2.0
        threshold_set.add(mid)
    
    # Convert to a sorted numpy array
    thresholds = np.array(sorted(threshold_set))  # shape (n_thresh,)
    
    n_samples = len(scores)
    lengths = np.array([len(s) for s in normalized_scores])  # actual lengths of each time series
    max_length = lengths.max()
    
    # --- Pad scores into a 2D tensor ---
    # Pad with -1 so that padded positions (outside the valid data) never trigger detection.
    padded_scores = np.full((n_samples, max_length), -1.0)
    for i, s in enumerate(normalized_scores):
        padded_scores[i, :len(s)] = s
    
    # --- Vectorized detection time computation ---
    # Compare every time point in every series against every threshold.
    # Shape transformations:
    # - padded_scores: (n_samples, max_length)
    # - thresholds: (n_thresh,)
    # We create a boolean tensor with shape: (n_samples, n_thresh, max_length)
    detection_mask = padded_scores[:, np.newaxis, :] >= thresholds[np.newaxis, :, np.newaxis]
    
    # For each sample and threshold, check if any detection occurred.
    has_detection = np.any(detection_mask, axis=2)  # shape: (n_samples, n_thresh)
    
    # Get the first index where detection occurs; if none, np.argmax returns 0 by default.
    first_detection = np.argmax(detection_mask, axis=2)  # shape: (n_samples, n_thresh)
    
    # If no detection occurred, set detection time to the actual length of the time series.
    detection_times = np.where(has_detection, first_detection, lengths[:, np.newaxis]) # shape: (n_samples, n_thresh)
    
    # --- Compute relative detection times for ground truth positive samples ---
    # Divide by each sample's original length so the value is in [0,1].
    relative_detection_times = detection_times / lengths[:, np.newaxis]
    
    # Only consider ground truth positives for the average detection time.
    pos_mask = labels == 1 # shape: (n_samples,)
    avg_det_time = np.mean(relative_detection_times[pos_mask, :], axis=0) # shape: (n_thresh,)
    
    # --- Compute classification metrics ---
    # A time series is predicted positive if any time point exceeds the threshold.
    predicted = has_detection  # shape: (n_samples, n_thresh)
    
    pos_samples = pos_mask
    neg_samples = ~pos_mask
    
    tp = np.sum(predicted[pos_samples, :], axis=0)  # true positives for each threshold
    fn = np.sum(~predicted[pos_samples, :], axis=0)   # false negatives for each threshold
    fp = np.sum(predicted[neg_samples, :], axis=0)    # false positives for each threshold
    tn = np.sum(~predicted[neg_samples, :], axis=0)   # true negatives for each threshold
    
    # Safe division for metrics
    with np.errstate(divide='ignore', invalid='ignore'):
        tpr = np.where((tp + fn) > 0, tp / (tp + fn), 0) 
        tnr = np.where((tn + fp) > 0, tn / (tn + fp), 0)
        acc = (tp + tn) / n_samples
        f1 = np.where((2 * tp + fp + fn) > 0, 2 * tp / (2 * tp + fp + fn), 0)
        bal_acc = (tpr + tnr) / 2
    
    # --- Build results ---
    results = []
    for j, th in enumerate(thresholds):
        th_ori = th * (max_score - min_score) + min_score
        results.append({
            "threshold": float(th_ori),
            "avg_det_time": float(avg_det_time[j]),  # average relative detection time over positive samples
            "tpr": float(tpr[j]),
            "tnr": float(tnr[j]),
            "acc": float(acc[j]),
            "bal_acc": float(bal_acc[j]),
            "f1": float(f1[j]),
        })
    
    return results




