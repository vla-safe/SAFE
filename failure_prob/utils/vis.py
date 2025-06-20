import warnings
from matplotlib import pyplot as plt
import numpy as np

from failure_prob.data.utils import Rollout

def compute_mean_std(scores, max_length = None):
    '''
    Compute the mean and std of a list of lists of varying lengths
    
    Args:
        scores: List of lists
        max_length: The maximum length of the lists
    '''
    if len(scores) == 0:
        return np.array([]), np.array([])
    
    if max_length is None:
        max_length = max([len(s) for s in scores])
        
    scores_paded = np.full((len(scores), max_length), np.nan)
    for i, scores in enumerate(scores):
        scores_paded[i, :len(scores)] = scores
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(scores_paded, axis=0)
        std = np.nanstd(scores_paded, axis=0)
    return mean, std


def plot_curves(success_scores, fail_scores, ax, individual):
    if individual:
        for scores in success_scores:
            ax.plot(scores, color="green", alpha=0.5, label="Success")
        for scores in fail_scores:
            ax.plot(scores, color="red", alpha=0.5, label="Fail")
    else:
        max_length = max([len(s) for s in success_scores + fail_scores])
        
        success_mean, success_std = compute_mean_std(success_scores, max_length)
        ax.plot(success_mean, color="green", label="Success")
        ax.fill_between(range(max_length), success_mean - success_std, success_mean + success_std, color="green", alpha=0.3)

        fail_mean, fail_std = compute_mean_std(fail_scores, max_length)
        ax.plot(fail_mean, color="red", label="Fail")
        ax.fill_between(range(max_length), fail_mean - fail_std, fail_mean + fail_std, color="red", alpha=0.3)


def plot_scores_by_splits(
    scores_by_split_name: dict[str, list[np.ndarray]],
    rollouts_by_split_name: dict[str, list[Rollout]],
    individual=False
):
    n_axes = len(scores_by_split_name)
    fig, axes = plt.subplots(1, n_axes, figsize=(6*n_axes, 6))
    for i, (split_name, scores) in enumerate(scores_by_split_name.items()):
        ax = axes[i]
        rollouts = rollouts_by_split_name[split_name]
        assert len(scores) == len(rollouts)
        success_scores = [scores[i] for i in range(len(rollouts)) if rollouts[i].episode_success == 1]
        fail_scores = [scores[i] for i in range(len(rollouts)) if rollouts[i].episode_success == 0]
        plot_curves(success_scores, fail_scores, ax, individual)
        ax.set_xlabel("Time Step")
        ax.set_title(split_name)
        
    return fig, axes


def plot_roc_curves(
    data: list,
    method_name: str
):
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {method_name}")
    
    for fpr, tpr, name, roc_auc in data:
        plt.plot(fpr, tpr, label=f"{name}, AUC={roc_auc*100:.2f}")
        
    plt.legend()
    return fig


def plot_prc_curves(
    data: list,
    method_name: str
):
    fig = plt.figure()
    plt.plot([0, 1], [1, 0], linestyle="--", color="gray")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PRC Curve for {method_name}")
    
    for rec, pre, split, prc_auc in data:
        plt.plot(rec, pre, label=f"{split}, AUC={prc_auc*100:.2f}")
        
    plt.legend()
    return fig