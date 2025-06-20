import argparse
import itertools
import logging
import os, math

from glob import glob

from pathlib import Path
import json, csv

from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import pandas as pd
import wandb

from failure_prob.utils.wandb import load_summary_tables_from_runs
from failure_prob.utils.constants import (
    METHOD_NAME_2_DISPLAY_NAME, 
    METHOD_NAME_2_GROUP_ID,
    METHOD_NAME_2_MARKER,
    EXP_NAME_2_DISPLAY_NAME,
)
from failure_prob.utils.figure import get_method_colors

METRIC_KEY_TO_DISPLAY_NAME = {
    "bal_acc": "Bal-acc: Balanced Accuracy",
    "avg_det_time": "Average Detection Time (Normalized)",
    "f1": "F1 Score",
    "tpr": "TPR: True Positive Rate",
    "tnr": "TNR: True Negative Rate",
    "fpr": "FPR: False Positive Rate",
    "fnr": "FNR: False Negative Rate",
    "alpha": "Significance Level $\\alpha$"
}

SEED_2_UNSEEN_TASK_IDS_LIBERO = {
    0: [3, 0, 5],
    1: [7, 8, 5], 
    2: [6, 9, 8],
}
SEED_2_UNSEEN_TASK_IDS_SIMPLER = {
    0: [0],
    1: [1],
    2: [0],
}


CSV_PATHS_V2 = [
    "notebooks/wandb_metrics_batch_v2/openvla_libero_v2.csv",
    "notebooks/wandb_metrics_batch_v2/opi0_simpler_v1.csv",
    "notebooks/wandb_metrics_batch_v2/pi0fast_libero_v4.csv",
    "notebooks/wandb_metrics_batch_v2/pi0diff_libero_v1.csv",
]

CSV_PATHS = {
    "v2": CSV_PATHS_V2,
}


def read_aggregated_training_results_v1(
    csv_path: str,
):
    df_best = pd.read_csv(csv_path)

    df_best['subset'] = "default"
    df_best['seed'] = 0
    if "openvla" in os.path.basename(csv_path):
        df_best = df_best.rename(columns={
            "10-_entity": "_entity",
            "10-_project": "_project",
            "10-_id": "_id",
        })
    elif "simpler" in os.path.basename(csv_path):
        df_new = []
        for subset in ["bridge", "fractal"]:
            df_best_subset = df_best.copy()
            df_best_subset["subset"] = subset
            df_best_subset = df_best_subset.rename(columns={
                f"{subset}-_entity": "_entity",
                f"{subset}-_project": "_project",
                f"{subset}-_id": "_id",
            })
            df_new.append(df_best_subset)
        df_best = pd.concat(df_new, ignore_index=True)
    
    return df_best


def read_aggregated_training_results_v2(
    csv_path: str,
):
    df_best = pd.read_csv(csv_path)
    
    df_best['subset'] = "default"
    
    subsets = ['default']
    if "simpler" in os.path.basename(csv_path):
        subsets = ['bridge', 'fractal']
        
    n_seeds = 3
    if "droid" in os.path.basename(csv_path):
        n_seeds = 5
    
    df_new = []
    for subset in subsets:
        # For each training seed, extract the wandb run ID
        for seed in range(n_seeds):
            df_best_subset = df_best.copy()
            df_best_subset["subset"] = subset
            df_best_subset["seed"] = seed
            if subset == "default":
                prefix = f"{seed}-"
            else:
                prefix = f"{seed}-{subset}-"
                
            df_best_subset = df_best_subset.rename(columns={
                f"{prefix}_entity": "_entity",
                f"{prefix}_project": "_project",
                f"{prefix}_id": "_id",
            })
            df_new.append(df_best_subset)
            
    df_best = pd.concat(df_new, ignore_index=True)
    return df_best



def plot_metrics(
    df: pd.DataFrame,
    method_to_color: dict,
    exp_name: str,
    x_name: str,
    y_name: str,
):
    if x_name == "alpha":
        fig, ax = plt.subplots(figsize=(3.3, 3.3), dpi=300)
    else:
        fig, ax = plt.subplots(figsize=(5, 3.3), dpi=300)

    # for method_name in df['method_name'].unique():
    for method_name in list(METHOD_NAME_2_DISPLAY_NAME.keys()):
        if method_name not in df['method_name'].unique():
            continue
        df_method = df[df['method_name'] == method_name]
        x = df_method[x_name].values
        y = df_method[y_name].values

        # Sort the x and y values
        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        y = y[sorted_indices]
        
        # Add (0, 0.5) and (1, 0.5) to two ends
        x = np.concatenate(([0], x, [1]))
        if y_name == "bal_acc":
            y = np.concatenate(([0.5], y, [0.5]))
        else:
            if y[0] > y[-1]:
                y = np.concatenate(([1], y, [0]))
            else:
                y = np.concatenate(([0], y, [1]))
        
        kwargs = {}
        kwargs['linewidth'] = 0.75
        kwargs['markersize'] = 3
        kwargs['alpha'] = 0.6
        kwargs['marker'] = METHOD_NAME_2_MARKER.get(method_name, 'o')
        kwargs['label'] = METHOD_NAME_2_DISPLAY_NAME.get(method_name, method_name)
        kwargs['color'] = method_to_color.get(method_name, 'black')
        if method_name in ['lstm', 'indep']:
            kwargs['linewidth'] = 1.5
            kwargs['alpha'] = 0.8
            kwargs['markersize'] = 5
        
        # Plot the line
        ax.plot(x, y, **kwargs)

    plt.xlabel(METRIC_KEY_TO_DISPLAY_NAME.get(x_name, x_name))
    plt.ylabel(METRIC_KEY_TO_DISPLAY_NAME.get(y_name, y_name))
    plt.title(EXP_NAME_2_DISPLAY_NAME.get(exp_name, exp_name))
    
    plt.tight_layout()
    
    return fig, ax


def plot_legend(
    ax: plt.Axes,
):
    # 1. Extract handles & labels from the main plot
    handles, labels = ax.get_legend_handles_labels()

    # 2. Create a new figure just for the legend
    fig_leg = plt.figure(figsize=(2, 4), dpi=300)
    # draw the legend in the center, you can tweak ncol, fontsize, etc.
    fig_leg.legend(
        handles, labels,
        loc='center',
        ncol=1,
        frameon=False,       # optional: turn off the legend frame
        title="Method",      # if you want a title
        fontsize=10,
        title_fontsize=12,
    )

    # 3. Hide the axes on this legend‐only figure
    plt.axis('off')
    plt.tight_layout()
    
    return fig_leg


def get_avg_fail_time(
    exp_name: str,
    seed: int,
) -> float:
    if "openvla_libero" in exp_name:
        annotation_paths = [
            "/rvl-home/guqiao/src/failure_prob/video_annotator/results/openvla-10.json"
        ]
        max_steps = [520]
        unseen_task_ids = SEED_2_UNSEEN_TASK_IDS_LIBERO[seed]
    elif "opi0_simpler" in exp_name:
        '''
        (openvla) guqiao@samoa:~/src/failure_prob$ python video_annotator/count_frame_per_task.py /rvl-home/guqiao/src/failure_prob/video_annotator/data/open_pizero-bridge.json
        Task 0: 60 frames in 'video_annotator/videos/open_pizero-bridge/task00-episode075-succ0.mp4'
        Task 1: 120 frames in 'video_annotator/videos/open_pizero-bridge/task01-episode001-succ0.mp4'
        Task 2: 60 frames in 'video_annotator/videos/open_pizero-bridge/task02-episode001-succ0.mp4'
        Task 3: 60 frames in 'video_annotator/videos/open_pizero-bridge/task03-episode001-succ0.mp4'
        (openvla) guqiao@samoa:~/src/failure_prob$ python video_annotator/count_frame_per_task.py /rvl-home/guqiao/src/failure_prob/video_annotator/data/open_pizero-fractal.json
        Task 0: 80 frames in 'video_annotator/videos/open_pizero-fractal/task00-episode019-succ0.mp4'
        Task 1: 113 frames in 'video_annotator/videos/open_pizero-fractal/task01-episode001-succ0.mp4'
        Task 2: 113 frames in 'video_annotator/videos/open_pizero-fractal/task02-episode037-succ0.mp4'
        Task 3: 200 frames in 'video_annotator/videos/open_pizero-fractal/task03-episode001-succ0.mp4'
        '''
        annotation_paths = [
            "/rvl-home/guqiao/src/failure_prob/video_annotator/results/open_pizero-bridge.json", # bridge
            "/rvl-home/guqiao/src/failure_prob/video_annotator/results/open_pizero-fractal.json",  # fractal
        ]
        max_steps = {
            0: [60, 80], 
            1: [120, 113],
            2: [60, 113],
            3: [60, 200],
        }[seed]
        unseen_task_ids = SEED_2_UNSEEN_TASK_IDS_SIMPLER[seed]
    elif "pi0_libero" in exp_name:
        annotation_paths = [
            "/rvl-home/guqiao/src/failure_prob/video_annotator/results/pizero_fast-default.json"
        ]
        max_steps = [520]
        unseen_task_ids = SEED_2_UNSEEN_TASK_IDS_LIBERO[seed]
    elif "pi0diff_libero" in exp_name:
        annotation_paths = [
            "/rvl-home/guqiao/src/failure_prob/video_annotator/results/pizero-default.json"
        ]
        max_steps = [520]
        unseen_task_ids = SEED_2_UNSEEN_TASK_IDS_LIBERO[seed]
    elif "pi0fast_droid_0510" in exp_name:
        # annotation_paths = [
        #     "/rvl-home/guqiao/src/failure_prob/video_annotator/results/pizero_fast_droid-0510.json"
        # ]
        return 2.0
    else:
        raise ValueError(f"Unknown exp_name: {exp_name}")
    
    # Load the json file
    avg_fail_times = []
    for annotation_path, max_step in zip(annotation_paths, max_steps):
        annotations = json.load(open(annotation_path, "r"))
        rollouts = [r for r in annotations if r['task_id'] in unseen_task_ids]
        fail_times = [r['frame'] / max_step for r in rollouts]
        avg_fail_time = np.mean(fail_times)
        avg_fail_times.append(avg_fail_time)
        
    avg_fail_time = np.mean(avg_fail_times)
    
    print(f"Avg fail time for {exp_name} with seed {seed}: {avg_fail_time}")
    
    return avg_fail_time


def main(args: argparse.Namespace):
    # Use different function for reading the CSV files
    read_aggregated_training_results = {
        "v1": read_aggregated_training_results_v1,
        "v2": read_aggregated_training_results_v2,
    }[args.version]
    
    # Use the csv_paths for the given version
    csv_paths = CSV_PATHS[args.version]
    
    for csv_path in csv_paths:
        api = wandb.Api()
        exp_name = csv_path.split("/")[-1].split(".")[0]
        print(f"Working on {exp_name}")

        # Compute the ground truth average fail time
        gt_avg_fail_time = get_avg_fail_time(exp_name, args.use_seed)
        
        df_best = read_aggregated_training_results(csv_path)
        df_best = df_best[df_best['seed'] == args.use_seed]
        
        print(f"Loaded {len(df_best)} rows from {csv_path}")
        
        # Load the summary and tables from the runs for each best method
        loaded_data = {}
        for i, row in df_best.iterrows():
            run_path = f"{row['_entity']}/{row['_project']}/{row['_id']}"
            if run_path in loaded_data:
                continue
            
            print(f"Loading {run_path}")
            run = api.run(run_path)
            loaded_data[run_path] = load_summary_tables_from_runs([run])[0]
            
        
        dfs_cp = []
        for i, row in df_best.iterrows():
            run_path = f"{row['_entity']}/{row['_project']}/{row['_id']}"
            original_method_name = row['method_full'] # the original text in the wandb logs
            method_name = row['model.name'] # The renamed method name for the paper

            if isinstance(original_method_name, float) and math.isnan(original_method_name):
                original_method_name = "model"
            
            data_run = loaded_data[run_path]
            
            for eval_protocol in ['classify_cp_functional', 'classify_cp_maxsofar']:
                key = f"{eval_protocol}/{original_method_name}"
                if key not in data_run:
                    print(f"Key {key} not found in run {run_path}")
                    continue
                df = data_run[key].copy()
                df['method_name'] = method_name
                df['method_name_full'] = original_method_name
                df['eval_protocol'] = eval_protocol
                df['subset'] = row['subset']
                dfs_cp.append(df)

        df_cp = pd.concat(dfs_cp, ignore_index=True)

        if exp_name == "pi0_libero_v3":
            df_cp['calib on'] = df_cp['calib on'].fillna("neg")

        group_by_cols = [
            'detect_method', 'cal split', 'test split', 'calib on', 'task',
            'thresh_method', 'alpha', 'time', 'method_name', 'method_name_full','eval_protocol', 
        ]
        mean_value_cols = [
            'tpr', 'tnr', 'fpr', 'fnr', 'acc', 'bal_acc', 'f1', 'weighted-acc',
            'roc_auc', 'prc_auc', 'threshold', 'avg_det_time', 
        ]
        df_cp = df_cp.groupby(group_by_cols)[mean_value_cols].agg(['mean']).reset_index()
        df_cp.columns = df_cp.columns.droplevel(1)

        print(f"Loaded {len(df_cp)} rows from {len(dfs_cp)} runs")
        
        # Check the values for the following keys
        keys = [
            'cal split',
            'test split',
            'calib on',
            'task',
            'thresh_method',
            'time',
            'method_name',
            "subset",
        ]

        for k in keys:
            if k in df_cp.columns:
                for v in df_cp[k].unique():
                    n = len(df_cp[df_cp[k] == v])
                    # print(f"{k}: {v} -> {n}")
                    logging.debug(f"{k}: {v} -> {n}")
                    
        # Filter the metrics based on argument
        df = df_cp.copy()
        if args.eval_proto == "functional":
            df = df[df['time'] == 'by final end']
            df = df[df['calib on'] == 'neg']
            df = df[df['thresh_method'] == 'functional CP']
        elif args.eval_proto == "max_neg":
            df = df[df['time'] == 'by final end']
            df = df[df['calib on'] == 'neg']
            df = df[df['thresh_method'] == 'split CP, cal on val_seen']
        elif args.eval_proto == "max_pos":
            df = df[df['time'] == 'by final end']
            df = df[df['calib on'] == 'pos']
            df = df[df['thresh_method'] == 'split CP, cal on val_seen']
        else:
            raise ValueError(f"Unknown eval_proto: {args.eval_proto}")

        print(f"number of rows: {len(df)}")

        method_to_color = get_method_colors(METHOD_NAME_2_GROUP_ID)
        
        
        # Plot and save the Bal Acc vs. Avg Det Time
        fig, ax = plot_metrics(
            df,
            method_to_color,
            exp_name,
            x_name="avg_det_time",
            y_name="bal_acc",
        )

        # Plot a vertical line at gt_avg_fail_time
        plt.axvline(x=gt_avg_fail_time, color='blue', linestyle='--', label="GT Avg Fail Time")
        
        plt.xlim(0, 0.7)
        ax.set_ylim(bottom=0.45)
        
        save_folder = f"notebooks/figures/{args.version}/{args.eval_proto}-seed{args.use_seed}/"
        
        save_path = f"{save_folder}/{exp_name}-balacc_tdet.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Saved to {save_path}")
        
        # Plot and save the legend figure
        fig_leg = plot_legend(ax)
        
        save_path = f"{save_folder}/{exp_name}-legend.pdf"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Saved to {save_path}")
        
        plt.close(fig)
        plt.close(fig_leg)
        
        
        # # Plot and save the TPR vs. Avg Det Time
        # fig, ax = plot_metrics(
        #     df,
        #     method_to_color,
        #     exp_name,
        #     x_name="avg_det_time",
        #     y_name="tpr",
        # )

        # # Plot a vertical line at gt_avg_fail_time
        # plt.axvline(x=gt_avg_fail_time, color='blue', linestyle='--', label="GT Avg Fail Time")
        
        # plt.xlim(0, 0.7)
        # ax.set_ylim(bottom=0.45)
        
        # save_path = f"{save_folder}/{exp_name}-tpr_tdet.pdf"
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # plt.savefig(save_path, bbox_inches='tight')
        # logging.info(f"Saved to {save_path}")
        
        # plt.close(fig)
        
        
        # Plot other metrics vs. alpha
        for metric_key in ['fpr', "fnr", "tpr", "tnr", "bal_acc"]:
            fig, ax = plot_metrics(
                df,
                method_to_color,
                exp_name,
                x_name="alpha",
                y_name=metric_key,
            )

            if args.eval_proto in ['functional', 'max_neg']: # use negative for CP
                if metric_key in ['tnr']:
                    plt.plot([0, 1], [1, 0], color="gray", linestyle="--")
                    # plt.xlim(0, 0.5)
                    # plt.ylim(0.5, 1)
                elif metric_key in ['fpr']:
                    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
                    # plt.xlim(0, 0.5)
                    # plt.ylim(0, 0.5)
            elif args.eval_proto == 'max_pos':
                if metric_key in ['tpr']:
                    plt.plot([0, 1], [1, 0], color="gray", linestyle="--")
                    # plt.xlim(0, 0.5)
                    # plt.ylim(0.5, 1)
                elif metric_key in ['fnr']:
                    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
                    # plt.xlim(0, 0.5)
                    # plt.ylim(0, 0.5)
                    
            # if metric_key != "bal_acc":
            #     ax.set_aspect('equal', adjustable='box')
            plt.xlim(0, 0.5)
            plt.tight_layout()
            
            save_path = f"{save_folder}/{exp_name}-{metric_key}.pdf"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            logging.info(f"Saved to {save_path}")
            
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Conformal Prediction")
    parser.add_argument("--version", type=str, default="v2", help="Version of the data to use")
    parser.add_argument("--use_seed", type=int, default=0, help="Seed to use for the data")
    parser.add_argument("--eval_proto", type=str, default="functional", help="Evaluation protocol to use")
    args = parser.parse_args()
    main(args)