from dataclasses import dataclass
import os, sys, glob
import argparse
import pickle
from argparse import Namespace

import cv2
import hydra
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio

# Import PCA, t-SNE
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap, SpectralEmbedding

from omegaconf import OmegaConf
from hydra import compose, initialize
from tqdm import tqdm
import umap

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from failure_prob.data import load_rollouts, split_rollouts
from failure_prob.conf import Config, process_cfg
from failure_prob.utils.video import read_frames_and_frame_rate


@hydra.main(version_base=None, config_path="../failure_prob/conf", config_name="feat_vis")
def main(cfg: Config) -> None:
    cfg = process_cfg(cfg)
    feat_skip = 1
    if cfg.dataset.name == "openvla":
        feat_skip = 5
    
    print("Loading rollouts...")
    all_rollouts = load_rollouts(cfg)
    all_rollouts = sorted(all_rollouts, key=lambda x: (x.task_id, x.episode_idx))

    print(f"Loaded {len(all_rollouts)} rollouts")
    print(f"hidden_feature: {all_rollouts[0].hidden_states.shape} {all_rollouts[0].hidden_states.dtype}")

    rollouts = all_rollouts
    
    '''Project features on 2D space'''
    # Construct the feature matrix for down projection
    feats = []
    labels = []
    rollout_indices = []
    for i, r in enumerate(rollouts):
        feat = r.hidden_states[::feat_skip] # (seq_len, hidden_size)
        feats.append(feat.numpy())
        labels.append(np.ones(feat.shape[0]) * (1 - r.episode_success))
        rollout_indices.append(np.ones(feat.shape[0]) * i)
        
    feats = np.concatenate(feats, axis=0)
    labels = np.concatenate(labels, axis=0)
    rollout_indices = np.concatenate(rollout_indices, axis=0)

    print(f"feats: {feats.shape} {feats.dtype}")
    print(f"labels: {labels.shape} {labels.dtype}")
    print(f"rollout_indices: {rollout_indices.shape} {rollout_indices.dtype}")
    
    folder_name = f"{cfg.train.exp_name}-{cfg.projector}"
    save_folder = f"./notebooks/feat_vis/{folder_name}"
    
    os.makedirs(save_folder, exist_ok=True)
    save_pkl_path = f"{save_folder}/feats_projected_skip{feat_skip}.pkl"

    if os.path.exists(save_pkl_path):
        with open(save_pkl_path, "rb") as f:
            data = pickle.load(f)
            projector = data["projector"]
            feats_projected = data["feats_projected"]
            labels = data["labels"]
            rollout_indices = data["rollout_indices"]
    else:
        if cfg.projector == "pca":
            projector = PCA(n_components=2)
        elif cfg.projector == "tsne":
            projector = TSNE(n_components=2)
        elif cfg.projector == "umap":
            projector = umap.UMAP(n_components=2)
        elif cfg.projector == "isomap":
            projector = Isomap(n_components=2)
        elif cfg.projector == "mds":
            projector = MDS(n_components=2)
        elif cfg.projector == "spectral":
            projector = SpectralEmbedding(n_components=2)
        else:
            raise ValueError(f"Unknown projector {cfg.projector}")
        
        feats_projected = projector.fit_transform(feats)
        print(f"feats_projected: {feats_projected.shape} {feats_projected.dtype}")

        # Save the projected features
        with open(save_pkl_path, "wb") as f:
            pickle.dump({
                "projector": projector,
                "feats_projected": feats_projected,
                "labels": labels,
                "rollout_indices": rollout_indices
            }, f)
    
    '''Visualize projected features and save as images'''
    # Compute the color vector
    task_ids = []
    colors_by_success = []
    for i, r in enumerate(rollouts):
        feat_proj = feats_projected[rollout_indices == i]
        task_ids.append(np.ones(feat_proj.shape[0]) * r.task_id)
        if r.episode_success == 0:
            colors_by_success.append(np.linspace(0, 1, feat_proj.shape[0]))
        else:
            colors_by_success.append(np.zeros(feat_proj.shape[0]))
    colors_by_success = np.concatenate(colors_by_success, axis=0) 
    task_ids = np.concatenate(task_ids, axis=0)
    
    # Plot the features and save as images
    # Colored by task success
    plt.figure(dpi=200)
    plt.scatter(
        feats_projected[:, 0], feats_projected[:, 1], 
        c=colors_by_success, cmap='coolwarm', s=0.5, alpha=0.5,
    )
    plt.axis("off"); plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f"{save_folder}/feats_vis_skip{feat_skip}-succ.png", bbox_inches='tight')
    plt.close()
    
    # Colored by task id
    if cfg.custom_cmap:
        # a 10-color palette with no pure reds or blues:
        custom_colors = [
            '#98df8a',  # light green
            '#c5b0d5',  # lavender
            '#8c564b',  # brown
            '#ff7f0e',  # orange
            '#9467bd',  # purple
            '#bcbd22',  # olive
            '#7f7f7f',  # gray
            '#e377c2',  # pink
            '#2ca02c',  # green
            '#c49c94',  # tan
        ]

        cmap_task_ids = mpl.colors.ListedColormap(custom_colors)
    else:
        cmap_task_ids = mpl.cm.get_cmap("tab10", 10)
    
    plt.figure(dpi=200)
    plt.scatter(
        feats_projected[:, 0], feats_projected[:, 1], 
        c=task_ids, cmap=cmap_task_ids, s=0.5, alpha=0.7,
    )
    plt.axis("off"); plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f"{save_folder}/feats_vis_skip{feat_skip}-taskid.png", bbox_inches='tight')
    plt.close()

    '''Save videos and video frames'''
    if cfg.save_video:
        video_save_folder = f"{save_folder}/videos"
        os.makedirs(video_save_folder, exist_ok=True)

        indices = np.arange(0, len(rollouts), 10)
        
        rollouts_simpler = [r.get_simple_meta() for r in rollouts]
        
        # Parallel processing with progress bar
        with ProcessPoolExecutor(max_workers=8) as executor:
            tasks = executor.map(
                process_rollout,
                indices,
                repeat(rollouts_simpler),
                repeat(feats_projected),
                repeat(rollout_indices),
                repeat(feat_skip),
                repeat(colors_by_success),
                repeat(cfg),
                repeat(save_folder),
            )
            # Consume iterator with tqdm for visual feedback
            for _ in tqdm(tasks, total=len(indices), desc="Saving rollouts"):  # noqa: F841
                pass
        

def process_rollout(
    i,
    rollouts,
    feats_projected,
    rollout_indices,
    feat_skip,
    colors_by_success,
    cfg,
    save_folder,
):
    r = rollouts[i]
    raw_frames_bgr, fps = read_frames_and_frame_rate(r.mp4_path)
    feats_proj = feats_projected[rollout_indices == i]
    exec_horizon = r.exec_horizon if r.exec_horizon is not None else cfg.dataset.exec_horizon
    n_feats = len(feats_proj)
    frames = []

    rollout_save_name = f"task{r.task_id}_ep{r.episode_idx}_succ{r.episode_success}"

    if cfg.save_indiv_frames:
        frame_save_folder = os.path.join(save_folder, "videos", rollout_save_name)
        os.makedirs(frame_save_folder, exist_ok=True)

    for j, frame in enumerate(raw_frames_bgr):
        score_plot_end = j // (exec_horizon * feat_skip) + 1

        # Save individual frame and t-SNE snapshot
        if cfg.save_indiv_frames:
            # Raw frame
            frame_path = os.path.join(frame_save_folder, f"t{j:04d}_frame.jpg")
            cv2.imwrite(frame_path, frame)

            # t-SNE plot
            fig = plt.figure(figsize=(6, 6), dpi=200)
            plt.scatter(
                feats_projected[:, 0], feats_projected[:, 1],
                c=colors_by_success, cmap='coolwarm', s=0.5, alpha=0.05
            )
            plt.scatter(
                feats_proj[score_plot_end-1:score_plot_end, 0], feats_proj[score_plot_end-1:score_plot_end, 1],
                marker='*', s=25, alpha=1.0,
                c=[score_plot_end-1], cmap='turbo', vmin=0, vmax=n_feats
            )
            plt.axis('off')
            plt.tight_layout()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(os.path.join(frame_save_folder, f"t{j:04d}_tsne.jpg"), bbox_inches='tight')
            plt.close(fig)

        # Combined plot for video
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=200)
        axes[0].imshow(frame[:, :, ::-1])
        axes[0].axis('off')
        axes[0].set_title(f"RGB obs frame {j}")

        axes[1].scatter(
            feats_projected[:, 0], feats_projected[:, 1],
            c=colors_by_success, cmap='coolwarm', s=0.5, alpha=0.05
        )
        axes[1].scatter(
            feats_proj[:score_plot_end, 0], feats_proj[:score_plot_end, 1],
            marker='*', s=25, alpha=1.0,
            c=np.arange(score_plot_end), cmap='turbo', vmin=0, vmax=n_feats
        )
        axes[1].axis('off')
        axes[1].set_aspect('equal', adjustable='box')

        fig.suptitle(f"{r.task_description}\nEp {r.episode_idx}, Succ {r.episode_success}")
        fig.tight_layout()
        fig.canvas.draw()
        plot_img = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        frames.append(plot_img)

    # Write video
    save_path = os.path.join(save_folder, "videos", f"{rollout_save_name}.mp4")
    imageio.mimsave(save_path, frames, fps=fps)
            
if __name__ == "__main__":
    main()
    
