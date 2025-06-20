#!/bin/bash

# Run all experiments for open-pi-zero model on SimplerEnv rollouts

GROUP_NAME=opi0_simpler_v1

# LSTM and MLP
for DATASET in open_pizero_simpler_bridge open_pizero_simpler_fractal; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${SAFE_OPENPIZERO_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,mean,concat-2 \
        model=lstm \
        model.lr=1e-4,3e-4,1e-3 \
        model.lambda_reg=1e-3,1e-2,1e-1,1 \
        train.seed=0-1-2 \
        train.exp_suffix=lstm
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${SAFE_OPENPIZERO_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,mean,concat-2 \
        model=indep \
        model.lr=1e-4,3e-4,1e-3 \
        model.lambda_reg=1e-3,1e-2,1e-1,1 \
        train.seed=0-1-2 \
        train.exp_suffix=mlp
done

# embed
for DATASET in open_pizero_simpler_bridge open_pizero_simpler_fractal; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${SAFE_OPENPIZERO_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,mean,concat-2 \
        model=embed \
        model.n_epochs=1 \
        model.distance=cosine,euclid \
        model.use_success_only=False \
        model.topk=1,5,10 \
        model.cumsum=False,True \
        train.seed=0-1-2 \
        train.exp_suffix=embed
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${SAFE_OPENPIZERO_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,mean,concat-2 \
        model=embed \
        model.n_epochs=1 \
        model.distance=mahala \
        model.use_success_only=False \
        model.cumsum=False,True \
        train.seed=0-1-2 \
        train.exp_suffix=embed
done

for DATASET in open_pizero_simpler_bridge open_pizero_simpler_fractal; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${SAFE_OPENPIZERO_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,mean,concat-2 \
        model=embed \
        model.distance=pca_kmeans \
        model.pca_dim=32,64,128 \
        model.n_clusters=16,32,64 \
        model.use_success_only=False \
        model.cumsum=False,True \
        train.seed=0-1-2 \
        train.exp_suffix=embed
done

# Chen's baselines
for DATASET in open_pizero_simpler_bridge open_pizero_simpler_fractal; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${SAFE_OPENPIZERO_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.load_to_cuda=False \
        model=rnd \
        train.roc_every=50 \
        model.batch_size=32 \
        model.use_success_only=False \
        train.seed=0-1-2 \
        train.exp_suffix=chen
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${SAFE_OPENPIZERO_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,mean,concat-2 \
        dataset.load_to_cuda=False \
        model=logpzo \
        train.roc_every=50 \
        model.batch_size=32 \
        model.forward_chunk_size=512 \
        model.use_success_only=False \
        train.seed=0-1-2 \
        train.exp_suffix=chen
done

# Handcrafted
for DATASET in open_pizero_simpler_bridge open_pizero_simpler_fractal; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${SAFE_OPENPIZERO_ROLLOUT_ROOT} \
        train.log_precomputed_only=True \
        train.seed=0-1-2 \
        train.exp_suffix=handcrafted
done

for DATASET in open_pizero_simpler_bridge open_pizero_simpler_fractal; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${SAFE_OPENPIZERO_ROLLOUT_ROOT} \
        train.log_precomputed_only=True \
        train.seed=0-1-2 \
        train.exp_suffix=handcrafted_multi
done