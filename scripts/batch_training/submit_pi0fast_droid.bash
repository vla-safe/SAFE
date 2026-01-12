#!/bin/bash

# Run all experiments for Pi0FAST model on the real-world Franka dataset

GROUP_NAME=pizero_fast_droid_0510
DATASET=pizero_fast_droid_0510

## LSTM and MLP
python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=indep,lstm \
    model.n_layers=1,2 \
    model.hidden_dim=128,256 \
    model.lr=1e-4,3e-4,1e-3,3e-3 \
    model.lambda_reg=1e-3,1e-2,1e-1 \
    train.seed=0-1-2-3-4 \
    train.exp_suffix=13task

## Baselines
python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=embed \
    model.n_epochs=1 \
    model.distance=cosine,euclid \
    model.use_success_only=False \
    model.topk=1,5,10 \
    model.cumsum=False,True \
    train.seed=0-1-2-3-4 \
    train.exp_suffix=13task_embed

python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=embed \
    model.n_epochs=1 \
    model.distance=mahala \
    model.use_success_only=False \
    model.cumsum=False,True \
    train.seed=0-1-2-3-4 \
    train.exp_suffix=13task_embed

python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=embed \
    model.distance=pca_kmeans \
    model.pca_dim=32,64,128 \
    model.n_clusters=16,32,64 \
    model.use_success_only=False \
    model.cumsum=False,True \
    train.seed=0-1-2-3-4 \
    train.exp_suffix=13task_embed

for MODEL in rnd logpzo; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=${DATASET} \
        dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
        dataset.feat_name=pre_logits \
        dataset.token_idx_rel=mean \
        model=${MODEL} \
        model.use_success_only=False \
        train.seed=0-1-2-3-4 \
        train.exp_suffix=13task_chen
done

python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=${DATASET} \
    dataset.data_path_prefix=${SAFE_OPENPI_DROID_ROLLOUT_ROOT} \
    train.log_precomputed_only=True \
    train.seed=0-1-2-3-4 \
    train.exp_suffix=13task_handcrafted