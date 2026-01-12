#!/bin/bash

# Run all experiments for OpenVLA model on the LIBERO rollouts

GROUP_NAME=openvla_widowx_v2

# LSTM and MLP
for REG in 1e-3 1e-2 1e-1 1; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=openvla_widowx \
        dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
        dataset.token_idx_rel=mean,0.0,1.0 \
        dataset.load_to_cuda=False \
        model=lstm \
        model.batch_size=64 \
        model.lr=1e-4,3e-4,1e-3 \
        model.lambda_reg=${REG} \
        train.seed=0-1-2 \
        train.exp_suffix=lstm
done

for REG in 1e-2 1e-1 1; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=openvla_widowx \
        dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
        dataset.token_idx_rel=mean,0.0,1.0 \
        dataset.load_to_cuda=False \
        model=indep \
        model.batch_size=64 \
        model.lr=1e-4,3e-4,1e-3 \
        model.lambda_reg=${REG} \
        train.seed=0-1-2 \
        train.exp_suffix=mlp
done

# Embedding-based method

python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=openvla_widowx \
    dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    dataset.token_idx_rel=mean,0.0,1.0 \
    dataset.load_to_cuda=False \
    model=embed \
    model.n_epochs=1 \
    model.distance=euclid \
    model.use_success_only=False \
    model.topk=1,5,10 \
    model.cumsum=False,True \
    train.seed=0-1-2 \
    train.exp_suffix=embed

python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=openvla_widowx \
    dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    dataset.token_idx_rel=mean,0.0,1.0 \
    dataset.load_to_cuda=False \
    model=embed \
    model.n_epochs=1 \
    model.distance=cosine \
    model.use_success_only=False \
    model.topk=1,5,10 \
    model.cumsum=False,True \
    train.seed=0-1-2 \
    train.exp_suffix=embed

python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=openvla_widowx \
    dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    dataset.token_idx_rel=mean,0.0,1.0 \
    dataset.load_to_cuda=False \
    model=embed \
    model.n_epochs=1 \
    model.distance=mahala \
    model.use_success_only=False \
    model.cumsum=False,True \
    train.seed=0-1-2 \
    train.exp_suffix=embed

python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=openvla_widowx \
    dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    dataset.token_idx_rel=mean,0.0,1.0 \
    dataset.load_to_cuda=False \
    model=embed \
    model.distance=pca_kmeans \
    model.pca_dim=32,64,128 \
    model.n_clusters=16,32,64 \
    model.use_success_only=False \
    model.cumsum=False,True \
    train.seed=0-1-2 \
    train.exp_suffix=embed


# Chen's baselines

python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=openvla_widowx \
    dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    dataset.token_idx_rel=mean,0.0,1.0 \
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
    dataset=openvla_widowx \
    dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    dataset.token_idx_rel=mean,0.0,1.0 \
    dataset.load_to_cuda=False \
    model=logpzo \
    train.roc_every=50 \
    model.batch_size=32 \
    model.forward_chunk_size=512 \
    model.use_success_only=False \
    train.seed=0-1-2 \
    train.exp_suffix=chen

# Handcrafted metrics

python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=openvla_widowx \
    dataset.data_path_prefix=${SAFE_OPENVLA_WIDOWX_ROLLOUT_ROOT} \
    train.log_precomputed_only=True \
    train.seed=0-1-2 \
    train.exp_suffix=handcrafted
