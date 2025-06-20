#!/bin/bash

# Run all experiments for OpenVLA model on the LIBERO rollouts

GROUP_NAME=openvla_libero_v2

# LSTM and MLP
for SUITE_NAME in 10; do
for SEED in 0 1 2; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=openvla_libero_${SUITE_NAME} \
        dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
        dataset.token_idx_rel=mean,0.0,1.0 \
        dataset.load_to_cuda=False \
        model=lstm \
        model.batch_size=64 \
        model.lr=1e-4,3e-4,1e-3 \
        model.lambda_reg=1e-3,1e-2,1e-1,1 \
        train.seed=${SEED} \
        train.exp_suffix=lstm
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=openvla_libero_${SUITE_NAME} \
        dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
        dataset.token_idx_rel=mean,0.0,1.0 \
        dataset.load_to_cuda=False \
        model=indep \
        model.batch_size=64 \
        model.lr=1e-4,3e-4,1e-3 \
        model.lambda_reg=1e-3,1e-2,1e-1,1 \
        train.seed=${SEED} \
        train.exp_suffix=mlp
done
done

# Embedding-based method
for SUITE_NAME in 10; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=openvla_libero_${SUITE_NAME} \
        dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
        dataset.token_idx_rel=mean,0.0,1.0 \
        dataset.load_to_cuda=False \
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
        dataset=openvla_libero_${SUITE_NAME} \
        dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
        dataset.token_idx_rel=mean,0.0,1.0 \
        dataset.load_to_cuda=False \
        model=embed \
        model.n_epochs=1 \
        model.distance=mahala \
        model.use_success_only=False \
        model.cumsum=False,True \
        train.seed=0-1-2 \
        train.exp_suffix=embed
done

for SUITE_NAME in 10; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=openvla_libero_${SUITE_NAME} \
        dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
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
done


# Chen's baselines
for SUITE_NAME in 10; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=openvla_libero_${SUITE_NAME} \
        dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
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
        dataset=openvla_libero_${SUITE_NAME} \
        dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
        dataset.token_idx_rel=mean,0.0,1.0 \
        dataset.load_to_cuda=False \
        model=logpzo \
        train.roc_every=50 \
        model.batch_size=32 \
        model.forward_chunk_size=512 \
        model.use_success_only=False \
        train.seed=0-1-2 \
        train.exp_suffix=chen
done

# Handcrafted metrics
for SUITE_NAME in 10; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=openvla_libero_${SUITE_NAME} \
        dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
        train.log_precomputed_only=True \
        train.seed=0-1-2 \
        train.exp_suffix=handcrafted
done

# Multi-sample handcrafted metrics 
for SUITE_NAME in 10; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=openvla_libero_${SUITE_NAME}_multi \
        dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
        train.log_precomputed_only=True \
        train.seed=0-1-2 \
        train.exp_suffix=handcrafted_multi
done