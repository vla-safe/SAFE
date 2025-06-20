#!/bin/bash

# Run all experiments for Pi0-FAST model on the LIBERO rollouts

GROUP_NAME=pi0fast_libero_v4

# LSTM
python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=pizero_fast \
    dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
    dataset.feat_name=encoded,pre_logits \
    dataset.token_idx_rel=0.0,1.0,mean \
    model=lstm \
    model.lr=3e-5,1e-4,3e-4,1e-3 \
    model.lambda_reg=1e-3,1e-2,1e-1 \
    train.seed=0-1-2 \
    train.exp_suffix=lstm

# MLP
python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=pizero_fast \
    dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
    dataset.feat_name=encoded,pre_logits \
    dataset.token_idx_rel=0.0,1.0,mean \
    model=indep \
    model.lr=1e-5,1e-4,3e-4,1e-3 \
    model.lambda_reg=1e-3,1e-2,1e-1 \
    train.seed=0-1-2 \
    train.exp_suffix=mlp

# The embed baseline
python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=pizero_fast \
    dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
    dataset.feat_name=encoded,pre_logits \
    dataset.token_idx_rel=0.0,1.0,mean \
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
    dataset=pizero_fast \
    dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
    dataset.feat_name=encoded,pre_logits \
    dataset.token_idx_rel=0.0,1.0,mean \
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
    dataset=pizero_fast \
    dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
    dataset.feat_name=encoded,pre_logits \
    dataset.token_idx_rel=0.0,1.0,mean \
    model=embed \
    model.distance=pca_kmeans \
    model.pca_dim=32,64,128 \
    model.n_clusters=16,32,64 \
    model.use_success_only=False \
    model.cumsum=False,True \
    train.seed=0-1-2 \
    train.exp_suffix=embed

# Chen's method
for MODEL in rnd logpzo; do
for FEAT in encoded pre_logits; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=pizero_fast \
        dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
        dataset.feat_name=${FEAT} \
        dataset.token_idx_rel=0.0,1.0,mean \
        model=${MODEL} \
        model.use_success_only=False \
        model.batch_size=32 \
        train.roc_every=50 \
        train.seed=0-1-2 \
        train.exp_suffix=chen
done
done

# The hand-crafted baselines
python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=pizero_fast \
    dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
    train.log_precomputed_only=True \
    train.seed=0-1-2 \
    train.exp_suffix=handcrafted


# Handcreafted baselines with multiple action samples
python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=pizero_fast_libero_sample10 \
    dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
    train.log_precomputed_only=True \
    train.seed=0-1-2 \
    train.exp_suffix=handcrafted_multi

