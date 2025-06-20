#!/bin/bash

# Run all experiments for Pi0 model on the LIBERO rollouts

GROUP_NAME=pi0diff_libero_v1

# LSTM and MLP
for REG in 1e-3 1e-2 1e-1; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=pizero \
        dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,concat-2 \
        model=lstm \
        model.lr=1e-5,3e-5,1e-4,3e-4,1e-3 \
        model.lambda_reg=${REG} \
        train.seed=0-1-2 \
        train.exp_suffix=lstm
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=pizero \
        dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,concat-2 \
        model=indep \
        model.lr=1e-5,3e-5,1e-4,3e-4,1e-3 \
        model.lambda_reg=${REG} \
        train.seed=0-1-2 \
        train.exp_suffix=mlp
done

# Embed model
for DIST in cosine euclid; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=pizero \
        dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,concat-2 \
        dataset.load_to_cuda=False \
        model=embed \
        model.n_epochs=1 \
        model.distance=${DIST} \
        model.use_success_only=False \
        model.topk=1,5,10 \
        model.cumsum=False,True \
        train.seed=0-1-2 \
        train.exp_suffix=embed
done

python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=pizero \
    dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
    dataset.horizon_idx_rel=0.0,1.0,concat-2 \
    dataset.diff_idx_rel=0.0,1.0,concat-2 \
    dataset.load_to_cuda=False \
    model=embed \
    model.n_epochs=1 \
    model.distance=mahala \
    model.use_success_only=False \
    model.cumsum=False,True \
    train.seed=0-1-2 \
    train.exp_suffix=embed

for PCA_DIM in 32 64 128; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=pizero \
        dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=0.0,1.0,concat-2 \
        dataset.diff_idx_rel=0.0,1.0,concat-2 \
        dataset.load_to_cuda=False \
        model=embed \
        model.distance=pca_kmeans \
        model.pca_dim=${PCA_DIM} \
        model.n_clusters=16,32,64 \
        model.use_success_only=False \
        model.cumsum=False,True \
        train.seed=0-1-2 \
        train.exp_suffix=embed
done

# Chen's baselines
for HORIZON_IDX in 0.0 1.0 concat-2; do
    python -m failure_prob.train \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        dataset=pizero \
        dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=${HORIZON_IDX} \
        dataset.diff_idx_rel=0.0,1.0,concat-2 \
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
        dataset=pizero \
        dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
        dataset.horizon_idx_rel=${HORIZON_IDX} \
        dataset.diff_idx_rel=0.0,1.0,concat-2 \
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
python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=pizero \
    dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
    train.log_precomputed_only=True \
    train.seed=0-1-2 \
    train.exp_suffix=handcrafted


# Handcreafted baselines with multiple action samples
python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    dataset=pizero_libero_sample10 \
    dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
    train.log_precomputed_only=True \
    train.seed=0-1-2 \
    train.exp_suffix=handcrafted_multi
