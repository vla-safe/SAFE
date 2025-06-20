# Automatic the processing of metrics from wandb
import argparse
import os
import pandas as pd

import wandb

from failure_prob.utils.wandb import (
    pull_metrics_from_group_v2,
)

# initialize the API
WANDB_USERNAME = wandb.Api().viewer.username

WANBD_PROJECT_NAME = f"{WANDB_USERNAME}/vla-safe"


WANDB_META_V2 = {
    "pi0fast_libero_v4": [
        # Pi0-FAST on the LIBERO benchmark
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["pi0fast_libero_v4"],
            "exp_suffixes": ["lstm", "mlp"],
            "ablated_configs": ["model.name", "dataset.feat_name", "dataset.token_idx_rel", "model.lr", "model.lambda_reg"],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["pi0fast_libero_v4"],
            "exp_suffixes": ["embed"],
            "ablated_configs": ["model.name", "dataset.feat_name", "dataset.token_idx_rel", "model.distance", "model.topk", "model.cumsum", "model.pca_dim", "model.n_clusters"],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["pi0fast_libero_v4"],
            "exp_suffixes": ["chen"],
            "ablated_configs": ["model.name", "dataset.feat_name", "dataset.token_idx_rel", "model.use_success_only"],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["pi0fast_libero_v4"],
            "exp_suffixes": ["handcrafted"],
            "ablated_configs": [],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["pi0fast_libero_v4"],
            "exp_suffixes": ["handcrafted_multi"],
            "ablated_configs": [],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
    ],
    "openvla_libero_v2": [
        # OpenVLA on the LIBERO benchmark
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["openvla_libero_v2"],
            "exp_suffixes": ["lstm", "mlp"],
            "ablated_configs": ["model.name", "dataset.token_idx_rel", "model.lr", "model.lambda_reg"],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["openvla_libero_v2"],
            "exp_suffixes": ["embed"],
            "ablated_configs": ["model.name", "dataset.token_idx_rel", "model.distance", "model.topk", "model.cumsum", "model.pca_dim", "model.n_clusters"],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["openvla_libero_v2"],
            "exp_suffixes": ["chen"],
            "ablated_configs": ["model.name", "dataset.token_idx_rel"],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["openvla_libero_v2"],
            "exp_suffixes": ["handcrafted"],
            "ablated_configs": [],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["openvla_libero_v2"],
            "exp_suffixes": ["handcrafted_multi"],
            "ablated_configs": [],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
    ],
    "pi0diff_libero_v1": [
        # Pi0 (diffusion version) on LIBERO
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["pi0diff_libero_v1"],
            "exp_suffixes": ["lstm", "mlp"],
            "ablated_configs": ["model.name", "dataset.horizon_idx_rel", "dataset.diff_idx_rel", "model.lr", "model.lambda_reg"],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["pi0diff_libero_v1"],
            "exp_suffixes": ["embed"],
            "ablated_configs": ["model.name", "dataset.horizon_idx_rel", "dataset.diff_idx_rel", "model.distance", "model.topk", "model.cumsum", "model.pca_dim", "model.n_clusters"],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["pi0diff_libero_v1"],
            "exp_suffixes": ["chen"],
            "ablated_configs": ["model.name", "dataset.horizon_idx_rel", "dataset.diff_idx_rel"],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["pi0diff_libero_v1"],
            "exp_suffixes": ["handcrafted"],
            "ablated_configs": [],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["pi0diff_libero_v1"],
            "exp_suffixes": ["handcrafted_multi"],
            "ablated_configs": [],
            "group_configs": ["train.seed"],
            "extra_filters": {},
        },
    ],
    "opi0_simpler_v1": [
        # open-pi-zero on SimplerEnv
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["opi0_simpler_v1"],
            "exp_suffixes": ["lstm", "mlp"],
            "ablated_configs": ["model.name", "dataset.horizon_idx_rel", "dataset.diff_idx_rel", "model.lr", "model.lambda_reg"],
            "group_configs": ["train.seed", "dataset.subset_name"],
            "extra_filters": {},
        },
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["opi0_simpler_v1"],
            "exp_suffixes": ["embed"],
            "ablated_configs": ["model.name", "dataset.horizon_idx_rel", "dataset.diff_idx_rel", "model.distance", "model.topk", "model.cumsum", "model.pca_dim", "model.n_clusters"],
            "group_configs": ["train.seed", "dataset.subset_name"],
            "extra_filters": {},
        },
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["opi0_simpler_v1"],
            "exp_suffixes": ["chen"],
            "ablated_configs": ["model.name", "dataset.horizon_idx_rel", "dataset.diff_idx_rel"],
            "group_configs": ["train.seed", "dataset.subset_name"],
            "extra_filters": {},
        },
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["opi0_simpler_v1"],
            "exp_suffixes": ["handcrafted"],
            "ablated_configs": [],
            "group_configs": ["train.seed", "dataset.subset_name"],
            "extra_filters": {},
        },
        {
            "project_name": WANBD_PROJECT_NAME,
            "group_names": ["opi0_simpler_v1"],
            "exp_suffixes": ["handcrafted_multi"],
            "ablated_configs": [],
            "group_configs": ["train.seed", "dataset.subset_name"],
            "extra_filters": {},
        },
    ],
}


META_MAP = {
    "v2": WANDB_META_V2,
}

METRIC_MAP = {
    "v2": "falert_early_roc_auc",
}

HANDCRAFTED_METRICS_SINGLE = [
    "max_token_prob",
    "avg_token_prob",
    "max_token_entropy",
    "avg_token_entropy",
    "stac_single",
]

HANDCRAFTED_METRICS_MULTI = [
    "total_var",
    "pos_var",
    "rot_var",
    "gripper_var",
    "entropy_linkage",
    "stac_mmd",
]

MODEL_NAME_ORDER = [
    "max_token_prob",
    "avg_token_prob",
    "max_token_entropy",
    "avg_token_entropy",
    "embed-mahala",
    "embed-euclid",
    "embed-cosine",
    "embed-pca_kmeans",
    "rnd",
    "logpZO",
    "total_var",
    "pos_var",
    "rot_var",
    "gripper_var",
    "entropy_linkage",
    "stac_mmd",
    "stac_single",
    "lstm",
    "indep",
]

def main(args: argparse.Namespace):
    
    METRIC = METRIC_MAP[args.meta]

    if args.meta not in META_MAP.keys():
        raise ValueError(f"Invalid meta version: {args.meta}")
    else:
        SAVE_ROOT = f"./scripts/wandb_metrics_batch_{args.meta}"
        WANDB_META = META_MAP[args.meta]

    os.makedirs(SAVE_ROOT, exist_ok=True)
    
    if args.benchmark == "all":
        benchmark_names = list(WANDB_META.keys())
    else:
        assert args.benchmark in WANDB_META.keys(), f"Invalid benchmark name: {args.benchmark}"
        benchmark_names = [args.benchmark]
    
    for benchmark_name in benchmark_names:
        benchmark = WANDB_META[benchmark_name]
        print(f"Processing {benchmark_name}...")
        benchmark_best = []
        for group in benchmark:
            project_name = group["project_name"]
            group_names = group["group_names"]
            exp_suffixes = group["exp_suffixes"]
            ablated_configs = group["ablated_configs"]
            group_configs = group["group_configs"]
            extra_filters = group["extra_filters"]
            
            print(f"Processing {group_names} {exp_suffixes}")
            
            filters = {
                "group": {"$in": group_names},
                "config.train.exp_suffix": {"$in": exp_suffixes},
                **extra_filters,
            }
            
            compare_df = pull_metrics_from_group_v2(
                project_name, group_names, ablated_configs, group_configs, filters
            )
            
            # Convert the learning rate to string
            if "model.lr" in compare_df.columns:
                compare_df["model.lr"] = compare_df["model.lr"].astype(str)
            if "model.lambda_reg" in compare_df.columns:
                compare_df["model.lambda_reg"] = compare_df["model.lambda_reg"].astype(str)

            if "extra_name" in group:
                save_path = f"{SAVE_ROOT}/{group_names[0]}-{'_'.join(exp_suffixes)}-{group['extra_name']}.csv"
            else:
                save_path = f"{SAVE_ROOT}/{group_names[0]}-{'_'.join(exp_suffixes)}.csv"
            print(f"Saving results to {save_path}. \n")
            
            
            # Rename columns
            rename_map = {
                "avg-train": "train",
                "avg-val_seen": "val_seen",
                "avg-val_unseen": "val_unseen",
                "avg-avg-train": "train",
                "avg-avg-val_seen": "val_seen",
                "avg-avg-val_unseen": "val_unseen",
            }
            compare_df.rename(columns=rename_map, inplace=True)
            
            compare_df.to_csv(
                save_path, index=False
            )
            
            # Handle handcrafted metrics
            if "handcrafted" in exp_suffixes[0]:
                # This is handcrafted metrics
                # Add a dummy column for model.name
                assert len(exp_suffixes) == 1
                metrics = HANDCRAFTED_METRICS_SINGLE
                if "handcrafted_multi" in exp_suffixes[0]:
                    metrics = HANDCRAFTED_METRICS_MULTI
                compare_df["method_full"] = compare_df["method"]
                compare_df["method"] = "handcrafted"
                compare_df['model.name'] = "handcrafted"
                to_drop = []
                for i, row in compare_df.iterrows():
                    match = False
                    for metric in metrics:
                        if metric in row["method_full"]:
                            compare_df.at[i, "model.name"] = metric
                            match = True
                            break
                    if not match:
                        to_drop.append(i)
                compare_df.drop(to_drop, inplace=True)
                compare_df.reset_index(drop=True, inplace=True)
                        
            # Handle embedding baselines
            if compare_df["model.name"].str.contains("embed").any():
                for i, row in compare_df.iterrows():
                    if "embed" == row["model.name"]:
                        compare_df.at[i, "model.name"] = row["model.name"] + "-" + row['model.distance']
                        
            # Get the best run for each combination of method and model.name
            compare_df = compare_df[compare_df["metric"] == METRIC]
            idx = compare_df.groupby(['model.name', 'method'])['val_seen'].idxmax()
            df_max = compare_df.loc[idx].reset_index(drop=True)
            benchmark_best.append(df_max)
        
        # Concatenate all the best runs
        benchmark_best = pd.concat(benchmark_best, ignore_index=True)
        
        # Put method model.name train val_seen val_unseen columns in the front
        # Put other columns after them
        cols = benchmark_best.columns.tolist()
        cols.insert(0, cols.pop(cols.index('method')))
        cols.insert(1, cols.pop(cols.index('model.name')))
        cols.insert(2, cols.pop(cols.index('train')))
        cols.insert(3, cols.pop(cols.index('val_seen')))
        cols.insert(4, cols.pop(cols.index('val_unseen')))
        benchmark_best = benchmark_best[cols]
        
        # Sort the dataframe by model.name, according to the order in MODEL_NAME_ORDER
        benchmark_best['model.name'] = pd.Categorical(
            benchmark_best['model.name'], categories=MODEL_NAME_ORDER, ordered=True
        )
        benchmark_best = benchmark_best.sort_values(by=['model.name'])
        
        # print(benchmark_best)
        
        # Save the benchmark best runs to a csv file
        benchmark_save_path = f"{SAVE_ROOT}/{benchmark_name}.csv"
        print(f"Saving benchmark best runs to {benchmark_save_path}. \n")
        benchmark_best.to_csv(
            benchmark_save_path, index=False, float_format="%.2f"
        )
        print(f"Finished processing {benchmark_name}.\n")
        print("=" * 50)
        print("\n\n")
        
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, default="v2", help="The meta version to use")
    parser.add_argument("--benchmark", type=str, default="all", help="The benchmark to process")
    args = parser.parse_args()
    main(args)