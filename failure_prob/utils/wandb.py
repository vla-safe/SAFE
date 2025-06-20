import json
import warnings
import wandb
import pandas as pd


def check_pivot_duplicate(
    df: pd.DataFrame,
    index_cols: list[str],
    column_cols: list[str],
):
    # # define your row‑index and column‑index keys
    # index_cols  = ["metric", "method"] + ablated_configs
    # column_cols = group_configs + ["split"]

    # count how many rows map to each (index, column) pair
    dup_counts = (
        df
        .groupby(index_cols + column_cols)
        .size()
        .reset_index(name="n")
    )

    # select just the ones that really are duplicates
    dupe_groups = dup_counts[dup_counts["n"] > 1]

    if not dupe_groups.empty:
        total_dup_rows = int(dupe_groups["n"].sum())
        total_dupe_cells = len(dupe_groups)
        warnings.warn(
            f"Found {total_dup_rows} rows across {total_dupe_cells} duplicate "
            f"(index×column) combinations — aggregating with mean."
        )

# v2 version of metric processing, for falert evaluation
def pull_metrics_from_group_v2(
    project_name: list[str],
    group_names: list[str],
    ablated_configs: list[str],
    group_configs: list[str],
    filters: dict = None,
    return_wandb_info: bool = True,
):
    assert isinstance(group_configs, list)

    # info_configs save extra information about each run
    info_configs = []
    if return_wandb_info:
        info_configs = ["_entity", "_project", "_id"]
        
    # Get a dataframe where each row is a run and columns are configs and metrics
    runs_df = get_runs_df(project_name, group_names, filters)

    print(f"Loaded {len(runs_df)} runs")

    # Parse each run into multiple rows representing according to metric and split
    split_df = parse_runs_df_to_split_df_v2(
        runs_df,
        group_configs + ablated_configs + info_configs
    )
    
    # Convert group_configs to string
    for col in group_configs:
        split_df[col] = split_df[col].astype(str)
    
    # Keep only the falert metrics
    compare_df = split_df[split_df['metric'].str.contains("falert")]

    # Split metric name into metric and method, move them to the front
    compare_df[['metric', 'method']] = compare_df['metric'].str.split('/', expand=True)
    cols = compare_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('metric')))
    cols.insert(1, cols.pop(cols.index('method')))
    compare_df = compare_df[cols]
    
    # Sort first by group_configs, then by split
    compare_df = compare_df.sort_values(by=group_configs + ['split'])
    
    # if there is any nan value in the ablated_configs columens, replace it with "0"
    for col in ablated_configs:
        if col == "model.pca_dim":
            compare_df[col] = compare_df[col].fillna(64)
        elif col == "model.n_clusters":
            compare_df[col] = compare_df[col].fillna(16)
        else:
            compare_df[col] = compare_df[col].fillna("default")
            
        
    # Make info_configs columns also be a kind of split
    if info_configs:
        new_rows = []
        for i, row in compare_df.iterrows():
            if row['split'] == "train":
                for col in info_configs:
                    # Add a new row for each info_config
                    new_row = row.to_dict().copy()
                    new_row['split'] = col
                    new_row['value'] = compare_df.loc[i, col]
                    new_rows.append(new_row)
        
        # Add the new rows to the DataFrame and drop the original info_configs columns
        new_df = pd.DataFrame(new_rows)
        compare_df = pd.concat([compare_df, new_df], ignore_index=True)
        compare_df = compare_df.drop(columns=info_configs)
            
    
    if group_configs:
        # Just the mean
        compare_df_suite_mean = df_group_mean_except(
            compare_df, 
            group_configs, 
            ['value'],
        )
        for col in group_configs:
            compare_df_suite_mean[col] = "avg"
        compare_df = pd.concat([compare_df_suite_mean, compare_df])
        
        # Mean and std
        compare_df_suite_mean = df_group_mean_except(
            compare_df, 
            group_configs, 
            ['value'],
            mean_std=True,
        )
        for col in group_configs:
            compare_df_suite_mean[col] = "avg_mstd"
        compare_df = pd.concat([compare_df_suite_mean, compare_df])
        
    # pivot_df = compare_df.pivot(
    #     index=["metric", "method"] + ablated_configs, 
    #     columns=group_configs + ["split"], 
    #     values=["value"],
    # ).reset_index()
    
    pivot_df = compare_df.pivot_table(
        index=["metric", "method"] + ablated_configs,
        columns=group_configs + ["split"],
        values="value",
        aggfunc="first",      # Take the first value in case of duplicates
    ).reset_index()
    
    if len(group_configs) == 0:
        # Drop the first level of columns
        pivot_df.columns = [a if len(b) == 0 else b for a, b in pivot_df.columns]
    elif len(group_configs) == 1:
        pivot_df.columns = [a if len(b) == 0 else f"{a}-{b}" for a,b in pivot_df.columns]
    elif len(group_configs) == 2:
        pivot_df.columns = [a if len(b) == 0 else f"{a}-{b}-{c}" for a,b,c in pivot_df.columns]
    
    return pivot_df
    
def parse_runs_df_to_split_df_v2(
    runs_df: pd.DataFrame,
    config_compared: list[str],
)-> pd.DataFrame:
    for config in config_compared:
        assert config in runs_df.columns, f"Column '{config}' not found in DataFrame."
        
    split_data = []
    for col in runs_df.columns:
        for split in ["train", "val_seen", "val_unseen"]:
            if not col.endswith(f"_{split}"):
                continue
            metric = col[: -len(f"_{split}")]
            for i, value in enumerate(runs_df[col]):
                new_row = {
                    "split": split,
                    "metric": metric,
                    "value": value,
                }
                for config in config_compared:
                    new_row[config] = runs_df[config].iloc[i]
                split_data.append(new_row)
                
    # Merge split_data into a DataFrame
    split_df = pd.DataFrame(split_data)
    split_df['value'] = split_df['value'].astype(float) * 100
    
    return split_df


def pull_metrics_from_group(
    project_name: list[str],
    group_names: list[str],
    ablated_configs: list[str],
    group_configs: list[str],
    only_tq1: bool = True,
):
    runs_df = get_runs_df(project_name, group_names)

    print(f"Loaded {len(runs_df)} runs")

    split_df = parse_runs_df_to_split_df(
        runs_df,
        group_configs + ablated_configs
    )

    compare_df = split_df[split_df['metric'].isin(["roc_auc/model", "prc_auc/model"])]

    if only_tq1:
        compare_df = compare_df[compare_df['time_quantile'] == "1.0"]

    if group_configs:
        compare_df_suite_mean = df_group_mean_except(
            compare_df, group_configs, ['value']
        )
        compare_df_suite_mean[group_configs[0]] = "z-mean"
        compare_df = pd.concat([compare_df, compare_df_suite_mean])
        
    compare_df = compare_df.pivot(
        index=group_configs + ablated_configs, 
        columns=["split", "metric"], 
        values="value"
    )

    # Merge two levels of columns
    compare_df.columns = [f"{a}_{b}" for a, b in compare_df.columns]
    compare_df.reset_index(inplace=True)
    
    return compare_df



def flatten_dict(d: dict, parent_key='', sep='.'):
    """
    Flattens a nested dictionary by concatenating keys.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key to use for the current level.
        sep (str): The separator between keys.

    Returns:
        dict: A flattened dictionary with dot-separated keys.
    """
    items = {}
    for k, v in d.items():
        # Build the new key by appending the current key to the parent key (if any)
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        # If the value is a dictionary, recursively flatten it
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def get_runs_df(
    project_name: str,
    group_names: list[str],
    filters: dict = None,
)-> pd.DataFrame:
    if isinstance(group_names, str):
        group_names = [group_names]
    
    api = wandb.Api()
    
    if filters is None:
        runs = api.runs(project_name)
        runs = [run for run in runs if run.group in group_names]
    else:
        runs = api.runs(project_name, filters=filters)
        runs = [run for run in runs if run.group in group_names]
    
    # Create a DataFrame with summaries and names
    data = []
    for run in runs:
        summary = run.summary._json_dict  # Access the summary as a dictionary
        config = flatten_dict(run.config)  # Access the config as a dictionary
        data.append({
            "name": run.name, 
            "_entity": run.entity,
            "_project": run.project,
            "_id": run.id,
            **summary, 
            **config
        })
        
    return pd.DataFrame(data)


def load_summary_tables_from_runs(
    runs: list[wandb.apis.public.Run],
    api = wandb.Api(),
) -> list[dict]:
    data_runs = []

    for run in runs:
        name = run.name
        summary = run.summary._json_dict
        config = flatten_dict(run.config)
        
        data_run = {
            "name":name, 
            "id": run.id,
            **summary, 
            **config
        }
        data_run_keys = list(data_run.keys())
        
        # Get all artifacts logged in this run
        for artifact in run.logged_artifacts():
            if artifact.type != "run_table":
                continue  # Only process table artifacts

            try:
                # Use qualified name to fetch and download
                qualified_name = artifact.qualified_name
                
                downloaded_artifact = api.artifact(qualified_name)
                artifact_dir = downloaded_artifact.download()
                
                logged_name = artifact.name.split("-")[-1].split(":")[0]

                # Find the .table.json file path
                for file in downloaded_artifact.manifest.entries:
                    if file.endswith(".table.json"):
                        file_path = f"{artifact_dir}/{file}"

                        # Load the table as DataFrame
                        table_json = json.load(open(file_path, 'r'))
                        df = pd.DataFrame(table_json['data'], columns=table_json['columns'])
                        
                        # Save the table to the data_run dictionary
                        for k in data_run_keys:
                            # remove "." and "/" from k
                            if k.replace(".", "").replace("/", "") == logged_name or k.replace("/", "") == logged_name:
                                data_run[k] = df

            except Exception as e:
                print(f"Failed to load artifact from run '{name}': {e}")
                
        # Append the data_run to the list
        data_runs.append(data_run)
        
    return data_runs


def parse_runs_df_to_split_df(
    runs_df: pd.DataFrame,
    config_compared: list[str],
)-> pd.DataFrame:
    for config in config_compared:
        assert config in runs_df.columns, f"Column '{config}' not found in DataFrame."
        
    split_data = []
    for col in runs_df.columns:
        if "_tq" not in col:
            continue
        metric_split, tq = col.split("_tq")
        for split in ["train", "val_seen", "val_unseen"]:
            if not metric_split.endswith(f"_{split}"):
                continue
            metric = metric_split[: -len(f"_{split}")]
            for i, value in enumerate(runs_df[col]):
                new_row = {
                    "split": split,
                    "metric": metric,
                    "value": value,
                    "time_quantile": tq,
                }
                for config in config_compared:
                    new_row[config] = runs_df[config].iloc[i]
                split_data.append(new_row)
                    
    # Merge split_data into a DataFrame
    split_df = pd.DataFrame(split_data)
    split_df['value'] = split_df['value'].astype(float) * 100
    
    return split_df


# def df_group_mean_except(
#     df: pd.DataFrame,
#     mean_over_cols: list[str],
#     mean_value_cols: list[str],
#     mean_std: bool = False,
# )-> pd.DataFrame:
#     # The mean will be computed over all different values of mean_over_cols. 
#     # The mean of mean_value_cols will be computed. 
#     if isinstance(mean_over_cols, str):
#         mean_over_cols = [mean_over_cols]
#     if isinstance(mean_value_cols, str):
#         mean_value_cols = [mean_value_cols]
    
#     # Group over all other columns     
#     group_cols = [col for col in df.columns if col not in (mean_over_cols+mean_value_cols)]
    
#     def agg_func(x):
#         if all(isinstance(i, str) for i in x):
#             return x.iloc[0]
#         elif all(isinstance(i, (float, int)) or (isinstance(i, str) and i.replace('.', '', 1).isdigit()) for i in x):
#             if mean_std:
#                 # TODO: return s string "\mstd{}{}" that will be used for latex table
                
#             else:
#                 return x.astype(float).mean()
#         else:
#             raise ValueError("Mixed types in group: cannot aggregate.")
        
#     agg = {col: agg_func for col in mean_value_cols}
#     df = df.groupby(group_cols)
#     df = df.agg(agg)
#     df = df.reset_index()
#     return df

def df_group_mean_except(
    df: pd.DataFrame,
    mean_over_cols: list[str],
    mean_value_cols: list[str],
    mean_std: bool = False,
) -> pd.DataFrame:
    # The mean will be computed over all different values of mean_over_cols. 
    # The mean of mean_value_cols will be computed. 
    if isinstance(mean_over_cols, str):
        mean_over_cols = [mean_over_cols]
    if isinstance(mean_value_cols, str):
        mean_value_cols = [mean_value_cols]
    
    # Group over all other columns     
    group_cols = [col for col in df.columns if col not in (mean_over_cols + mean_value_cols)]
    
    def agg_func(x):
        # Non-numeric (string) -> pass through
        if all(isinstance(i, str) for i in x):
            return x.iloc[0]
        # Numeric (or numeric-looking) -> aggregate
        elif all(
            (isinstance(i, (float, int)) or 
             (isinstance(i, str) and i.replace('.', '', 1).isdigit()))
            for i in x
        ):
            vals = x.astype(float)
            if mean_std:
                m = vals.mean()
                s = vals.std()  # pandas uses ddof=1 by default
                # returns a LaTeX macro \mstd{mean}{std}
                return f"\\mstd{{{m:.2f}}}{{{s:.2f}}}"
            else:
                return vals.mean()
        else:
            raise ValueError("Mixed types in group: cannot aggregate.")
        
    agg = {col: agg_func for col in mean_value_cols}
    df = df.groupby(group_cols).agg(agg).reset_index()
    return df

