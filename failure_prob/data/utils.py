from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from failure_prob.conf import Config

@dataclass
class Rollout:
    '''
    A single rollout of the experiment.
    '''
    hidden_states: torch.Tensor
    task_suite_name: str
    task_id: int
    task_description: int
    episode_idx: int
    episode_success: int
    mp4_path: str
    logs: Optional[pd.DataFrame] = None
    task_min_step: Optional[int] = None
    exec_horizon: Optional[int] = None
    action_vectors: Optional[torch.Tensor] = None
    
    def __post_init__(self):
        self.episode_success = int(self.episode_success)
        
    # Move the hidden states to different device (can help speed up for training)
    def to(self, device):
        self.hidden_states = self.hidden_states.to(device)
        return self
    
    def get_simple_meta(self):
        '''
        Return a simple metadata of the rollout, removing heavy attributes.
        '''
        new_rollout = Rollout(
            hidden_states=None,
            task_suite_name=self.task_suite_name,
            task_id=self.task_id,
            task_description=self.task_description,
            episode_idx=self.episode_idx,
            episode_success=self.episode_success,
            mp4_path=self.mp4_path,
            logs=None,
            task_min_step=self.task_min_step,
            exec_horizon=self.exec_horizon,
            action_vectors=None,
        )
        return new_rollout
        

class RolloutDataset(Dataset):
    '''
    A PyTorch Dataset for the rollout data.
    '''
    def __init__(
        self, 
        cfg: Config, 
        rollouts: list[Rollout], 
        device=None
    ):
        self.cfg = cfg
        self.rollouts = rollouts
        self.length = len(rollouts)

        self.device = device
        if self.device is None:
            self.device = "cuda" if cfg.dataset.load_to_cuda else "cpu"
        
        # Weigh the loss by the frequency of success/failure
        freq_0 = (sum([r.episode_success == 0 for r in rollouts]) + 1) / len(rollouts)
        freq_1 = (sum([r.episode_success == 1 for r in rollouts]) + 1) / len(rollouts)
        self.weights = [1./(freq_0), 1./(freq_1)]
        self.weights[0] *= self.cfg.model.lambda_fail
        self.weights[1] *= self.cfg.model.lambda_success
        
        features, valid_masks, labels, action_vectors = pad_rollout_batch(self.rollouts, self.device)
        
        self.features = features
        self.valid_masks = valid_masks
        self.labels = labels
        self.action_vectors = action_vectors
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        data = {
            'features': self.features[idx],
            'valid_masks': self.valid_masks[idx],
            'success_labels': self.labels[idx],
        }
        if self.action_vectors is not None:
            data['action_vectors'] = self.action_vectors[idx]
        return data
    
    def get_rollouts(self):
        return self.rollouts
    
    def get_features(self):
        return self.features
    
    def get_valid_masks(self):
        return self.valid_masks
    
    def get_labels(self):
        return self.labels
    
    def get_class_weights(self):
        return self.weights


def normalize_rollouts_hidden_states(
    rollouts: list[Rollout],
):
    """
    Normalize the hidden states of the rollouts to have zero mean and unit variance.
    This is done in place, modifying the original rollouts.
    """
    # Stack all hidden states into a single tensor
    all_hidden_states = torch.cat([r.hidden_states for r in rollouts], dim=0)
    
    # Compute mean and std
    mean = all_hidden_states.mean(dim=0)
    std = all_hidden_states.std(dim=0)
    
    # Normalize each rollout's hidden states
    for r in rollouts:
        r.hidden_states = (r.hidden_states - mean) / std
    
    return rollouts


def split_rollouts_by_seen_unseen(
    cfg: Config,
    all_rollouts: list[Rollout],
    seen_task_ids: list[int],
    unseen_task_ids: list[int],
):
    print(f"Seen tasks: {seen_task_ids}, Unseen tasks: {unseen_task_ids}")

    seen_rollouts = [r for r in all_rollouts if r.task_id in seen_task_ids]
    unseen_rollouts = [r for r in all_rollouts if r.task_id in unseen_task_ids]

    # Split the rollouts for training and evaluation
    train_rollouts = []
    val_seen_rollouts = []
    for task_id in seen_task_ids:
        task_rollouts = [r for r in seen_rollouts if r.task_id == task_id]
        # Split the seen tasks into training and val_seen sets
        permuted_indices = torch.randperm(len(task_rollouts))
        n_train_rollouts = int(cfg.dataset.seen_train_ratio * len(task_rollouts))
        train_rollouts += [task_rollouts[i] for i in permuted_indices[:n_train_rollouts]]
        val_seen_rollouts += [task_rollouts[i] for i in permuted_indices[n_train_rollouts:]]
    val_unseen_rollouts = unseen_rollouts
    
    rollouts_by_split_name = {
        "train": train_rollouts,
        "val_seen": val_seen_rollouts,
        "val_unseen": val_unseen_rollouts,
    }
    
    if len(val_unseen_rollouts) == 0:
        del rollouts_by_split_name["val_unseen"]
    if len(val_seen_rollouts) == 0:
        del rollouts_by_split_name["val_seen"]
    
    for split, rollouts in rollouts_by_split_name.items():
        n_success = sum([r.episode_success for r in rollouts])
        n_fail = len(rollouts) - n_success
        print(f"{split}: {len(rollouts)} rollouts, {n_success} success, {n_fail} fail")

    return rollouts_by_split_name


def pad_rollout_batch(
    rollouts: list[Rollout], device = None
):
    """
    Pad the hidden states to the same length (max length in the batch).

    Args:
        rollouts: list of rollouts, each containing:
            - hidden_states (Tensor): shape [sequence_length, hidden_dim]
            - episode_success (int): success flag (1 or 0)
            - action_vectors (Tensor or None), shape [sequence_length, action_dim]
        device: device to put the tensors on

    Returns:
        padded_features (Tensor): shape [batch_size, max_length, hidden_dim]
        valid_masks    (Tensor): shape [batch_size, max_length]
            0 indicates padding, 1 indicates valid
        labels          (Tensor): shape [batch_size]
            1 indicates rollout success, 0 indicates rollout failure
        action_vectors  (Tensor or None): shape [batch_size, max_length, action_dim]
    """
    # Extract all hidden states into a list
    batch_features = [r.hidden_states for r in rollouts]

    # Determine padding dimensions
    max_length = max(seq.shape[0] for seq in batch_features)
    hidden_dim = batch_features[0].shape[-1]
    batch_size = len(batch_features)

    # Infer dtype and device from the first sequence
    dtype = batch_features[0].dtype
    if device is None:
        device = batch_features[0].device

    # Pre-allocate output tensors
    padded_features = torch.zeros(
        batch_size, max_length, hidden_dim,
        dtype=dtype, device=device
    )
    padding_masks = torch.ones(
        batch_size, max_length,
        dtype=torch.float32, device=device
    )

    # Fill in values for each sequence
    for i, seq in enumerate(batch_features):
        seq_length = seq.shape[0]
        padded_features[i, :seq_length] = seq.to(device)
        padding_masks[i, :seq_length] = 0

    # Convert success flags to a tensor (still use torch.long for labels)
    labels = torch.tensor(
        [r.episode_success for r in rollouts],
        dtype=torch.float32,
        device=device
    )
    
    valid_masks = (1 - padding_masks)
    
    # Extract and pad action vectors if available
    if rollouts[0].action_vectors is None:
        action_vectors = None
    else:
        action_dim = rollouts[0].action_vectors.shape[-1]
        action_vectors = torch.zeros(
            batch_size, max_length, action_dim,
            dtype=dtype, device=device
        )
        for r in rollouts:
            seq_length = r.action_vectors.shape[0]
            action_vectors[i, :seq_length] = r.action_vectors.to(device)

    return padded_features, valid_masks, labels, action_vectors


def set_task_min_step(rollouts: list[Rollout]) -> list[Rollout]:
    '''
    Compute the minimum timestep for each task
    This operation modifies the input rollouts in place.
    '''
    task_ids = list(set([r.task_id for r in rollouts]))
    for task_id in task_ids:
        task_rollouts = [r for r in rollouts if r.task_id == task_id]
        min_timestep = min([r.hidden_states.shape[0] for r in task_rollouts])
        for r in rollouts:
            if r.task_id == task_id:
                r.task_min_step = min_timestep
                
    return rollouts


def parse_and_index_tensor_last(A, command):
    """
    Parse a command string to index into the last two dimensions of a multi-dimensional tensor A,
    and then flatten these last two dimensions.

    Supported commands:
      - "concat": 
          -> Flatten the entire last two dimensions (of shape (c, d) becomes (c*d,)).
      - "concat-:10" or "concat-::5": 
          -> Apply a Python slice on the second-to-last axis (i.e. the "row" axis of the last two dims),
             then flatten the last two dimensions.
      - "concat-2", "concat-5", etc.: 
          -> Uniformly index into the second-to-last dimension to obtain the specified number of features.
             For instance, "concat-2" selects the first and last positions along that dimension.
             "concat-5" selects 5 indices (first, last, and three equally spaced indices in between).

    Parameters:
      A (np.ndarray): A multi-dimensional tensor (e.g., shape (..., c, d)).
      command (str): A command starting with "concat" that specifies how to index the tensor.

    Returns:
      np.ndarray: The tensor where the operation has been applied on the last two dimensions and then flattened.

    Raises:
      ValueError: If the command format is not recognized.
    """
    # Case 1: When command is exactly "concat", flatten the last two dimensions.
    if command == "concat":
        new_last_dim = A.shape[-2] * A.shape[-1]
        return A.reshape(*A.shape[:-2], new_last_dim)
    
    prefix = "concat-"
    sub_cmd = command[len(prefix):]  # Extract portion after "concat-"
    
    # Check if the sub-command contains a colon (indicating slice notation)
    if ":" in sub_cmd:
        parts = sub_cmd.split(':')
        # Two-part slice, e.g. ":10"
        if len(parts) == 2:
            start_str, stop_str = parts
            start = int(start_str) if start_str != "" else None
            stop = int(stop_str) if stop_str != "" else None
            # Apply slicing on the second-to-last dimension.
            indexed = A[..., slice(start, stop), :]
        # Three-part slice, e.g. "::5"
        elif len(parts) == 3:
            start_str, stop_str, step_str = parts
            start = int(start_str) if start_str != "" else None
            stop = int(stop_str) if stop_str != "" else None
            step = int(step_str) if step_str != "" else None
            indexed = A[..., slice(start, stop, step), :]
        else:
            raise ValueError("Invalid slice format in command.")
        
        new_last_dim = indexed.shape[-2] * indexed.shape[-1]
        return indexed.reshape(*indexed.shape[:-2], new_last_dim)
    
    # Otherwise, check if the sub-command is simply an integer for uniform indexing.
    try:
        k = int(sub_cmd)
    except ValueError:
        raise ValueError("Invalid command format; expected a colon-based slice or an integer.")
    
    # Uniform indexing requires at least 2 features.
    if k < 2:
        raise ValueError("Uniform indexing requires at least 2 features.")
    
    # Determine the number of indices available along the second-to-last dimension.
    c = A.shape[-2]
    # Compute k indices uniformly spaced, including the endpoints.
    indices = np.linspace(0, c - 1, num=k)
    # Convert to integers by rounding.
    indices = np.round(indices).astype(int)
    # Use these indices to select along the second-to-last dimension.
    indexed = A[..., indices, :]
    new_last_dim = indexed.shape[-2] * indexed.shape[-1]
    return indexed.reshape(*indexed.shape[:-2], new_last_dim)
    
    
def process_tensor_idx_rel(A, command):
    """
    Process a multi-dimensional tensor A based on a provided command.

    The command specifies the operation to perform on A as follows:
    
      1. If `command` is a float (between 0 and 1):
         - Interprets it as a relative index to select a single token from the second-to-last dimension.
         - Example: token_idx = round((A.shape[-2]-1) * command) 
           and then returns A[..., token_idx, :].

      2. If `command` is the string "mean":
         - Computes the mean over the second last axis 
           (Note: the axis choice here is based on your snippet; if you intend to average over the horizon axis of the last two dimensions, you might use axis=-2 instead).
      
      3. If `command` is a string containing "concat":
         - Calls `parse_and_index_tensor_last(A, command)` to apply a slice to the last two dimensions and flatten them.
      
      4. Otherwise:
         - Raises a ValueError indicating an unknown token index.

    Parameters:
      A (np.ndarray): A multi-dimensional tensor.
      command (str or float): The command specifying which processing operation to apply.
      
    Returns:
      np.ndarray: The processed tensor.

    Raises:
      ValueError: If the command is not recognized.
    """
    assert len(A.shape) >= 2, "Tensor A must have at least two dimensions."
    
    if isinstance(command, float):
        # Validate the command as a float in the range [0, 1].
        assert 0 <= command <= 1, f"Invalid token index ratio: {command}"
        token_idx = round((A.shape[-2] - 1) * command)
        # Select the specific token along the second-to-last dimension.
        return A[..., token_idx, :]
    
    elif command == "mean":
        # Compute the mean over axis 0.
        # (Adjust the axis if you need the mean over a different dimension.)
        return A.mean(axis=-2)
    
    elif isinstance(command, str) and "concat" in command:
        return parse_and_index_tensor_last(A, command)
    
    else:
        raise ValueError(f"Unknown token index: {command}")
    

