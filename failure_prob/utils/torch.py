import torch

def move_to_device(data, device):
    """
    Recursively moves PyTorch tensors in a data structure to a specified device.
    
    Args:
        data (torch.Tensor, list, dict, tuple): A tensor or a container (possibly nested) 
            that holds tensors.
        device (str or torch.device): The target device (e.g., "cuda", "cpu").
    
    Returns:
        The same structure as `data` with all tensors moved to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    else:
        # If data is not a tensor or container of tensors, return it unchanged.
        return data