import importlib
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel
from failure_prob.conf import Config


def _import_model_module(model_type: str):
    """
    Dynamically import a module named after the model_type. 
    Assumes each module implements a function get_model(cfg, input_dim) -> BaseModel.
    For example, if model_type='indep', we try to import .indep under the current package.
    """
    try:
        module = importlib.import_module(f'.{model_type}', package=__name__)
        return module
    except ImportError as e:
        raise ValueError(
            f"Error importing model submodule for type '{model_type}': {e}" 
        ) from e


def get_model(cfg: Config, input_dim: int) -> BaseModel:
    """
    Dynamically load the appropriate model module based on cfg.model.name.
    Then call its get_model function.
    """
    module = _import_model_module(cfg.model.name)
    return module.get_model(cfg, input_dim)
