import importlib
import os

from failure_prob.conf import Config
from failure_prob.data.utils import Rollout

def _import_data_module(data_type: str):
    # Dynamically import a module named after the data_type
    # Assumes each module implements load_rollouts(cfg) and split_rollouts(cfg, all_rollouts)
    try:
        return importlib.import_module(f'.{data_type}', package=__name__)
    except ImportError:
        raise ValueError(f"No module named '{data_type}' found, or it doesn't implement the required functions.")


def load_rollouts(cfg: Config) -> list[Rollout]:
    # Dynamically load the appropriate module and call its load_rollouts
    module = _import_data_module(cfg.dataset.name)
    return module.load_rollouts(cfg)


def split_rollouts(cfg: Config, all_rollouts) -> dict[str, list[Rollout]]:
    # Dynamically load the appropriate module and call its split_rollouts
    module = _import_data_module(cfg.dataset.name)
    return module.split_rollouts(cfg, all_rollouts)
