import torch
import random
import datetime
import importlib
import numpy as np
from typing import Union, Dict
from omegaconf import OmegaConf, DictConfig


def check_freq(freq: int, step: int):
    """Check if the current step (0-indexed) is a multiple of the frequency."""
    return freq >= 1 and (step + 1) % freq == 0


def get_time_str():
    """Get the current time as a string in the format of "YYYY-mm-dd-HH-MM-SS"."""
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def set_seed(seed: int, deterministic: bool = False):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def instantiate_from_config(conf: Union[Dict, DictConfig], **extra_params):
    """Instantiate an object from a configuration dictionary.

    The configuration dictionary should follow the format:
    ---------------------------------
    target: "module.submodule.Class"
    params:
        param1: value1
        param2: value2
    ---------------------------------
    An object will be instantiated as `module.submodule.Class(param1=value1, param2=value2, **extra_params)`.

    Args:
        conf: The configuration dictionary.
        extra_params: Extra parameters to pass to the class constructor.

    Returns:
        The instantiated object.
    """
    if isinstance(conf, DictConfig):
        conf = OmegaConf.to_container(conf)
    module, cls = conf["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    params = conf.get("params", dict())
    params.update(extra_params)
    return cls(**params)
