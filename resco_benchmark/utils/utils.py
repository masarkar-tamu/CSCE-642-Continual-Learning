import logging
import numpy as np
from math import factorial
from itertools import permutations, islice

import torch.nn as nn

import pfrl.initializers

from resco_benchmark.config.config import config as cfg

logger = logging.getLogger(__name__)


def compute_safe_id(agent_id):
    total = len(cfg.uuid) + len(agent_id) + 8
    file_name_limit = 128
    safe_agt_id = agent_id.replace(":", "_")
    if total > file_name_limit:
        safe_agt_id = safe_agt_id[: len(agent_id) - (total - file_name_limit)]
    return safe_agt_id


def conv2d_size_out(size, kernel_size=2, stride=1):
    return (size - (kernel_size - 1) - 1) // stride + 1


def lecun_init(layer, gain=1):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        pfrl.initializers.init_lecun_normal(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    else:
        pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
        pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
        nn.init.zeros_(layer.bias_ih_l0)
        nn.init.zeros_(layer.bias_hh_l0)
    return layer


def permutations_without_rotations(lst):
    return islice(permutations(lst), factorial(max(len(lst) - 1, 0)))


def get_stats(logged_metrics, key, with_min_and_max=False):
    """
    Used by COIN TODO why is it here?
    """
    v = logged_metrics[key]
    vals = (
        np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
    )

    x = np.array(vals, dtype=np.float32)
    global_sum, global_n = [np.sum(x), len(x)]
    mean = global_sum / global_n

    global_sum_sq = np.sum((x - mean) ** 2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = np.min(x) if len(x) > 0 else np.inf
        global_max = np.max(x) if len(x) > 0 else -np.inf
        return mean, std, global_min, global_max
    return mean, std
