"""
Utility functions abbreviating common PyTorch patterns.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"


import random
import os
import numpy as np
import torch


# see https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
def apply_global_seed(seed: int):
    """
    Applies the same seed to all libraries that make use of randomness and enforces deterministic GPU computation.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def count_trainable_params(model):
    """
    Counts the number of trainable parameters in the given model.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_device(model):
    """
    Determines which device the given model is currently on.
    """

    return next(model.parameters()).device

