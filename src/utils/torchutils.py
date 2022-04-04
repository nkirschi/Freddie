"""
This file contains utility functions abbreviating common PyTorch patterns.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"


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

