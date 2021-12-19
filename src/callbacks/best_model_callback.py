"""
This file contains a callback for saving the best model throughout the training process.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"

import torch

from learning.fitter import Callback
from os import PathLike


class BestModelCallback(Callback):
    """
    Callback for saving the best model during training to the file system.
    """

    def __init__(self, target_file):
        """
        Constructs a best model callback with the given target file path.

        Parameters
        ----------
        target_file: PathLike
            The file path where the best model shall be stored.
        """
        self.target_file = target_file
        self.min_loss = float("inf")

    def after_eval_step(self, model, loss, metrics, epoch):
        if loss < self.min_loss:
            self.min_loss = loss
            torch.save(model.state_dict(), self.target_file)
