"""
This file contains a callback for saving model checkpoints.
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


class CheckpointingCallback(Callback):
    """
    Callback for saving model checkpoints after each epoch during training to the file system.
    """

    def __init__(self, ckpt_dir, file_pattern):
        """
        Constructs a checkpointing callback with the given target directory path.

        Parameters
        ----------
        ckpt_dir: PathLike
            The directory path where the checkpoints shall be stored.
        file_pattern: (int) -> str
            A function giving the checkpoint file name depending on the epoch number.

        """
        self.target_dir = ckpt_dir
        self.file_pattern = file_pattern

    def after_train_step(self, model, loss, metrics, epoch):
        torch.save(model.state_dict(), self.target_dir / self.file_pattern(epoch))
