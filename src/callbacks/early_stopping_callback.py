"""
This file contains a callback for early stopping of the training process.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"

from learning.fitter import Callback


class EarlyStoppingCallback(Callback):
    """
    Callback for early stopping when overfitting starts.
    """

    def __init__(self, *, patience):
        """
        Constructs an early stopping callback with the given patience.

        Parameters
        ----------
        patience: int
            The number of successive epochs with increasing evaluation loss to wait before stopping.
        """
        self.last_loss = float("inf")
        self.patience = patience
        self.current_patience = patience

    def after_eval_step(self, model, loss, metrics, epoch):
        if loss > self.last_loss:
            self.current_patience -= 1
        else:
            self.current_patience = self.patience
        self.last_loss = loss

    def after_epoch(self, epoch):
        return self.current_patience == 0
