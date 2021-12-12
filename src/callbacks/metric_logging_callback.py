"""
This file contains a callback for local logging of metrics during training.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"

import json

from collections import defaultdict
from fitter import Callback
from os import PathLike


class MetricLoggingCallback(Callback):
    """
    Callback for logging metrics values to a JSON file during training.
    """

    def __init__(self, target_file):
        """
        Constructs an early stopping callback with the given patience.

        Parameters
        ----------
        target_file: PathLike
            The file path where the metrics are logged to.
        """
        self.target_file = target_file
        self.metrics_history = defaultdict(list)

    def after_train_step(self, model, loss, metrics, epoch):
        metrics = {key: val.tolist() for key, val in metrics.items()}
        self.metrics_history["train/loss"].append(loss)
        for key, val in metrics.items():
            self.metrics_history[key].append(val)

    def after_eval_step(self, model, loss, metrics, epoch):
        metrics = {key: val.tolist() for key, val in metrics.items()}
        self.metrics_history["eval/loss"].append(loss)
        for key, val in metrics.items():
            self.metrics_history[key].append(val)

    def after_epoch(self, epoch: int):
        with open(self.target_file, "w") as f:
            json.dump(self.metrics_history, f, indent=4)
