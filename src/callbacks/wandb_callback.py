"""
This file contains a callback for logging to weights and biases (WandB) during training.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"

import torch
import wandb

from fitter import Callback
from collections import defaultdict
from pathlib import Path


class WandBCallback(Callback):
    """
    Callback for logging to the weights and biases (WandB) service during training.
    """

    def __init__(self, file_pattern, class_dict, summary_metric):
        """
        Constructs a WandB callback.

        Parameters
        ----------
        file_pattern: (int) -> str
            A function giving the checkpoint file name depending on the epoch number.
        class_dict: dict of str
            Dictionary giving a label for each entry in multi-valued metrics.
        summary_metric: str
            The name of the metric whose best value should be used as run summary.
        """
        self.file_pattern = file_pattern
        self.class_dict = class_dict
        self.summary_metric = summary_metric

        self.max_metric = float("-inf")
        self.metrics_history = defaultdict(list)

    def after_train_step(self, model, loss, metrics, epoch):
        metrics = {key: val.tolist() for key, val in metrics.items()}
        self._append_history(loss, metrics)

        # save model for upload later
        path = Path(wandb.run.dir) / "checkpoints" / self.file_pattern(epoch)
        path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), path)

        # log training metrics
        wandb.log({"train/loss": loss}, step=epoch)
        wandb.log(self._transform_metrics(metrics), step=epoch)

    def after_eval_step(self, model, loss, metrics, epoch):
        metrics = {key: val.tolist() for key, val in metrics.items()}
        self._append_history(loss, metrics)

        if metrics[self.summary_metric] > self.max_metric:
            self.max_metric = metrics[self.summary_metric]

        # log evaluation metrics
        wandb.log({"eval/loss": loss}, step=epoch)
        wandb.log(self._transform_metrics(metrics), step=epoch)

    def after_epoch(self, epoch: int):
        wandb.run.summary[self.summary_metric] = self.max_metric
        wandb.log({})  # finally commit for this epoch

    def _append_history(self, loss, metrics):
        self.metrics_history["train/loss"].append(loss)
        for key, val in metrics.items():
            self.metrics_history[key].append(val)

    def _transform_metrics(self, metrics):
        return {key: val if type(val) != list else wandb.plot.line_series(
            xs=list(range(len(self.metrics_history[key]))),
            ys=list(map(list, zip(*self.metrics_history[key]))),
            keys=self.class_dict,
            title=key,
            xname="epoch"
        ) for key, val in metrics.items()}
