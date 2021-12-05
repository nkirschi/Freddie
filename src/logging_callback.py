"""
This file contains a callback for logging during training.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"


import torch
import json
import wandb
import constants as c

from fitter import Callback
from collections import defaultdict


class LoggingCallback(Callback):
    """
    This class defines logging hooks for critical execution points during training.
    """

    def __init__(self, run_path, wandb_enabled):
        self.run_path = run_path
        self.wandb_enabled = wandb_enabled
        self.metrics_history = defaultdict(list)

    def after_train_step(self, model, loss, metrics, epoch):
        dest = self.run_path / c.CKPT_SUBDIR / c.CKPT_FILE(epoch)
        torch.save(model.state_dict(), dest)

        metrics = {key: val.tolist() for key, val in metrics.items()}
        self.metrics_history["train/loss"].append(loss)
        for key, val in metrics.items():
            self.metrics_history[key].append(val)

        if self.wandb_enabled:
            wandb.save(str(dest), str(self.run_path))
            wandb.log({"train/loss": loss}, step=epoch)
            wandb.log(metrics, step=epoch)

    def after_eval_step(self, model, loss, metrics, epoch):
        metrics = {key: val.tolist() for key, val in metrics.items()}
        self.metrics_history["eval/loss"].append(loss)
        for key, val in metrics.items():
            self.metrics_history[key].append(val)

        with open(self.run_path / c.METRICS_FILE, "w") as f:
            json.dump(self.metrics_history, f, indent=4)

        if self.wandb_enabled:
            wandb.log({"eval/loss": loss}, step=epoch)
            scalar_metrics, vector_metrics = {}, {}
            for key, val in metrics.items():
                if type(val) != list:
                    scalar_metrics[key] = val
                else:
                    vector_metrics[key] = wandb.plot.line_series(
                        xs=list(range(len(self.metrics_history[key]))),
                        ys=list(map(list, zip(*self.metrics_history[key]))),
                        keys=c.CLASSES,
                        title=key,
                        xname="epoch"
                    )
            wandb.log(scalar_metrics, step=epoch)
            wandb.log(vector_metrics, step=epoch)

    def after_fitting(self):
        if self.wandb_enabled:
            wandb.log({})  # finally commit
