import torch
import json
import wandb
import utils.torchutils as torchutils
import constants as c

from fitter import Callback
from collections import defaultdict


class LoggingCallback(Callback):

    def __init__(self, run_path, wandb_enabled):
        self.run_path = run_path
        self.wandb_enabled = wandb_enabled
        self.metrics_history = defaultdict(list)

    def after_train_step(self, model, loss, metrics, epoch):
        dest = self.run_path / c.CKPT_SUBDIR / c.CKPT_FILE(epoch)
        torch.save(torchutils.unwrap_model(model).state_dict(), dest)

        metrics = {key: val.tolist() for key, val in metrics.items()}
        self.metrics_history["train/loss"].append(loss)
        for key, val in metrics.items():
            self.metrics_history[key].append(val)

        if self.wandb_enabled:
            wandb.save(str(dest), str(self.run_path))
            wandb.log({"train/loss": loss}, step=epoch)
            wandb.log(metrics, step=epoch)

        return False

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

        return False

    def after_fitting(self):
        with open(self.run_path / c.METRICS_FILE, "w") as f:
            json.dump(self.metrics_history, f, indent=4)

        if self.wandb_enabled:
            wandb.log({})  # finally commit
