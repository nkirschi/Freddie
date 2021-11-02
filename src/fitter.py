import torch

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch._C import device
from torch.nn.modules.loss import _Loss
from torchmetrics.collections import MetricCollection
from abc import ABC

import utils.torchutils as torchutils


class Callback(ABC):

    def after_train_step(self, model: Module, loss: float, metrics: dict, epoch: int) -> bool:
        return False

    def after_eval_step(self, model: Module, loss: float, metrics: dict, epoch: int) -> bool:
        return False

    def after_fitting(self):
        pass


class Fitter:
    """
    A generic trainer for deep learning models based on PyTorch and TorchMetrics.
    """

    def __init__(self, optimizer, criterion, train_metrics, eval_metrics, *,
                 max_epochs, log_every, device, callback=None):
        """
        Constructs a new fitter with the given optimization parameters.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer to use for fitting.
        criterion : _Loss
            The target loss function for the optimization process.
        train_metrics, eval_metrics : MetricCollection
            The metrics to track during training and on after-epoch evaluation.
        max_epochs : int
            The maximum number of epochs to perform.
        log_every : int
            The period duration after which to log progress to the console.
            A value of 0 disables intermediate logging.
        device : device
            The device to use for fitting.
        callback : Callback
            A callback object with hooks being called at key points during fitting.
        """

        self.optimizer = optimizer
        self.criterion = criterion
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics

        self.max_epochs = max_epochs
        self.log_every = log_every
        self.device = device
        self.callback = callback

        criterion.to(device, non_blocking=True)
        train_metrics.to(device, non_blocking=True)
        eval_metrics.to(device, non_blocking=True)

    def fit(self, model, dl_train, dl_eval):
        """
        Fits the given model to the training data while evaluating on a holdout set.

        Parameters
        ----------
        model : Module
            The model to fit.
        dl_train : DataLoader
            The data loader yielding the training data.
        dl_eval : DataLoader
            The data loader yielding the evaluation data.
        """

        model.to(self.device, non_blocking=True)

        print(f"Training on {len(dl_train.dataset)} samples")
        print(f"Evaluating on {len(dl_eval.dataset)} samples")

        for epoch in range(1, self.max_epochs + 1):
            stop_early = self.__train_step(model, dl_train, epoch)
            stop_early += self.__eval_step(model, dl_eval, epoch)

            if stop_early:
                print("Stopping early")
                break

        if self.callback:
            self.callback.after_fitting()

    def __train_step(self, model: Module, dl_train: DataLoader, epoch: int):
        print(f"Epoch {epoch}/{self.max_epochs}")

        model.train()
        self.train_metrics.reset()
        epoch_loss = 0.0
        running_loss = 0.0

        # one pass on entire training set
        for batch, (x, y) in enumerate(dl_train):

            # move tensors to the same device as the model
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # forward propagation
            pred = model(x)

            # loss and metric calculation
            loss = self.criterion(pred, y)
            self.train_metrics(pred, y)
            epoch_loss += loss.item()
            running_loss += loss.item()

            # backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # intermediate logging
            if self.log_every > 0 and (batch + 1) % self.log_every == 0:
                self.__log_progress(running_loss / 100, self.train_metrics.compute(), batch + 1, len(dl_train)),
                running_loss = 0.0

        # calculate total loss and metrics
        loss = epoch_loss / len(dl_train)
        metrics = self.train_metrics.compute()
        self.__log_progress(loss, metrics, len(dl_train), len(dl_train), end="\n")

        if self.callback:
            return self.callback.after_train_step(model, loss, metrics, epoch)

        return False

    @torch.no_grad()
    def __eval_step(self, model: Module, dl_eval: DataLoader, epoch: int):
        model.eval()
        module = torchutils.unwrap_model(model).to("cpu")
        self.eval_metrics.to("cpu")
        self.eval_metrics.reset()
        self.criterion.to("cpu")
        epoch_loss = 0.0

        for x, y in dl_eval:
            x = x.to("cpu", non_blocking=True)
            y = y.to("cpu", non_blocking=True)

            pred = module(x)
            epoch_loss += self.criterion(pred, y).item()
            self.eval_metrics(pred, y)

        # compute final result
        loss = epoch_loss / len(dl_eval)
        metrics = self.eval_metrics.compute()

        model.to(self.device)
        self.criterion.to(self.device)

        if self.callback:
            return self.callback.after_eval_step(model, loss, metrics, epoch)

        return False

    @staticmethod
    def __log_progress(loss, metrics, batch, size, end="\r"):
        progress = (20 * batch) // size
        print(f"{batch}/{size}", end=" ")
        print(f"[{progress * '=':20}]", end=" ")
        print(f"loss: {loss:.8f}", end=" ")
        for key, val in metrics.items():
            print(f"| {key}: {float(val):.8f}", end=" ")
        print(end=end)
