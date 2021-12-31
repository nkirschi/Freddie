"""
This file contains the main training logic.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"

import torch

from torch.nn import Module, DataParallel
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch._C import device
from torch.nn.modules.loss import _Loss
from torchmetrics.collections import MetricCollection
from abc import ABC


class Fitter:
    """
    A generic trainer for deep learning models based on PyTorch and TorchMetrics.
    """

    def __init__(self, optimizer, criterion, train_metrics, eval_metrics, *,
                 max_epochs, log_every, train_device, eval_device, callbacks=()):
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
        train_device : device or str
            The device to use for the training step.
        eval_device : device or str
            The device to use for the evaluation step.
        callbacks : list of Callback
            A list of callback objects with hooks being called at key points during fitting.
        """

        self.optimizer = optimizer
        self.criterion = criterion
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics

        self.max_epochs = max_epochs
        self.log_every = log_every
        self.train_device = torch.device(train_device)
        self.eval_device = torch.device(eval_device)
        self.callbacks = callbacks

    def fit(self, model, dl_train, dl_eval):
        """
        Fits the given model to the training data while evaluating on a holdout set.

        Parameters
        ----------
        model : Module
            The model to fit. Do NOT wrap it with DataParallel.
        dl_train : DataLoader
            The data loader yielding the training data.
        dl_eval : DataLoader
            The data loader yielding the evaluation data.
        """

        print(f"Training on {len(dl_train.dataset)} samples")
        print(f"Evaluating on {len(dl_eval.dataset)} samples")

        model = DataParallel(model)

        for epoch in range(1, self.max_epochs + 1):
            print(f"Epoch {epoch}/{self.max_epochs}")
            self.__train_step(model, dl_train, epoch)
            self.__eval_step(model, dl_eval, epoch)

            if any(cb.after_epoch(epoch) for cb in self.callbacks):
                print("Stopping early")
                break

        torch.cuda.empty_cache()

    def __train_step(self, model: Module, dl_train: DataLoader, epoch: int):
        model = model.module if self.train_device.type == "cpu" else model
        self._ensure_device(model, self.train_device)

        model.train()
        epoch_loss = 0.0
        running_loss = 0.0
        self.train_metrics.reset()

        # one pass on entire training set
        for batch, (x, y) in enumerate(dl_train):

            # move tensors to the training device
            x = x.to(self.train_device)
            y = y.to(self.train_device)

            # forward propagation
            pred = model(x)

            # loss and metric calculation
            loss = self.criterion(pred, y)
            self.train_metrics(pred, y)
            epoch_loss += loss.item()
            running_loss += loss.item()

            # backward propagation
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # intermediate logging
            if self.log_every > 0 and (batch + 1) % self.log_every == 0:
                self._log_progress(running_loss / self.log_every, self.train_metrics.compute(),
                                   batch + 1, len(dl_train)),
                running_loss = 0.0

        # calculate total loss and metrics
        loss = epoch_loss / len(dl_train)
        metrics = self.train_metrics.compute()
        self.train_metrics.reset()
        self._log_progress(loss, metrics, len(dl_train), len(dl_train), end="\n")

        for cb in self.callbacks:
            cb.after_train_step(self._unwrap(model), loss, metrics, epoch)

    @torch.no_grad()
    def __eval_step(self, model: Module, dl_eval: DataLoader, epoch: int):
        model = model.module if self.eval_device.type == "cpu" else model
        self._ensure_device(model, self.eval_device)

        model.eval()
        epoch_loss = 0.0
        running_loss = 0.0
        self.eval_metrics.reset()

        for batch, (x, y) in enumerate(dl_eval):

            # move tensors to the evaluation device
            x = x.to(self.eval_device)
            y = y.to(self.eval_device)

            # forward propagation
            pred = model(x)

            # loss and metric calculation
            loss = self.criterion(pred, y)
            self.eval_metrics(pred, y)
            epoch_loss += loss.item()
            running_loss += loss.item()

            # intermediate logging
            if self.log_every > 0 and (batch + 1) % self.log_every == 0:
                self._log_progress(running_loss / self.log_every, self.eval_metrics.compute(),
                                   batch + 1, len(dl_eval)),
                running_loss = 0.0

        # calculate total loss and metrics
        loss = epoch_loss / len(dl_eval)
        metrics = self.eval_metrics.compute()
        self.eval_metrics.reset()
        self._log_progress(loss, metrics, len(dl_eval), len(dl_eval), end="\n")

        for cb in self.callbacks:
            cb.after_eval_step(self._unwrap(model), loss, metrics, epoch)

    def _ensure_device(self, model, device):
        model.to(device)
        self.criterion.to(device)
        self.train_metrics.to(device)
        self.eval_metrics.to(device)

    @staticmethod
    def _unwrap(model):
        return model.module if isinstance(model, DataParallel) else model

    @staticmethod
    def _log_progress(loss, metrics, batch, size, end="\r"):
        print(f"{f'{batch}/{size}':>16}", end=" ")
        print(f"[{(24 * batch) // size * '=':24}]", end=" ")
        print(f"loss: {loss:.8f}", end=" ")
        for key, val in metrics.items():
            if val.dim() == 0:
                print(f"| {key}: {float(val):.8f}", end=" ")
        print(end=end)


class Callback(ABC):
    """
    An abstract callback for 'Fitter' defining hooks for different execution points during training.
    """

    def after_train_step(self, model: Module, loss: float, metrics: dict, epoch: int):
        """
        Hook called after a training step.

        Parameters
        ----------
        model: Module
            The current state of the model being trained.
        loss: float
            The current training loss after the training step.
        metrics: dict
            A dictionary of all current training metrics values.
        epoch: int
            The current epoch number.
        """
        pass

    def after_eval_step(self, model: Module, loss: float, metrics: dict, epoch: int):
        """
        Hook called after an evaluation step.

        Parameters
        ----------
        model: Module
            The current state of the model being trained.
        loss: float
            The current evaluation loss after the evaluation step.
        metrics: dict
            A dictionary of all current evaluation metrics values.
        epoch: int
            The current epoch number.
        """
        pass

    def after_epoch(self, epoch: int) -> bool:
        """
        Hook called after an entire epoch is finished.

        Parameters
        ----------
        epoch: int
            The current epoch number.
        """
        pass
