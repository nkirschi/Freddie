import torch

from torch.nn import Module, DataParallel
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch._C import device
from torch.nn.modules.loss import _Loss
from torchmetrics.collections import MetricCollection
from abc import ABC


class Callback(ABC):

    def after_train_step(self, model: Module, loss: float, metrics: dict, epoch: int) -> bool:
        pass

    def after_eval_step(self, model: Module, loss: float, metrics: dict, epoch: int) -> bool:
        pass

    def after_fitting(self):
        pass


class Fitter:
    """
    A generic trainer for deep learning models based on PyTorch and TorchMetrics.
    """

    def __init__(self, optimizer, criterion, train_metrics, eval_metrics, *,
                 max_epochs, log_every, train_device, eval_device, callback=None):
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
        callback : Callback
            A callback object with hooks being called at key points during fitting.
        """

        self.optimizer = optimizer
        self.criterion = criterion
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics

        self.max_epochs = max_epochs
        self.log_every = log_every
        self.train_device = torch.device(train_device)
        self.eval_device = torch.device(eval_device)
        self.callback = callback

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

            # TODO implement early stopping

        if self.callback:
            self.callback.after_fitting()

    def __train_step(self, model: Module, dl_train: DataLoader, epoch: int):
        model = model.module if self.train_device.type == "cpu" else model
        self._ensure_device(model, self.train_device)

        model.train()
        self.train_metrics.reset()
        epoch_loss = 0.0
        running_loss = 0.0

        # one pass on entire training set
        for batch, (x, y) in enumerate(dl_train):

            # move tensors to the training device
            x = x.to(self.train_device, non_blocking=True)
            y = y.to(self.train_device, non_blocking=True)

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
        self.train_metrics.reset()
        self.__log_progress(loss, metrics, len(dl_train), len(dl_train), end="\n")

        if self.callback:
            return self.callback.after_train_step(self._unwrap(model), loss, metrics, epoch)

        return False

    @torch.no_grad()
    def __eval_step(self, model: Module, dl_eval: DataLoader, epoch: int):
        model = model.module if self.eval_device.type == "cpu" else model
        self._ensure_device(model, self.eval_device)

        model.eval()
        self.eval_metrics.reset()
        epoch_loss = 0.0

        for x, y in dl_eval:
            x = x.to(self.eval_device, non_blocking=True)
            y = y.to(self.eval_device, non_blocking=True)

            pred = model(x)
            epoch_loss += self.criterion(pred, y).item()
            self.eval_metrics(pred, y)

        # compute final result
        loss = epoch_loss / len(dl_eval)
        metrics = self.eval_metrics.compute()
        self.eval_metrics.reset()

        if self.callback:
            return self.callback.after_eval_step(self._unwrap(model), loss, metrics, epoch)

        return False

    def _ensure_device(self, model, device):
        model.to(device, non_blocking=True)
        self.criterion.to(device, non_blocking=True)
        self.train_metrics.to(device, non_blocking=True)
        self.eval_metrics.to(device, non_blocking=True)

    @staticmethod
    def _unwrap(model):
        return model.module if isinstance(model, DataParallel) else model

    @staticmethod
    def __log_progress(loss, metrics, batch, size, end="\r"):
        progress = (20 * batch) // size
        print(f"{batch}/{size}", end=" ")
        print(f"[{progress * '=':20}]", end=" ")
        print(f"loss: {loss:.8f}", end=" ")
        for key, val in metrics.items():
            print(f"| {key}: {float(val):.8f}", end=" ")
        print(end=end)
