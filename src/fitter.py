import torch

from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchmetrics.collections import MetricCollection


class Fitter:

    def __init__(self, optimizer: Optimizer, criterion: Module, metrics: MetricCollection, *,
                 max_epochs: int, device="cuda", train_step_callback=None, eval_step_callback=None):

        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics

        self.max_epochs = max_epochs
        self.device = device
        self.train_step_callback = train_step_callback
        self.eval_step_callback = eval_step_callback

    def fit(self, model: Module, dl_train: DataLoader, dl_eval: DataLoader):
        for epoch in range(1, self.max_epochs + 1):
            self.__train_step(model, dl_train, epoch)
            self.__eval_step(model, dl_eval, epoch)

    def __train_step(self, model: Module, dl_train: DataLoader, epoch: int):
        print("Training...", end="\n\n")
        print(f"Epoch {epoch}/{self.max_epochs}\n" + 16 * "-")

        model.train()
        train_metrics = self.metrics.clone(prefix="train_")
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
            train_metrics(pred, y)
            epoch_loss += loss.item()
            running_loss += loss.item()

            # backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # intermediate logging
            if (batch + 1) % 100 == 0:
                self.__log_progress(running_loss / 100, train_metrics.compute(), batch + 1, len(dl_train)),
                running_loss = 0.0

        # calculate total loss and metrics
        loss = epoch_loss / len(dl_train)
        metrics = train_metrics.compute()
        self.__log_progress(loss, metrics, len(dl_train), len(dl_train), end="\n\n")

        if self.train_step_callback:
            self.train_step_callback(model, loss, metrics, epoch)

    @torch.no_grad()
    def __eval_step(self, model: Module, dl_eval: DataLoader, epoch: int):
        print("Evaluating...", end="\n\n")

        model.eval()
        eval_metrics = self.metrics.clone(prefix="eval_")
        epoch_loss = 0.0

        for x, y in dl_eval:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            pred = model(x)
            epoch_loss += self.criterion(pred, y).item()
            eval_metrics(pred, y)

        # compute final result
        loss = epoch_loss / len(dl_eval)
        metrics = eval_metrics.compute()
        print(metrics)

        if self.eval_step_callback:
            self.eval_step_callback(model, loss, metrics, epoch)

    @staticmethod
    def __log_progress(loss, metrics, batch, size, end="\r"):
        print(f"loss: {loss:.8f}", end=" ")
        for key, val in metrics.items():
            print(f"| {key}: {float(val):.8f}", end=" ")
        print(f" [{batch}/{size}]", end=end)
