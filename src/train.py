# %%

import torch
import torch.nn as nn
import utils
import neptune.new as neptune

from constants import *
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, F1
from messenger_dataset import ClassificationDataset, PredictionDataset
from models import *

# %%

MODEL_NAME = "cnn"
EPOCHS = 20
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
NUM_WORKERS = 32  # 4 per GPU seems to be a rule of thumb
DROPOUT = 0.2

SEED = 42
EVAL_SPLIT = 0.2
WINDOW_SIZE = 10

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parallelized = device != "cpu" and torch.cuda.device_count() > 1
print(f"Device: {device}", f"#GPUs: {torch.cuda.device_count()}", sep="\n")

# %%

run = neptune.init(
    project="nelorth/freddie",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwYWE3NzA2NS0yMTMwLTQ4YzMtYmYzYy0zYjEyNmVmNTBjMGMifQ==",
)

# %%

utils.apply_global_seed(SEED)

# %%

ds = ClassificationDataset(utils.resolve_path(DATA_DIR, TRAIN_FILE), window_size=WINDOW_SIZE, partial=False)
train_ds, test_ds = ds.split(EVAL_SPLIT)

# %%

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

# %%

model = BaseCNN(window_size=WINDOW_SIZE, num_channels=2)
if parallelized:
    model = nn.DataParallel(model)
model.to(device, non_blocking=True)

# %%

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
metrics = MetricCollection({
    "accuracy": Accuracy(num_classes=len(CLASSES),
                         dist_sync_on_step=True,
                         compute_on_step=False),
    "macro_f1": F1(num_classes=len(CLASSES),
                   average="macro",
                   mdmc_average="global",
                   dist_sync_on_step=True,
                   compute_on_step=False)
}).to(device, non_blocking=True)
train_metrics = metrics.clone()
eval_metrics = metrics.clone()

# %%

run["params"] = {
    "batch_size": BATCH_SIZE,
    "dropout": DROPOUT,
    "learning_rate": LEARNING_RATE,
    "optimizer": type(optimizer).__name__,
    "criterion": type(criterion).__name__
}

# %%


def save_model(model, filename):
    module = model.module if parallelized else model
    path = utils.resolve_path(MODELS_DIR, MODEL_NAME, filename + ".pth")
    torch.save(module.state_dict(), path)
    return path


# %%

console_log = lambda loss, metrics: f"loss: {loss:.8f}".join(
    f" | {key}: {float(val):.8f}" for (key, val) in metrics.items()) + f" [{batch + 1}/{size}]"
size = len(train_dl)

for epoch in range(1, EPOCHS + 1):
    model.train()
    print("Training...", end="\n\n")
    print(f"Epoch {epoch}/{EPOCHS}\n" + 16 * "-")

    running_loss = 0.0

    # one pass on entire training set
    for batch, (X, y) in enumerate(train_dl):
        # move tensors to device
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # forward propagation
        pred = model(X)

        # metric calculation
        loss = criterion(pred, y)
        train_metrics(pred, y)
        running_loss += loss.item()

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # intermediate logging
        if (batch + 1) % 100 == 0:
            print(console_log(running_loss / size, train_metrics.compute()), end="\r")

    # calculate metrics and log them
    loss = running_loss / size
    metrics = train_metrics.compute()
    run["train/loss"].log(loss)
    for key, val in metrics.items():
        run[f"train/{key}"].log(val)
    print(console_log(loss, metrics), end="\n\n")

    # save model checkpoint
    path = save_model(model, f"epoch_{epoch:02d}")
    run[f"checkpoints/epoch{epoch:02d}"].upload(path)

    model.eval()
    print("Evaluating...", end="\n\n")
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        running_loss = 0.0

        for X, y in test_dl:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(X)
            running_loss += criterion(pred, y).item()
            eval_metrics(pred, y)

        # compute final result
        loss = running_loss / size
        metrics = eval_metrics.compute()

        # log metrics to Neptune
        run["eval/loss"].log(loss)
        for key, val in metrics.items():
            run[f"eval/{key}"].log(val)

        # log metrics to console
        print(metrics)

# %%

run.stop()
