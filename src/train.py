
# %%

import torch
import torch.nn as nn
import utils
import neptune.new as neptune

from constants import *
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1
from messenger_dataset import ClassificationDataset, PredictionDataset
from models import BaseNet

# %%

MODEL_NAME = "baseline"
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

model = BaseNet(num_bands=2)
if parallelized:
    model = nn.DataParallel(model)
model.to(device, non_blocking=True)

# %%

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
accuracy = Accuracy(num_classes=len(CLASSES)).to(device, non_blocking=True)
f1_macro = F1(num_classes=len(CLASSES), average="macro", mdmc_average="global").to(device, non_blocking=True)

# %%

params = {
    "batch_size": BATCH_SIZE,
    "dropout": DROPOUT,
    "learning_rate": LEARNING_RATE,
    "optimizer": type(optimizer).__name__,
    "criterion": type(criterion).__name__
}
run["params"] = params

# %%

def save_model(model, filename):
    module = model.module if parallelized else model
    path = utils.resolve_path(MODELS_DIR, MODEL_NAME, filename + ".pth")
    torch.save(module.state_dict(), path)
    return path


# %%

print("Training...", end="\n\n")
model.train()
console_log = lambda loss, acc, f1: f"loss: {loss:.8f} | acc: {acc:.8f} | f1: {f1:.8f} [{batch + 1}/{size}]"

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}\n" + 16 * "-")
    size = len(train_dl)

    for batch, (X, y) in enumerate(train_dl):
        # move tensors to device
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # forward propagation
        pred = model(X)

        # metric calculation
        loss = criterion(pred, y)
        accuracy(pred, y)
        f1_macro(pred, y)

        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # intermediate logging
        if (batch + 1) % 10 == 0:
            loss = loss.item()
            acc = accuracy.compute()
            f1 = f1_macro.compute()
            print(console_log(loss, acc, f1), end="\r")

    # calculate metrics and log them
    loss = loss.item()
    acc = accuracy.compute()
    f1 = f1_macro.compute()

    run["train/loss"].log(loss)
    run["train/accuracy"].log(acc)
    run["train/f1_macro"].log(f1)

    print(console_log(loss, acc, f1), end="\n\n")

    # save model checkpoint
    path = save_model(model, f"epoch_{epoch:02d}")
    run[f"checkpoints/epoch{epoch:02d}"].upload(path)

# %%

print("Evaluating...", end="\n\n")
model.eval()
accuracy.reset()
f1_macro.reset()

with torch.no_grad():
    for X, y in test_dl:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        accuracy(model(X), y)
        f1_macro(model(X), y)

    # calculate metrics
    acc = accuracy.compute()
    f1 = f1_macro.compute()

    # log metrics to Neptune
    run["eval/accuracy"] = acc
    run["eval/f1_macro"] = f1

    # log metrics to console
    print(f"accuracy: {acc}")
    print(f"f1_macro: {f1}")

# %%

run.stop()
