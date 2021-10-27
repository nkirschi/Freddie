# %%

import os
import shutil
import torch
import torch.nn as nn
import utils
import json
import wandb
import models
import constants as const

from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, F1
from messenger_dataset import MessengerDataset
from fitter import Fitter


def load_params():
    with open(utils.resolve_path(const.HPARAMS_FILE)) as h, \
            open(utils.resolve_path(const.TPARAMS_FILE)) as t:
        return json.load(h), json.load(t)


def create_run_directory():
    path = utils.resolve_path(const.RUNS_DIR)  # runs directory
    run_ids = list(map(int, next(os.walk(path))[1]))  # subfolders as ints
    next_id = (max(run_ids) + 1) if run_ids else 0  # highest id plus one
    run_path = os.path.join(path, f"{next_id:04d}")  # path for new run
    os.mkdir(run_path)

    return run_path


def set_up_devices():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parallelized = device != "cpu" and torch.cuda.device_count() > 1
    print(f"Device: {device}", f"#GPUs: {torch.cuda.device_count()}", sep="\n")
    return device, parallelized


def save_model(model, path, filename):
    module = model.module if parallelized else model
    dest = os.path.join(path, filename + ".pth")
    torch.save(module.state_dict(), dest)
    return dest


def train_step_callback(model, loss, metrics, epoch):
    model_path = save_model(model, run_path, f"epoch_{epoch:02d}")
    if TPARAMS["wandb_enabled"]:
        wandb.save(model_path, run_path)
        wandb.log({"train_loss": loss}, step=epoch)
        wandb.log({key: val for key, val in metrics.items()}, step=epoch)


def eval_step_callback(model, loss, metrics, epoch):
    if TPARAMS["wandb_enabled"]:
        wandb.log({"eval_loss": loss}, step=epoch)
        wandb.log({key: val for key, val in metrics.items()}, step=epoch)


# initialize training environment
HPARAMS, TPARAMS = load_params()

run_path = create_run_directory()
shutil.copy(utils.resolve_path(const.HPARAMS_FILE), os.path.join(run_path, const.HPARAMS_FILE))
if TPARAMS["wandb_enabled"]:
    wandb.init(project=TPARAMS["wandb_project"], entity=TPARAMS["wandb_entity"], config=HPARAMS)
device, parallelized = set_up_devices()
utils.apply_global_seed(HPARAMS["seed"])  # apply the same seed to all libraries for reproducability

# prepare data loaders
ds = MessengerDataset(utils.resolve_path(const.DATA_DIR),
                      features=HPARAMS["features"],
                      window_size=HPARAMS["window_size"],
                      future_size=HPARAMS["future_size"],
                      # use_orbits=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                      )
ds_train, ds_eval = ds.split(HPARAMS["eval_split"])
dl_train = DataLoader(ds_train,
                      batch_size=HPARAMS["batch_size"],
                      num_workers=TPARAMS["num_workers"],
                      pin_memory=True)
dl_eval = DataLoader(ds_eval,
                     batch_size=HPARAMS["batch_size"],
                     num_workers=TPARAMS["num_workers"],
                     pin_memory=True)

# create model
Arch = getattr(models, HPARAMS["model_arch"])
model = Arch(window_size=HPARAMS["window_size"],
             future_size=HPARAMS["future_size"],
             num_channels=len(HPARAMS["features"]))
if parallelized:
    model = nn.DataParallel(model)
model.to(device, non_blocking=True)
if TPARAMS["wandb_enabled"]:
    wandb.watch(model.module if parallelized else model, log="all")

# define optimization criterion
class_dist = ds.get_class_distribution()
weights = sum(class_dist) / class_dist
criterion = nn.CrossEntropyLoss(weight=weights).to(device)

# define gradient descent optimizer
Optim = getattr(torch.optim, HPARAMS["optimizer"])
optimizer = Optim(model.parameters(), lr=HPARAMS["learning_rate"])

# define evaluation metrics
metrics = MetricCollection({
    "accuracy": Accuracy(num_classes=len(const.CLASSES),
                         dist_sync_on_step=True,
                         compute_on_step=False),
    "macro_f1": F1(num_classes=len(const.CLASSES),
                   average="macro",
                   mdmc_average="global",
                   dist_sync_on_step=True,
                   compute_on_step=False)
}).to(device, non_blocking=True)

# fit the model to the training set
fitter = Fitter(optimizer, criterion, metrics,
                max_epochs=TPARAMS["max_epochs"],
                device=device,
                train_step_callback=train_step_callback,
                eval_step_callback=eval_step_callback)
fitter.fit(model, dl_train, dl_eval)
