"""
This is the main training script.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"

###############################################################################

import torch
import utils.ioutils as ioutils
import utils.torchutils as torchutils
import sys
import yaml
import wandb
import models
import warnings
import constants as const

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, F1, AUROC
from auprc import AUPRC
from messenger_dataset import MessengerDataset
from fitter import Fitter
from pprint import pprint
from logging_callback import LoggingCallback
from torchinfo import summary


################################################################################

def load_config():
    # load hyperparameter configuration from file
    with open(ioutils.resolve_path(const.CONFIG_DIR) / const.HPARAMS_FILE) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    # load technical parameter configuration from file
    with open(ioutils.resolve_path(const.CONFIG_DIR) / const.TPARAMS_FILE) as f:
        tparams = yaml.load(f, Loader=yaml.FullLoader)

    # hyperparameters given via CLI have higher priority
    hparams.update(ioutils.parse_cli_keyword_args(sys.argv[1:]))

    return hparams, tparams


def prepare_dataloaders():
    ds = MessengerDataset(ioutils.resolve_path(const.DATA_DIR),
                          features=HPARAMS["features"],
                          window_size=HPARAMS["window_size"],
                          future_size=HPARAMS["future_size"],
                          use_orbits=HPARAMS["use_orbits"],
                          )
    ds_train, ds_eval = ds.split(HPARAMS["eval_split"])
    dl_train = DataLoader(ds_train,
                          batch_size=HPARAMS["batch_size"],
                          num_workers=TPARAMS["num_workers"],
                          pin_memory=True
                          )
    dl_eval = DataLoader(ds_eval,
                         batch_size=HPARAMS["batch_size"],
                         num_workers=TPARAMS["num_workers"],
                         pin_memory=True
                         )
    return dl_train, dl_eval, ds.get_class_frequencies()


def define_metrics():
    # ignore annoying memory footprint warnings from torchmetrics
    warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics")

    base_metrics = MetricCollection({
        "accuracy": Accuracy(num_classes=len(const.CLASSES),
                             compute_on_step=False,
                             dist_sync_on_step=True),
        "macro_f1": F1(num_classes=len(const.CLASSES),
                       average="macro",
                       mdmc_average="global",
                       compute_on_step=False,
                       dist_sync_on_step=True)
    })

    train_metrics = base_metrics.clone(prefix="train/")
    eval_metrics = base_metrics.clone(prefix="eval/")
    eval_metrics.add_metrics({
        # "auroc":      AUROC(num_classes=len(const.CLASSES),
        #                     compute_on_step=False,
        #                     dist_sync_on_step=True,
        #                     average=None),
        # "auprc":      AUPRC(num_classes=len(const.CLASSES),
        #                     compute_on_step=False,
        #                     dist_sync_on_step=True),
        "claccuracy": Accuracy(num_classes=len(const.CLASSES),
                               compute_on_step=False,
                               dist_sync_on_step=True,
                               average="none")
    })

    return train_metrics, eval_metrics


################################################################################

# initialize training environment
HPARAMS, TPARAMS = load_config()
run_path, run_id = ioutils.create_run_directory(HPARAMS)
torchutils.apply_global_seed(HPARAMS["seed"])  # apply the same seed to all libraries for reproducability

# set up connection the WandB service
if TPARAMS["wandb_enabled"]:
    wandb.init(project=TPARAMS["wandb_project"],
               entity=TPARAMS["wandb_entity"],
               group=TPARAMS["wandb_group"],
               dir=TPARAMS["wandb_dir"],
               name=const.RUN_NAME(run_id),
               notes=TPARAMS["wandb_notes"],
               config=HPARAMS)

# log basic information about the run
print(80 * "=", f"Run #{run_id}", 80 * "=", sep="\n")
print(f"{torch.cuda.device_count()} GPUs")
print("hyperparameters:")
pprint(HPARAMS)
print("technical parameters:")
pprint(TPARAMS)

# create model
Arch = getattr(models, HPARAMS["model_arch"])
model = Arch(num_channels=len(HPARAMS["features"]),
             window_size=HPARAMS["window_size"],
             future_size=HPARAMS["future_size"],
             hidden_sizes=[HPARAMS[f"hidden_size{i}"] for i in range(HPARAMS["hidden_layers"])],
             channel_sizes=[HPARAMS[f"channel_size{i}"] for i in range(HPARAMS["conv_layers"])],
             kernel_sizes=[HPARAMS[f"kernel_size{i}"] for i in range(HPARAMS["conv_layers"])],
             stride_sizes=[HPARAMS[f"stride_size{i}"] for i in range(HPARAMS["conv_layers"])],
             pool_sizes=[HPARAMS[f"pool_size{i}"] for i in range(HPARAMS["conv_layers"])],
             rnn_layers=HPARAMS["rnn_layers"],
             rnn_state_size=HPARAMS["rnn_state_size"],
             dropout_rate=HPARAMS["dropout_rate"])

# let WandB track the model parameters and gradients
if TPARAMS["wandb_enabled"]:
    wandb.watch(model, log="all")

# print important info about the model
print(model)
summary(model, input_size=(len(HPARAMS["features"]), HPARAMS["window_size"]), batch_dim=0)

# prepare data loaders
dl_train, dl_eval, class_dist = prepare_dataloaders()

# define optimization criterion
weights = sum(class_dist) / class_dist
criterion = CrossEntropyLoss(weight=weights)
print(f"weighting CE loss with {weights}")

# define gradient descent optimizer
Optim = getattr(torch.optim, HPARAMS["optimizer"])
optimizer = Optim(model.parameters(), lr=HPARAMS["learning_rate"])

# define evaluation metrics
train_metrics, eval_metrics = define_metrics()

# fit the model to the training set
fitter = Fitter(optimizer, criterion, train_metrics, eval_metrics,
                max_epochs=TPARAMS["max_epochs"],
                log_every=10,
                train_device="cuda",
                eval_device="cuda",
                callback=LoggingCallback(run_path, TPARAMS["wandb_enabled"]))
fitter.fit(model, dl_train, dl_eval)
