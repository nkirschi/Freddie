"""
Utility functions for training runs.
"""

from learning import models
import numpy as np
import os
import random
import sys
import torch
import yaml
import wandb

from callbacks.best_model_callback import BestModelCallback
from callbacks.checkpointing_callback import CheckpointingCallback
from callbacks.early_stopping_callback import EarlyStoppingCallback
from callbacks.metric_logging_callback import MetricLoggingCallback
from callbacks.wandb_callback import WandBCallback
from learning.fitter import Fitter
from learning.datasets import MessengerDataset
from pprint import pprint
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy, F1, MetricCollection
from utils import constants as const, io as ioutils


# see https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
def apply_global_seed(seed: int):
    """
    Applies the same seed to all libraries that make use of randomness and enforces deterministic GPU computation.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def define_metrics():
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


def load_datasets(hparams):
    ds_train = MessengerDataset(ioutils.resolve_path(const.DATA_DIR),
                                split="train",
                                features=hparams["features"],
                                window_size=hparams["window_size"],
                                future_size=hparams["future_size"],
                                use_orbits=hparams["train_orbits"])
    ds_eval = MessengerDataset(ioutils.resolve_path(const.DATA_DIR),
                               split="eval",
                               features=hparams["features"],
                               window_size=hparams["window_size"],
                               future_size=hparams["future_size"],
                               use_orbits=hparams["eval_orbits"])
    return ds_train, ds_eval


def prepare_dataloaders(ds_train, ds_eval, hparams, tparams):
    dl_train = DataLoader(ds_train,
                          batch_size=hparams["batch_size"],
                          num_workers=tparams["num_workers"],
                          shuffle=True,
                          pin_memory=True
                          )
    dl_eval = DataLoader(ds_eval,
                         batch_size=hparams["batch_size"],
                         num_workers=tparams["num_workers"],
                         pin_memory=True
                         )
    return dl_train, dl_eval


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


def construct_model(hparams):
    Arch = getattr(models, hparams["model_arch"])
    return Arch(num_channels=len(hparams["features"]),
                window_size=hparams["window_size"],
                future_size=hparams["future_size"],
                hidden_sizes=[hparams[f"hidden_size{i}"] for i in range(hparams["hidden_layers"])],
                channel_sizes=[hparams[f"channel_size{i}"] for i in range(hparams["conv_layers"])],
                kernel_sizes=[hparams[f"kernel_size{i}"] for i in range(hparams["conv_layers"])],
                stride_sizes=[hparams[f"stride_size{i}"] for i in range(hparams["conv_layers"])],
                dilation_sizes=[hparams[f"dilation_size{i}"] for i in range(hparams["conv_layers"])],
                pool_sizes=[hparams[f"pool_size{i}"] for i in range(hparams["conv_layers"])],
                state_sizes=[hparams[f"state_size{i}"] for i in range(hparams["rnn_layers"])],
                attn_sizes=[hparams[f"attn_size{i}"] for i in range(hparams["attn_layers"])],
                head_sizes=[hparams[f"head_size{i}"] for i in range(hparams["attn_layers"])],
                dropout_rate=hparams["dropout_rate"],
                batch_normalization=hparams["batch_normalization"])


def perform_train(model, hparams, tparams):
    # initialize training environment
    run_path, run_id = ioutils.create_run_directory(hparams)

    # apply the same seed to all libraries for reproducability
    apply_global_seed(hparams["seed"])

    # set up connection the WandB service
    if tparams["wandb_enabled"]:
        wandb.init(project=tparams["wandb_project"],
                   entity=tparams["wandb_entity"],
                   group=tparams["wandb_group"],
                   dir=tparams["wandb_dir"],
                   name=const.RUN_NAME(run_id),
                   notes=tparams["wandb_notes"],
                   config=hparams)

    # log basic information about the run
    print(80 * "=", f"Run #{run_id}", 80 * "=", sep="\n")
    print(f"{torch.cuda.device_count()} GPUs")
    print("hyperparameters:")
    pprint(hparams)
    print("technical parameters:")
    pprint(tparams)

    # let WandB track the model parameters and gradients
    if tparams["wandb_enabled"]:
        wandb.watch(model, log="all")

    # print important info about the model
    print(model)
    summary(model, input_size=(hparams["batch_size"], len(hparams["features"]), hparams["window_size"]), depth=2)

    # prepare dataloaders
    print("Loading dataset...")
    ds_train, ds_eval = load_datasets(hparams)
    dl_train, dl_eval = prepare_dataloaders(ds_train, ds_eval, hparams, tparams)

    # define optimization criterion
    class_dist = ds_train.get_class_frequencies()
    weights = sum(class_dist) / class_dist
    criterion = CrossEntropyLoss(weight=weights)
    print("Weighting CE loss with", weights.tolist())

    # define gradient descent optimizer
    Optim = getattr(torch.optim, hparams["optimizer"])
    optimizer = Optim(model.parameters(), lr=hparams["learning_rate"])

    # define evaluation metrics
    train_metrics, eval_metrics = define_metrics()

    # prepare training hooks
    callbacks = [MetricLoggingCallback(run_path / const.METRICS_FILE),
                 BestModelCallback(run_path / const.BEST_MODEL_FILE),
                 CheckpointingCallback(run_path / const.CKPT_SUBDIR, const.CKPT_FILE)]
    if tparams["wandb_enabled"]:
        callbacks.append(WandBCallback(const.CKPT_FILE, const.CLASSES, summary_metric="eval/macro_f1"))
    if hparams["early_stopping"]:
        callbacks.append(EarlyStoppingCallback(patience=hparams["patience"]))

    # fit the model to the training set
    fitter = Fitter(optimizer, criterion, train_metrics, eval_metrics,
                    max_epochs=tparams["max_epochs"],
                    log_every=10,
                    train_device="cuda",
                    eval_device="cuda",
                    callbacks=callbacks)
    fitter.fit(model, dl_train, dl_eval)

    model.load_state_dict(torch.load(run_path / const.BEST_MODEL_FILE))
