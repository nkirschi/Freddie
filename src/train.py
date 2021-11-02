# %%

import torch
import utils.ioutils as ioutils
import utils.torchutils as torchutils
import yaml
import wandb
import models
import warnings
import constants as const

from torch.nn import DataParallel, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, F1, AUROC
from auprc import AUPRC
from messenger_dataset import MessengerDataset
from fitter import Fitter
from pprint import pprint
from logging_callback import LoggingCallback


def load_config():
    # load hyperparameter configuration from file
    with open(ioutils.resolve_path(const.HPARAMS_FILE)) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    # load technical parameter configuration from file
    with open(ioutils.resolve_path(const.TPARAMS_FILE)) as f:
        tparams = yaml.load(f, Loader=yaml.FullLoader)

    # hyperparameters given via CLI have higher priority
    hparams.update(ioutils.parse_cli_keyword_args())

    return hparams, tparams


def set_up_devices():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parallelized = device.type != "cpu" and torch.cuda.device_count() > 1
    print(f"Device: {device}", f"#GPUs: {torch.cuda.device_count()}", sep="\n")
    return device, parallelized


def print_section(title):
    print(80 * "=", f"\n{title}\n", 80 * "=", sep="")


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
                          pin_memory=True)
    dl_eval = DataLoader(ds_eval,
                         batch_size=HPARAMS["batch_size"],
                         num_workers=TPARAMS["num_workers"],
                         pin_memory=True)
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
        "auroc": AUROC(num_classes=len(const.CLASSES),
                       compute_on_step=False,
                       dist_sync_on_step=True,
                       average=None),
        "auprc": AUPRC(num_classes=len(const.CLASSES),
                       compute_on_step=False,
                       dist_sync_on_step=True)
    })

    return train_metrics, eval_metrics

# initialize training environment
HPARAMS, TPARAMS = load_config()
run_path, run_id = ioutils.create_run_directory()
torchutils.apply_global_seed(HPARAMS["seed"])  # apply the same seed to all libraries for reproducability
device, parallelized = set_up_devices()

print_section(f"Run #{run_id}")
print("hyperparameters:")
pprint(HPARAMS)
print("technical parameters:")
pprint(TPARAMS)

if TPARAMS["wandb_enabled"]:
    wandb.init(project=TPARAMS["wandb_project"],
               entity=TPARAMS["wandb_entity"],
               group=TPARAMS["wandb_group"],
               dir=TPARAMS["wandb_dir"],
               name=const.RUN_NAME(run_id),
               config=HPARAMS)

# prepare data loaders
dl_train, dl_eval, class_dist = prepare_dataloaders()

# create model
Arch = getattr(models, HPARAMS["model_arch"])
model = Arch(window_size=HPARAMS["window_size"],
             future_size=HPARAMS["future_size"],
             num_channels=len(HPARAMS["features"]))
if TPARAMS["wandb_enabled"]:
    wandb.watch(model, log="all")
if parallelized:
    model = DataParallel(model)


# define optimization criterion
weights = sum(class_dist) / class_dist
criterion = CrossEntropyLoss(weight=weights)

# define gradient descent optimizer
Optim = getattr(torch.optim, HPARAMS["optimizer"])
optimizer = Optim(model.parameters(), lr=HPARAMS["learning_rate"])

# define evaluation metrics
train_metrics, eval_metrics = define_metrics()

# fit the model to the training set
callback = LoggingCallback(run_path, TPARAMS["wandb_enabled"])
fitter = Fitter(optimizer, criterion, train_metrics, eval_metrics,
                max_epochs=TPARAMS["max_epochs"],
                log_every=10,
                device=device,
                callback=callback)
fitter.fit(model, dl_train, dl_eval)
