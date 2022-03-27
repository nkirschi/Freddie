"""
This is the active learning script.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"

import torch
import utils.constants as c
import utils.io as io
import ordinary_training
import json

from collections import defaultdict
from torch.nn.functional import softmax
from tqdm import tqdm


def entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Calculates the entropy of a discrete distribution along an axis in the given tensor.

    Parameters
    ----------
    p: torch.Tensor
        A tensor of probabilities.
    dim: int, default -1
        The dimension along which to compute entropy.

    Returns
    -------
    torch.Tensor
        A tensor with one dimension less containing the entropy scores.
    """

    return -torch.where(p > 0, p * p.log(), p.new([0.])).sum(dim=dim)


def orbit_uncert(logits: torch.Tensor) -> float:
    """
    Calculates the orbit uncertainty score for an orbit's predictions as defined in the paper.

    Parameters
    ----------
    logits: torch.Tensor
        A tensor with log-probabilities as produced by a model for the Freddie task.

    Returns
    -------
    float:
        The orbit-integrated uncertainty for the given predictions.
    """

    return float(entropy(softmax(logits, dim=1), dim=1).max(dim=1).values.mean())


def worst_pred(logits: torch.Tensor) -> torch.Tensor:
    """
    Determines the worst single time step prediction among the given orbit predictions.

    Parameters
    ----------
    logits: torch.Tensor
        A tensor with log-probabilities as produced by a model for the Freddie task.

    Returns
    -------
    torch.Tensor
        A one-dimensional tensor with the most uncertain single time step prediction.
    """

    expits = softmax(logits.transpose(1, 2), dim=2)
    return expits.flatten(0, 1)[entropy(expits, dim=2).argmax()]


def predict_chunked(model, x, device):
    """
    Performs a model prediction on the given input in a chunked manner.

    Parameters
    ----------
    model: torch.nn.Module
        The model to predict with.
    x: torch.Tensor
        The input to predict for.
    device: torch._C.device
        The device to predict on.

    Returns
    -------
    A big tensor with all chunk-wise predictions combined.
    """
    return torch.cat([model(chunk.to(device)) for chunk in x.split(5000)])


def prepare_orbits():
    """
    Loads all training orbits and extracts windows containing an SK or MP label.

    Returns
    -------
    dict(int):
        A dictionary of filtered orbit tensors.
    """

    print("Preparing training data...")
    ds_train, _ = ordinary_training.load_datasets(hparams)
    path = io.resolve_path(c.TEMP_DIR) / "critical_windows_full.pt"
    if path.is_file():
        print("Loading cached critical windows...")
        all_orbits = torch.load(path)
    else:
        print("Filtering critical windows...")
        all_orbits = {k: torch.stack([sample for sample, label in v if 1 in label or 3 in label]).detach()
                      for k, v in tqdm(ds_train.explode_orbits().items())}
        torch.save(all_orbits, path)
    return all_orbits

################################################################################


# load config, model and data
hparams, tparams = ordinary_training.load_config()
model = ordinary_training.construct_model(hparams)
device = torch.device(tparams["train_device"])
all_orbits = prepare_orbits()

# active learning loop initialization
iteration = 0
train_orbits = []
history = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

# training set increment function as discussed in the paper
increment = lambda n: max(10, int(n / 4))


while len(train_orbits) < len(all_orbits):
    print(f"iteration #{iteration}: trained on {len(train_orbits)} so far")

    with torch.no_grad():
        model.to(device)
        model.eval()

        # determine uncertainty scores and worst predictions for unseen orbits
        uncerts = {}
        for key, orbit_tensor in all_orbits.items():
            if key not in train_orbits:
                preds = predict_chunked(model, orbit_tensor, device)
                uncerts[key] = orbit_uncert(preds)
                history[iteration]["orbit_uncertainty"][int(key)] = uncerts[key]
                history[iteration]["worst_prediction"][int(key)] = worst_pred(preds).tolist()

        # add increment(#train_orbits) many orbits with highest uncertainty to the training set
        worst_orbits = sorted(uncerts, key=uncerts.get, reverse=True)[:increment(len(train_orbits))]
        train_orbits += worst_orbits

    with open(io.resolve_path(c.TEMP_DIR) / "al_results.json", "w") as f:
        json.dump(history, f, indent=4)

    # retrain with updated set of orbits
    hparams["train_orbits"] = train_orbits
    tparams["wandb_notes"] = f"AL #{iteration} ({len(train_orbits)}/{len(all_orbits)})"
    ordinary_training.perform_train(model, hparams, tparams)
    iteration += 1
