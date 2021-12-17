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


import constants as c
import random
import heapq
import torch

from torch.nn.functional import softmax

import utils.training
from utils import torchutils, training
from utils.timer import Timer


def confidence(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return ((p - 1 / len(p)) ** 2).sum(dim=dim)


def entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -torch.where(p > 0, p * p.log(), p.new([0.])).sum(dim=dim)


def orbit_confidence(preds: torch.Tensor):
    return float(entropy(softmax(preds, dim=1), dim=1).max(dim=1).values.mean())


hparams, tparams = training.load_config()
utils.training.apply_global_seed(hparams["seed"])

ds = training.load_dataset(hparams)
orbits = {k: torch.stack([sample for sample, label in v if 1 in label or 3 in label]) for k, v in ds.explode_orbits().items()}
model = training.construct_model(hparams)

# start with single random orbit
train_orbits = [random.choice(list(orbits.keys()))]
start_index = 0

increment = 1

while len(train_orbits) < 0.7 * len(orbits):
    print(f"train_orbits: {train_orbits}")

    hparams["train_split"] = train_orbits[start_index:]
    # training.perform_train(model, ds, hparams, tparams)
    # now model should have params from lowest loss

    unseen_orbits = {k: v for (k, v) in orbits.items() if k not in train_orbits}

    conf_vals = {}
    for key, orbit_tensor in unseen_orbits.items():
        with torch.no_grad():
            preds = model(orbit_tensor)
        conf = orbit_confidence(preds)
        conf_vals[key] = conf

        print(f"confidence on orbit #{key}:", conf)
        print(f"-> number of relevant samples: {len(orbit_tensor)}")

    start_index = len(train_orbits)
    train_orbits += heapq.nlargest(increment, conf_vals, conf_vals.get)

    print(f"most inconfident orbits: {train_orbits[start_index:]}")

