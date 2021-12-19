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

import random
import torch

from torch.nn.functional import softmax

from utils import training


def entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -torch.where(p > 0, p * p.log(), p.new([0.])).sum(dim=dim)


def orbit_confidence(preds: torch.Tensor):
    return float(entropy(softmax(preds, dim=1), dim=1).max(dim=1).values.mean())


hparams, tparams = training.load_config()

ds_train, _ = training.load_datasets(hparams)
orbits = {k: torch.stack([sample for sample, label in v if 1 in label or 3 in label]) for k, v in ds_train.explode_orbits().items()}
model = training.construct_model(hparams)

# start with single random orbit
train_orbits = [random.choice(list(orbits.keys()))]
start_index = 0

increment = 1

while len(train_orbits) < len(orbits):
    print(f"train_orbits: {train_orbits}")

    hparams["train_split"] = train_orbits[start_index:]
    # training.perform_train(model, hparams, tparams)
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
    worst_orbits = sorted(conf_vals, key=conf_vals.get, reverse=True)[:increment]
    # worst_orbits = heapq.nlargest(increment, conf_vals, conf_vals.get)
    train_orbits += worst_orbits

    print(f"most inconfident orbits: {train_orbits[start_index:]}")

