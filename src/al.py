import constants as c
import random
import heapq
import torch

from torch.nn.functional import softmax
from utils import torchutils, training
from utils.timer import Timer


def confidence(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return ((p - 1 / len(p)) ** 2).sum(dim=dim)


def entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.where(p > 0, p * p.log(), p.new([0.])).sum(dim=dim)


def orbit_confidence(preds: torch.Tensor):
    return float(entropy(softmax(preds, dim=1), dim=1).min(dim=1).values.mean())


hparams, tparams = training.load_config()
torchutils.apply_global_seed(hparams["seed"])


ds = training.load_dataset(hparams)
orbits = ds.explode_orbits()
model = training.construct_model(hparams)

# start with single random orbit
train_orbits = [random.choice(list(ds.get_orbits().keys()))]

increment = 1
done = False

while not done:
    print(f"train_orbits: {train_orbits}")
    ds_train, ds_eval = ds.split(train_orbits)

    if len(ds_eval) == 0:
        break

    dl_train, dl_eval = training.prepare_dataloaders(ds_train, ds_eval, hparams, tparams)
    #training.perform_train(model, ds, dl_train, dl_eval, hparams, tparams)
    # now model should have params from lowest loss

    ds_orbits = {k: v for (k, v) in orbits.items() if k not in train_orbits}

    conf_vals = []
    for key, ds_orbit in ds_orbits.items():
        with Timer("stacking"):
            all_windows = torch.stack([sample for sample, label in ds_orbit])
        with Timer("inferring"):
            conf = orbit_confidence(model(all_windows))
        heapq.heappush(conf_vals, (conf, key))

        print(f"confidence on orbit #{key}:", conf)

    # top_indices = torch.topk(torch.tensor(conf_vals), increment).indices
    # keys = list(ds_orbits.keys())
    # most_inconfident_orbits = [keys[i] for i in top_indices]
    most_inconfident_orbits = [key for _, key in conf_vals[:increment]]
    print(f"most inconfident orbits: {most_inconfident_orbits}")
    train_orbits += most_inconfident_orbits
