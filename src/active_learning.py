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
import utils.training as training
import json

from collections import defaultdict
from torch.nn.functional import softmax
from tqdm import tqdm


def entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -torch.where(p > 0, p * p.log(), p.new([0.])).sum(dim=dim)


def orbit_uncertainty(logits: torch.Tensor):
    return float(entropy(softmax(logits, dim=1), dim=1).max(dim=1).values.mean())


def worst_prediction(logits: torch.Tensor) -> torch.Tensor:
    expits = softmax(logits.transpose(1, 2), dim=2)
    return expits.flatten(0, 1)[entropy(expits, dim=2).argmax()]


def predict(model, x, device):
    half = len(x) // 2
    first = model(x[:half].to(device))
    second = model(x[half:].to(device))
    return torch.cat([first, second])


hparams, tparams = training.load_config()
model = training.construct_model(hparams)
device = torch.device(tparams["train_device"])

# load all training orbits and extract windows containing an SK or MP label
print("Loading training data...")
ds_train, _ = training.load_datasets(hparams)
path = io.resolve_path(c.TEMP_DIR) / "critical_windows_full.pt"
if path.is_file():
    print("Loading critical windows...")
    all_orbits = torch.load(path)
else:
    print("Filtering critical windows...")
    all_orbits = {k: torch.stack([sample for sample, label in v if 1 in label or 3 in label]).detach()
                  for k, v in tqdm(ds_train.explode_orbits().items())}
    torch.save(all_orbits, path)


history = defaultdict(dict)
train_orbits = []
increment = 10
iteration = 0


while len(train_orbits) < len(all_orbits):
    print(f"iteration #{iteration}: trained on {len(train_orbits)} so far")

    with torch.no_grad():
        model.to(device)
        model.eval()

        history[iteration]["uncertainty"] = {}
        history[iteration]["worst_prediction"] = {}

        # calculate uncertainty scores for unseen orbits
        uncerts = {}
        for key, orbit_tensor in all_orbits.items():
            if key not in train_orbits:
                preds = predict(model, orbit_tensor, device)
                uncerts[key] = orbit_uncertainty(preds)
                history[iteration]["uncertainty"][int(key)] = uncerts[key]

        worst_orbits = sorted(uncerts, key=uncerts.get, reverse=True)[:increment]
        train_orbits += worst_orbits

        for o in worst_orbits:
            preds = predict(model, all_orbits[o], device)
            history[iteration]["worst_prediction"][int(o)] = worst_prediction(preds).tolist()

    with open(io.resolve_path(c.TEMP_DIR) / "al_linear.json", "w") as f:
        json.dump(history, f, indent=4)

    hparams["train_orbits"] = train_orbits
    training.perform_train(model, hparams, tparams)
    iteration += 1
