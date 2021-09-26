import random, os
import numpy as np
import torch


def normalize_decoupled(data, cols):
    data[cols] = (data[cols] - data[cols].mean()) / data[cols].std()


def normalize_coupled(data, cols):
    centered = data[cols] - data[cols].mean()
    distdev = (centered ** 2).sum(axis=1).mean() ** 0.5
    data[cols] = centered / distdev


def apply_global_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True