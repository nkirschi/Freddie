import random
import os
import numpy as np
import torch

from torch.nn import DataParallel


# see https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
def apply_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def is_model_on_gpu(model):
    return next(model.parameters()).is_cuda

