"""
This file wraps a metric for more convenient use.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"


import torch
from torchmetrics import AveragePrecision


class AUPRC(AveragePrecision):
    """
    Wrapper for the AveragePrecision metric from torchmetrics.

    Its 'compute' method returns the class-wise results as tensor and not a list, therefore complying with the
    overall convention in the library.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        result = super().compute()
        return torch.tensor(result) if isinstance(result, list) else result
