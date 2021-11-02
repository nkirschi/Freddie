import torch
from torchmetrics import AveragePrecision


class AUPRC(AveragePrecision):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self):
        result = super().compute()
        return torch.tensor(result) if isinstance(result, list) else result
