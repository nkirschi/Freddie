import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)