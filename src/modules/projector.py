"""
This file contains a custom PyTorch module for projecting onto tensor dimensions.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"


from torch import nn


class Projector(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x[self.dim]

    def __repr__(self):
        return f"Projector(dim={self.dim})"