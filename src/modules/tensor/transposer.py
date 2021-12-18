"""
This file contains a custom PyTorch module frequently needed in model definitions.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"


from torch import nn


class Transpose(nn.Module):
    """
    Custom module for swapping two dimensions of a tensor.

    Swapping two axes of a tensor is a commonly needed operation when defining deep learning models.
    For instance, when building a CRNN the shape needs to be changed from (batch_size, channel_num, time_steps)
    for convolution modules to (batch_size, time_steps, channel_num) for RNN modules. Thanks to this module,
    the swapping operation becomes part of a model's definition and, e.g., thus shows up in its printout.
    """

    def __init__(self, dim1, dim2):
        super().__init__()

        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

    def __repr__(self):
        return f"Transposer(dim1={self.dim1}, dim2={self.dim2})"
