"""
This file contains definitions of deep learning models for the Freddie task.
All models inherit from the base class 'FreddieModel'. They expect an input
sample of shape (batch_size, num_channels, window_size) and yield an output
prediction of shape (batch_size, num_classes, window_size + future_size).
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"

import torch.nn as nn

from abc import ABC, abstractmethod
from utils.swap_last import SwapLast


################################################################################

class FreddieModel(nn.Module, ABC):
    """
    Base class for all models.

    Inheriting from this class provides predefined attributes for the input and output shape, respectively.
    """

    def __init__(self, num_channels, window_size, future_size, num_classes):
        super().__init__()

        self.in_shape = (num_channels, window_size)
        self.out_shape = (num_classes, window_size + future_size)

    @abstractmethod
    def forward(self, x):
        ...


################################################################################

# TODO move these utility methods to a separate file

def build_linear_stack(in_size, out_size, hidden_sizes, dropout_rate):
    linear_stack = []
    hidden_sizes = [in_size] + hidden_sizes
    for i in range(len(hidden_sizes) - 1):
        linear_stack.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        if dropout_rate > 0:
            linear_stack.append(nn.Dropout(dropout_rate))
        linear_stack.append(nn.ReLU())
    linear_stack.append(nn.Linear(hidden_sizes[-1], out_size))
    return nn.Sequential(*linear_stack)


def build_conv_stack(in_size, channel_sizes, kernel_sizes, stride_sizes, pool_sizes, dropout_rate):
    conv_stack = []
    channel_sizes = [in_size] + channel_sizes
    for i in range(len(channel_sizes) - 1):
        conv_stack.append(nn.Conv1d(channel_sizes[i], channel_sizes[i + 1],
                                    kernel_size=(kernel_sizes[i],),
                                    stride=(stride_sizes[i],),
                                    padding=kernel_sizes[i] // 2))
        if dropout_rate > 0:
            conv_stack.append(nn.Dropout(dropout_rate))
        conv_stack.append(nn.ReLU())
        if pool_sizes[i] > 1:
            conv_stack.append(nn.MaxPool1d(pool_sizes[i]))
    return nn.Sequential(*conv_stack)


def determine_conv_out_size(initial_size, kernel_sizes, stride_sizes, pool_sizes):
    if len(kernel_sizes) == 0:
        return initial_size

    in_size = determine_conv_out_size(initial_size, kernel_sizes[:-1], stride_sizes[:-1], pool_sizes[:-1])
    return ((in_size - kernel_sizes[-1] % 2) // stride_sizes[-1] + 1) // pool_sizes[-1]


################################################################################

class MLP(FreddieModel):
    """
    Simple multi-layer perceptron with ReLU activations and optional dropout.
    """

    def __init__(self, num_channels, window_size, future_size, num_classes=5, **kwargs):
        super().__init__(num_channels, window_size, future_size, num_classes)

        self.flatten = nn.Flatten(-2, -1)
        self.hidden_stack = build_linear_stack(self.in_shape[0] * self.in_shape[1],
                                               self.out_shape[0] * self.out_shape[1],
                                               kwargs["hidden_sizes"],
                                               kwargs["dropout_rate"])
        self.unflatten = nn.Unflatten(-1, self.out_shape)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden_stack(x)
        x = self.unflatten(x)

        return x


################################################################################

class CNN(FreddieModel):
    """
    Standard CNN with 1D convolutions, optional pooling and optional dropout.
    """

    def __init__(self, num_channels, window_size, future_size, num_classes=5, **kwargs):
        super().__init__(num_channels, window_size, future_size, num_classes)

        conv_out_size = determine_conv_out_size(window_size,
                                                kwargs["kernel_sizes"],
                                                kwargs["stride_sizes"],
                                                kwargs["pool_sizes"])

        self.conv_stack = build_conv_stack(num_channels,
                                           kwargs["channel_sizes"],
                                           kwargs["kernel_sizes"],
                                           kwargs["stride_sizes"],
                                           kwargs["pool_sizes"],
                                           kwargs["dropout_rate"])
        self.flatten = nn.Flatten(-2, -1)
        self.linear_stack = build_linear_stack(kwargs["channel_sizes"][-1] * conv_out_size,
                                               self.out_shape[0] * self.out_shape[1],
                                               kwargs["hidden_sizes"],
                                               kwargs["dropout_rate"])
        self.unflatten = nn.Unflatten(-1, self.out_shape)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.linear_stack(x)
        x = self.unflatten(x)

        return x


################################################################################


class FCN(FreddieModel):
    """
    Fully convolutional network with global average pooling (GAP) instead of linear layers.
    """

    def __init__(self, num_channels, window_size, future_size, num_classes=5, **kwargs):
        super().__init__(num_channels, window_size, future_size, num_classes)

        self.conv_stack = build_conv_stack(num_channels,
                                           kwargs["channel_sizes"],
                                           kwargs["kernel_sizes"],
                                           kwargs["stride_sizes"],
                                           kwargs["pool_sizes"],
                                           kwargs["dropout_rate"])
        self.out_conv = nn.Conv1d(kwargs["channel_sizes"][-1],
                                  self.out_shape[0] * self.out_shape[1],
                                  kernel_size=(1,))
        self.gap = nn.AvgPool1d(determine_conv_out_size(window_size,
                                                        kwargs["kernel_sizes"],
                                                        kwargs["stride_sizes"],
                                                        kwargs["pool_sizes"]))
        self.unflatten = nn.Unflatten(-1, self.out_shape)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.out_conv(x)
        x = self.gap(x)[..., 0]
        x = self.unflatten(x)

        return x


################################################################################

class RNN(FreddieModel):
    """
    Basic RNN consisting of a stack of long short-term memory (LSTM) modules.
    """

    def __init__(self, num_channels, window_size, future_size, num_classes=5, **kwargs):
        super().__init__(num_channels, window_size, future_size, num_classes)

        self.zero_pad = nn.ConstantPad1d((0, future_size), 0)
        self.swap_in = SwapLast()
        self.lstm = nn.LSTM(input_size=num_channels,
                            hidden_size=kwargs["rnn_state_size"],
                            bidirectional=True,
                            batch_first=True,
                            num_layers=kwargs["rnn_layers"],
                            dropout=kwargs["dropout_rate"]
                            )
        self.fc = nn.Linear(2 * kwargs["rnn_state_size"], num_classes)
        self.swap_out = SwapLast()

    def forward(self, x):
        x = self.zero_pad(x)
        x = self.swap_in(x)
        x = self.lstm(x)[0]
        x = self.fc(x)
        x = self.swap_out(x)

        return x


################################################################################

class CRNN(FreddieModel):
    """
    Convolutional recurrent network combining convolutions and LSTM components.
    """

    def __init__(self, num_channels, window_size, future_size, num_classes=5, **kwargs):
        super().__init__(num_channels, window_size, future_size, num_classes)

        self.conv_stack = build_conv_stack(num_channels,
                                           kwargs["channel_sizes"],
                                           kwargs["kernel_sizes"],
                                           kwargs["stride_sizes"],
                                           kwargs["pool_sizes"],
                                           kwargs["dropout_rate"])
        self.zero_pad = nn.ConstantPad1d((0, future_size), 0)
        self.swap_last1 = SwapLast()
        self.lstm = nn.LSTM(input_size=kwargs["channel_sizes"][-1],
                            hidden_size=kwargs["rnn_state_size"],
                            bidirectional=True,
                            batch_first=True,
                            num_layers=kwargs["rnn_layers"],
                            dropout=kwargs["dropout_rate"]
                            )
        self.fc = nn.Linear(2 * kwargs["rnn_state_size"], num_classes)
        self.swap_last2 = SwapLast()

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.zero_pad(x)
        x = self.swap_last1(x)
        x = self.lstm(x)[0]
        x = self.fc(x)
        x = self.swap_last2(x)

        return x
