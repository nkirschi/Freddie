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
from math import prod

from modules.transposer import Transpose
from modules.linear_stack import LinearStack
from modules.conv_stack import ConvStack
from modules.recurrent_stack import RecurrentStack


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


class MLP(FreddieModel):
    """
    Simple multi-layer perceptron with ReLU activations and optional dropout.
    """

    def __init__(self, num_channels, window_size, future_size, num_classes=5, **kwargs):
        super().__init__(num_channels, window_size, future_size, num_classes)

        self.flatten = nn.Flatten(-2, -1)
        self.hidden_stack = LinearStack(prod(self.in_shape),
                                        prod(self.out_shape),
                                        kwargs["hidden_sizes"],
                                        kwargs["dropout_rate"],
                                        kwargs["batch_normalization"])
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

        self.conv_stack = ConvStack(num_channels,
                                    window_size,
                                    kwargs["channel_sizes"],
                                    kwargs["kernel_sizes"],
                                    kwargs["stride_sizes"],
                                    kwargs["pool_sizes"],
                                    kwargs["dropout_rate"],
                                    kwargs["batch_normalization"])
        self.flatten = nn.Flatten(-2, -1)
        self.linear_stack = LinearStack(prod(self.conv_stack.output_size()),
                                        prod(self.out_shape),
                                        kwargs["hidden_sizes"],
                                        kwargs["dropout_rate"],
                                        kwargs["batch_normalization"])
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

        self.conv_stack = ConvStack(num_channels,
                                    window_size,
                                    kwargs["channel_sizes"],
                                    kwargs["kernel_sizes"],
                                    kwargs["stride_sizes"],
                                    kwargs["pool_sizes"],
                                    kwargs["dropout_rate"],
                                    kwargs["batch_normalization"])
        self.out_conv = nn.Conv1d(kwargs["channel_sizes"][-1],
                                  self.out_shape[0] * self.out_shape[1],
                                  kernel_size=(1,))
        self.gap = nn.AvgPool1d(self.conv_stack.output_size()[1])
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
        self.swap_in = Transpose(-1, -2)
        self.lstm_stack = RecurrentStack(num_channels,
                                         num_classes,
                                         window_size + future_size,
                                         kwargs["state_sizes"],
                                         kwargs["dropout_rate"],
                                         kwargs["batch_normalization"])
        self.swap_out = Transpose(-1, -2)

    def forward(self, x):
        x = self.zero_pad(x)
        x = self.swap_in(x)
        x = self.lstm_stack(x)
        x = self.swap_out(x)

        return x


################################################################################


class CRNN(FreddieModel):
    """
    Convolutional recurrent network combining convolutions and LSTM components.
    """

    def __init__(self, num_channels, window_size, future_size, num_classes=5, **kwargs):
        super().__init__(num_channels, window_size, future_size, num_classes)

        self.conv_stack = ConvStack(num_channels,
                                    window_size,
                                    kwargs["channel_sizes"],
                                    kwargs["kernel_sizes"],
                                    kwargs["stride_sizes"],
                                    kwargs["pool_sizes"],
                                    kwargs["dropout_rate"],
                                    kwargs["batch_normalization"])
        self.zero_pad = nn.ConstantPad1d((0, future_size), 0)
        self.swap_last1 = Transpose(-1, -2)
        self.lstm_stack = RecurrentStack(self.conv_stack.output_size()[0],
                                         num_classes,
                                         self.conv_stack.output_size()[1] + future_size,
                                         kwargs["state_sizes"],
                                         kwargs["dropout_rate"],
                                         kwargs["batch_normalization"])
        self.swap_last2 = Transpose(-1, -2)

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.zero_pad(x)
        x = self.swap_last1(x)
        x = self.lstm_stack(x)
        x = self.swap_last2(x)

        return x


################################################################################


class TNN(FreddieModel):
    """
    Transformer model.
    """

    def __init__(self, num_channels, window_size, future_size, num_classes=5, **kwargs):
        super().__init__(num_channels, window_size, future_size, num_classes)

        self.zero_pad = nn.ConstantPad1d((0, 0, 0, future_size), 0)
        self.swap_last1 = Transpose(-1, -2)
        self.embed = nn.Linear(num_channels, kwargs["transformer_dim"])
        self.transformer = nn.Transformer(d_model=kwargs["transformer_dim"],
                                          nhead=kwargs["attention_heads"],
                                          batch_first=True,
                                          dim_feedforward=kwargs["transformer_dim"],
                                          num_encoder_layers=kwargs["transformer_encoders"],
                                          num_decoder_layers=kwargs["transformer_decoders"])
        self.debed = nn.Linear(kwargs["transformer_dim"], num_classes)
        self.swap_last2 = Transpose(-1, -2)

    def forward(self, x):
        x = self.swap_last1(x)
        x = self.embed(x)
        x = self.transformer(x, self.zero_pad(x))
        x = self.debed(x)
        x = self.swap_last2(x)

        return x
