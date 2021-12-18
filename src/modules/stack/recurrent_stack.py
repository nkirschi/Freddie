import torch.nn as nn

from modules.tensor.projector import Projector


class RecurrentStack(nn.Sequential):

    def __init__(self, in_size, out_size, seq_len, state_sizes, dropout_rate, use_bn):
        super().__init__()

        state_sizes.insert(0, in_size / 2)
        for i in range(len(state_sizes) - 1):
            self.add_module(f"lstm{i}", nn.LSTM(input_size=int(2 * state_sizes[i]),
                                                hidden_size=int(state_sizes[i + 1]),
                                                bidirectional=True,
                                                batch_first=True,
                                                dropout=dropout_rate
                                                ))
            self.add_module(f"proj{i}", Projector(0))
            if use_bn:
                self.add_module(f"batch_norm{i}", nn.BatchNorm1d(seq_len))
        self.add_module(f"linear", nn.Linear(2 * int(state_sizes[-1]), out_size))

    def forward(self, x):
        return super().forward(x)
