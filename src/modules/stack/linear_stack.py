import torch.nn as nn


class LinearStack(nn.Sequential):

    def __init__(self, in_size, out_size, hidden_sizes, dropout_rate, use_bn):
        super().__init__()

        hidden_sizes = [in_size] + hidden_sizes
        for i in range(len(hidden_sizes) - 1):
            self.add_module(f"linear{i}", nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            if dropout_rate > 0:
                self.add_module(f"dropout{i}", nn.Dropout(dropout_rate))
            self.add_module(f"relu{i}", nn.ReLU())
            if use_bn:
                self.add_module(f"batch_norm{i}", nn.BatchNorm1d(hidden_sizes[i + 1]))
        self.add_module(f"linear{len(hidden_sizes) - 1}", nn.Linear(hidden_sizes[-1], out_size))

    def forward(self, x):
        return super().forward(x)
