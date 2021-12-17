from torch import nn


class ConvStack(nn.Sequential):

    def __init__(self, in_size, seq_len, channel_sizes, kernel_sizes, stride_sizes, pool_sizes, dropout_rate, use_bn):
        super().__init__()

        self.out_len = seq_len
        self.out_channels = channel_sizes[-1]
        channel_sizes = [in_size] + channel_sizes
        for i in range(len(channel_sizes) - 1):
            self.add_module(f"conv{i}", nn.Conv1d(channel_sizes[i], channel_sizes[i + 1],
                                                  kernel_size=(kernel_sizes[i],),
                                                  stride=(stride_sizes[i],),
                                                  padding=kernel_sizes[i] // 2))
            if dropout_rate > 0:
                self.add_module(f"dropout{i}", nn.Dropout(dropout_rate))
            self.add_module(f"relu{i}", nn.ReLU())
            if use_bn:
                self.add_module(f"batch_norm{i}", nn.BatchNorm1d(channel_sizes[i + 1]))
            if pool_sizes[i] > 1:
                self.add_module(f"max_pool{i}", nn.MaxPool1d(pool_sizes[i]))
            self.out_len = ((self.out_len - kernel_sizes[i] % 2) // stride_sizes[i] + 1) // pool_sizes[i]

    def forward(self, x):
        return super().forward(x)

    def output_size(self):
        return self.out_channels, self.out_len
