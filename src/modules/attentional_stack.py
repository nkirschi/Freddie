import torch.nn as nn

from modules.multihead_self_attention import MultiheadSelfAttention


class AttentionalStack(nn.Sequential):

    def __init__(self, out_size, seq_len, attn_sizes, head_sizes, dropout_rate, use_bn):
        super().__init__()

        attn_sizes.append(out_size)
        for i in range(len(attn_sizes) - 1):
            self.add_module(f"attn{i}", MultiheadSelfAttention(attn_sizes[i], head_sizes[i]))
            # self.add_module(f"proj{i}", Projector(0))
            if dropout_rate > 0:
                self.add_module(f"dropout{i}", nn.Dropout(dropout_rate))
            self.add_module(f"relu{i}", nn.ReLU())
            if use_bn:
                self.add_module(f"batch_norm{i}", nn.BatchNorm1d(seq_len))
            self.add_module(f"linear{i}", nn.Linear(attn_sizes[i], attn_sizes[i + 1]))

    def forward(self, x):
        return super().forward(x)
