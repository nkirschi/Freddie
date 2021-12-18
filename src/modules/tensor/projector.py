from torch import nn


class Projector(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x[self.dim]

    def __repr__(self):
        return f"Projector(dim={self.dim})"