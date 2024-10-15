import torch
from torch import nn
import torch.nn.functional as F


class SquaredReLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.pow(F.relu(x), 2)
