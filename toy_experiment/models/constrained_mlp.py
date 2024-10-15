from typing import Tuple

import torch
from torch import nn

from .mlp import Mlp


class ConstrainedMlp(Mlp):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        n_layers: int,
        out_features: int = 1,
        act_layer: nn.Module = nn.ReLU,
        radius: float = 1.0,
    ):
        super().__init__(
            in_features, hidden_features, out_features, n_layers, act_layer
        )
        self.radius = radius

    def polar2cartesian(
        self,
        theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.radius * torch.cos(theta)
        y = self.radius * torch.sin(theta)
        return x, y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = super().forward(x)
        x, y = self.polar2cartesian(theta)
        return torch.concat([x, y], dim=1)
