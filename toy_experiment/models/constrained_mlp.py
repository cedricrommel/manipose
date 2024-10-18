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

class ConstrainedMlpV2(Mlp):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        n_layers: int,
        out_features: int = 2,
        act_layer: nn.Module = nn.ReLU,
        major_radius: float = 1.0,
        minor_radius: float = 1.0,
    ):
        super().__init__(
            in_features, hidden_features, out_features, n_layers, act_layer
        )
        self.major_radius = major_radius
        self.minor_radius = minor_radius

    def torusanglestocartesian(self,major_radius, minor_radius, angles) :
        """Converts angles on a torus to points the 3D euclidean space"""
        # angles: array of shape (n_samples, 2)
        # radius: float

        x = (major_radius + minor_radius*torch.cos(angles[:,0]))*torch.cos(angles[:,1])
        y = (major_radius + minor_radius*torch.cos(angles[:,0]))*torch.sin(angles[:,1])
        z = minor_radius*torch.sin(angles[:,0])

        return torch.stack((x,y,z), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = super().forward(x)
        cartesian_position_predicted = self.torusanglestocartesian(major_radius=self.major_radius, minor_radius=self.minor_radius, angles=angles)
        return cartesian_position_predicted
