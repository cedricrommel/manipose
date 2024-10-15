import torch
from torch import nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        n_layers: int,
        act_layer: nn.Module = nn.Tanh,
    ):
        super().__init__()
        self.act = act_layer()

        self.fc_in = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            self.act,
            nn.BatchNorm1d(hidden_features),
        )
        self.fcs = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(hidden_features, hidden_features),
                    self.act,
                    nn.BatchNorm1d(hidden_features),
                )
                for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.fcs(x)
        x = self.fc_out(x)
        return x
