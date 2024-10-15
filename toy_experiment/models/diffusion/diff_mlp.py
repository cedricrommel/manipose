import torch
from torch import nn

from .utils import DiffusionEmbedding
from ..mlp import Mlp


class DiffMlp(Mlp):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        n_layers: int,
        num_diff_steps: int,
        act_layer: nn.Module = nn.Tanh,
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            n_layers=n_layers,
            act_layer=act_layer,
        )
        self.diffusion_step_enc = DiffusionEmbedding(
            num_steps=num_diff_steps,
            embedding_dim=hidden_features,
        )

    def forward(
        self,
        x: torch.Tensor,
        diffusion_step: torch.Tensor
    ) -> torch.Tensor:
        x = self.fc_in(x)
        diffusion_emb = self.diffusion_step_enc(diffusion_step)
        x += diffusion_emb

        x = self.fcs(x)
        x = self.fc_out(x)
        return x
