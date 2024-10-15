from typing import Tuple, Callable, Optional
import math

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


def betas_for_alpha_bar(
  num_diffusion_timesteps: int,
  alpha_bar: Callable,
  max_beta: float = 0.5,
) -> np.array:
    # """
    # Create a beta schedule that discretizes the given alpha_t_bar
    # function, which defines the cumulative product of (1-beta) over time
    # from t = [0,1].
    # :param num_diffusion_timesteps: the number of betas to produce.
    # :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
    #                   produces the cumulative product of (1-beta) up to
    #                   that part of the diffusion process.
    # :param max_beta: the maximum beta to use; use values lower than 1 to
    #                  prevent singularities.
    # """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def compute_noise_scheduling(
    schedule: str,
    beta_start: float,
    beta_end: float,
    num_steps: int,
) -> Tuple[np.array, np.array, np.array, np.array]:
    if schedule == "quad":
        beta = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_steps,
            )
            ** 2
        )
    elif schedule == "linear":
        beta = np.linspace(
            beta_start,
            beta_end,
            num_steps,
        )
    elif schedule == "cosine":
        beta = betas_for_alpha_bar(
            num_steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            max_beta=beta_end
        )

    alpha_hat = 1 - beta
    alpha = np.cumprod(alpha_hat)

    sigma = (
        (1.0 - alpha[:-1])
        / (1.0 - alpha[1:])
        * beta[1:]
    ) ** 0.5
    return beta, alpha, alpha_hat, sigma


class DiffusionEmbedding(nn.Module):
    def __init__(
        self,
        num_steps: int,
        embedding_dim: int = 128,
        projection_dim: Optional[int] = None,
        project: bool = True,
    ):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.project = project
        if project:
            self.projection1 = nn.Linear(embedding_dim, projection_dim)
            self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step: int) -> torch.Tensor:
        x = self.embedding[diffusion_step]
        if self.project:
            x = self.projection1(x)
            x = F.silu(x)
            x = self.projection2(x)
            x = F.silu(x)
        return x

    def _build_embedding(self, num_steps: int, dim: int = 64) -> torch.Tensor:
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(
            0
        )  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat(
            [torch.sin(table), torch.cos(table)], dim=1
        )  # (T,dim*2)
        return table
