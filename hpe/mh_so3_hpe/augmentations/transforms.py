import torch
from torch import nn
from ..data import Skeleton
from .functional import pose_flip


class PoseFlip(nn.Module):
    def __init__(
        self,
        skeleton: Skeleton,
        probability: float,
    ) -> None:
        super().__init__()
        self.skeleton = skeleton
        self.probability = probability

    def forward(
        self,
        *poses_tuple: torch.Tensor,
    ) -> torch.Tensor:
        # augment only with probability=self.probability
        if torch.rand(1).item() <= self.probability:
            return pose_flip(
                poses_tuple=poses_tuple,
                skeleton=self.skeleton
            )

        return poses_tuple
