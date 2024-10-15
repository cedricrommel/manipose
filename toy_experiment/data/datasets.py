from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader

from .distributions import LiftingDist1Dto2D


class LiftingDataset:
    def __init__(
        self,
        distribution: LiftingDist1Dto2D,
        n_train: int,
        n_val: int,
        n_test: int,
    ):
        self.distribution = distribution
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test

        # Sample training, validation and test data ONCE and convert to torch
        self.X_train, self.Y_train = to_torch(*distribution.sample(n_train))
        self.X_val, self.Y_val = to_torch(*distribution.sample(n_val))
        self.X_test, self.Y_test = to_torch(*distribution.sample(n_test))

        # Create corresponding torch dataset objects
        self.training_set = TensorDataset(self.X_train, self.Y_train)
        self.validation_set = TensorDataset(self.X_val, self.Y_val)
        self.test_set = TensorDataset(self.X_test, self.Y_test)

    def get_tr_loader(
            self,
            **kwargs
    ) -> DataLoader:
        if "shuffle" in {**kwargs}:
            raise ValueError(
                "shuffle kwarg is set to True for training loaders and "
                "False for others and should not be manually set by user."
            )
        return DataLoader(self.training_set, shuffle=True, **kwargs)

    def get_loaders(
            self,
            **kwargs
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if "shuffle" in {**kwargs}:
            raise ValueError(
                "shuffle kwarg is set to True for training loaders and "
                "False for others and should not be manually set by user."
            )
        return (
            DataLoader(self.training_set, shuffle=True, **kwargs),
            DataLoader(self.validation_set, shuffle=False, **kwargs),
            DataLoader(self.test_set, shuffle=False, **kwargs),
        )


def to_torch(X: np.array, y: np.array) -> Tuple[Tensor, Tensor]:
    X = torch.from_numpy(X[:, None]).float()
    y = torch.from_numpy(y).float()
    return X, y
