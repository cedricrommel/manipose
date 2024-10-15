import numbers
from typing import Tuple

import numpy as np
from scipy.special import i0

from utils.utils import check_random_state, polar2cartesian


class MixtureVonMises:
    def __init__(
        self,
        weights: Tuple,
        modes: Tuple,
        dispersions: Tuple,
        random_state: int,
    ):

        assert sum(weights) == 1
        self.weights = np.array(weights)
        assert all(self.weights >= 0)
        self.modes = np.array(modes)
        self.dispersions = np.array(dispersions)
        assert (
            self.weights.shape[0]
            == self.modes.shape[0]
            == self.dispersions.shape[0]
        )

        self.rng = check_random_state(random_state)
        self.components = np.arange(0, self.weights.shape[0])

    def sample(self, size: int) -> np.array:
        picked_components = self.rng.choice(
            self.components,
            size=size,
            p=self.weights,
        )

        samples = np.empty(size)
        for c, mu, kappa in zip(self.components, self.modes, self.dispersions):
            mask = picked_components == c
            size_c = sum(mask)
            samples[mask] = self.rng.vonmises(mu, kappa=kappa, size=size_c)

        return samples

    def pdf(self, theta):
        theta = np.array(theta)
        if len(theta.shape) > 0:
            theta = theta[:, None]
        return np.sum(
            self.weights
            * np.exp(self.dispersions * np.cos(theta - self.modes))
            / (2 * np.pi * i0(self.dispersions)),
            axis=1,
        )


class LiftingDist1Dto2D(MixtureVonMises):
    def __init__(
        self,
        radius: float,
        weights: Tuple,
        modes: Tuple,
        dispersions: Tuple,
        random_state: int,
    ):
        super().__init__(weights, modes, dispersions, random_state)
        assert isinstance(radius, numbers.Real)
        assert radius > 0
        self.radius = radius

    def sample(self, size: int) -> Tuple[np.array, np.array]:
        angles = super().sample(size)
        x, y = polar2cartesian(self.radius, angles)
        return x, np.hstack([x[:, None], y[:, None]])
