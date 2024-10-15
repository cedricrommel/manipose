import numpy as np

from .distributions import LiftingDist1Dto2D


class EasyDist(LiftingDist1Dto2D):
    def __init__(self, radius: float, random_state: int):
        super().__init__(
            radius=radius,
            weights=[1.0],
            modes=[4 * np.pi / 10],
            dispersions=[20],
            random_state=random_state,
        )


class HardUnimodalDist(LiftingDist1Dto2D):
    def __init__(self, radius: float, random_state: int):
        super().__init__(
            radius=radius,
            weights=[1.0],
            modes=[0.],
            dispersions=[20],
            random_state=random_state,
        )


class HardBimodalDist(LiftingDist1Dto2D):
    def __init__(self, radius: float, random_state: int):
        super().__init__(
            radius=radius,
            weights=[2/3, 1/3],
            modes=[np.pi / 3, -np.pi / 3],
            dispersions=[20] * 2,
            random_state=random_state,
        )


class HardQuadmodalDist(LiftingDist1Dto2D):
    def __init__(self, radius: float, random_state: int):
        super().__init__(
            radius=radius,
            weights=[0.3, 0.1, 0.4, 0.2],
            modes=[5 * np.pi / 6, 7 * np.pi / 6, np.pi / 3, -np.pi / 3],
            dispersions=[20] * 4,
            random_state=random_state,
        )
