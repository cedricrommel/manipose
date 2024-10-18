from typing import Tuple

import torch
from ..data import Skeleton


def pose_flip(
    poses_tuple: Tuple[torch.Tensor],
    skeleton: Skeleton,
) -> torch.Tensor:
    assert isinstance(poses_tuple, tuple)

    aug_poses = list()
    for pose in poses_tuple:
        assert pose.shape[-1] in [2, 3]
        assert pose.shape[-2] == skeleton.num_joints

        # reflect horizontal coordinates (u or x)
        pose[..., 0] *= -1

        # swap left and right joints (necessary to ensure we are not creating
        # impossible poses)
        pose[..., skeleton.joints_left + skeleton.joints_right, :] = pose[
            ..., skeleton.joints_right + skeleton.joints_left, :
        ]
        aug_poses.append(pose)

    return tuple(aug_poses)
