import torch

from ...data.skeleton import Skeleton


def forward_kinematics(
    t_pose: torch.Tensor,
    rotations: torch.Tensor,
    root_positions: torch.Tensor,
    skeleton: Skeleton,
) -> torch.Tensor:
    """
    Uses a skeleton T-pose representation together with predicted
    root-joint positions and joints' rotations to compute the forward
    kinematics and derive 3D keypoints' coordinates.
    """
    assert len(rotations.shape) == 4
    assert rotations.shape[-1] == rotations.shape[-2] == 3

    positions_world = []
    rotations_world = []

    _, n_joints, _, _ = rotations.shape

    for j in range(n_joints):
        if skeleton.parents[j] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotations[:, 0])
        else:
            parent = skeleton.parents[j]
            offset = (
                t_pose[:, j, :] - t_pose[:, parent, :]
            ).view(-1, 3, 1)
            parent_rot_mat = rotations_world[parent]
            rot_mat = parent_rot_mat.matmul(rotations[:, j])

            positions_world.append(
                rot_mat.matmul(offset).view(-1, 3) +
                positions_world[parent]
            )
            if skeleton.has_children[j]:
                rotations_world.append(rot_mat)
            else:
                # This joint is a terminal node
                # -> it would be useless to compute the transformation
                rotations_world.append(None)

    return torch.stack(positions_world, dim=2).permute(0, 2, 1)
