import torch
from torch import nn

from .utils.forward_kinematics import forward_kinematics
from .utils.rotation_tools import compute_rotation_matrix_from_ortho6d
from .utils.rotation_tools import compute_rotation_matrix_from_ortho4d
from ..data.skeleton import Skeleton


class PoseDecoder(nn.Module):
    """ Base class implementing decoders from disentangled representations
    to 3D keypoints positions

    Parameters
    ----------
    skeleton: common.skeleton.Skeleton
        Chosen human body skeleton object.
    """
    def __init__(
        self,
        skeleton: Skeleton,
        rot_rep_dim: int = 6,
    ):
        super().__init__()
        self.skeleton = skeleton
        self.rot_rep_dim = rot_rep_dim
        assert rot_rep_dim in [4, 6], (
            "Unsupported rotations representation dimension: "
            f"{self.rot_rep_dim}"
        )

    def forward(
        self,
        rotations_repr: torch.Tensor,
        bones_lengths_repr: torch.Tensor,
        root_positions: torch.Tensor,
    ) -> torch.Tensor:
        assert rotations_repr.shape[-1] == self.rot_rep_dim

        bones_length = self._compute_bones_length(
            bones_lengths_repr,
            rotations_repr.shape[0],
        )
        rotations = self._compute_rotation_mats(rotations_repr)

        t_pose = self.build_t_pose_from_bone_lengths(
            bones_length=bones_length,
        )
        pose = forward_kinematics(
            t_pose=t_pose,
            rotations=rotations,
            root_positions=root_positions,
            skeleton=self.skeleton,
        )
        return pose

    def _compute_rotation_mats(
        self,
        rot_representations: torch.Tensor,  # (B*L, J, 6) or (B*L, J, 4)
    ):
        BL, J, _ = rot_representations.shape

        # (B*L*J, 6)
        stacked_rot_representations = rot_representations.reshape(
            (-1, self.rot_rep_dim)
        )

        # (B*L*J, 3, 3)
        if self.rot_rep_dim == 6:
            stacked_rot_mat = compute_rotation_matrix_from_ortho6d(
                stacked_rot_representations
            )
        elif self.rot_rep_dim == 4:
            stacked_rot_mat = compute_rotation_matrix_from_ortho4d(
                stacked_rot_representations
            )
        else:
            raise ValueError(
                "Unsupported rotations representation dimension: "
                f"{self.rot_rep_dim}"
            )

        return stacked_rot_mat.reshape(BL, J, 3, 3)

    def _compute_bones_length(
        self,
        bones_lengths_repr: torch.Tensor,  # (B, J, 1)
        BL: int,
    ) -> torch.Tensor:
        """ Dummy method for computing bones lengths (useful to implement
        sagittal symmetry if needed).
        """
        B = bones_lengths_repr.shape[0]
        assert BL % B == 0
        L = int(BL / B)
        return torch.stack([bones_lengths_repr] * L, dim=1).reshape(BL, -1, 1)

    def build_t_pose_from_bone_lengths(
        self,
        bones_length: torch.Tensor,
    ) -> torch.Tensor:
        """ Compute 3D keypoints positions of a skeleton in T-pose using bone
        lengths
        """
        batch_size, n_parts, _ = bones_length.shape
        assert n_parts == self.skeleton.num_bones
        device = bones_length.device

        t_pose = torch.zeros(
            (batch_size, self.skeleton.num_joints, 3),
            dtype=torch.float,
            device=device
        )

        for b in range(n_parts):
            t_pose[:, b + 1, :] = (
                t_pose[:, self.skeleton.parents[b + 1], :] +
                self.skeleton.t_pose_operators[b + 1].to(device) * bones_length[:, b]
            )
        return t_pose
