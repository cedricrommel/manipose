import torch

from ..data.skeleton import Skeleton

from .utils import measure_bones_length


def _segments_time_consistency_no_agg(
    joints_coords: torch.Tensor,
    skeleton: Skeleton,
    mode: str,
) -> float:
    bones_lengths = measure_bones_length(
        joints_coords, skeleton.bones
    )  # (batch_size, num_bones, series_length)

    stat = torch.var
    if mode == "average":
        aggregator = torch.mean
    elif mode == "sum":
        aggregator = torch.sum
    elif mode == "std":
        aggregator = torch.mean
        stat = torch.std
    elif mode == "min":
        aggregator = torch.min
    elif mode == "max":
        aggregator = torch.max
    else:
        raise ValueError(
            f"Unexpected value for 'mode' encoutered: {mode}."
            "Accepted values are 'average', 'sum' and 'std."
        )
    return stat(bones_lengths, dim=2), aggregator


def segments_time_consistency(
    joints_coords: torch.Tensor,
    skeleton: Skeleton,
    mode: str,
) -> float:
    seg_var, aggregator = _segments_time_consistency_no_agg(
        joints_coords=joints_coords,
        skeleton=skeleton,
        mode=mode,
    )
    return aggregator(seg_var)


def segments_time_consistency_per_bone(
    joints_coords: torch.Tensor,
    skeleton: Skeleton,
    mode: str,
) -> float:
    seg_var, aggregator = _segments_time_consistency_no_agg(
        joints_coords=joints_coords,
        skeleton=skeleton,
        mode=mode,
    )
    return aggregator(seg_var, dim=0)


def segments_max_strech_per_bone(
    joints_coords: torch.Tensor,
    skeleton: Skeleton,
) -> float:
    bones_lengths = measure_bones_length(
        joints_coords, skeleton.bones
    )  # (batch_size, num_bones, series_length)
    bones_lengths_no_batch = bones_lengths.permute(0, 2, 1).reshape(
        -1,
        skeleton.num_bones,
    )
    return bones_lengths_no_batch.min(dim=0)[0], bones_lengths_no_batch.max(dim=0)[0]


def segments_max_diff_strech_per_bone(
    joints_coords: torch.Tensor,
    skeleton: Skeleton,
) -> float:
    bones_lengths = measure_bones_length(
        joints_coords, skeleton.bones
    )  # (batch_size, num_bones, series_length)
    diff_bone_lengths = torch.abs(
        torch.diff(
            bones_lengths,
            dim=2,
        )
    )
    diff_bone_lengths_no_batch = diff_bone_lengths.permute(0, 2, 1).reshape(
        -1,
        skeleton.num_bones,
    )
    return diff_bone_lengths_no_batch.max(dim=0)


def _sagittal_symmetry_no_agg(
    joints_coords: torch.Tensor,
    skeleton: Skeleton,
    mode: str,
    squared: bool,
) -> float:
    bones_lengths = measure_bones_length(
        joints_coords, skeleton.bones
    )  # (batch_size, num_bones, series_length)

    if mode == "average":
        aggregator = torch.mean
    elif mode == "sum":
        aggregator = torch.sum
    else:
        raise ValueError(
            f"Unexpected value for 'mode' encoutered: {mode}."
            "Accepted values are 'average' and 'sum'."
        )

    diff = (
        bones_lengths[:, skeleton.bones_left, :]
        - bones_lengths[:, skeleton.bones_right, :]
    ).abs()
    if squared:
        diff = diff**2.0

    return diff, aggregator


def sagittal_symmetry(
    joints_coords: torch.Tensor,
    skeleton: Skeleton,
    mode: str,
    squared: bool = True,
) -> float:
    unnag_sym, aggregator = _sagittal_symmetry_no_agg(
        joints_coords=joints_coords,
        skeleton=skeleton,
        mode=mode,
        squared=squared,
    )
    return aggregator(unnag_sym)


def sagittal_symmetry_per_bone(
    joints_coords: torch.Tensor,
    skeleton: Skeleton,
    mode: str,
    squared: bool = True,
) -> float:
    unnag_sym, aggregator = _sagittal_symmetry_no_agg(
        joints_coords=joints_coords,
        skeleton=skeleton,
        mode=mode,
        squared=squared,
    )
    return aggregator(
        unnag_sym.permute(0, 2, 1).reshape(-1, len(skeleton.bones_left)),
        dim=0,
    )


def smoothness_regularization(
    prediction: torch.Tensor,
    weights: torch.Tensor = None,
    axis: int = 1,
) -> torch.Tensor:
    velocity_predicted = torch.diff(prediction, dim=axis)

    if weights is None:
        weights = torch.ones_like(velocity_predicted[0, 0, :, 0])

    assert weights.shape[0] == velocity_predicted.shape[-2]
    return torch.mean(
        weights[None, None, :, None].to(velocity_predicted.device) *
        velocity_predicted**2
    )
