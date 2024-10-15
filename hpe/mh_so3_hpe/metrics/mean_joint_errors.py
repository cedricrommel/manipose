import numpy as np
import torch

from .utils import measure_bones_length
from ..data.skeleton import Skeleton


def _mean_joint_error_helper(batch_imp, batch_gt, mode, shape):
    assert batch_imp.shape[-1] == batch_gt.shape[-1] == 3
    batch_imp = batch_imp.contiguous().view(*shape, 3)
    batch_gt = batch_gt.contiguous().view(*shape, 3)

    if mode == "average":
        aggregator = torch.mean
    elif mode == "sum":
        aggregator = torch.sum
    elif mode == "no_agg":

        def aggregator(x, dim=None):
            return x

    else:
        raise ValueError(
            f"Unexpected value for 'mode' encoutered: {mode}."
            "Accepted values are 'average' and 'sum'."
        )

    return batch_imp, batch_gt, aggregator


def mpjpe_error(batch_imp, batch_gt, mode):
    batch_imp, batch_gt, aggregator = _mean_joint_error_helper(
        batch_imp, batch_gt, mode, shape=(-1,)
    )

    return aggregator(torch.norm(batch_gt - batch_imp, 2, 1))


def mse_error(batch_imp, batch_gt, mode):
    batch_imp, batch_gt, aggregator = _mean_joint_error_helper(
        batch_imp, batch_gt, mode, shape=(-1,)
    )

    return aggregator(torch.sum((batch_gt - batch_imp)**2, dim=1))


def jointwise_error(batch_imp, batch_gt, mode):
    J = batch_gt.shape[-2]
    batch_imp, batch_gt, aggregator = _mean_joint_error_helper(
        batch_imp,
        batch_gt,
        mode,
        shape=(
            -1,
            J,
        ),
    )

    return aggregator(
        torch.norm(batch_gt - batch_imp, 2, 2),
        dim=0,
    )


def jointwise_mse(batch_imp, batch_gt, mode):
    J = batch_gt.shape[-2]
    batch_imp, batch_gt, aggregator = _mean_joint_error_helper(
        batch_imp,
        batch_gt,
        mode,
        shape=(
            -1,
            J,
        ),
    )

    return aggregator(
        torch.sum((batch_gt - batch_imp)**2, dim=2),
        dim=0,
    )


def _segments_len_err_no_agg(
    batch_imp: torch.Tensor,
    batch_gt: torch.Tensor,
    skeleton: Skeleton,
    mode: str,
) -> float:
    B, _, _, L = batch_imp.shape

    # (batch_size * series_length, num_bones)
    pred_bones_lengths = measure_bones_length(
        batch_imp, skeleton.bones
    ).permute(0, 2, 1).reshape(B*L, -1)
    gt_bones_lengths = measure_bones_length(
        batch_gt, skeleton.bones
    ).permute(0, 2, 1).reshape(B*L, -1)

    if mode == "average":
        aggregator = torch.mean
    elif mode == "sum":
        aggregator = torch.sum
    elif mode == "no_agg":
        def aggregator(x, dim=None):
            return x
    else:
        raise ValueError(
            f"Unexpected value for 'mode' encoutered: {mode}."
            "Accepted values are 'average' and 'sum'."
        )
    return pred_bones_lengths, gt_bones_lengths, aggregator


def segments_len_err(
    batch_imp: torch.Tensor,
    batch_gt: torch.Tensor,
    skeleton: Skeleton,
    mode: str,
    signed: bool = True,
) -> float:
    pred_len, gt_len, aggregator = _segments_len_err_no_agg(
        batch_imp=batch_imp,
        batch_gt=batch_gt,
        skeleton=skeleton,
        mode=mode,
    )
    diff = gt_len - pred_len
    if not signed:
        diff = torch.abs(diff)
    return aggregator(diff)


def coordwise_error(batch_imp, batch_gt, mode):
    batch_imp, batch_gt, aggregator = _mean_joint_error_helper(
        batch_imp, batch_gt, mode, shape=(-1,)
    )

    return aggregator(
        torch.abs(batch_gt - batch_imp),
        dim=0,
    )


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    assert predicted.shape[-1] == target.shape[-1] == 3
    _, _, J, _ = predicted.shape
    predicted = predicted.contiguous().view(-1, J, 3).detach().cpu().numpy()
    target = target.contiguous().view(-1, J, 3).detach().cpu().numpy()

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    # Return MPJPE
    return np.mean(
        np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
    )
