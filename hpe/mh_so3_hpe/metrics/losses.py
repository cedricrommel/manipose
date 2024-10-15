from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F

# From MixSTE code
STANDARD_H36M_WEIGHTS = torch.Tensor(
    [1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]
)
STANDARD_HEVA_WEIGHTS = torch.Tensor(
    [1, 1, 2.5, 2.5, 1, 2.5, 2.5, 1, 1.5, 1.5, 4, 4, 1.5, 4, 4]
)


def weighted_mpjpe_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor = None,
    dims: Optional[List[int]] = None,
) -> torch.Tensor:
    if weights is None:
        weights = torch.ones(target.shape[-2])

    assert weights.shape[0] == target.shape[-2]
    weights = weights[None, None, :].to(prediction.device)
    if dims is None:
        return torch.mean(
            weights * torch.norm(
                prediction - target,
                p=2,
                dim=len(target.shape)-1
            )
        )
    else:
        ret = (
            weights * torch.norm(
                prediction - target,
                p=2,
                dim=len(target.shape)-1
            )
        )
        for dim in dims:
            ret = ret.mean(dim=dim)
        return ret


def weighted_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor = None,
    dims: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    Note: compared to original implementation from MixSTE, replaced Euclidean
    distances by squared distances).
    """
    if weights is None:
        return F.mse_loss(prediction, target)

    assert weights.shape[0] == target.shape[-2]
    if dims is None:
        return torch.mean(
            weights[None, None, :, None].to(prediction.device) *
            (prediction - target)**2
        )
    else:
        ret = (
            weights[None, None, :, None].to(prediction.device) *
            (prediction - target)**2
        )
        for dim in dims:
            ret = ret.mean(dim=dim)
        return ret


def mean_velocity_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
    axis: torch.Tensor = 1,
    squared: bool = False,
) -> torch.Tensor:
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st
    derivative)
    """
    if len(predicted.shape) > len(target.shape):
        target = target.unsqueeze(1).expand_as(predicted)
    else:
        assert predicted.shape == target.shape

    velocity_predicted = torch.diff(predicted, dim=axis)
    velocity_target = torch.diff(target, dim=axis)

    if squared:
        return torch.mean((velocity_predicted - velocity_target)**2)
    else:
        return torch.mean(
            torch.norm(
                velocity_predicted - velocity_target,
                dim=len(target.shape)-1
            )
        )


def _l2_loss_per_hyp(
    hypothesis: torch.Tensor,  # supposed (B, H, L, J, 3)
    y: torch.Tensor,  # supposed (B, L, J, 3)
    weights: torch.Tensor = None,  # (J,)
    squared: bool = False,
) -> torch.Tensor:  # should be (B, H)
    if squared:
        return weighted_mse_loss(
            prediction=hypothesis,
            target=y[:, None, :].expand_as(hypothesis),
            weights=weights,
            dims=[4, 3],
        )
    else:
        return weighted_mpjpe_loss(
            prediction=hypothesis,
            target=y[:, None, :].expand_as(hypothesis),
            weights=weights,
            dims=[3],
        )


def wta_l2_loss_and_activate_head(
    hypothesis: torch.Tensor,  # supposed (B, H, L, J, 3)
    y: torch.Tensor,  # supposed (B, L, J, 3)
    weights: torch.Tensor = None,  # (J,)
    squared: bool = False,
) -> Tuple[torch.Tensor]:
    base_loss = _l2_loss_per_hyp(  # (B, H, L)
        hypothesis=hypothesis,
        y=y,
        weights=weights,
        squared=squared,
    )
    return torch.min(base_loss, dim=1)


def wta_with_scoring_loss(
    hypothesis: torch.Tensor,  # supposed (B, H, L, J, 3)
    scores: torch.Tensor,  # supposed (B, H, L, 1)
    y: torch.Tensor,  # supposed (B, L, J, 3)
    beta: float,
    weights: torch.Tensor = None,  # (J,)
    squared: bool = False,
):
    unagg_wta_loss, active_heads_idx = wta_l2_loss_and_activate_head(
        hypothesis=hypothesis,
        y=y,
        weights=weights,
        squared=squared,
    )
    if beta == 0:
        return unagg_wta_loss.mean()

    batch_size, H, L = hypothesis.shape[:3]
    gt_scores = torch.zeros((batch_size, L, H))  # (B, L, H)
    batch_indices = torch.arange(batch_size)[:, None].repeat(1, L)
    seq_indices = torch.arange(L).repeat(batch_size, 1)
    gt_scores[batch_indices, seq_indices, active_heads_idx] = 1.
    gt_scores = gt_scores.permute(0, 2, 1).to(scores.device)  # (B, H, L)

    scoring_loss = F.binary_cross_entropy(
        scores.view(batch_size, H, L),
        gt_scores,
    )

    return unagg_wta_loss.mean() + beta * scoring_loss, beta * scoring_loss
