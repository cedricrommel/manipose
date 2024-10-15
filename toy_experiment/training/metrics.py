import torch


def calc_mpjpe(
    pred: torch.Tensor,  # supposed to be (B, 2)
    gt: torch.Tensor,  # supposed to be (B, 2)
) -> float:
    return torch.norm(pred - gt, p=2, dim=1).mean().item()


def oracle_multihyp_mpjpe(
    hypothesis: torch.Tensor,  # supposed to be (B, H, 2)
    gt: torch.Tensor,  # supposed to be (B, 2)
) -> float:
    hypothesis_pred = hypothesis[..., :2]
    expanded_gt = gt[:, None, :].expand_as(hypothesis_pred)  # (B, H, 2)
    mpjpe_per_hyp = torch.norm(  # (B, H)
        hypothesis_pred - expanded_gt,
        p=2,
        dim=2,
    )
    return mpjpe_per_hyp.min(dim=1).mean().item()


def distance_to_circle(
    pred: torch.Tensor,  # supposed to be (B, 2)
) -> float:
    return 1 - torch.norm(pred, p=2, dim=1).mean().item()
