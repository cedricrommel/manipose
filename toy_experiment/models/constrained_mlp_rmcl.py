import torch
from torch import nn
import torch.nn.functional as F

from .constrained_mlp import ConstrainedMlp


class ConstrainedMlpRmcl(ConstrainedMlp):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        n_layers: int,
        out_features: int = 1,
        act_layer: nn.Module = nn.ReLU,
        radius: float = 1.0,
        n_hyp: int = 5,
        beta: float = 1.,
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            n_layers=n_layers,
            act_layer=act_layer,
            radius=radius,
        )

        # number of hypothesis
        self.n_hyp = n_hyp

        # reg weight
        self.beta = beta

        # override fc_out, creating one head per hypothesis
        self.fc_out = nn.ModuleList([
            nn.Linear(hidden_features, out_features + 1)  # +1 for hyp score
            for _ in range(self.n_hyp)
        ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.fc_in(input)
        input = self.fcs(input)

        predictions = list()
        for head in self.fc_out:
            theta, scores_logits = torch.split(head(input), 1, dim=1)
            x, y = self.polar2cartesian(theta)
            predictions.append(
                torch.concat(
                    [x, y, scores_logits],
                    dim=1,
                )
            )

        hypothesis = torch.stack(predictions, dim=1)  # (B, H, 2 + 1)

        # normalize scores
        hypothesis[..., 2] = hypothesis[..., 2].softmax(dim=1)
        return hypothesis

    def aggregate(
        self,
        hypothesis: torch.Tensor,
        mode: str = "weighted_ave",
    ) -> torch.Tensor:
        if mode == "best_score":
            best_score_idx = torch.argmax(hypothesis[..., -1], dim=1)
            return hypothesis[:, best_score_idx, :2]
        elif mode == "weighted_ave":
            return torch.sum(
                hypothesis[..., :2] * hypothesis[..., 2].unsqueeze(-1),
                dim=1,
            )
        else:
            raise ValueError(
                "Only best_score and weighted_ave modes are implemented."
                f"Got {mode}."
            )

    def wta_with_scoring_l2_loss(
        self,
        hypothesis: torch.Tensor,  # supposed (B, H, 3)
        y: torch.Tensor,  # supposed (B, 2)
    ):
        if self.beta == 0:
            return wta_l2_loss(hypothesis, y).mean()

        base_loss = _l2_loss_per_hyp(hypothesis=hypothesis, y=y)  # (B, H)
        wta_loss, active_heads_idx = torch.min(base_loss, dim=1)

        gt_scores = torch.zeros_like(base_loss)  # (B, H)
        batch_size = base_loss.shape[0]
        batch_indices = torch.arange(batch_size)
        gt_scores[batch_indices, active_heads_idx] = 1.

        pred_scores = hypothesis[:, :, 2]  # (B, H)
        scoring_loss = F.binary_cross_entropy(
            pred_scores,
            gt_scores,
        )

        return wta_loss.mean() + self.beta * scoring_loss


def _l2_loss_per_hyp(
    hypothesis: torch.Tensor,  # supposed (B, H, 3)
    y: torch.Tensor,  # supposed (B, 2)
) -> torch.Tensor:
    hypothesis_pred = hypothesis[..., :2]
    return torch.mean(
        (hypothesis_pred - y[:, None, :].expand_as(hypothesis_pred))**2,
        dim=2,
    )


def wta_l2_loss(
    hypothesis: torch.Tensor,  # supposed (B, H, 3)
    y: torch.Tensor,  # supposed (B, 2)
) -> torch.Tensor:
    base_loss = _l2_loss_per_hyp(hypothesis=hypothesis, y=y)  # (B, H)
    return torch.min(base_loss, dim=1)[0]
