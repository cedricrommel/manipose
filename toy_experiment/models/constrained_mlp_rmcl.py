import torch
from torch import nn
import torch.nn.functional as F

from .constrained_mlp import ConstrainedMlp, ConstrainedMlpV2


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


class ConstrainedMlpRmclV2(ConstrainedMlpV2):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        n_layers: int,
        out_features: int = 2,
        act_layer: nn.Module = nn.ReLU,
        major_radius: float = 1.0,
        minor_radius: float = 1.0,
        n_hyp: int = 5,
        beta: float = 1.,
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            n_layers=n_layers,
            act_layer=act_layer,
            major_radius=major_radius,
            minor_radius=minor_radius,
        )
        self.major_radius = major_radius
        self.minor_radius = minor_radius

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
            angles, scores_logits = torch.split(head(input), split_size_or_sections=[2,1], dim=1) #theta: (B,2), score_logits: (B,1)
            cartesian_positions_predicted = self.torusanglestocartesian(major_radius=self.major_radius, minor_radius=self.minor_radius, angles=angles) #Shape (B,3)
            predictions.append(
                torch.concat(
                    [cartesian_positions_predicted, scores_logits],
                    dim=1,
                )
            )

        hypothesis = torch.stack(predictions, dim=1)  # (B, H, 3 + 1)

        # normalize scores
        # a = hypothesis[..., -1].softmax(dim=1)
        # b = hypothesis[..., -1] 
        hypothesis[..., -1] = hypothesis[..., -1].softmax(dim=1)
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
                hypothesis[..., :-1] * hypothesis[..., -1].unsqueeze(-1),
                dim=1,
            )
        else:
            raise ValueError(
                "Only best_score and weighted_ave modes are implemented."
                f"Got {mode}."
            )

    def wta_with_scoring_l2_loss(
        self,
        hypothesis: torch.Tensor,  # supposed (B, H, 4)
        y: torch.Tensor,  # supposed (B, 3)
    ):
        if self.beta == 0:
            return wta_l2_loss(hypothesis, y).mean()

        base_loss = joint_l2_loss_per_hyp(hypothesis=hypothesis, y=y,major_radius=self.major_radius, minor_radius=self.minor_radius)  # (B, H)
        wta_loss, active_heads_idx = torch.min(base_loss, dim=1)

        gt_scores = torch.zeros_like(base_loss)  # (B, H)
        batch_size = base_loss.shape[0]
        batch_indices = torch.arange(batch_size)
        gt_scores[batch_indices, active_heads_idx] = 1.

        pred_scores = hypothesis[:, :, -1]  # (B, H)
        scoring_loss = F.binary_cross_entropy(
            pred_scores,
            gt_scores,
        )

        return wta_loss.mean() + self.beta * scoring_loss
    
def joint_l2_loss_per_hyp(
    hypothesis: torch.Tensor,  # supposed (B, H, 3+1)
    y: torch.Tensor,  # supposed (B, 3)
    major_radius,
    minor_radius
) -> torch.Tensor:
    hypothesis_pred = hypothesis[..., :3]  # (B, H, 3)
    joint1_hypothesis_pred , joint2_hypothesis_pred = torushyps_to_joints(hypothesis_pred,R=major_radius,r=minor_radius) # (B, H, 3), (B, H, 3)
    joint1_y , joint2_y = toruspoints_to_joints(y,R=major_radius,r=minor_radius) # (B, 3)

    err_joint_1 = torch.mean(
        (joint1_hypothesis_pred - joint1_y[:, None, :].expand_as(joint1_hypothesis_pred))**2,
        dim=2,
    ) #(B,H)
    err_joint_2= torch.mean(
        (joint2_hypothesis_pred - joint2_y[:, None, :].expand_as(joint2_hypothesis_pred))**2,
        dim=2,
    ) #(B,H)
    assert err_joint_1.shape == err_joint_2.shape
    assert err_joint_1.shape == (joint1_hypothesis_pred.shape[0],joint1_hypothesis_pred.shape[1])
    return 1/2*(err_joint_1+err_joint_2)

def _l2_loss_per_hyp(
    hypothesis: torch.Tensor,  # supposed (B, H, 3+1)
    y: torch.Tensor,  # supposed (B, 3)
) -> torch.Tensor:
    hypothesis_pred = hypothesis[..., :3] 

    return torch.mean(
        (hypothesis_pred - y[:, None, :].expand_as(hypothesis_pred))**2,
        dim=2,
    )

def wta_l2_loss(
    hypothesis: torch.Tensor,  # supposed (B, H, 3+1)
    y: torch.Tensor,  # supposed (B, 3)
) -> torch.Tensor:
    base_loss = joint_l2_loss_per_hyp(hypothesis=hypothesis, y=y)  # (B, H)
    return torch.min(base_loss, dim=1)[0]

def torushyps_to_joints(vector,R=2,r=1):
    # vector = (B,H,3)
    B = vector.shape[0]
    H = vector.shape[1]
    norm_xy_plane = torch.sqrt(vector[:,:,0]**2+vector[:,:,1]**2).unsqueeze(-1) # shape (B,H,1)
    norm_xy_plane = torch.repeat_interleave(norm_xy_plane, repeats=2, dim=2) # shape (B,H,2)
    # assert norm_xv_plane.shape == (B,2)
    joint1 = R*vector[:,:,:2]/norm_xy_plane # shape (B,H,2)
    joint1 = joint1.reshape(B,H,2) # shape (B,H,2)
    joint1 = torch.cat((joint1,torch.zeros(size=(B,H,1), device=joint1.device)),axis=-1)
    joint2 = vector

    return (joint1, joint2) #((B,H,3),(B,H,3))

def toruspoints_to_joints(vector,R=2,r=1):
    # vector = (B,3)
    B = vector.shape[0]
    norm_xy_plane = torch.sqrt(vector[:,0]**2+vector[:,1]**2).unsqueeze(1) # shape (B,1)
    norm_xy_plane = torch.repeat_interleave(norm_xy_plane, repeats=2, dim=1)
    # assert norm_xv_plane.shape == (B,2)
    joint1 = R*vector[:,:2]/norm_xy_plane
    joint1 = joint1.reshape(B,2)
    joint1 = torch.cat((joint1,torch.zeros(size=(B,1), device=joint1.device)),axis=1)
    joint2 = vector

    return (joint1, joint2) #((B,3),(B,3))