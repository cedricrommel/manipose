from typing import Optional

import torch
import torch.nn as nn

from einops import rearrange  # equivalent to reshape, but reader-friendly
from .mix_ste import MixSTE
from .manifold_mix_ste import ManifoldMixSTE
from ..data.skeleton import Skeleton
from mh_so3_hpe.metrics import wta_l2_loss_and_activate_head

from mup import MuReadout


class RMCLManifoldMixSTE(ManifoldMixSTE):
    def __init__(
        self,
        skeleton: Skeleton,
        num_frame: int = 243,
        num_joints: int = 17,
        num_bones: int = 16,
        in_chans: int = 2,
        rot_rep_dim: int = 6,
        embed_dim_rot: int = 512,
        depth_rot: int = 8,
        num_heads_rot: int = 8,
        embed_dim_seg: int = 128,
        depth_seg: int = 2,
        num_heads_seg: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer: nn.Module = None,
        n_hyp: int = 5,
        mup: bool = False,
    ):
        super().__init__(
            skeleton=skeleton,
            num_frame=num_frame,
            num_joints=num_joints,
            num_bones=num_bones,
            in_chans=in_chans,
            rot_rep_dim=rot_rep_dim,
            embed_dim_rot=embed_dim_rot,
            depth_rot=depth_rot,
            num_heads_rot=num_heads_rot,
            embed_dim_seg=embed_dim_seg,
            depth_seg=depth_seg,
            num_heads_seg=num_heads_seg,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            mup=mup,
        )
        self.n_hyp = n_hyp

        self.rotations_module = RMCLRotMixSTE(
            num_frame=num_frame,
            num_joints=num_joints,
            in_chans=in_chans,
            out_dim=rot_rep_dim,
            embed_dim=embed_dim_rot,
            depth=depth_rot,
            num_heads=num_heads_rot,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            n_hyp=n_hyp,
            mup=mup,
        )

    def forward(self, x):
        B, L, _, _ = x.shape
        # Predict rotations and scores for each head
        rotations, scores = self.rotations_module(x)  # (B,H,L,J,6) & (B,H,L,1)

        # Compute common bones lengths
        bones_lengths = self.segments_module(x)  # (B, S, 1)

        # We suppose that the root is always the reference (e.g. at 0,0,0)
        root_positions = torch.zeros(B * L * self.n_hyp, 3, device=x.device)

        # Decode poses for all sequences, hypothesis and instants
        poses = self.decoder(
            rotations_repr=rearrange(rotations, "B H L J D -> (B H L) J D"),
            bones_lengths_repr=bones_lengths,
            root_positions=root_positions,
        )
        poses = rearrange(  # (B, H, L, J, 3)
            poses,
            "(B H L) J D -> B H L J D",
            B=B, H=self.n_hyp, L=L
        )

        return poses, scores

    def concat_hyp_and_scores(
        self,
        hypothesis: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                hypothesis,
                scores.unsqueeze(3).expand(-1, -1, -1, self.num_joints, -1)
            ),
            dim=-1,
        )

    def poses_from_hyp_idx(
        self,
        hypothesis: torch.Tensor,  # (B, H, L, J, 3)
        hyp_indices: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:  # (B, L, J, 3)
        B, _, L, J, _ = hypothesis.shape
        batch_indices = torch.arange(B)[:, None, None, None].repeat(1, L, J, 3)
        seq_indices = torch.arange(L)[None, :, None, None].repeat(B, 1, J, 3)
        joint_indices = torch.arange(J)[None, None, :, None].repeat(B, L, 1, 3)
        coord_indices = torch.arange(3)[None, None, None, :].repeat(B, L, J, 1)
        hypothesis = hypothesis.permute(0, 2, 3, 4, 1)

        return hypothesis[
            batch_indices,
            seq_indices,
            joint_indices,
            coord_indices,
            hyp_indices[..., None, None].repeat(1, 1, J, 3),
        ]

    def aggregate(
        self,
        hypothesis: torch.Tensor,
        scores: torch.Tensor = None,
        mode: str = "weighted_ave",
        ground_truth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mode == "best_score":
            assert scores is not None, (
                "Scores required to compute hypothesis with best confidence."
            )
            best_score_idx = torch.argmax(scores, dim=1)[..., 0]
            return self.poses_from_hyp_idx(
                hypothesis=hypothesis,
                hyp_indices=best_score_idx,
            )
        elif mode == "weighted_ave":
            assert scores is not None, (
                "Scores required to compute weighted hypothesis average."
            )
            return torch.sum(
                hypothesis * scores.unsqueeze(-1),
                dim=1,
            )
        elif mode == "oracle":
            assert ground_truth is not None, (
                "Ground-truth required to compute best hypothesis."
            )
            oracle_mpjpe, oracle_hyp_idx = wta_l2_loss_and_activate_head(
                hypothesis=hypothesis,
                y=ground_truth,
                squared=False,
                weights=None,
            )

            oracle_poses = self.poses_from_hyp_idx(
                hypothesis=hypothesis,
                hyp_indices=oracle_hyp_idx,
            )
            return oracle_mpjpe, oracle_poses
        else:
            raise ValueError(
                "Only best_score and weighted_ave modes are implemented."
                f"Got {mode}."
            )


class RMCLRotMixSTE(MixSTE):
    def __init__(
        self,
        num_frame: int = 243,
        num_joints: int = 17,
        in_chans: int = 2,
        out_dim=6,  # <--
        embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        qk_scale: float = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        norm_layer: nn.Module = None,
        n_hyp: int = 5,
        mup: bool = False,
    ):
        super().__init__(
            num_frame,
            num_joints,
            in_chans,
            out_dim,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
        )
        self.n_hyp = n_hyp

        # override linear head of rotations module, creating one head per
        # hypothesis
        self.head = nn.ModuleList(
            [
                MCLHead(
                    embed_dim=embed_dim,
                    out_dim=out_dim,
                    num_joints=num_joints,
                    mup=mup,
                ) for _ in range(self.n_hyp)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, J, _ = x.shape
        # now x is (B, L, J, C) = (batch_size, seq lentgh, joint num, 2)

        x = self.STE_forward(x)
        # now x shape is (B*J, L, C)

        x = self.TTE_foward(x)

        x = rearrange(x, "(B J) L C -> B L J C", J=J)
        x = self.ST_foward(x)

        predictions = list()
        scores_logits = list()
        for head in self.head:
            rotations, score_logit = head(x)  # (B, L, J, 6) and (B, L, 1)
            predictions.append(rotations)
            scores_logits.append(score_logit)

        hypothesis = torch.stack(predictions, dim=1)  # (B, H, L, J, 6)
        scores_logits = torch.stack(scores_logits, dim=1)  # (B, H, L, 1)

        # normalize scores
        scores = scores_logits.softmax(dim=1)

        return hypothesis, scores


class MCLHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        out_dim: int,
        num_joints: int,
        mup: bool = False
    ):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)
        if mup:
            self.prediction_head = MuReadout(
                embed_dim,
                out_dim + 1,
            )
            self.score_head = MuReadout(
                num_joints,
                1,
            )
        else:
            self.prediction_head = nn.Linear(embed_dim, out_dim + 1)
            self.score_head = nn.Linear(num_joints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        prediction_and_score_emb = self.prediction_head(x)  # (B, L, J, 6 + 1)
        prediction = prediction_and_score_emb[..., :-1]  # (B, L, J, 6)
        score_emb = prediction_and_score_emb[..., -1]  # (B, L, J)

        score_logit = self.score_head(score_emb)  # (B, L, 1)
        return prediction, score_logit
