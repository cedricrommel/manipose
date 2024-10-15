import torch
import torch.nn as nn

from einops import rearrange  # equivalent to reshape, but reader-friendly
from .mix_ste import MixSTE
from .pose_decoder import PoseDecoder
from ..data.skeleton import Skeleton


class ManifoldMixSTE(nn.Module):
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
        mup: bool = False,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.rotations_module = MixSTE(
            num_frame=num_frame,
            num_joints=num_joints,
            in_chans=in_chans,
            out_dim=rot_rep_dim,  # 6d or 4d rotation representation per joint
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
            mup=mup,
        )

        self.segments_module = BonesMixSTE(
            num_frame=num_frame,
            num_joints=num_joints,
            num_bones=num_bones,  # <--
            in_chans=in_chans,
            out_dim=1,  # predict segments 1D lengths
            embed_dim=embed_dim_seg,
            depth=depth_seg,
            num_heads=num_heads_seg,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            mup=mup,
        )

        self.decoder = PoseDecoder(skeleton=skeleton, rot_rep_dim=rot_rep_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _, _ = x.shape
        rotations = self.rotations_module(x)  # (B, L, J, 6)
        bones_lengths = self.segments_module(x)  # (B, S, 1)

        # We suppose that the root is always the reference (e.g. at 0,0,0)
        root_positions = torch.zeros(B*L, 3, device=x.device)

        poses = self.decoder(
            rotations_repr=rearrange(rotations, 'B L J D -> (B L) J D'),
            bones_lengths_repr=bones_lengths,
            root_positions=root_positions,
        )
        return rearrange(poses, '(B L) J D -> B L J D', B=B, L=L)


class BonesMixSTE(MixSTE):
    def __init__(
        self,
        num_frame: int = 243,
        num_joints: int = 17,
        num_bones: int = 16,
        in_chans: int = 2,
        out_dim: int = 1,
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: int = 2,
        qkv_bias: bool = True,
        qk_scale: float = None,
        drop_rate: float = 0,
        attn_drop_rate: float = 0,
        drop_path_rate: float = 0.2,
        norm_layer: nn.Module = None,
        mup: bool = False,
    ):
        super().__init__(
            num_frame=num_frame,
            num_joints=num_bones,  # <-- this is intentional
            in_chans=in_chans,
            out_dim=out_dim,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            mup=mup,
        )
        self.num_joints = num_joints
        self.num_bones = num_bones
        self.embed_dim = embed_dim

        # Replace first linear layer by one mapping from joints to segments
        self.Spatial_patch_to_embedding = nn.Identity()
        self.joints_to_segments_proj = nn.Linear(
            in_features=num_joints * in_chans,
            out_features=num_bones * embed_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Map joints to segments
        B, L, _, _ = x.shape
        x = rearrange(x, "B L J C  -> (B L) (J C)")
        x = self.joints_to_segments_proj(x)
        x = rearrange(
            x,
            "(B L) (S C) -> B L S C",
            B=B, L=L, S=self.num_bones, C=self.embed_dim
        )

        x = super().forward(x)  # (B, L, S, 1), where S is the # of segments

        # Average over time dimension
        x = torch.mean(x, dim=1)
        return x
