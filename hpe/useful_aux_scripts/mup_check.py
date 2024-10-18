# %%
from pathlib import Path
import pickle

from omegaconf import OmegaConf

from mup.coord_check import get_coord_data, plot_coord_data
from mup import make_base_shapes, set_base_shapes

from mh_so3_hpe.architectures import MixSTE, ManifoldMixSTE, RMCLManifoldMixSTE
from main_h36m_lifting import create_dataloader, fetch_and_prepare_data


# %%
data_dir = Path("/home/crommel/shared/evalle/h36m_lifting_ds")
preproc_dataset_path = data_dir / "preproc_data_3d_h36m_17_mh_so3_hpe.pkl"
with open(preproc_dataset_path, "rb") as f:
    dataset = pickle.load(f)

# %%
cfg = OmegaConf.load("./conf/config.yaml")

# %%

keypoints, dataset = fetch_and_prepare_data(cfg, proj_name="mh_so3_hpe")

# %%
cfg.data.seq_len = 27
dataloader = create_dataloader(
    keypoints=keypoints,
    dataset=dataset,
    action_filter=["sittingdown"],
    subjects=["S1"],
    cfg=cfg,
    train=True,
)

# %%


def create_model(
    width,
    cfg=cfg,
    arch="mixste",
    seq_len=27,
    skeleton=dataset.skeleton,
):
    if arch == "mixste":
        model = MixSTE(
            num_frame=seq_len,
            num_joints=skeleton.num_joints,
            in_chans=2,
            out_dim=3,
            num_heads=cfg.model.nheads,
            depth=cfg.model.layers,
            embed_dim=width,
            drop_path_rate=cfg.model.drop_path_rate,
            mup=True,
        )
    elif arch == "manifold":
        model = ManifoldMixSTE(
            skeleton=skeleton,
            num_frame=seq_len,
            num_joints=skeleton.num_joints,
            num_bones=skeleton.num_bones,
            in_chans=2,
            rot_rep_dim=cfg.model.rot_dim,
            num_heads_rot=cfg.model.nheads,
            depth_rot=cfg.model.layers,
            embed_dim_rot=width,
            num_heads_seg=cfg.model.nheads_seg,
            depth_seg=cfg.model.layers_seg,
            embed_dim_seg=width,  # ??
            drop_path_rate=cfg.model.drop_path_rate,
            mup=True,
        )
    elif arch == "rmcl_manifold":
        model = RMCLManifoldMixSTE(
            skeleton=skeleton,
            num_frame=seq_len,
            num_joints=skeleton.num_joints,
            num_bones=skeleton.num_bones,
            in_chans=2,
            rot_rep_dim=cfg.model.rot_dim,
            num_heads_rot=cfg.model.nheads,
            depth_rot=cfg.model.layers,
            embed_dim_rot=width,
            num_heads_seg=cfg.model.nheads_seg,
            depth_seg=cfg.model.layers_seg,
            embed_dim_seg=width,
            drop_path_rate=cfg.model.drop_path_rate,
            n_hyp=cfg.multi_hyp.n_hyp,
            mup=True,
        )
    else:
        raise ValueError(
            "Only MixSTE, Manifold-MixSTE and RMCL-Manifold-MixSTE implemented"
            f" for now. Got option {cfg.model.arch}."
        )

    return model


# %% Instantiate a base model

base_model = create_model(width=64, seq_len=27)
# %%
delta_model = create_model(width=128, seq_len=81)

# %%

base_shape_file_path = "./base_shapes/mixste_width-seq_scaling.bsh"
make_base_shapes(base_model, delta_model, base_shape_file_path)

# %%
# construct a dictionary of lazy μP models with differing widths


def lazy_model(width, seq_len=27):
    # `set_base_shapes` returns the model
    return lambda: set_base_shapes(
        create_model(width, seq_len=seq_len),
        base_shape_file_path
    )
    # Note: any custom initialization with `mup.init` would need to
    # be done inside the lambda as well


widths_to_check = [64, 128, 256, 512]
models = {w: lazy_model(w) for w in widths_to_check}

# %%

df = get_coord_data(models, dataloader, optimizer="adam", lr=1e-2)
# %%

# This saves the coord check plots to filename.
plot_coord_data(df, save_to="./figures/mixste_mup_width-seq_coord_check.pdf")
# %%

df = get_coord_data(models, dataloader, optimizer="adam", lr=1e-2, mup=False)
# %%

# This saves the coord check plots to filename.
plot_coord_data(df, save_to="./figures/mixste_nomup_coord_check.pdf")
# %%
