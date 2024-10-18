# %%
import os
import pickle
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import OmegaConf

from mh_so3_hpe.data.h36m_lifting import TRAIN_SUBJECTS, TEST_SUBJECTS, Human36mDataset
from mh_so3_hpe.data.utils import create_2d_data, fetch, read_3d_data
from main_h36m_lifting import create_dataloader


# %% Function to create the general configs for 27 frames case
def gen_config():
    config = OmegaConf.load("conf/config.yaml")

    config_data = OmegaConf.load("conf/data/lifting_gt.yaml")
    config_data_2 = OmegaConf.load("conf/data/lifting_cpn17_test.yaml")
    config_data_3 = OmegaConf.load("conf/data/lifting_cpn17_test_seq27.yaml")
    config_data.update(config_data_2)
    config_data.update(config_data_3)
    # config_data.seq_len = 27
    # config_data.use_valid = False
    config_train = OmegaConf.load("conf/train/lifting.yaml")
    config_train.workers = 4
    config.data.update(config_data)
    config.train.update(config_train)
    return config



# %% - load minimum data config
os.chdir("..")
config = gen_config()

# %% - load data

# data_dir = Path("/home/crommel/shared/crommel/h36m_data")
data_dir = Path("/home/crommel/shared/crommel/h36m_data")
preproc_dataset_path = data_dir / "preproc_data_3d_h36m_17_mh_so3_hpe.pkl"

if preproc_dataset_path.exists():
    print("==> Loading preprocessed dataset...")
    with open(preproc_dataset_path, "rb") as f:
        dataset = pickle.load(f)
else:
    print("==> Loading raw dataset...")
    dataset_path = data_dir / f"data_3d_{config.data.dataset}.npz"

    dataset = Human36mDataset(dataset_path, n_joints=config.data.joints)

    print("==> Preparing data...")
    dataset = read_3d_data(dataset)

    print("==> Caching data...")
    with open(preproc_dataset_path, "wb") as f:
        pickle.dump(dataset, f)

# And 2D data
# inputs_path = data_dir / ("data_2d_h36m_gt.npz")
inputs_path = data_dir / ("data_2d_h36m_cpn_ft_h36m_dbb.npz")
keypoints = create_2d_data(inputs_path, dataset)

# %% - Plotting constants and function to setup some style params
# CVPR PAGE SIZES
TEXT_WIDTH = 3.25
PAGE_WIDTH = 6.875
FONTSIZE = 10

def setup_style(grid=False, column_fig=False, fontsize=FONTSIZE):
    # plt.style.use("seaborn-paper")
    plt.style.use("seaborn-v0_8")

    if column_fig:
        plt.rcParams["figure.figsize"] = (TEXT_WIDTH, TEXT_WIDTH / 2)
    else:
        plt.rcParams["figure.figsize"] = (PAGE_WIDTH, PAGE_WIDTH / 2)
    plt.rcParams["axes.grid"] = grid
    # lw = 1.0 if column_fig else 0.5
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "legend.fontsize": "medium",
            "axes.labelsize": "medium",
            "xtick.labelsize": "medium",
            "ytick.labelsize": "medium",
            "axes.titlesize": "medium",
            "lines.linewidth": 1.0,
            "lines.markersize": 7,
        }
    )


setup_style(grid=False, column_fig=True, fontsize=7)

# %% - set what subject, action, joint and camera to use

figures_dir = Path("figures")

pretty_joint_names = {
    "RElbow": "right elbow",
    "LElbow": "left elbow",
    "RKnee": "right knee",
    "LKnee": "left knee",
    "RWrist": "right wrist",
    "LWrist": "left wrist",
    "RFoot": "right foot",
    "LFoot": "left foot",
}

# %% - fetch corresponding data and put on a dataframe

def get_data(subject, joint, action, camera):
    out_poses_3d, out_poses_2d, out_actions, out_camera_params = fetch(
        subjects=[subject],
        dataset=dataset,
        keypoints=keypoints,
        action_filter=[action],
    )

    joint_index = np.where(
        np.array(dataset.skeleton.joints_names) == joint
    )[0][0]

    out_poses_3d = out_poses_3d[camera]

    out_poses_2d = out_poses_2d[camera]

    out_camera_params = out_camera_params[camera]

    coordinates_df = pd.DataFrame(
        np.stack(
        # np.concatenate(
            out_poses_3d,
            axis=0
        )[:, joint_index, :],
        columns=["x", "y", "z"]
    )

    keypoints_df = pd.DataFrame(
        np.stack(
        # np.concatenate(
            out_poses_2d,
            axis=0
        )[:, joint_index, :],
        columns=["u", "v"]
    )

    return pd.concat([coordinates_df, keypoints_df], axis=1)

# %%
def get_data_all_cams(subject, joint, action):
    dfs = [
        get_data(
            subject=subject,
            joint=joint,
            action=action,
            camera=camera
        )
        for camera in range(4)
    ]

    return pd.concat(dfs, axis=0).reset_index(drop=True)

# %% - plot and save GT joint position propability density projected onto
# (x,z) and (y,z) planes

def plot_dist(coordinates_df, u_cond=None, v_cond=None):
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        # figsize=(PAGE_WIDTH, 2 * TEXT_WIDTH),
        figsize=(TEXT_WIDTH, TEXT_WIDTH / 2),
        sharex=True,
        sharey=True
    )

    sns.kdeplot(data=coordinates_df, x="u", y="z", ax=ax1, fill=True)
    sns.kdeplot(data=coordinates_df, x="v", y="z", ax=ax2, fill=True)

    if u_cond is not None:
        ax1.vlines(
            u_cond,
            *ax1.get_ylim(),
            colors="r",
            linestyle="--",
        )
    if v_cond is not None:
        ax2.vlines(
            v_cond,
            *ax2.get_ylim(),
            colors="r",
            linestyle="--",
        )
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    fig.savefig(
        figures_dir / f"multimod_density_{joint}_{subject}_{action}_{camera}.pdf"
    )
# %%

subject = "S1"
joint = "RWrist"
action = "sittingdown"
# camera = 0

coordinates_df = get_data_all_cams(
# coordinates_df = get_data(
    subject=subject,
    joint=joint,
    action=action,
    # camera=camera,
)

camera = "all"
plot_dist(coordinates_df, u_cond=0.35, v_cond=0.13)
# plot_dist(coordinates_df, u_cond=-0.15, v_cond=0.1)

# %%
subject = "S1"
joint = "RElbow"
action = "greeting"
# camera = 0

coordinates_df = get_data_all_cams(
# coordinates_df = get_data(
    subject=subject,
    joint=joint,
    action=action,
    # camera=camera,
)

camera = "all"
plot_dist(coordinates_df, u_cond=0.03, v_cond=-0.27)
# %%
subject = "S9"
joint = "RWrist"
action = "walking"
# camera = 1

coordinates_df = get_data_all_cams(
# coordinates_df = get_data(
    subject=subject,
    joint=joint,
    action=action,
    # camera=camera,
)

camera = "all"
plot_dist(coordinates_df, u_cond=0.55, v_cond=-0.16)
# plot_dist(coordinates_df, u_cond=-0.3, v_cond=-0.25)

# %%
subject = "S11"
joint = "LWrist"
action = "directions"
# camera = 0

coordinates_df = get_data_all_cams(
# coordinates_df = get_data(
    subject=subject,
    joint=joint,
    action=action,
    # camera=camera,
)

camera = "all"
plot_dist(coordinates_df, u_cond=0.1, v_cond=-0.24)


# %%
subject = "S11"
joint = "LWrist"
action = "directions"

coordinates_df = get_data_all_cams(
    subject=subject,
    joint=joint,
    action=action,
)

camera="all"
plot_dist(coordinates_df, u_cond=-0.19, v_cond=-0.24)
# %%
