from logging import warn

import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from mh_so3_hpe.data.camera import camera_to_world, image_coordinates
from mh_so3_hpe.data.generators import PoseSequenceGenerator


# PAGE SIZES
TEXT_WIDTH = 3.25
PAGE_WIDTH = 6.875
FONTSIZE = 10


def setup_style(grid=False, column_fig=False, fontsize=FONTSIZE, lw=1.0):
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
            "lines.linewidth": lw,
            "lines.markersize": 7,
        }
    )


def prep_data_for_viz(cfg, dataset, keypoints):
    poses_2d_subj = {
        k.lower().split(" ")[0]: v
        for k, v in keypoints[cfg.viz.viz_subject].items()
    }
    poses_2d = poses_2d_subj[cfg.viz.viz_action]
    out_poses_2d = poses_2d[cfg.viz.viz_camera]

    poses_3d_subj = {
        k.lower().split(" ")[0]: v
        for k, v in dataset[cfg.viz.viz_subject].items()
    }
    poses_3d = poses_3d_subj[cfg.viz.viz_action]["positions_3d"]
    assert len(poses_3d) == len(poses_2d), "Camera count mismatch"
    out_poses_3d = poses_3d[cfg.viz.viz_camera]

    # NOTE: For some reason, input and output length sometimes mismatch by a
    # few frames. The following code handles it.
    L_3d = out_poses_3d.shape[0]
    L_2d = out_poses_2d.shape[0]
    if L_2d > L_3d:
        out_poses_2d = out_poses_2d[:L_3d]
        diff = L_2d - L_3d
        warn(
            "Lengths of 2D and 3D videos don't match. "
            f"Removing the last {diff} frames of 2D video."
        )
    elif L_2d < L_3d:
        out_poses_3d = out_poses_3d[:L_2d]
        diff = L_3d - L_2d
        warn(
            "Lengths of 2D and 3D videos don't match. "
            f"Removing the last {diff} frames of 3D video."
        )

    out_actions = [cfg.viz.viz_camera] * out_poses_2d.shape[0]
    ground_truth = out_poses_3d.copy()
    input_keypoints = out_poses_2d.copy()

    cam = dataset.cameras[cfg.viz.viz_subject][cfg.viz.viz_camera]
    input_keypoints = image_coordinates(
        input_keypoints[..., :2], w=cam["res_w"], h=cam["res_h"]
    )

    render_loader = DataLoader(
        PoseSequenceGenerator(
            [out_poses_3d],
            [out_poses_2d],
            [out_actions],
            seq_len=cfg.data.seq_len,
            random_start=False,
            drop_last=False,
        ),
        batch_size=cfg.train.batch_size_test,
        shuffle=False,
        num_workers=cfg.train.workers,
        pin_memory=True,
    )
    return render_loader, input_keypoints, ground_truth, cam


def prepare_prediction_for_viz(prediction, cam, multihyp=False):
    if multihyp:
        scores = prediction[..., -1:]
        prediction = prediction[..., :-1]

    # Invert camera transformation
    prediction = camera_to_world(prediction, R=cam["orientation"], t=0)
    prediction[..., 2] -= np.min(prediction[..., 2])

    # Glue scores and predictions together again when appropriate
    if multihyp:
        prediction = np.concatenate((prediction, scores), axis=-1)
    return prediction
