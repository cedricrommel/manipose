from __future__ import absolute_import, division, print_function

import os
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from eval_utils import lift_action
from hydra.utils import get_original_cwd
from main_h36m_lifting import create_model, fetch_and_prepare_data
from omegaconf import DictConfig, OmegaConf
from torch import nn

from mh_so3_hpe.architectures import RMCLManifoldMixSTE
from mh_so3_hpe.visualization import (prep_data_for_viz,
                                      prepare_prediction_for_viz,
                                      render_animation,
                                      render_frame_prediction,
                                      render_rotated_frame_prediction)

METHODS = {
    "RMCLManifoldMixSTE": "MHMC",
    "MixSTE": "MixSTE",
}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("==> Using settings:")
    print(OmegaConf.to_yaml(cfg))

    orig_cwd = Path(get_original_cwd())
    figures_dir = orig_cwd / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    original_cwd = Path(get_original_cwd())
    proj_name = original_cwd.parents[0].name
    keypoints, dataset = fetch_and_prepare_data(cfg, proj_name=proj_name)

    cudnn.benchmark = True
    device = torch.device("cuda")

    (
        render_loader,
        input_keypoints,
        ground_truth,
        cam,
    ) = prep_data_for_viz(cfg, dataset=dataset, keypoints=keypoints)

    # Check if there is more than one model to visualize
    models_to_viz = cfg.model.arch.split(",")
    all_checkpoints = cfg.run.checkpoint_model.split(",")
    assert len(models_to_viz) == len(all_checkpoints)

    anim_output = {}
    for arch, checkpoint_path in zip(models_to_viz, all_checkpoints):
        cfg.model.arch = arch
        cfg.run.checkpoint_model = checkpoint_path

        # Create model
        print("==> Creating model...")
        model = create_model(
            cfg,
            skeleton=dataset.skeleton,
        )

        if torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        model.to(device)

        # Resume from a checkpoint
        cwd = Path(os.getcwd())
        if cfg.run.checkpoint_model != "":
            model_path = cwd / cfg.run.checkpoint_model
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded checkpoint from {model_path}!")

        prediction = lift_action(
            data_loader=render_loader,
            model=model,
            device=device,
            config=cfg,
            skeleton=dataset.skeleton,
            return_hyps=cfg.viz.hypothesis,
        )

        multihyp = cfg.viz.hypothesis and isinstance(model, RMCLManifoldMixSTE)
        prediction = prepare_prediction_for_viz(
            prediction=prediction,
            cam=cam,
            multihyp=multihyp
        )
        method = METHODS[type(model).__name__]
        anim_output[method] = prediction

    ground_truth = prepare_prediction_for_viz(
        prediction=ground_truth,
        cam=cam,
    )
    anim_output["Ground truth"] = ground_truth

    if cfg.viz.viz_output != "":
        output_name = cfg.viz.viz_output
    else:
        hyps_tag = "_hyps" if cfg.viz.hypothesis else ""
        timestamp = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        output_name = (
            f"{cfg.model.arch}{hyps_tag}_"
            f"{cfg.viz.viz_subject}_{cfg.viz.viz_action}_"
            f"{cfg.viz.viz_camera}_{timestamp}"
        )

    print("==> Rendering...")

    if cfg.viz.frame_index == -1:
        output_name = figures_dir / f"{output_name}.{cfg.viz.extension}"

        render_animation(
            keypoints=input_keypoints,
            poses=anim_output,
            skeleton=dataset.skeleton,
            fps=dataset.fps,
            bitrate=cfg.viz.viz_bitrate,
            azim=cfg.viz.azim if cfg.viz.azim != "" else cam["azimuth"],
            output=str(output_name),
            limit=cfg.viz.viz_limit,
            downsample=cfg.viz.viz_downsample,
            size=cfg.viz.viz_size,
            input_video_path=cfg.viz.viz_video,
            viewport=(cam["res_w"], cam["res_h"]),
            input_video_skip=cfg.viz.viz_skip,
            elev=cfg.viz.elev if cfg.viz.elev != "" else 15,
        )
    else:
        if cfg.viz.azim_max == "":
            output_name = figures_dir / f"{output_name}_{cfg.viz.frame_index}.pdf"

            render_frame_prediction(
                frame_index=cfg.viz.frame_index,
                keypoints=input_keypoints,
                poses=anim_output,
                skeleton=dataset.skeleton,
                azim=cfg.viz.azim if cfg.viz.azim != "" else cam["azimuth"],
                output=str(output_name),
                size=cfg.viz.viz_size,
                input_video_path=cfg.viz.viz_video,
                viewport=(cam["res_w"], cam["res_h"]),
                input_video_skip=cfg.viz.viz_skip,
                elev=cfg.viz.elev if cfg.viz.elev != "" else 15,
            )
        else:
            # Create file name
            azim_min = (
                float(cfg.viz.azim) if cfg.viz.azim != "" else cam["azimuth"]
            )
            azim_max = float(cfg.viz.azim_max)
            output_name = (
                figures_dir / (
                    f"{output_name}_{cfg.viz.frame_index}_"
                    f"azim{azim_min}-{azim_max}.{cfg.viz.extension}"
                )
            )

            #  Create list of azimut values
            frames_per_phase = cfg.viz.stationary_frames
            azim_list = np.linspace(
                azim_min, azim_max,
                num=frames_per_phase,
            )

            # Pad it so that we can clearly see the starting and end azims
            azim_list = np.concatenate([
                azim_min * np.ones(frames_per_phase),
                azim_list,
                azim_max * np.ones(frames_per_phase),
            ])

            render_rotated_frame_prediction(
                frame_index=cfg.viz.frame_index,
                keypoints=input_keypoints,
                poses=anim_output,
                skeleton=dataset.skeleton,
                azim_list=azim_list,
                fps=dataset.fps,
                bitrate=cfg.viz.viz_bitrate,
                output=str(output_name),
                size=cfg.viz.viz_size,
                input_video_path=cfg.viz.viz_video,
                viewport=(cam["res_w"], cam["res_h"]),
                input_video_skip=cfg.viz.viz_skip,
                elev=cfg.viz.elev if cfg.viz.elev != "" else 15,
            )


if __name__ == "__main__":
    main()
