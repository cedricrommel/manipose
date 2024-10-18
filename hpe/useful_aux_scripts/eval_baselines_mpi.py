# %%
from pathlib import Path
import pickle

import numpy as np
import torch
from tqdm import tqdm

from mh_so3_hpe.metrics import sagittal_symmetry, segments_time_consistency
from mh_so3_hpe.metrics import keypoint_3d_auc, keypoint_3d_pck, mpjpe_error
from mh_so3_hpe.data.dataset_3dhp import MAP_H36M_TO_MPI_JOINTS



# %% FETCH DATASET SKELETON

from mh_so3_hpe.data import Dataset3DHP
from omegaconf import OmegaConf

cfg = OmegaConf.load("./conf/config.yaml")
cfg_3dhp = OmegaConf.load("./conf/data/mpi_inf_3dhp.yaml")
cfg.data.update(cfg_3dhp)

dataset = Dataset3DHP(
    config=cfg,
    root_path=cfg.data.data_dir,
    train=False
)

# %% - LOAD PREDICTED POSES
pstmo_preds_path = "/home/crommel/workspace/projects/human_pose_lifting/P-STMO/checkpoint/model_81_STMO/inference_data.pkl"
with open(pstmo_preds_path, "rb") as f:
    pstmo_preds_dict = pickle.load(f)

gt_path = "/home/crommel/workspace/projects/human_pose_lifting/P-STMO/checkpoint/model_81_STMO/gt_inference_data.pkl"
with open(gt_path, "rb") as f:
    gt_dict = pickle.load(f)

# %% - CONVERT TO LIST OF TENSORS

pstmo_preds = [
    torch.tensor(act_pred).float().permute(2, 3, 1, 0)
    for act_pred in pstmo_preds_dict.values()
]
  # (1, action_leng, J, 3)

# %%

with torch.no_grad():
    sag_syms = list()
    seg_stds = list()
    for act_pred in tqdm(pstmo_preds):
        act_pred = act_pred.permute(0, 3, 2, 1)[:, :, MAP_H36M_TO_MPI_JOINTS, :]
        sag_syms.append(
            sagittal_symmetry(
                joints_coords=act_pred,
                skeleton=dataset.skeleton,
                mode="average",
                squared=False,
            )
            .cpu()
            .numpy()
        )

        seg_stds.append(
            segments_time_consistency(
                joints_coords=act_pred,
                skeleton=dataset.skeleton,
                mode="std",
            )
            .cpu()
            .numpy()
        )
pstmo_sag_sym = np.mean(sag_syms)
print("pstmo_sag_sym: ", pstmo_sag_sym)
pstmo_seg_std = np.mean(seg_stds)
print("pstmo_seg_std:", pstmo_seg_std)

# %%
pck_per_subj = {}
for subj, preds in pstmo_preds_dict.items():
    _, J, _, L = preds.shape
    preds = np.transpose(preds.reshape(3, J, L), (2, 1, 0))
    gt = np.transpose(gt_dict[subj].reshape(3, J, L), (2, 1, 0))
    pck_per_subj[subj] = keypoint_3d_pck(
        pred=preds,
        gt=gt,
        mask=np.ones((L, J)).astype(bool),
        threshold=150,
    )
    # pck_per_subj[subj] = mpjpe_error(
    #     torch.from_numpy(preds).float(),
    #     torch.from_numpy(gt).float(),
    #     # torch.from_numpy(gt_3d_poses[subj]).float() * 1000,
    #     mode="average",
    # )

average_pck = np.mean(list(pck_per_subj.values()))
print("average_pck:", average_pck)

# %%

pck_per_subj = {}
for subj, preds in pstmo_preds_dict.items():
    _, J, _, L = preds.shape
    preds = np.transpose(preds.reshape(3, J, L), (2, 1, 0))
    gt = np.transpose(gt_dict[subj].reshape(3, J, L), (2, 1, 0))
    pck_per_subj[subj] = keypoint_3d_auc(
        pred=preds,
        gt=gt,
        mask=np.ones((L, J)).astype(bool),
    )

average_pck = np.mean(list(pck_per_subj.values()))
print("average_pck:", average_pck)

# %% - LOAD PREDICTED POSES FOR ANATOMY3D
anatomy3d_preds_path = "../../Anatomy3D/anatomy3d_preds.pkl"
with open(anatomy3d_preds_path, "rb") as f:
    anatomy3d_preds_dict = pickle.load(f)

# %% - CONVERT TO LIST OF TENSORS

for key, arr_list in anatomy3d_preds_dict.items():
    anatomy3d_preds_dict[key] = np.stack(
        arr_list, axis=0
    )

# %%

anatomy3d_preds_dict = [
    torch.tensor(act_pred).float().reshape(1, -1, 17, 3)
    for act_pred in anatomy3d_preds_dict.values()
]  # (1, action_leng, J, 3)

# %%

with torch.no_grad():
    sag_syms = list()
    seg_stds = list()
    for act_pred in tqdm(anatomy3d_preds_dict):
        act_pred = act_pred.permute(0, 3, 2, 1) * 1000
        sag_syms.append(
            sagittal_symmetry(
                joints_coords=act_pred,
                skeleton=dataset.skeleton,
                mode="average",
                squared=False,
            )
            .cpu()
            .numpy()
        )

        seg_stds.append(
            segments_time_consistency(
                joints_coords=act_pred,
                skeleton=dataset.skeleton,
                mode="std",
            )
            .cpu()
            .numpy()
        )
anatomy3d_sag_sym = np.mean(sag_syms)
anatomy3d_seg_std = np.mean(seg_stds)

# %% - LOAD PREDICTED POSES FOR MHFORMER
mhformer_preds_path = "../../MHFormer/mhformer_preds_dict.pkl"
with open(mhformer_preds_path, "rb") as f:
    mhformer_preds_dict = pickle.load(f)

# %% - CONVERT TO LIST OF TENSORS

for key, arr_list in mhformer_preds_dict.items():
    mhformer_preds_dict[key] = np.concatenate(
        [
            arr.reshape(-1, 1, 17, 3)
            for arr in arr_list
        ], axis=0
    )

# %%

mhformer_preds_dict = [
    torch.tensor(act_pred).float().reshape(1, -1, 17, 3)
    for act_pred in mhformer_preds_dict.values()
]  # (1, action_leng, J, 3)

# %%

with torch.no_grad():
    sag_syms = list()
    seg_stds = list()
    for act_pred in tqdm(mhformer_preds_dict):
        act_pred = act_pred.permute(0, 3, 2, 1) * 1000
        sag_syms.append(
            sagittal_symmetry(
                joints_coords=act_pred,
                skeleton=dataset.skeleton,
                mode="average",
                squared=False,
            )
            .cpu()
            .numpy()
        )

        seg_stds.append(
            segments_time_consistency(
                joints_coords=act_pred,
                skeleton=dataset.skeleton,
                mode="std",
            )
            .cpu()
            .numpy()
        )
mhformer_sag_sym = np.mean(sag_syms)
mhformer_seg_std = np.mean(seg_stds)

# %% - LOAD PREDICTED POSES FOR ST-GCN
stgcn_preds_path = "../../ST-GCN-original/stgcn_preds_dict.pkl"
with open(stgcn_preds_path, "rb") as f:
    stgcn_preds_dict = pickle.load(f)

# %% - CONVERT TO LIST OF TENSORS

for key, arr_list in stgcn_preds_dict.items():
    stgcn_preds_dict[key] = np.concatenate(
        [
            arr.reshape(-1, 1, 17, 3)
            for arr in arr_list
        ], axis=0
    )

# %%

stgcn_preds_dict = [
    torch.tensor(act_pred).float().reshape(1, -1, 17, 3)
    for act_pred in stgcn_preds_dict.values()
]  # (1, action_leng, J, 3)

# %%

with torch.no_grad():
    sag_syms = list()
    seg_stds = list()
    for act_pred in tqdm(stgcn_preds_dict):
        act_pred = act_pred.permute(0, 3, 2, 1) * 1000
        sag_syms.append(
            sagittal_symmetry(
                joints_coords=act_pred,
                skeleton=dataset.skeleton,
                mode="average",
                squared=False,
            )
            .cpu()
            .numpy()
        )

        seg_stds.append(
            segments_time_consistency(
                joints_coords=act_pred,
                skeleton=dataset.skeleton,
                mode="std",
            )
            .cpu()
            .numpy()
        )
stgcn_sag_sym = np.mean(sag_syms)
stgcn_seg_std = np.mean(seg_stds)

# %% - LOAD PREDICTED POSES FOR VidePose3D
videopose_preds_path = "../../VideoPose3D-original/videopose3d_preds.pkl"
with open(videopose_preds_path, "rb") as f:
    videpose_preds_dict = pickle.load(f)

# %% - CONVERT TO LIST OF TENSORS

for key, arr_list in videpose_preds_dict.items():
    videpose_preds_dict[key] = np.stack(
        arr_list, axis=0
    )

# %%

videpose_preds_dict = [
    torch.tensor(act_pred).float().reshape(1, -1, 17, 3)
    for act_pred in videpose_preds_dict.values()
]  # (1, action_leng, J, 3)

# %%

with torch.no_grad():
    sag_syms = list()
    seg_stds = list()
    for act_pred in tqdm(videpose_preds_dict):
        act_pred = act_pred.permute(0, 3, 2, 1) * 1000
        sag_syms.append(
            sagittal_symmetry(
                joints_coords=act_pred,
                skeleton=dataset.skeleton,
                mode="average",
                squared=False,
            )
            .cpu()
            .numpy()
        )

        seg_stds.append(
            segments_time_consistency(
                joints_coords=act_pred,
                skeleton=dataset.skeleton,
                mode="std",
            )
            .cpu()
            .numpy()
        )
videopose_sag_sym = np.mean(sag_syms)
videopose_seg_std = np.mean(seg_stds)

# %%
