# %%
from pathlib import Path
import pickle

import numpy as np
import torch
from tqdm import tqdm

from mh_so3_hpe.metrics import sagittal_symmetry, segments_time_consistency



# %% FETCH DATASET SKELETON

data_dir = Path("/home/crommel/shared/crommel/h36m_data")
preproc_dataset_path = data_dir / (
    "preproc_data_3d_h36m_17_mh_so3_hpe.pkl"
)

if preproc_dataset_path.exists():
    print("==> Loading preprocessed dataset...")
    with open(preproc_dataset_path, "rb") as f:
        dataset = pickle.load(f)

# %% - LOAD PREDICTED POSES
poseformer_preds_path = "../../PoseFormer/pose_former_preds.pkl"
with open(poseformer_preds_path, "rb") as f:
    poseformer_preds_dict = pickle.load(f)

# %% - CONVERT TO LIST OF TENSORS

poseformer_preds = [
    torch.tensor(act_pred).float().permute(1, 0, 2, 3)
    for act_pred in poseformer_preds_dict.values()
]
  # (1, action_leng, J, 3)

# %%

with torch.no_grad():
    sag_syms = list()
    seg_stds = list()
    for act_pred in tqdm(poseformer_preds):
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
poseformer_sag_sym = np.mean(sag_syms)
poseformer_seg_std = np.mean(seg_stds)

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

# >>> Normalizing Flow one <<<
# %% - LOAD PREDICTED POSES
nfpose_preds_path = "../../NFPose/nfpose_predictions.pkl"
with open(nfpose_preds_path, "rb") as f:
    nfpose_preds_dict = pickle.load(f)

# %% - CONVERT TO LIST OF TENSORS

nfpose_preds = [
    torch.cat(act_pred, dim=1).reshape(1, -1, 3, 17).permute(0, 1, 3, 2)
    for act_pred in nfpose_preds_dict.values()
]
  # (1, action_leng, J, 3)

# %%

with torch.no_grad():
    sag_syms = list()
    seg_stds = list()
    for act_pred in tqdm(nfpose_preds):
        act_pred = act_pred.permute(0, 3, 2, 1)
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
nfpose_sag_sym = np.mean(sag_syms)
print("nfpose_sag_sym:", nfpose_sag_sym)
nfpose_seg_std = np.mean(seg_stds)
print("nfpose_seg_std:", nfpose_seg_std)

# >>> Sharma et al. <<<
# %% - LOAD PREDICTED POSES
sharma_preds_path = "/home/crommel/shared/crommel/pre-trained-models/mh_so3_hpe/baselines/sharma_preds.pkl"
with open(sharma_preds_path, "rb") as f:
    sharma_preds_dict = pickle.load(f)

# %% - CONVERT TO LIST OF TENSORS

sharma_preds = []
for subj_pred in sharma_preds_dict.values():
    for act_pred in subj_pred.values():
        sharma_preds.append(
            torch.from_numpy(act_pred["pred"]).float().reshape(1, -1, 17, 3)
        )
  # (1, action_leng, J, 3)

# %%

with torch.no_grad():
    sag_syms = list()
    seg_stds = list()
    for act_pred in tqdm(sharma_preds):
        act_pred = act_pred.permute(0, 3, 2, 1)
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
sharma_sag_sym = np.mean(sag_syms)
print("sharma_sag_sym:", sharma_sag_sym)
sharma_seg_std = np.mean(seg_stds)
print("sharma_seg_std:", sharma_seg_std)

# %% >>> D3DP <<<
d3dp_pbest_preds_path = "/home/crommel/shared/crommel/pre-trained-models/mh_so3_hpe/baselines/D3DP_pbest_poses.pt"
d3dp_jbest_preds_path = "/home/crommel/shared/crommel/pre-trained-models/mh_so3_hpe/baselines/D3DP_jbest_poses.pt"

d3dp_pbest_preds = torch.load(d3dp_pbest_preds_path)
d3dp_jbest_preds = torch.load(d3dp_jbest_preds_path)


# %% - COMPUTE CONSISTENCY METRICS PER SUBJECT
d3dp_pbest_sag_sym = list()
d3dp_pbest_seg_std = list()
d3dp_jbest_sag_sym = list()
d3dp_jbest_seg_std = list()

for subject, preds in d3dp_pbest_preds.items():
    preds = preds[None, ...].permute(0, 3, 2, 1) * 1000
    d3dp_pbest_sag_sym.append(
        sagittal_symmetry(
            joints_coords=preds,
            skeleton=dataset.skeleton,
            mode="average",
            squared=False,
        )
        .cpu()
        .numpy()
    )

    d3dp_pbest_seg_std.append(
        segments_time_consistency(
            joints_coords=preds,
            skeleton=dataset.skeleton,
            mode="std",
        )
        .cpu()
        .numpy()
    )

for subject, preds in d3dp_jbest_preds.items():
    preds = preds[None, ...].permute(0, 3, 2, 1) * 1000
    d3dp_jbest_sag_sym.append(
        sagittal_symmetry(
            joints_coords=preds,
            skeleton=dataset.skeleton,
            mode="average",
            squared=False,
        )
        .cpu()
        .numpy()
    )

    d3dp_jbest_seg_std.append(
        segments_time_consistency(
            joints_coords=preds,
            skeleton=dataset.skeleton,
            mode="std",
        )
        .cpu()
        .numpy()
    )

# %%

d3dp_pbest_ave_sag_sym = np.mean(d3dp_pbest_sag_sym)
print("d3dp_pbest_sag_sym:", d3dp_pbest_ave_sag_sym)
d3dp_pbest_ave_seg_std = np.mean(d3dp_pbest_seg_std)
print("d3dp_pbest_seg_std:", d3dp_pbest_ave_seg_std)

d3dp_jbest_ave_sag_sym = np.mean(d3dp_jbest_sag_sym)
print("d3dp_jbest_sag_sym:", d3dp_jbest_ave_sag_sym)
d3dp_jbest_ave_seg_std = np.mean(d3dp_jbest_seg_std)
print("d3dp_jbest_seg_std:", d3dp_jbest_ave_seg_std)

# d3dp_pbest_sag_sym: 6.909872
# d3dp_pbest_seg_std: 8.970846
# d3dp_jbest_sag_sym: 6.806175
# d3dp_jbest_seg_std: 8.84001

# %% >>> ManiPose - J-Best MPJPE eval <<<

manipose_hyps_path = "/home/crommel/shared/crommel/mlflow_files/mlruns/550333576130974034/f078246cece241fda4a78b99a726d306/artifacts/all_pred_hyps.pkl"
with open(manipose_hyps_path, "rb") as f:
    manipose_hyps_per_act = pickle.load(f)

# %%
from einops import rearrange
    
def calc_jbest_mpjpe(predicted, target):
    B, H, L, J, D = predicted.shape
    target = target.unsqueeze(1).repeat(1, H, 1, 1, 1)
    errors = torch.norm(predicted - target, dim=len(target.shape)-1)
    errors = rearrange(errors, 'B H L J  -> H B L J', )
    min_errors = torch.min(errors, dim=0, keepdim=False).values  # XXX: minimize across hypotheses --> (B, L, J)
    return torch.mean(min_errors)

    
def calc_jbest_pose(predicted, target):
    B, H, L, J, D = predicted.shape
    target = target.unsqueeze(1).repeat(1, H, 1, 1, 1)
    errors = torch.norm(predicted - target, dim=len(target.shape)-1)
    errors = rearrange(errors, 'B H L J  -> H B L J', )
    jbest_idx = torch.argmin(errors, dim=0, keepdim=False)  # XXX: minimize across hypotheses --> (B, L, J)

    # create J-Best pose
    batch_indices = torch.arange(B)[:, None, None, None].repeat(1, L, J, D)
    seq_indices = torch.arange(L)[None, :, None, None].repeat(B, 1, J, D)
    joint_indices = torch.arange(J)[None, None, :, None].repeat(B, L, 1, D)
    coord_indices = torch.arange(D)[None, None, None, :].repeat(B, L, J, 1)
    jbest_pose = predicted.clone().permute(0, 2, 3, 4, 1)
    jbest_pose = jbest_pose[
        batch_indices,
        seq_indices,
        joint_indices,
        coord_indices,
        jbest_idx[..., None].repeat(1, 1, 1, D),
    ]

    return jbest_pose

# %%

jbest_mpjpe_per_act = []
for preds, target in manipose_hyps_per_act:
    concat_preds = torch.concatenate(preds, dim=0)
    concat_preds = concat_preds[..., :-1]  # drop scores
    concat_target = torch.concatenate(target, dim=0) * 1000
    jbest_mpjpe_per_act.append(
        calc_jbest_mpjpe(concat_preds, concat_target).cpu().numpy()
    )

# %%
print("ManiPose J-Best MPJPE:", np.mean(jbest_mpjpe_per_act))
# ManiPose J-Best MPJPE: 36.727062
# %%
jbest_mpsce_per_act = []
jbest_mpsse_per_act = []
for preds, target in manipose_hyps_per_act:
    concat_preds = torch.concatenate(preds, dim=0)
    concat_preds = concat_preds[..., :-1]  # drop scores
    concat_target = torch.concatenate(target, dim=0) * 1000
    jbest_pose = calc_jbest_pose(concat_preds, concat_target)

    jbest_pose = jbest_pose.permute(0, 3, 2, 1)
    jbest_mpsse_per_act.append(
        sagittal_symmetry(
            joints_coords=jbest_pose,
            skeleton=dataset.skeleton,
            mode="average",
            squared=False,
        )
        .cpu()
        .numpy()
    )

    jbest_mpsce_per_act.append(
        segments_time_consistency(
            joints_coords=jbest_pose,
            skeleton=dataset.skeleton,
            mode="std",
        )
        .cpu()
        .numpy()
    )

# %%
print("ManiPose J-Best MPSCE:", np.mean(jbest_mpsce_per_act))
print("ManiPose J-Best MPSSE:", np.mean(jbest_mpsse_per_act))
# ManiPose J-Best MPSCE: 4.9557548
# ManiPose J-Best MPSSE: 5.3416624

# %%
