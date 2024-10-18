# %%
import pickle
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import torch

from mh_so3_hpe.architectures.utils.forward_kinematics import forward_kinematics
from mh_so3_hpe.architectures.pose_decoder import PoseDecoder
from mh_so3_hpe.data import Human36mDataset
from mh_so3_hpe.data.utils import read_3d_data


# matplotlib.use('Agg')
# %%
data_dir = Path("/home/crommel/shared/crommel/h36m_data")
preproc_dataset_path = data_dir / (
    f"preproc_data_3d_h36m_17_"
    f"mh_so3_hpe.pkl"
)
if preproc_dataset_path.exists():
    with open(preproc_dataset_path, "rb") as f:
        dataset = pickle.load(f)
else:
    dataset_path = data_dir / f"data_3d_h36m.npz"
    dataset = Human36mDataset(dataset_path, n_joints=17)

    print("==> Preparing data...")
    dataset = read_3d_data(dataset)
    with open(preproc_dataset_path, "wb") as f:
        pickle.dump(dataset, f)

# %%
decoder = PoseDecoder(skeleton=dataset.skeleton)

# %%
azim = dataset.cameras["S1"][0]['azimuth']


def plot_pose(
    pose,
    skeleton,
    azim,
    size=6,
    radius=2.,
    annot=True,
    annot_ofs=0.05,
    savepath=None,
):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=15., azim=azim)
    ax.set_xlim3d([-radius / 2, radius / 2])
    ax.set_zlim3d([-radius / 2, radius / 2])
    ax.set_ylim3d([-radius / 2, radius / 2])
    ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.dist = 7.5
    # ax.set_title(title)  # , pad=35

    for j, j_parent in enumerate(skeleton.parents):
        if j_parent > -1:
            col = 'red' if j in skeleton.joints_right else 'black'
            pos = pose[0].detach().cpu()
            ax.plot(
                [pos[j, 0], pos[j_parent, 0]],
                [pos[j, 2], pos[j_parent, 2]],
                [pos[j, 1], pos[j_parent, 1]],
                zdir='z',
                c=col,
                marker='o',
            )
            if annot:
                ax.text(
                    s=f"{j}",
                    x=pos[j, 0] + annot_ofs,
                    y=pos[j, 1] + annot_ofs,
                    z=pos[j, 2] + annot_ofs,
                    color=col,
                )
    plt.show()
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")


# %%
unit_bones_lengths = torch.ones(1, 16, 1) * 0.2
unit_t_pose = decoder.build_t_pose_from_bone_lengths(unit_bones_lengths)

plot_pose(
    unit_t_pose,
    skeleton=dataset.skeleton,
    azim=azim,
    annot=False,
    savepath="figures/unit-t-pose.png",
)

# %%
bones_lengths = torch.Tensor(
    [0.2, 0.5, 0.5, 0.2, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.2, 0.4, 0.4]
).reshape(1, 16, 1)
t_pose = decoder.build_t_pose_from_bone_lengths(bones_lengths)

plot_pose(
    t_pose,
    skeleton=dataset.skeleton,
    azim=azim,
    annot=False,
    savepath="figures/t-pose.png",
)

# %%

def compute_rotation_matrix_from_euler(euler):
    batch=euler.shape[0]
        
    c1=torch.cos(euler[:,0]).view(batch,1)#batch*1 
    s1=torch.sin(euler[:,0]).view(batch,1)#batch*1 
    c2=torch.cos(euler[:,2]).view(batch,1)#batch*1 
    s2=torch.sin(euler[:,2]).view(batch,1)#batch*1 
    c3=torch.cos(euler[:,1]).view(batch,1)#batch*1 
    s3=torch.sin(euler[:,1]).view(batch,1)#batch*1 
        
    row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3) #batch*1*3
        
    matrix = torch.cat((row1, row2, row3), 1) #batch*3*3
     
        
    return matrix

# %%
angles = torch.tensor([
    [0, -35, 0],  # 0 - Root
    [0, 0, 0], # 1 - RHip
    [-45, 0, 0],  # 2 - RKnee
    [90, 0, 0],  # 3 - RFoot
    [0, 0, 0],  # 4 - LHip
    [30, 0, 0],  # 5 - LKnee
    [90, 0, 0],  # 6 - LFoot
    [0, 0, 0],  # 7 - Spine
    [0, 0, 0],  # 8 - Thorax
    [0, 0, 45],  # 9 - Neck/Nose
    [0, 0, -45],  # 10 - Head
    [30, 0, 0],  # 11 - LShoulder
    [-90, 0, 90],  # 12 - LElbow
    [0, 90, 0],  # 13 - LWrist
    [-30, 0, 0],  # 14 - RShould
    [-90, 0, 90],  # 15 - RElbow
    [0, 90, 0],  # 16 - RWrist
]) * np.pi / 180.

rotations = compute_rotation_matrix_from_euler(angles).reshape((1, 17, 3, 3))

rot_pose = forward_kinematics(
    t_pose=t_pose,
    rotations=rotations,
    root_positions=torch.zeros((1, 3)),
    skeleton=dataset.skeleton,
)

plot_pose(
    rot_pose,
    skeleton=dataset.skeleton,
    azim=azim,
    annot=False,
    savepath="figures/rot-pose.png"
)

# %%

# angles = torch.tensor([
#     [0, 0, 0],  # 0 - Root
#     [-90, 0, 0], # 1 - RHip
#     [0, 0, 0],  # 2 - RKnee
#     [90, 0, 0],  # 3 - RFoot
#     [0, 0, 0],  # 4 - LHip
#     [0, 0, 0],  # 5 - LKnee
#     [0, 0, 0],  # 6 - LFoot
#     [0, 0, 0],  # 7 - Spine
#     [0, 0, 0],  # 8 - Thorax
#     [0, 0, 0],  # 9 - Neck/Nose
#     [0, 0, 0],  # 10 - Head
#     [0, 0, -10],  # 11 - LShoulder
#     [0, 0, -35],  # 12 - LElbow
#     [0, 0, -45],  # 13 - LWrist
#     [0, 0, 10],  # 14 - RShould
#     [0, 0, 35],  # 15 - RElbow
#     [0, 0, 45],  # 16 - RWrist
# ]) * np.pi / 180.

angles = torch.tensor([
    [0, 0, 0],  # 0 - Root
    [0, 0, 0], # 1 - RHip
    [0, 0, 0],  # 2 - RKnee
    [0, 0, 0],  # 3 - RFoot
    [0, 0, 0],  # 4 - LHip
    [0, 0, 0],  # 5 - LKnee
    [0, 0, 0],  # 6 - LFoot
    [0, 0, 0],  # 7 - Spine
    [0, 0, 0],  # 8 - Thorax
    [0, 0, 0],  # 9 - Neck/Nose
    [0, 0, 0],  # 10 - Head
    [0, 0, 0],  # 11 - LShoulder
    [0, 0, 0],  # 12 - LElbow
    [0, 0, -90],  # 13 - LWrist
    [0, 0, 0],  # 14 - RShould
    [0, 0, 0],  # 15 - RElbow
    [0, 0, 0],  # 16 - RWrist
]) * np.pi / 180.

rotations = compute_rotation_matrix_from_euler(angles).reshape((1, 17, 3, 3))

# %%

rot_pose = forward_kinematics(
    t_pose=t_pose,
    rotations=rotations,
    root_positions=torch.zeros((1, 3)),
    skeleton=dataset.skeleton,
)
# %%
plot_pose(rot_pose, skeleton=dataset.skeleton, azim=azim, annot=False)

# %%
angles = torch.tensor([
    [0, 0, 0],  # 0 - Root
    [0, 0, 0], # 1 - RHip
    [0, 0, 0],  # 2 - RKnee
    [0, 0, 0],  # 3 - RFoot
    [0, 0, 0],  # 4 - LHip
    [0, 0, 0],  # 5 - LKnee
    [0, 0, 0],  # 6 - LFoot
    [0, 0, 0],  # 7 - Spine
    [0, 0, 0],  # 8 - Thorax
    [0, 0, 0],  # 9 - Neck/Nose
    [0, 0, 0],  # 10 - Head
    [0, 0, 0],  # 11 - LShoulder
    [-90, 0, 0],  # 12 - LElbow
    [0, 90, 0],  # 13 - LWrist
    [0, 0, 0],  # 14 - RShould
    [0, 0, 0],  # 15 - RElbow
    [0, 0, 0],  # 16 - RWrist
]) * np.pi / 180.

rotations = compute_rotation_matrix_from_euler(angles).reshape((1, 17, 3, 3))

# %%

rot_pose = forward_kinematics(
    t_pose=t_pose,
    rotations=rotations,
    root_positions=torch.zeros((1, 3)),
    skeleton=dataset.skeleton,
)
# %%
plot_pose(rot_pose, skeleton=dataset.skeleton, azim=azim, annot=False)


# %% -- TEST 3DHP
from mh_so3_hpe.data import Dataset3DHP
from omegaconf import OmegaConf

cfg = OmegaConf.load("./conf/config.yaml")
cfg_3dhp = OmegaConf.load("./conf/data/mpi_inf_3dhp.yaml")
cfg.data.update(cfg_3dhp)

dataset = Dataset3DHP(
    config=cfg,
    root_path=cfg.data.data_dir,
    train=True
)

# # %%

# bones_lengths = torch.Tensor(
#     [
#         0.2,  # 10
#         0.2,  # 8
#         0.2,  # 14
#         0.4,  # 15
#         0.4,  # 16
#         0.2,  # 11
#         0.4,  # 12
#         0.4,  # 13
#         0.2,  # 1
#         0.5,  # 2
#         0.5,  # 3
#         0.2,  # 4
#         0.5,  # 5
#         0.5,  # 6
#         0.2,  # 7
#         0.2,  # 9
#     ]
# ).reshape(1, 16, 1)

# %%

decoder = PoseDecoder(skeleton=dataset.skeleton)
t_pose = decoder.build_t_pose_from_bone_lengths(bones_lengths)
# %%
plot_pose(t_pose, skeleton=dataset.skeleton, azim=azim, annot=False)
# %%
angles = torch.tensor([
    [0, 0, 0],  # 0 - Root
    [0, 0, 0], # 1 - RHip
    [0, 0, 0],  # 2 - RKnee
    [0, 0, 0],  # 3 - RFoot
    [0, 0, 0],  # 4 - LHip
    [0, 0, 0],  # 5 - LKnee
    [0, 0, 0],  # 6 - LFoot
    [0, 0, 0],  # 7 - Spine
    [0, 0, 0],  # 8 - Thorax
    [0, 0, 0],  # 9 - Neck/Nose
    [0, 0, 0],  # 10 - Head
    [0, 0, 0],  # 11 - LShoulder
    [-90, 0, 0],  # 12 - LElbow
    [0, 90, 0],  # 13 - LWrist
    [0, 0, 0],  # 14 - RShould
    [0, 0, 0],  # 15 - RElbow
    [0, 0, 0],  # 16 - RWrist
]) * np.pi / 180.

rotations = compute_rotation_matrix_from_euler(angles).reshape((1, 17, 3, 3))

# %%

rot_pose = forward_kinematics(
    t_pose=t_pose,
    rotations=rotations,
    root_positions=torch.zeros((1, 3)),
    skeleton=dataset.skeleton,
)
# %%
plot_pose(rot_pose, skeleton=dataset.skeleton, azim=azim, annot=False)
# %%
