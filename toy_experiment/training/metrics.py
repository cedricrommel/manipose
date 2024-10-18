import torch
import numpy as np


def calc_mpjpe(
    pred: torch.Tensor,  # supposed to be (B, 2)
    gt: torch.Tensor,  # supposed to be (B, 2)
) -> float:
    return torch.norm(pred - gt, p=2, dim=1).mean().item()


def oracle_multihyp_mpjpe(
    hypothesis: torch.Tensor,  # supposed to be (B, H, 2)
    gt: torch.Tensor,  # supposed to be (B, 2)
) -> float:
    hypothesis_pred = hypothesis[..., :2]
    expanded_gt = gt[:, None, :].expand_as(hypothesis_pred)  # (B, H, 2)
    mpjpe_per_hyp = torch.norm(  # (B, H)
        hypothesis_pred - expanded_gt,
        p=2,
        dim=2,
    )
    return mpjpe_per_hyp.min(dim=1).mean().item()


def distance_to_circle(
    pred: torch.Tensor,  # supposed to be (B, 2)
) -> float:
    return 1 - torch.norm(pred, p=2, dim=1).mean().item()


def calc_mpjpe_3D(
    pred: torch.Tensor, # supposed to be ((B, 3),(B, 3))
    gt: torch.Tensor, # supposed to be (B, 3)
    joints_predictions=False,
    major_radius=2,
    minor_radius=1,
): 
    gt_joint1, gt_joint2 = toruspoints_to_joints(gt, major_radius=major_radius, minor_radius=minor_radius) # (B,3),(B,3)
    if joints_predictions is True : 
        preds_joint1 = pred[:,:3] #(B,3)
        preds_joint2 = pred[:,3:] #(B,3)
    else : 
        preds_joint1, preds_joint2 = toruspoints_to_joints(pred, major_radius=major_radius, minor_radius=minor_radius) # (B,3),(B,3)

    return (1/2)*(torch.norm(preds_joint1-gt_joint1, p=2, dim=1).mean().item()+torch.norm(preds_joint2-gt_joint2,p=2,dim=1).mean().item())

def oracle_multihyp_mpjpe_3D(
    hypothesis: torch.Tensor,  # supposed to be (B, H, 3)
    gt: torch.Tensor,  # supposed to be (B, 3)
    major_radius=2,
    minor_radius=1,
) :
    # assuming that the hypothesis lie on the torus
    B = hypothesis.shape[0]
    H = hypothesis.shape[1]
    mpjpe_per_hyp = torch.zeros(size=(B,H)) # (B,H)
    for batch in range(B) :
        for hyp in range(H):
            preds_joints_hyp = toruspoints_to_joints(hypothesis[batch,hyp,:].reshape(1,-1), major_radius=major_radius, minor_radius=minor_radius) #(1,3),(1,3)
            mpjpe_per_hyp[batch, hyp] = oracle_multihyp_mpjpe(pred=preds_joints_hyp, gt=gt)
    return mpjpe_per_hyp.min(dim=1).mean().item()


def torushyps_to_joints_np(vector,major_radius=2,minor_radius=1):
    # vector = (B,H,3)
    assert vector.shape[-1] == 3
    B = vector.shape[0]
    H = vector.shape[1]
    norm_xy_plane = np.expand_dims(np.sqrt(vector[:,:,0]**2+vector[:,:,1]**2),axis=-1) # shape (B,H,1)
    norm_xy_plane = np.repeat(norm_xy_plane, repeats=2, axis=2) # shape (B,H,2)
    # assert norm_xv_plane.shape == (B,2)
    joint1 = major_radius*vector[:,:,:2]/norm_xy_plane # shape (B,H,2)
    joint1 = joint1.reshape(B,H,2) # shape (B,H,2)
    joint1 = np.concatenate((joint1,np.zeros(shape=(B,H,1))),axis=-1)
    joint2 = vector

    return (joint1, joint2) #((B,H,3),(B,H,3))

def toruspoints_to_joints_np(vector,major_radius=2,minor_radius=1):
    # vector = (B,3)
    B = vector.shape[0]
    norm_xy_plane = np.expand_dims(np.sqrt(vector[:,0]**2+vector[:,1]**2),axis=-1) # shape (B,1)
    norm_xy_plane = np.repeat(norm_xy_plane, repeats=2, axis=1)
    # assert norm_xv_plane.shape == (B,2)
    joint1 = major_radius*vector[:,:2]/norm_xy_plane
    joint1 = joint1.reshape(B,2)
    joint1 = np.concatenate((joint1,np.zeros(shape=(B,1))),axis=1)
    joint2 = vector

    return (joint1, joint2) #((B,3),(B,3))

def toruspoints_to_joints(vector,major_radius=2,minor_radius=1):
    # vector = (B,3)
    B = vector.shape[0]
    norm_xy_plane = torch.sqrt(vector[:,0]**2+vector[:,1]**2).unsqueeze(1) # shape (B,1)
    norm_xy_plane = torch.repeat_interleave(norm_xy_plane, repeats=2, dim=1)
    # assert norm_xv_plane.shape == (B,2)
    joint1 = major_radius*vector[:,:2]/norm_xy_plane
    joint1 = joint1.reshape(B,2)
    joint1 = torch.cat((joint1,torch.zeros(size=(B,1), device=joint1.device)),axis=1)
    joint2 = vector

    return (joint1, joint2) #((B,3),(B,3))

def torushyps_to_joints(vector,major_radius=2,minor_radius=1):
    # vector = (B,H,3)
    B = vector.shape[0]
    H = vector.shape[1]
    norm_xy_plane = torch.sqrt(vector[:,:,0]**2+vector[:,:,1]**2).unsqueeze(1) # shape (B,H,1)
    norm_xy_plane = torch.repeat_interleave(norm_xy_plane, repeats=2, dim=1) # shape (B,H,2)
    # assert norm_xv_plane.shape == (B,2)
    joint1 = major_radius*vector[:,:,:2]/norm_xy_plane # shape (B,H,2)
    joint1 = joint1.reshape(B,H,2) # shape (B,H,2)
    joint1 = torch.cat((joint1,torch.zeros(size=(B,H,1), device=joint1.device)),axis=-1)
    joint2 = vector

    return (joint1, joint2) #((B,H,3),(B,H,3))

def std_length(pred, joint_prediction=False,mcl_version=False,major_radius=2,minor_radius=1) :
    # ((B,3),(B,3))
    if joint_prediction is True : 
        preds_joint1 = pred[:,:3] #(B,3)
        preds_joint2 = pred[:,3:] #(B,3)
    else : 
        if mcl_version is False :
            preds_joint1, preds_joint2 = toruspoints_to_joints_np(pred, major_radius=major_radius, minor_radius=minor_radius) # (B,3),(B,3)
            dist_joint1 = np.linalg.norm(preds_joint1, ord=2, axis=1)
            dist_joint2 = np.linalg.norm(preds_joint2-preds_joint1, ord=2, axis=1) # (B)
        else : 
            preds_joint1, preds_joint2 = torushyps_to_joints_np(pred[:,:,:-1], major_radius=major_radius, minor_radius=minor_radius) # (B,H,3),(B,H,3)
            H = preds_joint1.shape[1]
            dist_joint1 = np.mean([np.linalg.norm(preds_joint1[:,h,:], ord=2, axis=1) for h in range(H)]) # (B)
            dist_joint2 = np.mean([np.linalg.norm(preds_joint2[:,h,:]-preds_joint1[:,h,:], ord=2, axis=1) for h in range(H)]) # (B)
            return (dist_joint1.std()+dist_joint2.std())/2
    dist_joint1 = np.linalg.norm(preds_joint1, ord=2, axis=1)
    dist_joint2 = np.linalg.norm(preds_joint2-preds_joint1, ord=2, axis=1) # (B)
    return (dist_joint1.std()+dist_joint2.std())/2



