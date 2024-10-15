# Code adapted from https://github.com/papagina/RotationContinuity

import torch


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(
        v_mag,
        torch.autograd.Variable(  # I can probably clean this below
            torch.FloatTensor([1e-8]).cuda()
        )
    )
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1]*v[:, 2] - u[:, 2]*v[:, 1]
    j = u[:, 2]*v[:, 0] - u[:, 0]*v[:, 2]
    k = u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0]

    out = torch.cat(
        (i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)),
        1
    )  # batch*3

    return out


def compute_rotation_matrix_from_ortho6d(poses):
    """ Conversion from 6D rotation representation to 3x3 rotation matrix as
    proposed in [1]_.

    References
    ----------
    .. [1] Zhou et al. (2019), On the continuity of rotation representations
           in neural networks. Proceedings of the IEEE/CVF Conference on
           Computer Vision and Pattern Recognition 2019.
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def compute_rotation_matrix_from_ortho4d(poses):
    """ New conversion from 4D rotation representation to 3x3 rotation matrix
    constrained to an identity rotation in the last axis, thus continuously
    representing spherical coordinates
    """
    cs_theta_raw = poses[:, 0:2]  # batch*2
    cs_phi_raw = poses[:, 2:4]  # batch*2
    batch_size = poses.shape[0]

    cs_theta = normalize_vector(cs_theta_raw)  # batch*2
    cs_phi = normalize_vector(cs_phi_raw)  # batch*2

    theta_y = torch.cat(
        [
            cs_theta,
            torch.zeros((batch_size, 1), device=cs_theta.device)
        ],
        dim=1
    )
    theta_z = torch.tensor(
        [0, 0, 1],
        device=cs_theta.device,
    ).view(1, 3).expand(batch_size, -1)
    theta_x = cross_product(theta_y, theta_z)

    phi_y = torch.cat(
        [
            torch.zeros((batch_size, 1), device=cs_phi.device),
            cs_phi
        ],
        dim=1
    )
    phi_x = torch.tensor(
        [1, 0, 0],
        device=cs_phi.device
    ).view(1, 3).expand(batch_size, -1)
    phi_z = cross_product(phi_x, phi_y)

    R_theta = torch.cat(
        (
            theta_x.view(-1, 3, 1),
            theta_y.view(-1, 3, 1),
            theta_z.view(-1, 3, 1)
        ),
        dim=2
    )
    R_phi = torch.cat(
        (
            phi_x.view(-1, 3, 1),
            phi_y.view(-1, 3, 1),
            phi_z.view(-1, 3, 1)
        ),
        dim=2
    )

    matrix = R_theta.bmm(R_phi)  # batch*3*3
    return matrix


# def _invert_cos_sin_vec(vec):
#     sign_inv = torch.Tensor([-1, 1])
#     return vec[..., ::-1] * sign_inv


# def compute_rotation_matrix_from_ortho4d(poses):
#     """ New conversion from 4D rotation representation to 3x3 rotation matrix
#     constrained to an identity rotation in the last axis, thus continuously
#     representing spherical coordinates
#     """
#     cs_theta_raw = poses[:, 0:2]  # batch*2
#     cs_phi_raw = poses[:, 2:4]  # batch*2
#     batch_size = poses.shape[0]

#     cs_theta = normalize_vector(cs_theta_raw)  # batch*2
#     cs_phi = normalize_vector(cs_phi_raw)  # batch*2

#     R_theta = torch.stack(  # batch*3*3
#         [
#             torch.stack(
#                 [
#                     cs_theta,
#                     _invert_cos_sin_vec(cs_theta),
#                     torch.zeros(batch_size, 2)
#                 ],
#                 dim=2
#             ),
#             torch.Tensor([0, 0, 1]).view(1, 1, 3).expand(batch_size, 1, 3)
#         ],
#         dim=1,
#     )

#     R_phi = torch.stack(  # batch*3*3
#         [
#             torch.Tensor([1, 0, 0]).view(1, 1, 3).expand(batch_size, 1, 3),
#             torch.stack(
#                 [
#                     torch.zeros(batch_size, 2),
#                     cs_phi,
#                     _invert_cos_sin_vec(cs_phi)
#                 ],
#                 dim=2
#             )
#         ],
#         dim=1,
#     )

#     matrix = R_theta.mm(R_phi)  # batch*3*3
#     return matrix
