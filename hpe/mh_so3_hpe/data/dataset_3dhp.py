import hashlib

import torch
import torch.utils.data as data
import numpy as np

from .camera import normalize_screen_coordinates
from .generator_3dhp import ChunkedGenerator
from .skeleton import Skeleton
from .h36m_lifting import T_POSE_OPERATORS


# Translation actions allowing to obtain each joint's position in a T-pose,
# starting from its parent.
# T_POSE_OPERATORS = {
#     0: torch.tensor([0, 1, 0], dtype=torch.float),
#     1: torch.tensor([0, 1, 0], dtype=torch.float),
#     2: torch.tensor([1, 0, 0], dtype=torch.float),
#     3: torch.tensor([1, 0, 0], dtype=torch.float),
#     4: torch.tensor([1, 0, 0], dtype=torch.float),
#     5: torch.tensor([-1, 0, 0], dtype=torch.float),
#     6: torch.tensor([-1, 0, 0], dtype=torch.float),
#     7: torch.tensor([-1, 0, 0], dtype=torch.float),
#     8: torch.tensor([1, 0, 0], dtype=torch.float),
#     9: torch.tensor([0, -1, 0], dtype=torch.float),
#     10: torch.tensor([0, -1, 0], dtype=torch.float),
#     11: torch.tensor([-1, 0, 0], dtype=torch.float),
#     12: torch.tensor([0, -1, 0], dtype=torch.float),
#     13: torch.tensor([0, -1, 0], dtype=torch.float),
#     15: torch.tensor([0, 1, 0], dtype=torch.float),
#     16: torch.tensor([0, 1, 0], dtype=torch.float),
# }


MAP_MPI_TO_H36M_JOINTS = [
    10,
    8,
    14,
    15,
    16,
    11,
    12,
    13,
    1,
    2,
    3,
    4,
    5,
    6,
    0,
    7,
    9,
]


MAP_H36M_TO_MPI_JOINTS = [
    14,
    8,
    9,
    10,
    11,
    12,
    13,
    15,
    1,
    16,
    0,
    5,
    6,
    7,
    2,
    3,
    4,
]

JOINS_NAMES = (
    'Hip',
    'RHip',
    'RKnee',
    'RFoot',
    'LHip',
    'LKnee',
    'LFoot',
    'Spine',
    'Thorax',
    'Neck/Nose',
    'Head',
    'LShoulder',
    'LElbow',
    'LWrist',
    'RShoulder',
    'RElbow',
    'RWrist',
)


def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


class Dataset3DHP(data.Dataset):
    def __init__(self, config, root_path, train=True, MAE=False):
        self.data_type = config.data.dataset
        self.train = train
        self.keypoints_name = config.data.keypoints
        self.root_path = root_path
        self.data_augmentation = config.train.flip_aug
        self.reverse_augmentation = False  # <-- as in their config
        if train:
            self.batch_size = config.train.batch_size
        else:
            self.batch_size = config.train.batch_size_test

        self.action_filter = None if config.data.actions == '*' else config.data.actions.split(',')
        self.downsample = config.data.downsample
        self.seq_len = config.data.seq_len
        self.test_aug = config.train.tta
        self.pad = config.data.pad
        self.out_all = config.data.out_all
        self.MAE=MAE
        # self.kps_left, self.kps_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]
        # self.joints_left, self.joints_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]
        # self.skeleton = Skeleton(
        #     parents=[16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, -1, 14, 1],
        #     root_index=14,
        #     joints_left=self.joints_left,
        #     joints_right=self.joints_right,
        #     t_pose_operators=T_POSE_OPERATORS,
        # )
        self.skeleton = Skeleton(
            parents=[-1,  0,  1,  2,  0,  4,  5,  0,  7,  8,  9,  8, 11, 12,  8, 14, 15],
            joints_left=[4, 5, 6, 11, 12, 13],
            joints_right=[1, 2, 3, 14, 15, 16],
            joints_names=JOINS_NAMES,
            t_pose_operators=T_POSE_OPERATORS,
        )

        if self.train:
            self.poses, self.poses_2d = self.prepare_data(self.root_path, train=True)
        else:
            # self.poses_test, self.poses_test_2d, self.valid_frame = self.prepare_data(self.root_path, train=False)
            self.poses, self.poses_2d = self.prepare_data(self.root_path, train=False)

    def prepare_data(self, path, train=True):
        # out_poses_3d = {}
        # out_poses_2d = {}
        out_poses_3d = []
        out_poses_2d = []

        if train == True:
            data = np.load(path+"data_train_3dhp.npz",allow_pickle=True)['data'].item()
            for seq in data.keys():
                for cam in data[seq][0].keys():
                    anim = data[seq][0][cam]

                    # subject_name, seq_name = seq.split(" ")

                    data_3d = anim['data_3d']
                    # data_3d[:, :14] -= data_3d[:, 14:15]
                    # data_3d[:, 15:] -= data_3d[:, 14:15]
                    data_3d -= data_3d[:, 14:15]
                    out_poses_3d.append(
                        data_3d[
                            :,
                            MAP_H36M_TO_MPI_JOINTS,  # <-- permute joints index to match h36m
                        ] / 1000  # <-- converts to m, as in H36M
                    )
                    # out_poses_3d[(subject_name, seq_name, cam)] = data_3d

                    data_2d = anim['data_2d']

                    data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=2048, h=2048)
                    out_poses_2d.append(
                        data_2d[
                            :,
                            MAP_H36M_TO_MPI_JOINTS,  # <-- permute joints index to match h36m
                        ]
                    )
                    # out_poses_2d[(subject_name, seq_name, cam)]=data_2d

            return out_poses_3d, out_poses_2d
        else:
            data = np.load(path + "data_test_3dhp.npz", allow_pickle=True)['data'].item()
            for seq in data.keys():

                anim = data[seq]

                # valid_frame[seq] = anim["valid"]
                valid_frames = anim["valid"].astype(bool)

                data_3d = anim['data_3d']
                # data_3d[:, :14] -= data_3d[:, 14:15]
                # data_3d[:, 15:] -= data_3d[:, 14:15]
                data_3d -= data_3d[:, 14:15]
                out_poses_3d.append(
                    data_3d[valid_frames][
                        :, MAP_H36M_TO_MPI_JOINTS,  # <-- permute joints index to match h36m
                    ] / 1000  # <-- converts to m, as in H36M
                )
                # out_poses_3d[seq] = data_3d

                data_2d = anim['data_2d']

                if seq == "TS5" or seq == "TS6":
                    width = 1920
                    height = 1080
                else:
                    width = 2048
                    height = 2048
                data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=width, h=height)
                out_poses_2d.append(
                    data_2d[valid_frames][
                        :, MAP_H36M_TO_MPI_JOINTS,  # <-- permute joints index to match h36m
                    ]
                )
                # out_poses_2d[seq] = data_2d

            return out_poses_3d, out_poses_2d
            # return out_poses_3d, out_poses_2d, valid_frame


class OriginalDataset3DHP(data.Dataset):
    def __init__(self, config, root_path, train=True, MAE=False):
        self.data_type = config.data.dataset
        self.train = train
        self.keypoints_name = config.data.keypoints
        self.root_path = root_path
        self.data_augmentation = config.train.flip_aug
        self.reverse_augmentation = False  # <-- as in their config
        if train:
            self.batch_size = config.train.batch_size
        else:
            self.batch_size = config.train.batch_size_test

        self.action_filter = None if config.data.actions == '*' else config.data.actions.split(',')
        self.downsample = config.data.downsample
        self.seq_len = config.data.seq_len
        self.test_aug = config.train.tta
        self.pad = config.data.pad
        self.out_all = config.data.out_all
        self.MAE=MAE
        if self.train:
            self.poses_train, self.poses_train_2d = self.prepare_data(self.root_path, train=True)
            # self.cameras_train, self.poses_train, self.poses_train_2d = self.fetch(dataset, self.train_list,
            #                                                                        subset=self.subset)
            self.generator = ChunkedGenerator(self.batch_size // self.seq_len, None, self.poses_train,
                                              self.poses_train_2d, None, chunk_length=self.seq_len, pad=self.pad,
                                              augment=self.data_augmentation, reverse_aug=self.reverse_augmentation,
                                              kps_left=self.kps_left, kps_right=self.kps_right,
                                              joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=self.out_all, MAE=MAE, train = True)
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.poses_test, self.poses_test_2d, self.valid_frame = self.prepare_data(self.root_path, train=False)
            # self.cameras_test, self.poses_test, self.poses_test_2d = self.fetch(dataset, self.test_list,
            #                                                                     subset=self.subset)
            self.generator = ChunkedGenerator(
                batch_size=self.batch_size // self.seq_len,
                cameras=None,
                poses_3d=self.poses_test,
                poses_2d=self.poses_test_2d,
                valid_frame=self.valid_frame,
                # chunk_length=self.seq_len,
                pad=self.pad,
                augment=False,
                kps_left=self.kps_left,
                kps_right=self.kps_right,
                joints_left=self.joints_left,
                joints_right=self.joints_right,
                MAE=MAE,
                train=False
            )
            self.key_index = self.generator.saved_index
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self, path, train=True):
        out_poses_3d = {}
        out_poses_2d = {}
        valid_frame={}

        self.kps_left, self.kps_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]
        self.joints_left, self.joints_right = [5, 6, 7, 11, 12, 13], [2, 3, 4, 8, 9, 10]
        self.skeleton = Skeleton(
            parents=[16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, -1, 14, 1],
            root_index=14,
            joints_left=self.joints_left,
            joints_right=self.joints_right,
        )

        if train == True:
            data = np.load(path+"data_train_3dhp.npz",allow_pickle=True)['data'].item()
            for seq in data.keys():
                for cam in data[seq][0].keys():
                    anim = data[seq][0][cam]

                    subject_name, seq_name = seq.split(" ")

                    data_3d = anim['data_3d']
                    data_3d[:, :14] -= data_3d[:, 14:15]
                    data_3d[:, 15:] -= data_3d[:, 14:15]
                    out_poses_3d[(subject_name, seq_name, cam)] = data_3d

                    data_2d = anim['data_2d']

                    data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=2048, h=2048)
                    out_poses_2d[(subject_name, seq_name, cam)]=data_2d

            return out_poses_3d, out_poses_2d
        else:
            data = np.load(path + "data_test_3dhp.npz", allow_pickle=True)['data'].item()
            for seq in data.keys():

                anim = data[seq]

                valid_frame[seq] = anim["valid"]

                data_3d = anim['data_3d']
                data_3d[:, :14] -= data_3d[:, 14:15]
                data_3d[:, 15:] -= data_3d[:, 14:15]
                out_poses_3d[seq] = data_3d

                data_2d = anim['data_2d']

                if seq == "TS5" or seq == "TS6":
                    width = 1920
                    height = 1080
                else:
                    width = 2048
                    height = 2048
                data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=width, h=height)
                out_poses_2d[seq] = data_2d

            return out_poses_3d, out_poses_2d, valid_frame

    def __len__(self):
        return len(self.generator.pairs)
        #return 200

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]

        if self.MAE:
            cam, input_2D, seq, subject, cam_ind = self.generator.get_batch(seq_name, start_3d, end_3d, flip,
                                                                                      reverse)
            if self.train == False and self.test_aug:
                _, input_2D_aug, _, _,_ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
                input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
        else:
            cam, gt_3D, input_2D, seq, subject, cam_ind = self.generator.get_batch(seq_name, start_3d, end_3d, flip, reverse)

            if self.train == False and self.test_aug:
                _, _, input_2D_aug, _, _,_ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
                input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
            
        bb_box = np.array([0, 0, 1, 1])
        input_2D_update = input_2D

        scale = np.float(1.0)

        if self.MAE:
            if self.train == True:
                return cam, input_2D_update, seq, subject, scale, bb_box, cam_ind
            else:
                return cam, input_2D_update, seq, scale, bb_box
        else:
            if self.train == True:
                return cam, gt_3D, input_2D_update, seq, subject, scale, bb_box, cam_ind
            else:
                return cam, gt_3D, input_2D_update, seq, scale, bb_box


