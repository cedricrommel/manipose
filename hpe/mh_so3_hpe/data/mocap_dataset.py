from __future__ import absolute_import

from copy import deepcopy


class MocapDataset(object):
    def __init__(self, skeleton, fps=None):
        self._skeleton = deepcopy(skeleton)
        self._fps = fps
        self._data = None  # Must be filled by subclass
        self._cameras = None  # Must be filled by subclass

    def remove_joints(self, joints_to_remove):
        kept_joints = self._skeleton.remove_joints(joints_to_remove)
        for subject in self._data.keys():
            for action in self._data[subject].keys():
                s = self._data[subject][action]
                s["positions"] = s["positions"][:, kept_joints]

    def __getitem__(self, key):
        return self._data[key]

    @property
    def subjects(self):
        return self._data.keys()

    @property
    def fps(self):
        return self._fps

    @property
    def skeleton(self):
        return self._skeleton

    @property
    def cameras(self):
        return self._cameras

    def define_actions(self, action):
        # This method can be overridden
        return False
