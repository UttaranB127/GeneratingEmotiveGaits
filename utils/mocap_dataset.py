# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import os
import re
import scipy.ndimage.filters
import torch
import utils.motion.BVH as BVH
import utils.common as common

from utils.motion.Animation import Animation
from utils.Quaternions import Quaternions
from utils.Quaternions_torch import *
from utils.spline import Spline_AS, Spline


class MocapDataset:
    def __init__(self, V, C, joints_to_model, joints_dict):
        self.V = V
        self.C = C
        self.joints_to_model = joints_to_model
        self.joints_to_model = joints_dict['joints_to_model']
        self.joint_parents_all = joints_dict['joint_parents_all']
        self.joint_parents = joints_dict['joint_parents']
        self.joint_names_all = joints_dict['joint_names_all']
        self.joint_names = joints_dict['joint_names']
        self.joint_offsets_all = joints_dict['joint_offsets_all']
        self.joint_offsets = self.joint_offsets_all[self.joints_to_model]
        self.joints_left = joints_dict['joints_left']
        self.joints_right = joints_dict['joints_right']

    @staticmethod
    def has_children(joint, joint_parents):
        return np.isin(joint, joint_parents)

    @staticmethod
    def forward_kinematics(rotations, root_positions, parents, offsets):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
         -- parents: (J) numpy array where each element i contains the parent of joint i.
         -- offsets: (N, J, 3) tensor containing the offset of each joint in the batch.
        """
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4

        positions_world = []
        rotations_world = []

        expanded_offsets = offsets.expand(rotations.shape[0], rotations.shape[1],
                                          offsets.shape[-2], offsets.shape[-1]).contiguous()

        # Parallelize along the batch and time dimensions
        for i in range(offsets.shape[-2]):
            if parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(qrot(rotations_world[parents[i]],
                                            expanded_offsets[:, :, i]) + positions_world[parents[i]])
                if MocapDataset.has_children(i, parents):
                    rotations_world.append(qmul(rotations_world[parents[i]], rotations[:, :, i]))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

    @staticmethod
    def load_bvh(file_name, channel_map=None,
                 start=None, end=None, order=None, world=False):
        '''
        Reads a BVH file and constructs an animation

        Parameters
        ----------
        file_name: str
            File to be opened

        channel_map: Dict
            Mapping between the coordinates x, y, z and
            the positions X, Y, Z in the bvh file

        start : int
            Optional Starting Frame

        end : int
            Optional Ending Frame

        order : str
            Optional Specifier for joint order.
            Given as string E.G 'xyz', 'zxy'

        world : bool
            If set to true euler angles are applied
            together in world space rather than local
            space

        Returns
        -------

        (animation, joint_names, frame_time)
            Tuple of loaded animation and joint names
        '''

        if channel_map is None:
            channel_map = {'Xrotation': 'x', 'Yrotation': 'y', 'Zrotation': 'z'}
        f = open(file_name, 'r')

        i = 0
        active = -1
        end_site = False

        names = []
        orients = Quaternions.id(0)
        offsets = np.array([]).reshape((0, 3))
        parents = np.array([], dtype=int)

        for line in f:

            if 'HIERARCHY' in line:
                continue
            if 'MOTION' in line:
                continue

            root_match = re.match(r'ROOT (\w+)', line)
            if root_match:
                names.append(root_match.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = (len(parents) - 1)
                continue

            if '{' in line:
                continue

            if '}' in line:
                if end_site:
                    end_site = False
                else:
                    active = parents[active]
                continue

            offset_match = re.match(r'\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)', line)
            if offset_match:
                if not end_site:
                    offsets[active] = np.array([list(map(float, offset_match.groups()))])
                continue

            channel_match = re.match(r'\s*CHANNELS\s+(\d+)', line)
            if channel_match:
                channels = int(channel_match.group(1))
                if order is None:
                    channel_is = 0 if channels == 3 else 3
                    channel_ie = 3 if channels == 3 else 6
                    parts = line.split()[2 + channel_is:2 + channel_ie]
                    if any([p not in channel_map for p in parts]):
                        continue
                    order = ''.join([channel_map[p] for p in parts])
                continue

            joint_match = re.match('\s*JOINT\s+(\w+)', line)
            if joint_match:
                names.append(joint_match.group(1))
                offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                orients.qs = np.append(orients.qs, np.array([[1, 0, 0, 0]]), axis=0)
                parents = np.append(parents, active)
                active = (len(parents) - 1)
                continue

            if 'End Site' in line:
                end_site = True
                continue

            frame_match = re.match('\s*Frames:\s+(\d+)', line)
            if frame_match:
                if start and end:
                    frame_num = (end - start) - 1
                else:
                    frame_num = int(frame_match.group(1))
                joint_num = len(parents)
                positions = offsets[np.newaxis].repeat(frame_num, axis=0)
                rotations = np.zeros((frame_num, len(orients), 3))
                continue

            frame_match = re.match('\s*Frame Time:\s+([\d\.]+)', line)
            if frame_match:
                frame_time = float(frame_match.group(1))
                continue

            if (start and end) and (i < start or i >= end - 1):
                i += 1
                continue

            data_match = line.strip().split(' ')
            if data_match:
                data_block = np.array(list(map(float, data_match)))
                N = len(parents)
                fi = i - start if start else i
                if channels == 3:
                    positions[fi, 0:1] = data_block[0:3]
                    rotations[fi, :] = data_block[3:].reshape(N, 3)
                elif channels == 6:
                    data_block = data_block.reshape(N, 6)
                    positions[fi, :] = data_block[:, 0:3]
                    rotations[fi, :] = data_block[:, 3:6]
                elif channels == 9:
                    positions[fi, 0] = data_block[0:3]
                    data_block = data_block[3:].reshape(N - 1, 9)
                    rotations[fi, 1:] = data_block[:, 3:6]
                    positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
                else:
                    raise Exception('Too many channels! {}'.format(channels))

                i += 1

        f.close()

        rotations = qfix_np(Quaternions.from_euler(np.radians(rotations), order=order, world=world).qs)
        positions = MocapDataset.forward_kinematics(torch.from_numpy(rotations).cuda().float().unsqueeze(0),
                                                    torch.from_numpy(positions[:, 0]).cuda().float().unsqueeze(0),
                                                    parents,
                                                    torch.from_numpy(offsets).cuda().float()).squeeze().cpu().numpy()
        orientations, _ = Quaternions(rotations[:, 0]).angle_axis()
        return names, parents, offsets, positions, rotations

    @staticmethod
    def traverse_hierarchy(hierarchy, joint_names, joint_offsets, joint_parents,
                           joint, metadata, tabs, rot_string):
        if joint > 0:
            metadata += '{}JOINT {}\n{}{{\n'.format(tabs, joint_names[joint], tabs)
            tabs += '\t'
            metadata += '{}OFFSET {:.6f} {:.6f} {:.6f}\n'.format(tabs,
                                                                 joint_offsets[joint][0],
                                                                 joint_offsets[joint][1],
                                                                 joint_offsets[joint][2])
            metadata += '{}CHANNELS 3 {}\n'.format(tabs, rot_string)
        while len(hierarchy[joint]) > 0:
            child = hierarchy[joint].pop(0)
            metadata, tabs = MocapDataset.traverse_hierarchy(hierarchy, joint_names,
                                                             joint_offsets, joint_parents,
                                                             child, metadata, tabs, rot_string)
        if MocapDataset.has_children(joint, joint_parents):
            metadata += '{}}}\n'.format(tabs)
        else:
            metadata += '{}End Site\n{}{{\n{}\tOFFSET {:.6f} {:.6f} {:.6f}\n{}}}\n'.format(tabs,
                                                                                           tabs,
                                                                                           tabs, 0, 0, 0, tabs)
            tabs = tabs[:-1]
            metadata += '{}}}\n'.format(tabs)
        if len(hierarchy[joint_parents[joint]]) == 0:
            tabs = tabs[:-1]
        return metadata, tabs

    @staticmethod
    def save_as_bvh(animations, dataset_name=None, subset_name=None, save_file_names=None,
                    fill=6, fps=30.):
        '''
        Saves an animations as a BVH file

        Parameters
        ----------
        animations: Dict containing the joint names, offsets, parents, positions, and rotations
            Animation to be saved.

        dataset_name: str
            Name of the dataset, e.g., mpi.

        subset_name: str
            Name of the subset, e.g., gt, epoch_200.

        save_file_names: str
            Name of the files to be saved. If the files exist, they are overwritten.
            If this is None, then the files are saved in numerical order 0, 1, 2, ...

        fill: int
            Zero padding for file name, if save_file_names is None. Otherwise, it is not used.

        frame_time: float
            Time duration of each frame.
        '''

        os.makedirs('render', exist_ok=True)
        dir_name = os.path.join('render', 'bvh')
        os.makedirs(dir_name, exist_ok=True)
        dir_name = os.path.join(dir_name, dataset_name)
        os.makedirs(dir_name, exist_ok=True)
        if subset_name is not None:
            dir_name = os.path.join(dir_name, subset_name)
            os.makedirs(dir_name, exist_ok=True)

        num_samples = animations['rotations'].shape[0]
        num_frames = animations['rotations'].shape[1]
        frame_time = 1./fps
        num_joints = len(animations['joint_parents'])
        save_quats = animations['rotations'].contiguous().view(num_samples, num_frames,
                                                               num_joints, -1).detach().cpu().numpy()
        for s in range(num_samples):
            trajectory = animations['positions'][s, :, 0].detach().cpu().numpy()
            save_file_name = os.path.join(
                dir_name, (save_file_names[s] if save_file_names is not None else str(s).zfill(fill)) + '.bvh')
            hierarchy = [[] for _ in range(len(animations['joint_parents']))]
            for j in range(len(animations['joint_parents'])):
                if not animations['joint_parents'][j] == -1:
                    hierarchy[animations['joint_parents'][j]].append(j)
            string = ''
            tabs = ''
            joint = 0
            rot_string = 'Zrotation Yrotation Xrotation'
            joint_offsets = animations['joint_offsets'][s].detach().cpu().numpy()
            joint_offsets = np.concatenate((np.zeros_like(joint_offsets[0:1]), joint_offsets), axis=0)
            with open(save_file_name, 'w') as f:
                f.write('{}HIERARCHY\n'.format(tabs))
                f.write('{}ROOT {}\n{{\n'.format(tabs, animations['joint_names'][joint]))
                tabs += '\t'
                f.write('{}OFFSET {:.6f} {:.6f} {:.6f}\n'.format(tabs,
                                                                 joint_offsets[joint][0],
                                                                 joint_offsets[joint][1],
                                                                 joint_offsets[joint][2]))
                f.write('{}CHANNELS 6 Xposition Yposition Zposition {}\n'.format(tabs, rot_string))
                string, tabs = MocapDataset.traverse_hierarchy(hierarchy, animations['joint_names'],
                                                               joint_offsets, animations['joint_parents'],
                                                               joint, string, tabs, rot_string)
                f.write(string)
                f.write('MOTION\nFrames: {}\nFrame Time: {}\n'.format(num_frames + 1, frame_time))
                string = str(trajectory[0, 0]) + ' ' +\
                    str(trajectory[0, 1]) + ' ' + \
                    str(trajectory[0, 2])
                for j in range(num_joints * 3):
                    string += ' ' + '{:.6f}'.format(0)
                f.write(string + '\n')
                for t in range(num_frames):
                    string = str(trajectory[t, 0]) + ' ' + \
                             str(trajectory[t, 1]) + ' ' + \
                             str(trajectory[t, 2])
                    for j in range(num_joints):
                        eulers = np.degrees(Quaternions(save_quats[s, t, j]).euler(order='zyx'))[0]
                        string += ' ' + '{:.6f}'.format(eulers[2]) + \
                                  ' ' + '{:.6f}'.format(eulers[1]) + \
                                  ' ' + '{:.6f}'.format(eulers[0])
                    f.write(string + '\n')

    def get_positions_and_transformations(self, raw_data, mirrored=False):
        data = np.swapaxes(np.squeeze(raw_data), -1, 0)
        if data.shape[-1] == 73:
            positions, root_x, root_z, root_r = data[:, 3:-7], data[:, -7], data[:, -6], data[:, -5]
        elif data.shape[-1] == 66:
            positions, root_x, root_z, root_r = data[:, :-3], data[:, -3], data[:, -2], data[:, -1]
        else:
            raise AssertionError('Input data format not understood')
        num_frames = len(positions)
        positions_local = positions.reshape((num_frames, -1, 3))
        if mirrored:
            positions_local[:, self.joints_left], positions_local[:, self.joints_right] = \
            positions_local[:, self.joints_right], positions_local[:, self.joints_left]
            positions_local[:, :, [0, 2]] = -positions_local[:, :, [0, 2]]
        positions_world = np.zeros_like(positions_local)
        num_joints = positions_world.shape[1]

        trajectory = np.empty((num_frames, 3))
        orientations = np.empty(num_frames)
        rotations = np.zeros((num_frames, num_joints - 1, 4))
        cum_rotations = np.zeros((num_frames, 4))
        rotations_euler = np.zeros((num_frames, num_joints - 1, 3))
        cum_rotations_euler = np.zeros((num_frames, 3))
        translations = np.zeros((num_frames, num_joints, 3))
        cum_translations = np.zeros((num_frames, 3))
        offsets = []
        limbs_all = []

        for t in range(num_frames):
            positions_world[t, :, :] = (Quaternions(cum_rotations[t - 1]) if t > 0 else Quaternions.id(1)) * \
                                       positions_local[t]
            positions_world[t, :, 0] = positions_world[t, :, 0] + (cum_translations[t - 1, 0] if t > 0 else 0)
            positions_world[t, :, 2] = positions_world[t, :, 2] + (cum_translations[t - 1, 2] if t > 0 else 0)
            trajectory[t] = positions_world[t, 0]
            # if t > 0:
            #     rotations[t, 1:] = Quaternions.between(positions_world[t - 1, 1:], positions_world[t, 1:]).qs
            # else:
            #     rotations[t, 1:] = Quaternions.id(positions_world.shape[1] - 1).qs
            limbs = positions_world[t, 1:] - positions_world[t, self.joint_parents[1:]]
            rot_axes = np.cross(self.joint_offsets[1:], limbs)
            rotations[t] = Quaternions.between(self.joint_offsets[1:], limbs, rot_axes)
            limbs_all.append(limbs)
            # limb_recons = Quaternions(rotations[t, 1:]) * self._offsets[1:]
            # test_limbs = np.setdiff1d(np.arange(20), [12, 16])
            # if np.max(np.abs(limb_recons[test_limbs] - limbs[test_limbs])) > 1e-6:
            #     temp = 1
            rotations_euler[t] = Quaternions(rotations[t]).euler('yzx')
            orientations[t] = -root_r[t]
            # rotations[t, 0] = Quaternions.from_angle_axis(-root_r[t], np.array([0, 1, 0])).qs
            # rotations_euler[t, 0] = Quaternions(rotations[t, 0]).euler('yzx')
            cum_rotations[t] = (Quaternions.from_angle_axis(orientations[t], np.array([0, 1, 0])) *
                                (Quaternions(cum_rotations[t - 1]) if t > 0 else Quaternions.id(1))).qs
            # cum_rotations[t] = (Quaternions(rotations[t, 0]) *
            #                     (Quaternions(cum_rotations[t - 1]) if t > 0 else Quaternions.id(1))).qs
            cum_rotations_euler[t] = Quaternions(cum_rotations[t]).euler('yzx')
            offsets.append(Quaternions(cum_rotations[t]) * np.array([0, 0, 1]))
            translations[t, 0] = Quaternions(cum_rotations[t]) * np.array([root_x[t], 0, root_z[t]])
            cum_translations[t] = (cum_translations[t - 1] if t > 0 else np.zeros((1, 3))) + translations[t, 0]
        limbs_all = np.stack(limbs_all)
        limb_lengths = np.linalg.norm(positions_world[:, 1:] - positions_world[:, self.joint_parents[1:]], axis=-1)
        offset_lengths = np.linalg.norm(self.joint_offsets[1:], axis=-1)
        if np.sum(np.abs(limb_lengths - offset_lengths[None, ...])) > 0.001:
            temp = 1
        self.save_as_bvh(np.expand_dims(positions_world[:, 0], 0), np.expand_dims(orientations.reshape(-1, 1), 0),
                         np.expand_dims(rotations, 0), dataset_name='edin', subset_name='test')
        return positions_local, positions_world, trajectory, orientations, rotations, rotations_euler, \
               translations, cum_rotations, cum_rotations_euler, cum_translations, offsets

    def _mirror_sequence(self, sequence):
        mirrored_rotations = sequence['rotations'].copy()
        mirrored_trajectory = sequence['trajectory'].copy()

        joints_left = self._skeleton.joints_left()
        joints_right = self._skeleton.joints_right()

        # Flip left/right joints
        mirrored_rotations[:, joints_left] = sequence['rotations'][:, joints_right]
        mirrored_rotations[:, joints_right] = sequence['rotations'][:, joints_left]

        mirrored_rotations[:, :, [2, 3]] *= -1
        mirrored_trajectory[:, 0] *= -1

        return {
            'rotations': qfix_np(mirrored_rotations),
            'trajectory': mirrored_trajectory
        }

    def mirror(self):
        """
        Perform data augmentation by mirroring every sequence in the dataset.
        The mirrored sequences will have '_m' appended to the action name.
        """
        for subject in self._data.keys():
            for action in list(self._data[subject].keys()):
                if '_m' in action:
                    continue
                self._data[subject][action + '_m'] = self._mirror_sequence(self._data[subject][action])

    def compute_euler_angles(self, order):
        for subject in self._data.values():
            for action in subject.values():
                action['rotations_euler'] = qeuler_np(action['rotations'], order, use_gpu=self._use_gpu)

    def compute_positions(self):
        for subject in self._data.values():
            for action in subject.values():
                rotations = torch.from_numpy(action['rotations'].astype('float32')).unsqueeze(0)
                trajectory = torch.from_numpy(action['trajectory'].astype('float32')).unsqueeze(0)
                if self._use_gpu:
                    rotations = rotations.cuda()
                    trajectory = trajectory.cuda()
                action['positions_world'] = self._skeleton.forward_kinematics(rotations, trajectory).squeeze(
                    0).cpu().numpy()

                # Absolute translations across the XY plane are removed here
                trajectory[:, :, [0, 2]] = 0
                action['positions_local'] = self._skeleton.forward_kinematics(rotations, trajectory).squeeze(
                    0).cpu().numpy()

    @staticmethod
    def get_joints(data):
        hips = data[..., 0, :]
        left_hip_joint = data[..., 1, :]
        left_up_leg = data[..., 2, :]
        left_leg = data[..., 3, :]
        left_foot = data[..., 4, :]
        left_toe_base = data[..., 5, :]
        right_hip_joint = data[..., 6, :]
        right_up_leg = data[..., 7, :]
        right_leg = data[..., 8, :]
        right_foot = data[..., 9, :]
        right_toe_base = data[..., 10, :]
        lower_back = data[..., 11, :]
        spine = data[..., 12, :]
        spine1 = data[..., 13, :]
        neck = data[..., 14, :]
        neck1 = data[..., 15, :]
        head = data[..., 16, :]
        left_shoulder = data[..., 17, :]
        left_arm = data[..., 18, :]
        left_forearm = data[..., 19, :]
        left_hand = data[..., 20, :]
        left_finger_base = data[..., 21, :]
        left_hand_index1 = data[..., 22, :]
        left_thumb = data[..., 23, :]
        right_shoulder = data[..., 24, :]
        right_arm = data[..., 25, :]
        right_forearm = data[..., 26, :]
        right_hand = data[..., 27, :]
        right_finger_base = data[..., 28, :]
        right_hand_index1 = data[..., 29, :]
        right_thumb = data[..., 30, :]
        return hips,\
               left_hip_joint, left_up_leg, left_leg, left_foot, left_toe_base, \
               right_hip_joint, right_up_leg, right_leg, right_foot, right_toe_base, \
               lower_back, spine, spine1, neck, neck1, head, \
               left_shoulder, left_arm, left_forearm, left_hand, left_finger_base, left_hand_index1, left_thumb, \
               right_shoulder, right_arm, right_forearm, right_hand, right_finger_base, right_hand_index1, right_thumb

    @staticmethod
    def get_affective_features(data):
        # 0: hips,
        # 1: left_hip_joint,        2: left_up_leg,         3: left_leg,        4: left_foot,   5: left_toe_base,
        # 6: right_hip_joint,       7: right_up_leg,        8: right_leg,       9: right_foot,  10: right_toe_base,
        # 11: lower_back,           12: spine,              13: spine1,
        # 14: neck,                 15: neck1,              16: head,
        # 17: left_shoulder,        18: left_arm,           19: left_forearm,   20: left_hand,
        # 21: left_finger_base,     22: left_hand_index1,   23: left_thumb,
        # 24: right_shoulder,       25: right_arm,          26: right_forearm,  27: right_hand,
        # 28: right_finger_base,    29: right_hand_index1,  30: right_thumb,

        affs_dim = 18

        _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15,\
        _16, _17, _18, _19, _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30 = MocapDataset.get_joints(data)

        affective_features = np.zeros(data.shape[:-2] + (affs_dim,))
        fidx = 0
        affective_features[..., fidx] = common.angle_between_points(_25, _12, _18)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_27, _12, _20)
        fidx += 1
        affective_features[..., fidx] = common.dist_between(_22, _15) / common.dist_between(_22, _0)
        fidx += 1
        affective_features[..., fidx] = common.dist_between(_29, _15) / common.dist_between(_29, _0)
        fidx += 1
        affective_features[..., fidx] = common.dist_between(_22, _29) / common.dist_between(_15, _0)
        fidx += 1
        affective_features[..., fidx] = \
            common.area_of_triangle(_25, _12, _18) / common.area_of_triangle(_25, _0, _18)
        fidx += 1
        affective_features[..., fidx] = \
            common.area_of_triangle(_27, _12, _20) / common.area_of_triangle(_27, _0, _20)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_18, _19, _20)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_25, _26, _27)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_16, _15, _18)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_16, _15, _25)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_16, _0, _3)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_16, _0, _8)
        fidx += 1
        affective_features[..., fidx] = common.dist_between(_5, _10) / common.dist_between(_15, _0)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_10, _0, _5)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_2, _3, _4)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_7, _8, _9)
        fidx += 1
        affective_features[..., fidx] = \
            common.area_of_triangle(_29, _15, _22) / common.area_of_triangle(_10, _0, _5)
        fidx += 1

        return np.nan_to_num(affective_features)

    @staticmethod
    def build_speed_and_phase_track(positions_world):
        """
        Detect foot steps and extract a control signal that describes the current state of the walking cycle.
        This is based on the assumption that the speed of a foot is almost zero during a contact.
        """
        lf = 4  # Left foot index
        rf = 8  # Right foot index
        l_speed = np.linalg.norm(np.diff(positions_world[:, lf], axis=0), axis=1)
        r_speed = np.linalg.norm(np.diff(positions_world[:, rf], axis=0), axis=1)
        root_speed = np.linalg.norm(np.diff(positions_world[:, 0], axis=0), axis=1)
        displacements = np.cumsum(root_speed)
        left_contact = l_speed[0] < r_speed[0]
        epsilon = 0.1  # Hysteresis (i.e. minimum height difference before a foot switch is triggered)
        cooldown = 3  # Minimum # of frames between steps
        accumulator = np.pi if left_contact else 0
        phase_points = [(0, accumulator)]
        disp_points = [(0, displacements[0])]
        i = cooldown
        while i < len(l_speed):
            if left_contact and l_speed[i] > r_speed[i] + epsilon:
                left_contact = False
                accumulator += np.pi
                phase_points.append((i, accumulator))
                disp_points.append((i, displacements[i] - displacements[disp_points[-1][0]]))
                i += cooldown
            elif not left_contact and r_speed[i] > l_speed[i] + epsilon:
                left_contact = True
                accumulator += np.pi
                phase_points.append((i, accumulator))
                disp_points.append((i, displacements[i] - displacements[disp_points[-1][0]]))
                i += cooldown
            else:
                i += 1

        phase = np.zeros(l_speed.shape[0])
        end_idx = 0
        for i in range(len(phase_points) - 1):
            start_idx = phase_points[i][0]
            end_idx = phase_points[i + 1][0]
            phase[start_idx:end_idx] = np.linspace(phase_points[i][1], phase_points[i + 1][1], end_idx - start_idx,
                                                   endpoint=False)
        phase[end_idx:] = phase_points[-1][1]
        last_point = (phase[-1] - phase[-2]) + phase[-1]
        phase = np.concatenate((phase, [last_point]))
        root_speed = np.concatenate(([0], root_speed))
        return root_speed, phase

    @staticmethod
    def compute_translations_and_controls(data):
        """
        Extract the following features:
        - Translations: longitudinal speed along the spline; height of the root joint.
        - Controls: walking phase as a [cos(theta), sin(theta)] signal;
                    same for the facing direction and movement direction.
        """
        positions_world = data['positions']
        root_speed, phase = MocapDataset.build_speed_and_phase_track(positions_world)
        xy = np.diff(positions_world[:, 0, [0, 2]], axis=0)
        xy = np.concatenate((xy, xy[-1:]), axis=0)
        z = positions_world[:, 0, 1]  # We use a xzy coordinate system
        speeds_abs = np.linalg.norm(xy, axis=1)  # Instantaneous speed along the trajectory
        amplitude = scipy.ndimage.filters.gaussian_filter1d(speeds_abs, 5)  # Low-pass filter
        speeds_abs -= amplitude  # Extract high-frequency details
        # Integrate the high-frequency speed component to recover an offset w.r.t. the trajectory
        speeds_abs = np.cumsum(speeds_abs)

        xy /= np.linalg.norm(xy, axis=1).reshape(-1, 1) + 1e-9  # Epsilon to avoid division by zero

        return np.stack((speeds_abs, z,  # translations and speeds
                         np.cos(phase) * amplitude, np.sin(phase) * amplitude,  # walking phase
                         xy[:, 0] * amplitude, xy[:, 1] * amplitude,  # facing direction
                         root_speed),  # root speed
                        axis=1)

    @staticmethod
    def angle_difference_batch(y, x):
        """
        Compute the signed angle difference y - x,
        where x and y are given as versors.
        """
        return np.arctan2(y[:, :, 1] * x[:, :, 0] - y[:, :, 0] * x[:, :, 1],
                          y[:, :, 0] * x[:, :, 0] + y[:, :, 1] * x[:, :, 1])

    @staticmethod
    def phase_to_features(phase_signal):
        """
        Given a [A(t)*cos(phase), A(t)*sin(phase)] signal, extract a set of features:
        A(t), absolute phase (not modulo 2*pi), angular velocity.
        This function expects a (batch_size, seq_len, 2) tensor.
        """
        assert len(phase_signal.shape) == 3
        assert phase_signal.shape[-1] == 2
        amplitudes = np.linalg.norm(phase_signal, axis=2).reshape(phase_signal.shape[0], -1, 1)
        phase_signal = phase_signal / (amplitudes + 1e-9)
        phase_signal_diff = MocapDataset.angle_difference_batch(phase_signal[:, 1:], phase_signal[:, :-1])
        frequencies = np.pad(phase_signal_diff, ((0, 0), (0, 1)), 'edge').reshape(phase_signal_diff.shape[0], -1, 1)
        return amplitudes, np.cumsum(phase_signal_diff, axis=1), frequencies

    @staticmethod
    def compute_splines(data):
        """
        For each animation in the dataset, this method computes its equal-segment-length spline,
        along with its interpolated tracks (e.g. local speed, footstep frequency).
        """
        xy = data['positions'][:, 0, [0, 2]]
        spline = Spline(xy, closed=False)

        # Add extra tracks (facing direction, phase/amplitude)
        phase = data['trans_and_controls'][:, [2, 3]]
        amplitude, abs_phase, frequency = MocapDataset.phase_to_features(np.expand_dims(phase, 0))

        spline.add_track('amplitude', amplitude[0], interp_mode='linear')
        spline.add_track('phase', np.expand_dims(
            np.concatenate(([0.], abs_phase[0] % (np.pi)), axis=-1), axis=-1), interp_mode='linear')
        spline.add_track('frequency', frequency[0], interp_mode='linear')

        # spline = spline.re-parametrize(5, smoothing_factor=1)
        avg_speed_track = spline.get_track('amplitude')
        avg_speed_track[:] = np.mean(avg_speed_track)
        spline.add_track('average_speed', avg_speed_track, interp_mode='linear')

        return spline

    def get_features_from_data(self, dataset, raw_data=None, mirrored=False,
                               offsets=None, positions=None, orientations=None, rotations=None):
        data_dict = dict()
        if dataset == 'edin':
            data_dict['positions_local'], data_dict['positions_world'], data_dict['trajectory'], \
            data_dict['orientations'], data_dict['rotations'], data_dict['rotations_euler'], \
            data_dict['translations'], data_dict['cum_rotations'], data_dict['cum_rotations_euler'], \
            data_dict['cum_translations'], data_dict['offsets'] = \
                self.get_positions_and_transformations(np.swapaxes(raw_data, -1, 0), mirrored=mirrored)
        elif dataset == 'cmu':
            self.joint_offsets_all = offsets
            data_dict['offsets'] = offsets
            data_dict['positions_world'] = positions
            data_dict['trajectory'] = positions[:, 0]
            data_dict['orientations'] = orientations
            data_dict['rotations'] = rotations
        data_dict['affective_features'] = \
            MocapDataset.get_affective_features(data_dict['positions_world'])
        data_dict['trans_and_controls'] = MocapDataset.compute_translations_and_controls(data_dict)
        data_dict['spline'] = MocapDataset.compute_splines(data_dict)
        return data_dict

    def get_predicted_features(self, pos_past, traj, height, orient_pred, quat_pred):
        num_samples = quat_pred.shape[0]
        num_frames = quat_pred.shape[1]
        orient_pred = orient_pred.view(num_samples, num_frames, 1, -1)
        quat_pred = quat_pred.view(num_samples, num_frames, self.V - 1, -1)
        quats_world = torch.cat((orient_pred, quat_pred), dim=-2)
        root_pred = torch.zeros((num_samples, num_frames, self.C)).cuda().float()
        root_pred[:, :, [0, 2]] = traj
        root_pred[:, :, 1] = height

        pos_pred = MocapDataset.forward_kinematics(quats_world, root_pred,
                                                   self.joint_parents,
                                                   torch.from_numpy(self.joint_offsets).float().cuda())
        affs_pred = torch.tensor(MocapDataset.get_affective_features(pos_pred.detach().cpu().numpy())).cuda().float()
        spline = []
        for s in range(num_samples):
            data_pred_curr = dict()
            data_pred_curr['positions'] = torch.cat((pos_past[s], pos_pred[s]), dim=0).detach().cpu().numpy()
            data_pred_curr['trans_and_controls'] = MocapDataset.compute_translations_and_controls(data_pred_curr)
            spline.append(Spline.extract_spline_features(MocapDataset.compute_splines(data_pred_curr))[0][-1:])
        spline_pred = torch.from_numpy(np.stack(spline, axis=0)).cuda().float()
        return pos_pred, affs_pred, spline_pred

    def __getitem__(self, key):
        return self._data[key]

    def subjects(self):
        return self._data.keys()

    def subject_actions(self, subject):
        return self._data[subject].keys()

    def all_actions(self):
        result = []
        for subject, actions in self._data.items():
            for action in actions.keys():
                result.append((subject, action))
        return result

    def fps(self):
        return self._fps

    def skeleton(self):
        return self._skeleton
