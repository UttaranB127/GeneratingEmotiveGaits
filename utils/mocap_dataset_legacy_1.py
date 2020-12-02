# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import os
import scipy.ndimage.filters
import torch
import utils.motion.BVH as BVH
import utils.common as common

from utils.motion.Animation import Animation
from utils.Quaternions import Quaternions
from utils.Quaternions_torch import *
from utils.spline import Spline_AS, Spline


class MocapDataset:
    def __init__(self, V, C, joints_to_model, joint_parents_all, joint_parents,
                 joints_left=None, joints_right=None):
        self.V = V
        self.C = C
        self.joint_parents_all = joint_parents_all
        self.joint_parents = joint_parents
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.has_children_all = np.zeros_like(joint_parents_all)
        for j in range(len(joint_parents_all)):
            self.has_children_all[j] = np.isin(j, joint_parents_all)
        self.has_children = np.zeros_like(joint_parents)
        for j in range(len(joint_parents)):
            self.has_children[j] = np.isin(j, joint_parents)
        self.joint_names_all = ['Root',
                                'R_LH_Conn',    'LeftHip',          'LeftKnee',     'LeftHeel',     'LeftToe',
                                'R_RH_Conn',    'RightHip',         'RightKnee',    'RightHeel',    'RightToe',
                                'R_LB_Conn',    'LowerBack',        'Spine',        'Neck',
                                'N_H_Coonn',    'Head',
                                'N_LS_Conn',    'LeftShoulder',     'LeftElbow',    'LeftHand',     'LeftHandIndex',
                                'N_RS_Conn',    'RightShoulder',    'RightElbow',   'RightHand',    'RightHandIndex']
        self.joint_offsets_all = np.array([
            [0.,                    0.,         0.],        # root
            [0.,                    0.,         0.],        # root-left hip connector
            [1.36306,               -1.79463,   0.83929],   # left hip
            [2.44811,               -6.72613,   0.],        # left knee
            [2.5622,                -7.03959,   0.],        # left heel
            [0.15764,               -0.43311,   2.32255],   # left toe
            [0.,                    0.,         0.],        # root-right hip connector
            [-1.30552,              -1.79463,   0.83929],   # right hip
            [-2.54253,              -6.98555,   0.],        # right knee
            [-2.56826,              -7.05623,   0.],        # right heel
            [-0.16473,              -0.45259,   2.36315],   # right toe
            [0.,                    0.,         0.],        # root-lower back connector
            [0.02827,               2.03559,    -0.19338],  # lower back
            [0.05672,               2.04885,    -0.04275],  # spine
            [-0.05417,              1.74624,    0.17202],   # neck
            [0.,                    0.,         0.],        # neck-head connector
            [0.10407,               1.76136,    -0.12397],  # head
            [0.,                    0.,         0.],        # neck-left shoulder connector
            # [3.36241,   1.20089,    -0.31121],  # left shoulder
            [2.664712430242292,     1.20089,    -0.31121],  # left shoulder
            [4.983,                 -0.,        -0.],       # left elbow
            [3.48356,               -0.,        -0.],       # left hand
            [0.71526,               -0.,        -0.],       # left hand index
            [0.,                    0.,         0.],        # neck-right shoulder connector
            # [-3.1366,   1.37405,    -0.40465],  # right shoulder
            [-2.350174130038819,    1.37405,    -0.40465],  # right shoulder
            [-5.2419,               -0.,        -0.],       # right elbow
            [-3.44417,              -0.,        -0.],       # right hand
            [-0.62253,              -0.,        -0.]        # right hand index
        ])
        self.joint_offsets = self.joint_offsets_all[joints_to_model]
        self.joint_names = [self.joint_names_all[j] for j in joints_to_model]

    def traverse_hierarchy(self, hierarchy, joint, metadata, tabs, rot_string):
        if joint > 0:
            metadata += '{}JOINT {}\n{}{{\n'.format(tabs, self.joint_names_bvh[joint], tabs)
            tabs += '\t'
            metadata += '{}OFFSET {:.6f} {:.6f} {:.6f}\n'.format(tabs,
                                                                 self.joint_offsets[joint][0],
                                                                 self.joint_offsets[joint][1],
                                                                 self.joint_offsets[joint][2])
            metadata += '{}CHANNELS 3 {}\n'.format(tabs, rot_string)
        while len(hierarchy[joint]) > 0:
            child = hierarchy[joint].pop(0)
            metadata, tabs = self.traverse_hierarchy(hierarchy, child, metadata, tabs, rot_string)
        if self.has_children_all[joint]:
            metadata += '{}}}\n'.format(tabs)
        else:
            metadata += '{}End Site\n{}{{\n{}\tOFFSET {:.6f} {:.6f} {:.6f}\n{}}}\n'.format(tabs,
                                                                                           tabs,
                                                                                           tabs, 0, 0, 0, tabs)
            tabs = tabs[:-1]
            metadata += '{}}}\n'.format(tabs)
        if len(hierarchy[self.joint_parents[joint]]) == 0:
            tabs = tabs[:-1]
        return metadata, tabs

    # Rotations not working properly
    def save_as_bvh(self, trajectory, orientations, quaternions, save_path,
                    save_file_names=None, frame_time=0.032):

        quaternions = np.concatenate((Quaternions.from_angle_axis(-orientations, np.array([0., 1., 0.])).qs,
                                      quaternions), axis=-2)
        num_joints = len(self.joint_parents_all)
        num_samples = quaternions.shape[0]
        for s in range(num_samples):
            num_frames = quaternions[s].shape[0]
            positions = np.tile(self.joint_offsets_all, (num_frames, 1, 1))
            positions[:, 0] = trajectory[s]
            orients = Quaternions.id(num_joints)
            save_file_name = os.path.join(
                save_path, save_file_names[s] if save_file_names is not None else str(s).zfill(6) + '.bvh')
            save_quats = np.zeros((num_frames, num_joints, quaternions.shape[-1]))
            save_quats[..., 0] = 1.
            save_quats[:, 1] = Quaternions(quaternions[s, :, 0]) * \
                               Quaternions(quaternions[s, :, 1])
            save_quats[:, 2] = Quaternions(quaternions[s, :, 1]).__neg__() * \
                               Quaternions(quaternions[s, :, 2])
            save_quats[:, 3] = Quaternions(quaternions[s, :, 2]).__neg__() * \
                               Quaternions(quaternions[s, :, 3])
            save_quats[:, 4] = Quaternions(quaternions[s, :, 3]).__neg__() * \
                               Quaternions(quaternions[s, :, 4])
            save_quats[:, 6] = Quaternions(quaternions[s, :, 0]) * \
                               Quaternions(quaternions[s, :, 5])
            save_quats[:, 7] = Quaternions(quaternions[s, :, 5]).__neg__() * \
                               Quaternions(quaternions[s, :, 6])
            save_quats[:, 8] = Quaternions(quaternions[s, :, 6]).__neg__() * \
                               Quaternions(quaternions[s, :, 7])
            save_quats[:, 9] = Quaternions(quaternions[s, :, 7]).__neg__() * \
                               Quaternions(quaternions[s, :, 8])
            save_quats[:, 11] = Quaternions(quaternions[s, :, 0]) * \
                                Quaternions(quaternions[s, :, 9])
            save_quats[:, 12] = Quaternions(quaternions[s, :, 9]).__neg__() * \
                                Quaternions(quaternions[s, :, 10])
            save_quats[:, 13] = Quaternions(quaternions[s, :, 10]).__neg__() * \
                                Quaternions(quaternions[s, :, 11])
            save_quats[:, 15] = Quaternions(quaternions[s, :, 11]).__neg__() * \
                                Quaternions(quaternions[s, :, 12])
            save_quats[:, 17] = Quaternions(quaternions[s, :, 11]).__neg__() * \
                                Quaternions(quaternions[s, :, 13])
            save_quats[:, 18] = Quaternions(quaternions[s, :, 13]).__neg__() * \
                                Quaternions(quaternions[s, :, 14])
            save_quats[:, 19] = Quaternions(quaternions[s, :, 14]).__neg__() * \
                                Quaternions(quaternions[s, :, 15])
            save_quats[:, 20] = Quaternions(quaternions[s, :, 15]).__neg__() * \
                                Quaternions(quaternions[s, :, 16])
            save_quats[:, 22] = Quaternions(quaternions[s, :, 11]).__neg__() * \
                                Quaternions(quaternions[s, :, 17])
            save_quats[:, 23] = Quaternions(quaternions[s, :, 17]).__neg__() * \
                                Quaternions(quaternions[s, :, 18])
            save_quats[:, 24] = Quaternions(quaternions[s, :, 18]).__neg__() * \
                                Quaternions(quaternions[s, :, 19])
            save_quats[:, 25] = Quaternions(quaternions[s, :, 19]).__neg__() * \
                                Quaternions(quaternions[s, :, 20])
            # counter = 1
            # j = 2
            # while j < num_joints:
            #     # save_quats[:, counter] = Quaternions(save_quats[:, self.joint_parents[j]]).__neg__() * \
            #     #                          Quaternions(quaternions[s, :, j])
            #     save_quats[:, counter] = Quaternions(quaternions[s, :, self.joint_parents[j]]).__neg__() * \
            #                              Quaternions(quaternions[s, :, j])
            #     counter += 1 if self.has_children_all[j] else 2
            #     j += 1 if self.has_children_all[j] else 2
            BVH.save(save_file_name,
                     Animation(Quaternions(save_quats), positions, orients,
                               self.joint_offsets_all, self.joint_parents_all),
                     names=self.joint_names_all, frame_time=frame_time)
        # num_samples = quaternions.shape[0]
        # for s in range(num_samples):
        #     num_frames = quaternions[s].shape[0]
        #     hierarchy = [[] for _ in range(len(self.joint_parents))]
        #     for j in range(len(self.joint_parents)):
        #         if not self.joint_parents[j] == -1:
        #             hierarchy[self.joint_parents[j]].append(j)
        #     string = ''
        #     tabs = ''
        #     joint = 0
        #     rot_string = 'Xrotation Yrotation Zrotation'
        #     save_file_name = os.path.join(
        #         save_path, save_file_names[s] if save_file_names is not None else str(s).zfill(6) + '.bvh')
        #     with open(save_file_name, 'w') as f:
        #         f.write('{}HIERARCHY\n'.format(tabs))
        #         f.write('{}ROOT {}\n{{\n'.format(tabs, self.joint_names_bvh[joint]))
        #         tabs += '\t'
        #         f.write('{}OFFSET {:.6f} {:.6f} {:.6f}\n'.format(tabs,
        #                                                          self.joint_offsets[joint][0],
        #                                                          self.joint_offsets[joint][1],
        #                                                          self.joint_offsets[joint][2]))
        #         f.write('{}CHANNELS 6 Xposition Yposition Zposition {}\n'.format(tabs, rot_string))
        #         string, tabs = self.traverse_hierarchy(hierarchy, joint, string, tabs, rot_string)
        #         f.write(string)
        #         f.write('MOTION\nFrames: {}\nFrame Time: {}\n'.format(num_frames, frame_time))
        #         for t in range(num_frames):
        #             string = str(trajectory[s, t, 0]) + ' ' +\
        #                      str(trajectory[s, t, 1]) + ' ' +\
        #                      str(trajectory[s, t, 2])
        #             string += ' ' + '{:.6f}'.format(0) +\
        #                       ' ' + '{:.6f}'.format(0) +\
        #                       ' ' + '{:.6f}'.format(0)
        #             # -np.degrees(orientations[s, t, 0])
        #             for j in range(1, num_joints):
        #                 quaternions_local = Quaternions(quaternions[s, t, j]) *\
        #                                     Quaternions(quaternions[s, t, self.joint_parents[j]]).__neg__()
        #                 # quaternions_local = Quaternions(quaternions[s, t, j]) *\
        #                 #                     (Quaternions.id(1) if self.joint_parents[j] == 0
        #                 #                      else Quaternions(quaternions[s, t, self.joint_parents[j]]).__neg__())
        #
        #                 eulers = np.degrees(quaternions_local.euler(order='xyz'))[0]
        #                 string += ' ' + '{:.6f}'.format(eulers[0]) + \
        #                           ' ' + '{:.6f}'.format(-eulers[1]) + \
        #                           ' ' + '{:.6f}'.format(eulers[2])
        #             f.write(string + '\n')

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
            rotations[t] = Quaternions.between(self.joint_offsets[1:], limbs)
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
        # limb_lengths = np.zeros((num_frames, 20))
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
        root = data[..., 0, :]
        left_hip = data[..., 1, :]
        left_knee = data[..., 2, :]
        left_heel = data[..., 3, :]
        left_toe = data[..., 4, :]
        right_hip = data[..., 5, :]
        right_knee = data[..., 6, :]
        right_heel = data[..., 7, :]
        right_toe = data[..., 8, :]
        lower_back = data[..., 9, :]
        spine = data[..., 10, :]
        neck = data[..., 11, :]
        head = data[..., 12, :]
        left_shoulder = data[..., 13, :]
        left_elbow = data[..., 14, :]
        left_hand = data[..., 15, :]
        left_hand_index = data[..., 16, :]
        right_shoulder = data[..., 17, :]
        right_elbow = data[..., 18, :]
        right_hand = data[..., 19, :]
        right_hand_index = data[..., 20, :]
        return root, left_hip, left_knee, left_heel, left_toe, \
               right_hip, right_knee, right_heel, right_toe, \
               lower_back, spine, neck, head, \
               left_shoulder, left_elbow, left_hand, left_hand_index, \
               right_shoulder, right_elbow, right_hand, right_hand_index

    @staticmethod
    def get_affective_features(data):
        # 0: root,              1: left_hip,        2: left_knee,   3: left_heel,           4: left_toe,
        # 5: right_hip,         6: right_knee,      7: right_heel,  8: right_toe,
        # 9: lower_back,        10: spine,          11: neck,       12: head,
        # 13: left_shoulder,    14: left_elbow,     15: left_hand,  16: left_hand_index,
        # 17: right_shoulder,   18: right_elbow,    19: right_hand, 20: right_hand_index

        affs_dim = 18

        _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, \
        _11, _12, _13, _14, _15, _16, _17, _18, _19, _20 = MocapDataset.get_joints(data)

        affective_features = np.zeros(data.shape[:-2] + (affs_dim,))
        fidx = 0
        affective_features[..., fidx] = common.angle_between_points(_17, _9, _13)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_19, _9, _15)
        fidx += 1
        affective_features[..., fidx] = common.dist_between(_16, _11) / common.dist_between(_16, _0)
        fidx += 1
        affective_features[..., fidx] = common.dist_between(_20, _11) / common.dist_between(_20, _0)
        fidx += 1
        affective_features[..., fidx] = common.dist_between(_16, _20) / common.dist_between(_11, _0)
        fidx += 1
        affective_features[..., fidx] = \
            common.area_of_triangle(_17, _9, _13) / common.area_of_triangle(_17, _0, _13)
        fidx += 1
        affective_features[..., fidx] = \
            common.area_of_triangle(_19, _9, _15) / common.area_of_triangle(_19, _0, _15)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_13, _14, _15)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_17, _18, _19)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_12, _11, _13)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_12, _11, _17)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_12, _0, _2)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_12, _0, _6)
        fidx += 1
        affective_features[..., fidx] = common.dist_between(_4, _8) / common.dist_between(_11, _0)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_8, _0, _4)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_1, _2, _3)
        fidx += 1
        affective_features[..., fidx] = common.angle_between_points(_5, _6, _7)
        fidx += 1
        affective_features[..., fidx] = \
            common.area_of_triangle(_20, _11, _16) / common.area_of_triangle(_8, _0, _4)
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
        l_speed = torch.norm(positions_world[:, 1:, lf] - positions_world[:, :-1, lf], dim=-1)
        r_speed = torch.norm(positions_world[:, 1:, rf] - positions_world[:, :-1, rf], dim=-1)
        root_speed = torch.norm(positions_world[:, 1:, 0] - positions_world[:, :-1, 0], dim=-1)
        displacements = torch.cumsum(root_speed, dim=-1)
        left_contact = l_speed[:, 0] < r_speed[:, 0]
        epsilon = 0.1  # Hysteresis (i.e. minimum height difference before a foot switch is triggered)
        cooldown = 3  # Minimum # of frames between steps
        accumulator = torch.zeros_like(l_speed[:, 0])
        accumulator[left_contact] = np.pi
        num_samples = len(accumulator)
        phase = torch.zeros_like(l_speed)
        phase[:, 0] = accumulator.clone()
        for s in range(num_samples):
            curr_frame = cooldown
            last_phase_point = 0
            while curr_frame < l_speed.shape[1]:
                if left_contact[s] and l_speed[s, curr_frame] > r_speed[s, curr_frame] + epsilon:
                    left_contact[s] = False
                    phase[s, last_phase_point:curr_frame + 1] = torch.linspace(accumulator[s], accumulator[s] + np.pi,
                                                                               curr_frame - last_phase_point + 1)
                    accumulator[s] += np.pi
                    last_phase_point = curr_frame
                    curr_frame += cooldown
                elif not left_contact[s] and r_speed[s, curr_frame] > l_speed[s, curr_frame] + epsilon:
                    left_contact[s] = True
                    phase[s, last_phase_point:curr_frame + 1] = torch.linspace(accumulator[s], accumulator[s] + np.pi,
                                                                               curr_frame - last_phase_point + 1)
                    accumulator[s] += np.pi
                    last_phase_point = curr_frame
                    curr_frame += cooldown
                else:
                    curr_frame += 1
            phase[s, last_phase_point:] = accumulator[s]
        last_point = (phase[:, -1] - phase[:, -2]) + phase[:, -1]
        phase = torch.cat((phase, last_point.unsqueeze(-1)), dim=-1)
        root_speed = torch.cat((torch.zeros_like(accumulator).unsqueeze(-1), root_speed), dim=-1)
        return root_speed, phase

    @staticmethod
    def compute_translations_and_controls(positions_world, xy_orientation):
        """
        Extract the following features:
        - Translations: longitudinal speed along the spline; height of the root joint.
        - Controls: walking phase as a [cos(theta), sin(theta)] signal;
                    same for the facing direction and movement direction.
        """
        input_in_numpy = False
        if isinstance(positions_world, np.ndarray):
            positions_world = torch.from_numpy(positions_world).cuda().float()
            input_in_numpy = True
        elif not isinstance(positions_world, torch.cuda.FloatTensor):
            raise TypeError('Data type must be either numpy.ndarray oy torch.cuda.FloatTensor')
        root_speed, phase = MocapDataset.build_speed_and_phase_track(positions_world)
        xy = positions_world[:, 1:, 0, [0, 2]] - positions_world[:, :-1, 0, [0, 2]]
        xy = torch.cat((xy, xy[:, -1:]), dim=1)
        z = positions_world[:, :, 0, 1]  # We use a xzy coordinate system
        speeds_abs= torch.norm(xy, dim=-1)  # Instantaneous speed along the trajectory
        amplitude = torch.from_numpy(
            scipy.ndimage.filters.gaussian_filter1d(
                speeds_abs.detach().cpu().numpy(), 5)).cuda().float()  # Low-pass filter
        speeds_abs_hf = speeds_abs - amplitude  # Extract high-frequency details
        # Integrate the high-frequency speed component to recover an offset w.r.t. the trajectory
        traj_offsets = torch.cumsum(speeds_abs_hf, dim=-1)

        xy /= (speeds_abs[..., None] + 1e-9)  # Epsilon to avoid division by zero

        return torch.stack((traj_offsets, z,  # translations and speeds
                            torch.cos(phase) * amplitude, torch.sin(phase) * amplitude,  # walking phase
                            xy[:, :, 0] * amplitude, xy[:, :, 1] * amplitude,  # facing direction
                            torch.sin(xy_orientation), torch.cos(xy_orientation),  # movement direction
                            root_speed),  # root speed
               dim=-1)

    @staticmethod
    def angle_difference_batch(y, x):
        """
        Compute the signed angle difference y - x,
        where x and y are given as versors.
        """
        return torch.atan2(y[:, :, 1] * x[:, :, 0] - y[:, :, 0] * x[:, :, 1],
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
        amplitudes = torch.norm(phase_signal, dim=-1).contiguous().view(phase_signal.shape[0], -1, 1)
        phase_signal = phase_signal / (amplitudes + 1e-9)
        phase_signal_diff = MocapDataset.angle_difference_batch(phase_signal[:, 1:], phase_signal[:, :-1])
        frequencies = torch.cat((phase_signal_diff.unsqueeze(-1), phase_signal_diff[:, -1:].unsqueeze(-1)), dim=1)
        return amplitudes, torch.cumsum(phase_signal_diff, dim=1), frequencies

    @staticmethod
    def compute_splines(xy, xy_orientation, phase):
        """
        For each animation in the dataset, this method computes its equal-segment-length spline,
        along with its interpolated tracks (e.g. local speed, footstep frequency).
        """
        spline = Spline_AS(xy, closed=False)

        # Add extra tracks (facing direction, phase/amplitude)
        # xy_orientation = data['rotations_euler'][:, 0, 1]
        y_rot = torch.cat((torch.sin(xy_orientation).unsqueeze(-1),
                           torch.cos(xy_orientation).unsqueeze(-1)), dim=-1)
        spline.add_track('direction', y_rot, interp_mode='circular')
        amplitude, abs_phase, frequency = MocapDataset.phase_to_features(phase)

        spline.add_track('amplitude', amplitude, interp_mode='linear')
        spline.add_track('phase', torch.cat((torch.zeros((abs_phase.shape[0], 1)).cuda().float(),
                       abs_phase % np.pi), dim=1).unsqueeze(-1), interp_mode='linear')
        spline.add_track('frequency', frequency, interp_mode='linear')

        # spline = spline.re-parametrize(5, smoothing_factor=1)
        avg_speed_track = spline.get_track('amplitude')
        avg_speed_track = torch.mean(avg_speed_track, dim=1).repeat(1, avg_speed_track.shape[1]).unsqueeze(-1)
        spline.add_track('average_speed', avg_speed_track, interp_mode='linear')

        return spline

    def get_features_from_data(self, raw_data, mirrored=False):
        data_dict = dict()
        data_dict['positions_local'], data_dict['positions_world'], data_dict['trajectory'], \
        data_dict['orientations'], data_dict['rotations'], data_dict['rotations_euler'], \
        data_dict['translations'], data_dict['cum_rotations'], data_dict['cum_rotations_euler'], \
        data_dict['cum_translations'], data_dict['offsets'] =\
            self.get_positions_and_transformations(np.swapaxes(raw_data, -1, 0), mirrored=mirrored)
        data_dict['affective_features'] = \
            MocapDataset.get_affective_features(data_dict['positions_world'])
        data_dict['trans_and_controls'] = MocapDataset.compute_translations_and_controls(data_dict)
        data_dict['spline'] = MocapDataset.compute_splines(data_dict)
        return data_dict

    def get_predicted_features(self, pos_past, orient_past, root_pos, quat_pred, orient_pred):
        num_samples = quat_pred.shape[0]
        num_frames = quat_pred.shape[1]
        offsets = torch.from_numpy(self.joint_offsets).cuda().float(). \
            unsqueeze(0).unsqueeze(0).repeat(num_samples, num_frames, 1, 1)
        quat_pred = quat_pred.view(num_samples, num_frames, self.V - 1, -1)
        zeros = torch.zeros_like(orient_pred)
        quats_world = quat_pred.clone()
        quats_world = torch.cat((expmap_to_quaternion(torch.cat((zeros, orient_pred, zeros), dim=-1)).unsqueeze(-2),
                                 quats_world), dim=-2)
        pos_pred = torch.zeros((num_samples, num_frames, self.V, self.C)).cuda().float()
        pos_pred[:, :, 0] = root_pos

        # for joint in range(self.V):
        #     if self.joint_parents[joint] == -1:
        #         quats_world[:, :, joint] = expmap_to_quaternion(torch.cat((zeros, orient_pred, zeros), dim=-1))
        #     else:
        #         pos_pred[:, :, joint] = qrot(quats_world[:, :, self.joint_parents[joint]], offsets[:, :, joint]) \
        #                                 + pos_pred[:, :, self.joint_parents[joint]]
        #         quats_world[:, :, joint] = qmul(quats_world[:, :, self.joint_parents[joint]],
        #                                         quat_pred[:, :, joint - 1])
        for joint in range(1, self.V):
            pos_pred[:, :, joint] = qrot(quats_world[:, :, joint], offsets[:, :, joint]) \
                                    + pos_pred[:, :, self.joint_parents[joint]]
        affs_pred = torch.tensor(MocapDataset.get_affective_features(pos_pred.detach().cpu().numpy())).cuda().float()
        positions_world = torch.cat((pos_past, pos_pred), dim=1)
        orientations = torch.cat((orient_past, orient_pred), dim=1).squeeze()
        trajectory = positions_world[:, :, 0, [0, 2]]
        trans_and_controls = MocapDataset.compute_translations_and_controls(positions_world, orientations)
        splines = MocapDataset.compute_splines(trajectory, orientations, trans_and_controls[:, :, [2, 3]])
        spline_pred = Spline_AS.extract_spline_features(splines)[0][:, -1:]
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
