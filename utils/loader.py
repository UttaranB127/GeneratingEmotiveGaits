# sys
import csv
import glob
import numpy as np
import os

from utils.mocap_dataset import MocapDataset

# torch
import torch
from torchvision import datasets, transforms


def split_data_dict(data_dict, valid_size=0.1, randomized=True):
    num_samples = len(data_dict)
    num_samples_valid = int(round(valid_size * num_samples))
    samples_all = np.arange(num_samples)
    if randomized:
        samples_valid = np.random.choice(samples_all, num_samples_valid, replace=False)
    else:
        samples_valid = np.arange(num_samples - num_samples_valid, num_samples)
    samples_train = np.setdiff1d(samples_all, samples_valid)
    data_dict_train = dict()
    data_dict_valid = dict()
    for idx, sample_idx in enumerate(samples_train):
        data_dict_train[str(idx)] = data_dict[str(sample_idx)]
    for idx, sample_idx in enumerate(samples_valid):
        data_dict_valid[str(idx)] = data_dict[str(sample_idx)]
    return data_dict_train, data_dict_valid


def load_cmu_data(_path, V, C, joints_to_model=None, frame_drop=1, add_mirrored=False):
    data_path = os.path.join(_path, 'data_cmu_cleaned')
    cmu_data_dict_file = os.path.join(_path, 'cmu_data_dict_drop_' + str(frame_drop) + '.npz')
    try:
        data_dict = np.load(cmu_data_dict_file, allow_pickle=True)['data_dict'].item()
        min_time_steps = np.load(cmu_data_dict_file, allow_pickle=True)['min_time_steps'].item()
        print('Data file found. Returning data.')
    except FileNotFoundError:
        cmu_data_files = glob.glob(os.path.join(data_path, '*.bvh'))
        channel_map = {
            'Xrotation': 'x',
            'Yrotation': 'y',
            'Zrotation': 'z'
        }
        mocap = MocapDataset(V, C)
        data_dict = dict()
        labels_file = os.path.join(_path, 'labels_cmu/cmu_labels.csv')
        data_name = ['' for _ in range(len(cmu_data_files))]
        labels = []
        with open(os.path.join(labels_file)) as csv_file:
            read_lines = csv.reader(csv_file, delimiter=',')
            row_count = -1
            for row in read_lines:
                row_count += 1
                if row_count == 0:
                    labels_order = [x.lower() for x in row[1:]]
                    continue
                data_name[row_count - 1] = row[0]
                labels.append(list(map(float, row[1:])))
        labels = np.stack(labels)
        labels /= np.linalg.norm(labels, ord=1, axis=-1)[..., None]
        emo_idx = [labels_order.index(x) for x in ['happy', 'sad', 'angry', 'neutral']]
        labels = labels[:, emo_idx]
        num_files = len(cmu_data_files)
        min_time_steps = np.inf
        for data_counter, file in enumerate(cmu_data_files):
            offsets, positions, orientations, rot_in, rotations =\
                mocap.load_bvh(file, channel_map, joints_to_model=joints_to_model)
            if len(positions) - 1 < min_time_steps:
                min_time_steps = len(positions) - 1
            data_dict[str(data_counter)] = mocap.get_features_from_data('cmu',
                                                                        offsets=offsets,
                                                                        positions=positions[1::frame_drop],
                                                                        orientations=orientations[1::frame_drop],
                                                                        rotations=rotations[1::frame_drop])
            file_name = file.split('/')[-1].split('.')[0]
            # mocap.save_as_bvh(np.expand_dims(positions[:, 0], axis=0),
            #                   np.expand_dims(np.expand_dims(orientations, axis=-1), axis=0),
            #                   np.expand_dims(rotations, axis=0),
            #                   np.expand_dims(rot_in, axis=0),
            #                   dataset_name='cmu', subset_name='test', save_file_names=[file_name])
            data_dict[str(data_counter)]['labels'] = labels[data_name.index(file_name)]
            print('\rData file not found. Processing file {}/{}: {:3.2f}%'.format(
                data_counter + 1, num_files, data_counter * 100. / num_files), end='')
        min_time_steps = int(min_time_steps / frame_drop)
        print('\rData file not found. Processing files: done. Saving...', end='')
        np.savez_compressed(cmu_data_dict_file, data_dict=data_dict, min_time_steps=min_time_steps)
        print('done. Returning data.')
    return data_dict, min_time_steps


def load_edin_labels(_path, num_labels):
    labels_dirs = [os.path.join(_path, 'labels_edin_locomotion')]
                   # os.path.join(_path, 'labels_edin_xsens')]
    labels = []
    num_annotators = np.zeros(len(labels_dirs))
    for didx, labels_dir in enumerate(labels_dirs):
        annotators = os.listdir(labels_dir)
        num_annotators_curr = len(annotators)
        labels_curr = np.zeros((num_labels[didx], num_annotators_curr))
        for file in annotators:
            with open(os.path.join(labels_dir, file)) as csv_file:
                read_line = csv.reader(csv_file, delimiter=',')
                row_count = -1
                for row in read_line:
                    row_count += 1
                    if row_count == 0:
                        continue
                    try:
                        data_idx = int(row[0].split('_')[-1])
                    except ValueError:
                        data_idx = row[0].split('/')[-1]
                        data_idx = int(data_idx.split('.')[0])
                    emotion = row[1].split(sep=' ')
                    behavior = row[2].split(sep=' ')
                    personality = row[3].split(sep=' ')
                    try:
                        if len(emotion) == 1 and emotion[0].lower() == 'neutral':
                            labels_curr[data_idx, 3] += 1.
                        elif len(emotion) > 1:
                            counter = 0.
                            if emotion[0].lower() == 'extremely':
                                counter = 1.
                            elif emotion[0].lower() == 'somewhat':
                                counter = 1.
                            if emotion[1].lower() == 'happy':
                                labels_curr[data_idx, 0] += counter
                            elif emotion[1].lower() == 'sad':
                                labels_curr[data_idx, 1] += counter
                            elif emotion[1].lower() == 'angry':
                                labels_curr[data_idx, 2] += counter
                        if len(behavior) == 1 and behavior[0].lower() == 'neutral':
                            labels_curr[data_idx, 6] += 1.
                        elif len(behavior) > 1:
                            counter = 0.
                            if behavior[0].lower() == 'highly':
                                counter = 2.
                            elif behavior[0].lower() == 'somewhat':
                                counter = 1.
                            if behavior[1].lower() == 'dominant':
                                labels_curr[data_idx, 4] += counter
                            elif behavior[1].lower() == 'submissive':
                                labels_curr[data_idx, 5] += counter
                        if len(personality) == 1 and personality[0].lower() == 'neutral':
                            labels_curr[data_idx, 9] += 1.
                        elif len(personality) > 1:
                            counter = 0.
                            if personality[0].lower() == 'extremely':
                                counter = 2.
                            elif personality[0].lower() == 'somewhat':
                                counter = 1.
                            if personality[1].lower() == 'friendly':
                                labels_curr[data_idx, 7] += counter
                            elif personality[1].lower() == 'unfriendly':
                                labels_curr[data_idx, 8] += counter
                    except IndexError:
                        continue
        labels_curr /= (num_annotators_curr * 2.)
        labels.append(labels_curr)
        num_annotators[didx] = num_annotators_curr
    return np.vstack(labels), num_annotators


def load_edin_data(_path, num_labels, frame_drop=1, add_mirrored=False, randomized=True):
    if add_mirrored:
        edin_data_dict_file = os.path.join(_path, 'edin_data_dict_with_mirrored_drop_' + str(frame_drop) + '.npz')
    else:
        edin_data_dict_file = os.path.join(_path, 'edin_data_dict_drop_' + str(frame_drop) + '.npz')
    try:
        data_dict = np.load(edin_data_dict_file, allow_pickle=True)['data_dict'].item()
        print('Data file found. Returning data.')
    except FileNotFoundError:
        data_dict = dict()
        data_counter = 0
        bvh_files = glob.glob(os.path.join(_path, 'data_edin_original') + '/*.bvh')
        num_files = len(bvh_files)
        discard_idx = []
        for fidx, bvh_file in enumerate(bvh_files):
            names, parents, offsets, \
            positions, rotations = MocapDataset.load_bvh([f for f in bvh_files if str(fidx).zfill(6) in f][0])
            if len(positions) < 241:
                discard_idx.append(fidx)
                continue
            positions_down_sampled = positions[1::frame_drop]
            rotations_down_sampled = rotations[1::frame_drop]
            joints_dict = dict()
            joints_dict['joints_to_model'] = np.arange(len(parents))
            joints_dict['joint_parents_all'] = parents
            joints_dict['joint_parents'] = parents
            joints_dict['joint_names_all'] = names
            joints_dict['joint_names'] = names
            joints_dict['joint_offsets_all'] = offsets
            joints_dict['joints_left'] = [idx for idx, name in enumerate(names) if 'left' in name.lower()]
            joints_dict['joints_right'] = [idx for idx, name in enumerate(names) if 'right' in name.lower()]
            dict_key = str(data_counter)
            data_counter += 1
            data_dict[dict_key] = dict()
            data_dict[dict_key]['joints_dict'] = joints_dict
            data_dict[dict_key]['positions'] = positions_down_sampled
            data_dict[dict_key]['rotations'] = rotations_down_sampled
            data_dict[dict_key]['affective_features'] = MocapDataset.get_affective_features(positions_down_sampled)
            data_dict[dict_key]['trans_and_controls'] =\
                MocapDataset.compute_translations_and_controls(data_dict[dict_key])
            data_dict[dict_key]['spline'] = MocapDataset.compute_splines(data_dict[dict_key])
            print('\rData file not found. Processing file: {:3.2f}%'.format(fidx * 100. / num_files), end='')
        print('\rData file not found. Processing file: done. Saving...', end='')
        labels, num_annotators = load_edin_labels(_path, np.array([num_files], dtype='int'))
        if add_mirrored:
            labels = np.repeat(labels, 2, axis=0)
        label_partitions = np.append([0], np.cumsum(num_labels))
        for lpidx in range(len(num_labels)):
            labels[:, label_partitions[lpidx]:label_partitions[lpidx + 1]] = \
                labels[:, label_partitions[lpidx]:label_partitions[lpidx + 1]] / \
                np.linalg.norm(labels[:, label_partitions[lpidx]:label_partitions[lpidx + 1]], ord=1, axis=1)[:, None]
        for data_counter, idx in enumerate(np.setdiff1d(range(num_files), discard_idx)):
            data_dict[str(data_counter)]['labels'] = labels[idx]
        np.savez_compressed(edin_data_dict_file, data_dict=data_dict)
        print('done. Returning data.')
    return data_dict, split_data_dict(data_dict, randomized=randomized)


def scale_data(_data, data_max=None, data_min=None):
    _data = _data.astype('float32')
    if data_max is None:
        data_max = np.max(_data)
    if data_min is None:
        data_min = np.min(_data)
    return (_data - data_min) / (data_max - data_min), data_max, data_min


def scale_per_joint(_data, _nframes):
    max_per_joint = np.empty((_data.shape[0], _data.shape[2]))
    min_per_joint = np.empty((_data.shape[0], _data.shape[2]))
    for sidx in range(_data.shape[0]):
        max_per_joint[sidx, :] = np.amax(_data[sidx, :int(_nframes[sidx] - 1), :], axis=0)
        min_per_joint[sidx, :] = np.amin(_data[sidx, :int(_nframes[sidx] - 1), :], axis=0)
    max_per_joint = np.amax(max_per_joint, axis=0)
    min_per_joint = np.amin(min_per_joint, axis=0)
    data_scaled = np.empty_like(_data)
    for sidx in range(_data.shape[0]):
        max_repeated = np.repeat(np.expand_dims(max_per_joint, axis=0), _nframes[sidx] - 1, axis=0)
        min_repeated = np.repeat(np.expand_dims(min_per_joint, axis=0), _nframes[sidx] - 1, axis=0)
        data_scaled[sidx, :int(_nframes[sidx] - 1), :] =\
            np.nan_to_num(np.divide(_data[sidx, :int(_nframes[sidx] - 1), :] - min_repeated,
                                    max_repeated - min_repeated))
    return data_scaled, max_per_joint, min_per_joint


# descale generated data
def descale(data, data_max, data_min):
    data_descaled = data*(data_max-data_min)+data_min
    return data_descaled


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]
