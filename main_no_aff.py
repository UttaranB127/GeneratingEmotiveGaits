import argparse
import os
import numpy as np

from utils import loader, processor_no_aff as processor
from utils.visualizations import display_animations

import torch
from torchlight.torchlight import ngpu

import warnings
warnings.filterwarnings("ignore")


base_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(base_path, '../data')
dataset = 'edin'
spline_dim = 5
if dataset == 'edin':
    joints_to_model = [0,
                       2, 3, 4, 5,
                       7, 8, 9, 10,
                       12, 13,
                       15, 16,
                       18, 19, 20, 21,
                       23, 24, 25, 26]
elif dataset == 'cmu':
    joints_to_model = [0,
                       2, 3, 4, 5,
                       7, 8, 9, 10,
                       12, 13,
                       15, 16,
                       18, 19, 20, 21,
                       25, 26, 27, 28]
joint_offsets = None

num_labels = [4, 3, 3]

model_path = os.path.join(base_path, 'model')
if not os.path.exists(model_path):
    os.mkdir(model_path)


parser = argparse.ArgumentParser(description='Gait Gen')
parser.add_argument('--frame-drop', type=int, default=4, metavar='FD',
                    help='frame down-sample rate (default: 2)')
parser.add_argument('--add-mirrored', type=bool, default=False, metavar='AM',
                    help='perform data augmentation by mirroring all the sequences (default: False)')
parser.add_argument('--train', type=bool, default=True, metavar='T',
                    help='train the model (default: True)')
parser.add_argument('--load_last_best', type=bool, default=True, metavar='LB',
                    help='load the most recent best model (default: True)')
parser.add_argument('--batch-size', type=int, default=16, metavar='B',
                    help='input batch size for training (default: 32)')
parser.add_argument('--num-worker', type=int, default=4, metavar='W',
                    help='input batch size for training (default: 4)')
parser.add_argument('--start_epoch', type=int, default=0, metavar='SE',
                    help='starting epoch of training (default: 0)')
parser.add_argument('--num_epoch', type=int, default=5000, metavar='NE',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--optimizer', type=str, default='Adam', metavar='O',
                    help='optimizer (default: Adam)')
parser.add_argument('--base-lr', type=float, default=1e-3, metavar='LR',
                    help='base learning rate (default: 1e-3)')
parser.add_argument('--base-tr', type=float, default=1., metavar='TR',
                    help='base teacher rate (default: 1.0)')
parser.add_argument('--step', type=list, default=0.05 * np.arange(20), metavar='[S]',
                    help='fraction of steps when learning rate will be decreased (default: [0.5, 0.75, 0.875])')
parser.add_argument('--lr-decay', type=float, default=0.999, metavar='LRD',
                    help='learning rate decay (default: 0.999)')
parser.add_argument('--tf-decay', type=float, default=0.995, metavar='TFD',
                    help='teacher forcing ratio decay (default: 0.995)')
parser.add_argument('--gradient_clip', type=float, default=0.5, metavar='GC',
                    help='gradient clip threshold (default: 0.1)')
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='D',
                    help='Weight decay (default: 5e-4)')
parser.add_argument('--quat-norm-reg', type=float, default=0.1, metavar='QNR',
                    help='regularization for unit norm constraint (default: 0.01)')
parser.add_argument('--quat-reg', type=float, default=1.2, metavar='QR',
                    help='regularization for quaternion loss (default: 0.01)')
parser.add_argument('--spline-reg', type=float, default=1.2, metavar='SR',
                    help='regularization for spline loss (default: 0.01)')
parser.add_argument('--recons-reg', type=float, default=0.1, metavar='RCR',
                    help='regularization for reconstruction loss (default: 1.2)')
parser.add_argument('--eval-interval', type=int, default=1, metavar='EI',
                    help='interval after which model is evaluated (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='LI',
                    help='interval after which log is printed (default: 100)')
parser.add_argument('--save-interval', type=int, default=10, metavar='SI',
                    help='interval after which model is saved (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pavi-log', action='store_true', default=False,
                    help='pavi log')
parser.add_argument('--print-log', action='store_true', default=True,
                    help='print log')
parser.add_argument('--save-log', action='store_true', default=True,
                    help='save log')
# TO ADD: save_result

args = parser.parse_args()
device = 'cuda:0'
randomized = False

model_data_path = os.path.join(model_path, dataset + '_drop_' + str(args.frame_drop) + '_no_aff')
if not os.path.exists(model_data_path):
    os.mkdir(model_data_path)
args.work_dir = model_data_path

# if dataset == 'edin':
data_dict, [data_dict_train, data_dict_valid] = \
    loader.load_edin_data(data_path, num_labels,
                          frame_drop=args.frame_drop, add_mirrored=args.add_mirrored, randomized=randomized)
# data_dict['affective_features'], affs_max, affs_min = loader.scale_data(data_dict['affective_features'])
print('Data points for training:\t{}'.format(len(data_dict_train)))
print('Data points for validation:\t{}'.format(len(data_dict_valid)))
print('Total:\t\t\t\t\t\t{}'.format(len(data_dict)))
num_frames = data_dict['0']['positions'].shape[0]
joints_dict = data_dict['0']['joints_dict']
joint_names = joints_dict['joint_names']
joint_offsets = joints_dict['joint_offsets_all']
joint_parents = joints_dict['joint_parents']
num_joints = len(joint_parents)
coords = data_dict['0']['positions'].shape[-1]
data_loader = dict(train=data_dict_train, test=data_dict)
# elif dataset == 'cmu':
#     data_dict, num_frames = loader.load_cmu_data(data_path,
#                                                  joints_to_model=joints_to_model,
#                                                  frame_drop=args.frame_drop,
#                                                  add_mirrored=args.add_mirrored)
#     joint_offsets = np.zeros_like(data_dict['0']['offsets'])
#     joint_offsets = np.tile(joint_offsets, (len(data_dict), 1, 1))
#     for d in range(len(data_dict)):
#         joint_offsets[d] = data_dict[str(d)]['offsets']
#     joint_offsets = np.mean(joint_offsets, axis=0)
#     data_loader = dict(train=data_dict, test=data_dict)
prefix_length = int(0.3 * num_frames)
target_length = int(num_frames - prefix_length)
rots_dim = data_dict['0']['rotations'].shape[-1]
affs_dim = data_dict['0']['affective_features'].shape[-1]

pr = processor.Processor(args, dataset, data_loader, num_frames, num_joints, coords,
                         rots_dim, affs_dim, spline_dim,
                         joints_dict, joint_names, joint_offsets, joint_parents,
                         num_labels, prefix_length, target_length, generate_while_train=False,
                         save_path=base_path, device=device)

# idx = 1302
# display_animations(np.swapaxes(np.reshape(
#     np.expand_dims(data_dict[str(idx)]['positions_world'], axis=0),
#     (1, num_frames, -1)), 2, 1), num_joints, coords, joint_parents,
#     save=True,
#     dataset_name=dataset, subset_name='test',
#     save_file_names=[str(idx)],
#     overwrite=True)

if args.train:
    pr.train()
# pr.generate_motion(data_dict_valid['0']['spline'], data_dict_valid['0'])
pr.generate_motion(randomized=randomized)
