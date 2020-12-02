import math
import matplotlib.pyplot as plt
import os
import time
import torch.optim as optim
import torch.nn as nn
from net import quater_emonet as quater_emonet

from torchlight.torchlight.io import IO
from utils.common import generate_rvo_trajectories
from utils.mocap_dataset import MocapDataset
from utils.Quaternions import Quaternions
from utils.visualizations import display_animations
from utils import losses
from utils.Quaternions_torch import *
from utils.spline import Spline_AS, Spline

torch.manual_seed(1234)

rec_loss = losses.quat_angle_loss


def find_all_substr(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def get_best_epoch_and_loss(path_to_model_files):
    all_models = os.listdir(path_to_model_files)
    if len(all_models) < 2:
        return '', None, np.inf
    loss_list = -1. * np.ones(len(all_models))
    for i, model in enumerate(all_models):
        loss_val = str.split(model, '_')
        if len(loss_val) > 1:
            loss_list[i] = float(loss_val[3])
    if len(loss_list) < 3:
        best_model = all_models[np.argwhere(loss_list == min([n for n in loss_list if n > 0]))[0, 0]]
    else:
        loss_idx = np.argpartition(loss_list, 2)
        best_model = all_models[loss_idx[1]]
    all_underscores = list(find_all_substr(best_model, '_'))
    # return model name, best loss
    return best_model, int(best_model[all_underscores[0] + 1:all_underscores[1]]),\
        float(best_model[all_underscores[2] + 1:all_underscores[3]])


class Processor(object):
    """
        Processor for emotive gait generation
    """

    def __init__(self, args, dataset, data_loader, T, V, C, D, A, S,
                 joints_dict, joint_names, joint_offsets, joint_parents,
                 num_labels, prefix_length, target_length,
                 min_train_epochs=20, generate_while_train=False,
                 save_path=None, device='cuda:0'):

        self.args = args
        self.dataset = dataset
        self.mocap = MocapDataset(V, C, np.arange(V), joints_dict)
        self.joint_names = joint_names
        self.joint_offsets = joint_offsets
        self.joint_parents = joint_parents
        self.device = device
        self.data_loader = data_loader
        self.num_labels = num_labels
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        self.io = IO(
            self.args.work_dir,
            save_log=self.args.save_log,
            print_log=self.args.print_log)

        # model
        self.T = T
        self.V = V
        self.C = C
        self.D = D
        self.A = A
        self.S = S
        self.O = 4
        self.Z = 1
        self.RS = 1
        self.o_scale = 10.
        self.prefix_length = prefix_length
        self.target_length = target_length
        self.model = quater_emonet.QuaterEmoNet(V, D, S, A, self.O, self.Z, self.RS, num_labels[0])
        self.model.cuda(device)
        self.orient_h = None
        self.quat_h = None
        self.z_rs_loss_func = nn.L1Loss()
        self.affs_loss_func = nn.L1Loss()
        self.spline_loss_func = nn.L1Loss()
        self.best_loss = math.inf
        self.loss_updated = False
        self.mean_ap_updated = False
        self.step_epochs = [math.ceil(float(self.args.num_epoch * x)) for x in self.args.step]
        self.best_loss_epoch = None
        self.min_train_epochs = min_train_epochs

        # generate
        self.generate_while_train = generate_while_train
        self.save_path = save_path

        # optimizer
        if self.args.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.base_lr,
                momentum=0.9,
                nesterov=self.args.nesterov,
                weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.base_lr)
                # weight_decay=self.args.weight_decay)
        else:
            raise ValueError()
        self.lr = self.args.base_lr
        self.tf = self.args.base_tr

    def process_data(self, data, poses, quat, trans, affs):
        data = data.float().to(self.device)
        poses = poses.float().to(self.device)
        quat = quat.float().to(self.device)
        trans = trans.float().to(self.device)
        affs = affs.float().to(self.device)
        return data, poses, quat, trans, affs

    def load_best_model(self, ):
        model_name, self.best_loss_epoch, self.best_loss =\
            get_best_epoch_and_loss(self.args.work_dir)
        best_model_found = False
        try:
            loaded_vars = torch.load(os.path.join(self.args.work_dir, model_name))
            self.model.load_state_dict(loaded_vars['model_dict'])
            self.orient_h = loaded_vars['orient_h']
            self.quat_h = loaded_vars['quat_h']
            best_model_found = True
        except (FileNotFoundError, IsADirectoryError):
            print('No saved model found.')
        return best_model_found

    def adjust_lr(self):
        self.lr = self.lr * self.args.lr_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def adjust_tf(self):
        if self.meta_info['epoch'] > 20:
            self.tf = self.tf * self.args.tf_decay

    def show_epoch_info(self):

        print_epochs = [self.best_loss_epoch if self.best_loss_epoch is not None else 0]
        best_metrics = [self.best_loss]
        i = 0
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}. Best so far: {} (epoch: {:d}).'.
                              format(k, v, best_metrics[i], print_epochs[i]))
            i += 1
        if self.args.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):

        if self.meta_info['iter'] % self.args.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.args.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def yield_batch(self, batch_size, dataset):
        batch_pos = np.zeros((batch_size, self.T, self.V, self.C), dtype='float32')
        batch_quat = np.zeros((batch_size, self.T, (self.V - 1) * self.D), dtype='float32')
        batch_orient = np.zeros((batch_size, self.T, self.O), dtype='float32')
        batch_z_mean = np.zeros((batch_size, self.Z), dtype='float32')
        batch_z_dev = np.zeros((batch_size, self.T, self.Z), dtype='float32')
        batch_root_speed = np.zeros((batch_size, self.T, self.RS), dtype='float32')
        batch_affs = np.zeros((batch_size, self.T, self.A), dtype='float32')
        batch_spline = np.zeros((batch_size, self.T, self.S), dtype='float32')
        batch_labels = np.zeros((batch_size, 1, self.num_labels[0]), dtype='float32')
        pseudo_passes = (len(dataset) + batch_size - 1) // batch_size

        probs = []
        for k in dataset.keys():
            if 'spline' not in dataset[k]:
                raise KeyError('No splines found. Perhaps you forgot to compute them?')
            probs.append(dataset[k]['spline'].size())
        probs = np.array(probs) / np.sum(probs)

        for p in range(pseudo_passes):
            rand_keys = np.random.choice(len(dataset), size=batch_size, replace=True, p=probs)
            for i, k in enumerate(rand_keys):
                pos = dataset[str(k)]['positions'][:self.T]
                quat = dataset[str(k)]['rotations'][:self.T, 1:]
                orient = dataset[str(k)]['rotations'][:self.T, 0]
                affs = dataset[str(k)]['affective_features'][:self.T]
                spline, phase = Spline.extract_spline_features(dataset[str(k)]['spline'])
                spline = spline[:self.T]
                phase = phase[:self.T]
                z = dataset[str(k)]['trans_and_controls'][:, 1][:self.T]
                z_mean = np.mean(z[:self.prefix_length])
                z_dev = z - z_mean
                root_speed = dataset[str(k)]['trans_and_controls'][:, -1][:self.T]
                labels = dataset[str(k)]['labels'][:self.num_labels[0]]

                batch_pos[i] = pos
                batch_quat[i] = quat.reshape(self.T, -1)
                batch_orient[i] = orient.reshape(self.T, -1)
                batch_z_mean[i] = z_mean.reshape(-1, 1)
                batch_z_dev[i] = z_dev.reshape(self.T, -1)
                batch_root_speed[i] = root_speed.reshape(self.T, 1)
                batch_affs[i] = affs
                batch_spline[i] = spline
                batch_labels[i] = np.expand_dims(labels, axis=0)
            yield batch_pos, batch_quat, batch_orient, batch_z_mean, batch_z_dev,\
                  batch_root_speed, batch_affs, batch_spline, batch_labels

    def return_batch(self, batch_size, dataset, randomized=True):
        if len(batch_size) > 1:
            rand_keys = np.copy(batch_size)
            batch_size = len(batch_size)
        else:
            batch_size = batch_size[0]
            probs = []
            for k in dataset.keys():
                if 'spline' not in dataset[k]:
                    raise KeyError('No splines found. Perhaps you forgot to compute them?')
                probs.append(dataset[k]['spline'].size())
            probs = np.array(probs) / np.sum(probs)
            if randomized:
                rand_keys = np.random.choice(len(dataset), size=batch_size, replace=False, p=probs)
            else:
                rand_keys = np.arange(batch_size)

        batch_pos = np.zeros((batch_size, self.T, self.V, self.C), dtype='float32')
        batch_quat = np.zeros((batch_size, self.T, (self.V - 1) * self.D), dtype='float32')
        batch_orient = np.zeros((batch_size, self.T, self.O), dtype='float32')
        batch_z_mean = np.zeros((batch_size, self.Z), dtype='float32')
        batch_z_dev = np.zeros((batch_size, self.T, self.Z), dtype='float32')
        batch_root_speed = np.zeros((batch_size, self.T, self.RS), dtype='float32')
        batch_affs = np.zeros((batch_size, self.T, self.A), dtype='float32')
        batch_spline = np.zeros((batch_size, self.T, self.S), dtype='float32')
        batch_labels = np.zeros((batch_size, 1, self.num_labels[0]), dtype='float32')
        pseudo_passes = (len(dataset) + batch_size - 1) // batch_size

        for i, k in enumerate(rand_keys):
            pos = dataset[str(k)]['positions'][:self.T]
            quat = dataset[str(k)]['rotations'][:self.T, 1:]
            orient = dataset[str(k)]['rotations'][:self.T, 0]
            affs = dataset[str(k)]['affective_features'][:self.T]
            spline, phase = Spline.extract_spline_features(dataset[str(k)]['spline'])
            spline = spline[:self.T]
            phase = phase[:self.T]
            z = dataset[str(k)]['trans_and_controls'][:, 1][:self.T]
            z_mean = np.mean(z[:self.prefix_length])
            z_dev = z - z_mean
            root_speed = dataset[str(k)]['trans_and_controls'][:, -1][:self.T]
            labels = dataset[str(k)]['labels'][:self.num_labels[0]]

            batch_pos[i] = pos
            batch_quat[i] = quat.reshape(self.T, -1)
            batch_orient[i] = orient.reshape(self.T, -1)
            batch_z_mean[i] = z_mean.reshape(-1, 1)
            batch_z_dev[i] = z_dev.reshape(self.T, -1)
            batch_root_speed[i] = root_speed.reshape(self.T, 1)
            batch_affs[i] = affs
            batch_spline[i] = spline
            batch_labels[i] = np.expand_dims(labels, axis=0)

        return batch_pos, batch_quat, batch_orient, batch_z_mean, batch_z_dev,\
            batch_root_speed, batch_affs, batch_spline, batch_labels

    def per_train(self):

        self.model.train()
        train_loader = self.data_loader['train']
        batch_loss = 0.
        N = 0.

        for pos, quat, orient, z_mean, z_dev,\
                root_speed, affs, spline, labels in self.yield_batch(self.args.batch_size, train_loader):

            pos = torch.from_numpy(pos).cuda()
            orient = torch.from_numpy(orient).cuda()
            quat = torch.from_numpy(quat).cuda()
            z_mean = torch.from_numpy(z_mean).cuda()
            z_dev = torch.from_numpy(z_dev).cuda()
            root_speed = torch.from_numpy(root_speed).cuda()
            affs = torch.from_numpy(affs).cuda()
            spline = torch.from_numpy(spline).cuda()
            labels = torch.from_numpy(labels).cuda().repeat(1, quat.shape[1], 1)
            z_rs = torch.cat((z_dev, root_speed), dim=-1)
            quat_all = torch.cat((orient[:, self.prefix_length - 1:], quat[:, self.prefix_length - 1:]), dim=-1)

            pos_pred = pos.clone()
            orient_pred = orient.clone()
            quat_pred = quat.clone()
            z_rs_pred = z_rs.clone()
            affs_pred = affs.clone()
            spline_pred = spline.clone()
            pos_pred_all = pos.clone()
            orient_pred_all = orient.clone()
            quat_pred_all = quat.clone()
            z_rs_pred_all = z_rs.clone()
            affs_pred_all = affs.clone()
            spline_pred_all = spline.clone()
            orient_prenorm_terms = torch.zeros_like(orient_pred)
            quat_prenorm_terms = torch.zeros_like(quat_pred)

            # forward
            self.optimizer.zero_grad()
            for t in range(self.target_length):
                orient_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1],\
                    quat_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1],\
                    z_rs_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1],\
                    self.orient_h, self.quat_h,\
                    orient_prenorm_terms[:, self.prefix_length + t: self.prefix_length + t + 1],\
                    quat_prenorm_terms[:, self.prefix_length + t: self.prefix_length + t + 1] = \
                    self.model(
                        orient_pred[:, t:self.prefix_length + t],
                        quat_pred[:, t:self.prefix_length + t],
                        z_rs_pred[:, t:self.prefix_length + t],
                        affs_pred[:, t:self.prefix_length + t],
                        spline_pred[:, t:self.prefix_length + t],
                        labels[:, t:self.prefix_length + t],
                        orient_h=None if t == 0 else self.orient_h,
                        quat_h=None if t == 0 else self.quat_h, return_prenorm=True)
                pos_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1],\
                    affs_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1], \
                    spline_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1] = \
                    self.mocap.get_predicted_features(
                        pos_pred[:, :self.prefix_length + t],
                        pos_pred[:, self.prefix_length + t:self.prefix_length + t + 1, 0, [0, 2]],
                        z_rs_pred[:, self.prefix_length + t:self.prefix_length + t + 1, 0] + z_mean,
                        orient_pred[:, self.prefix_length + t:self.prefix_length + t + 1],
                        quat_pred[:, self.prefix_length + t:self.prefix_length + t + 1])
                if np.random.uniform(size=1)[0] > self.tf:
                    pos_pred[:, self.prefix_length + t:self.prefix_length + t + 1] = \
                        pos_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1].clone()
                    orient_pred[:, self.prefix_length + t:self.prefix_length + t + 1] = \
                        orient_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1].clone()
                    quat_pred[:, self.prefix_length + t:self.prefix_length + t + 1] = \
                        quat_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1].clone()
                    z_rs_pred[:, self.prefix_length + t:self.prefix_length + t + 1] = \
                        z_rs_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1].clone()
                    affs_pred[:, self.prefix_length + t:self.prefix_length + t + 1] = \
                        affs_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1].clone()
                    spline_pred[:, self.prefix_length + t:self.prefix_length + t + 1] = \
                        spline_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1].clone()

            prenorm_terms = torch.cat((orient_prenorm_terms, quat_prenorm_terms), dim=-1)
            prenorm_terms = prenorm_terms.view(prenorm_terms.shape[0], prenorm_terms.shape[1], -1, self.D)
            quat_norm_loss = self.args.quat_norm_reg * torch.mean((torch.sum(prenorm_terms ** 2, dim=-1) - 1) ** 2)

            quat_loss, quat_derv_loss = losses.quat_angle_loss(
                torch.cat((orient_pred_all[:, self.prefix_length - 1:],
                           quat_pred_all[:, self.prefix_length - 1:]), dim=-1),
                quat_all, self.V, self.D)
            quat_loss *= self.args.quat_reg

            z_rs_loss = self.z_rs_loss_func(z_rs_pred_all[:, self.prefix_length:],
                                            z_rs[:, self.prefix_length:])
            affs_loss = self.affs_loss_func(affs_pred_all[:, self.prefix_length:],
                                            affs[:, self.prefix_length:])
            fs_loss = losses.foot_speed_loss(pos_pred, pos)
            loss_total = quat_norm_loss + quat_loss + quat_derv_loss + z_rs_loss + affs_loss + fs_loss
            loss_total.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
            self.optimizer.step()

            # animation_pred = {
            #     'joint_names': self.joint_names,
            #     'joint_offsets': torch.from_numpy(self.joint_offsets[1:]).
            #         float().unsqueeze(0).repeat(pos_pred_all.shape[0], 1, 1),
            #     'joint_parents': self.joint_parents,
            #     'positions': pos_pred_all,
            #     'rotations': torch.cat((orient_pred_all, quat_pred_all), dim=-1)
            # }
            # MocapDataset.save_as_bvh(animation_pred,
            #                          dataset_name=self.dataset,
            #                          subset_name='test')

            # Compute statistics
            batch_loss += loss_total.item()
            N += quat.shape[0]

            # statistics
            self.iter_info['loss'] = loss_total.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.iter_info['tf'] = '{:.6f}'.format(self.tf)
            self.show_iter_info()
            self.meta_info['iter'] += 1

        batch_loss = batch_loss / N
        self.epoch_info['mean_loss'] = batch_loss
        self.show_epoch_info()
        self.io.print_timer()
        self.adjust_lr()
        self.adjust_tf()

    def per_test(self):

        self.model.eval()
        test_loader = self.data_loader['test']
        valid_loss = 0.
        N = 0.

        for pos, quat, orient, z_mean, z_dev,\
                root_speed, affs, spline, labels in self.yield_batch(self.args.batch_size, test_loader):
            with torch.no_grad():
                pos = torch.from_numpy(pos).cuda()
                orient = torch.from_numpy(orient).cuda()
                quat = torch.from_numpy(quat).cuda()
                z_mean = torch.from_numpy(z_mean).cuda()
                z_dev = torch.from_numpy(z_dev).cuda()
                root_speed = torch.from_numpy(root_speed).cuda()
                affs = torch.from_numpy(affs).cuda()
                spline = torch.from_numpy(spline).cuda()
                labels = torch.from_numpy(labels).cuda().repeat(1, quat.shape[1], 1)
                z_rs = torch.cat((z_dev, root_speed), dim=-1)
                quat_all = torch.cat((orient[:, self.prefix_length - 1:], quat[:, self.prefix_length - 1:]), dim=-1)

                pos_pred = pos.clone()
                orient_pred = orient.clone()
                quat_pred = quat.clone()
                z_rs_pred = z_rs.clone()
                affs_pred = affs.clone()
                spline_pred = spline.clone()
                orient_prenorm_terms = torch.zeros_like(orient_pred)
                quat_prenorm_terms = torch.zeros_like(quat_pred)

                # forward
                for t in range(self.target_length):
                    orient_pred[:, self.prefix_length + t:self.prefix_length + t + 1],\
                        quat_pred[:, self.prefix_length + t:self.prefix_length + t + 1],\
                        z_rs_pred[:, self.prefix_length + t:self.prefix_length + t + 1],\
                        self.orient_h, self.quat_h,\
                        orient_prenorm_terms[:, self.prefix_length + t: self.prefix_length + t + 1],\
                        quat_prenorm_terms[:, self.prefix_length + t: self.prefix_length + t + 1] = \
                        self.model(
                            orient_pred[:, t:self.prefix_length + t],
                            quat_pred[:, t:self.prefix_length + t],
                            z_rs_pred[:, t:self.prefix_length + t],
                            affs_pred[:, t:self.prefix_length + t],
                            spline[:, t:self.prefix_length + t],
                            labels[:, t:self.prefix_length + t],
                            orient_h=None if t == 0 else self.orient_h,
                            quat_h=None if t == 0 else self.quat_h, return_prenorm=True)
                    pos_pred[:, self.prefix_length + t:self.prefix_length + t + 1], \
                        affs_pred[:, self.prefix_length + t:self.prefix_length + t + 1],\
                        spline_pred[:, self.prefix_length + t:self.prefix_length + t + 1] = \
                        self.mocap.get_predicted_features(
                            pos_pred[:, :self.prefix_length + t],
                            pos_pred[:, self.prefix_length + t:self.prefix_length + t + 1, 0, [0, 2]],
                            z_rs_pred[:, self.prefix_length + t:self.prefix_length + t + 1, 0] + z_mean,
                            orient_pred[:, self.prefix_length + t:self.prefix_length + t + 1],
                            quat_pred[:, self.prefix_length + t:self.prefix_length + t + 1])

                prenorm_terms = torch.cat((orient_prenorm_terms, quat_prenorm_terms), dim=-1)
                prenorm_terms = prenorm_terms.view(prenorm_terms.shape[0], prenorm_terms.shape[1], -1, self.D)
                quat_norm_loss = self.args.quat_norm_reg *\
                    torch.mean((torch.sum(prenorm_terms ** 2, dim=-1) - 1) ** 2)

                quat_loss, quat_derv_loss = losses.quat_angle_loss(
                    torch.cat((orient_pred[:, self.prefix_length - 1:],
                               quat_pred[:, self.prefix_length - 1:]), dim=-1),
                    quat_all, self.V, self.D)
                quat_loss *= self.args.quat_reg

                recons_loss = self.args.recons_reg *\
                              (pos_pred[:, self.prefix_length:] - pos_pred[:, self.prefix_length:, 0:1] -
                               pos[:, self.prefix_length:] + pos[:, self.prefix_length:, 0:1]).norm()
                valid_loss += recons_loss
                N += quat.shape[0]

        valid_loss /= N
        # if self.meta_info['epoch'] > 5 and self.loss_updated:
        #     pos_pred_np = pos_pred.contiguous().view(pos_pred.shape[0], pos_pred.shape[1], -1).permute(0, 2, 1).\
        #         detach().cpu().numpy()
        #     display_animations(pos_pred_np, self.V, self.C, self.mocap.joint_parents, save=True,
        #                        dataset_name=self.dataset, subset_name='epoch_' + str(self.best_loss_epoch),
        #                        overwrite=True)
        #     pos_in_np = pos_in.contiguous().view(pos_in.shape[0], pos_in.shape[1], -1).permute(0, 2, 1).\
        #         detach().cpu().numpy()
        #     display_animations(pos_in_np, self.V, self.C, self.mocap.joint_parents, save=True,
        #                        dataset_name=self.dataset, subset_name='epoch_' + str(self.best_loss_epoch) +
        #                                                               '_gt',
        #                        overwrite=True)

        self.epoch_info['mean_loss'] = valid_loss
        if self.epoch_info['mean_loss'] < self.best_loss and self.meta_info['epoch'] > self.min_train_epochs:
            self.best_loss = self.epoch_info['mean_loss']
            self.best_loss_epoch = self.meta_info['epoch']
            self.loss_updated = True
        else:
            self.loss_updated = False
        self.show_epoch_info()

    def train(self):

        if self.args.load_last_best:
            best_model_found = self.load_best_model()
            self.args.start_epoch = self.best_loss_epoch if best_model_found else 0
        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            self.meta_info['epoch'] = epoch

            # training
            self.io.print_log('Training epoch: {}'.format(epoch))
            self.per_train()
            self.io.print_log('Done.')

            # evaluation
            if (epoch % self.args.eval_interval == 0) or (
                    epoch + 1 == self.args.num_epoch):
                self.io.print_log('Eval epoch: {}'.format(epoch))
                self.per_test()
                self.io.print_log('Done.')

            # save model and weights
            if self.loss_updated:
                torch.save({'model_dict': self.model.state_dict(),
                            'orient_h': self.orient_h,
                            'quat_h': self.quat_h},
                           os.path.join(self.args.work_dir, 'epoch_{}_loss_{:.4f}_model.pth.tar'.
                                        format(epoch, self.best_loss)))

                if self.generate_while_train:
                    self.generate_motion(load_saved_model=False, samples_to_generate=1)

    def copy_prefix(self, var, prefix_length=None):
        if prefix_length is None:
            prefix_length = self.prefix_length
        return [var[s, :prefix_length].unsqueeze(0) for s in range(var.shape[0])]

    def generate_linear_trajectory(self, traj, alpha=0.001):
        traj_markers = (traj[:, self.prefix_length - 2] +
                        (traj[:, self.prefix_length - 1] - traj[:, self.prefix_length - 2]) / alpha).unsqueeze(1)
        return traj_markers

    def generate_circular_trajectory(self, traj, alpha=5., num_segments=10):
        last_segment = alpha * traj[:, self.prefix_length - 1:self.prefix_length] -\
                       traj[:, self.prefix_length - 2:self.prefix_length - 1]
        last_marker = traj[:, self.prefix_length - 1:self.prefix_length]
        traj_markers = last_marker.clone()
        angle_per_segment = 2. * np.pi / num_segments
        for _ in range(num_segments):
            next_segment = qrot(expmap_to_quaternion(
                torch.tensor([0, -angle_per_segment, 0]).cuda().float().repeat(
                    last_segment.shape[0], last_segment.shape[1], 1)), torch.cat((
                last_segment[..., 0:1],
                torch.zeros_like(last_segment[..., 0:1]),
                last_segment[..., 1:]), dim=-1))[..., [0, 2]]
            next_marker = next_segment + last_marker
            traj_markers = torch.cat((traj_markers, next_marker), dim=1)
            last_segment = next_segment.clone()
            last_marker = next_marker.clone()
        traj_markers = traj_markers[:, 1:]
        return traj_markers

    def compute_next_traj_point(self, traj, traj_marker, rs_pred):
        tangent = traj_marker - traj
        tangent /= (torch.norm(tangent, dim=-1) + 1e-9)
        return tangent * rs_pred + traj

    def compute_next_traj_point_sans_markers(self, pos_last, quat_next, z_pred, rs_pred):
        # pos_next = torch.zeros_like(pos_last)
        offsets = torch.from_numpy(self.mocap.joint_offsets).cuda().float(). \
            unsqueeze(0).unsqueeze(0).repeat(pos_last.shape[0], pos_last.shape[1], 1, 1)
        pos_next = MocapDataset.forward_kinematics(quat_next.contiguous().view(quat_next.shape[0],
                                                                               quat_next.shape[1], -1, self.D),
                                                   pos_last[:, :, 0],
                                                   self.joint_parents,
                                                   torch.from_numpy(self.joint_offsets).float().cuda())
        # for joint in range(1, self.V):
        #     pos_next[:, :, joint] = qrot(quat_copy[:, :, joint - 1], offsets[:, :, joint]) \
        #                             + pos_next[:, :, self.mocap.joint_parents[joint]]
        root = pos_next[:, :, 0]
        l_shoulder = pos_next[:, :, 18]
        r_shoulder = pos_next[:, :, 25]
        facing = torch.cross(l_shoulder - root, r_shoulder - root, dim=-1)[..., [0, 2]]
        facing /= (torch.norm(facing, dim=-1)[..., None] + 1e-9)
        return rs_pred * facing + pos_last[:, :, 0, [0, 2]]

    def get_diff_from_traj(self, pos_pred, traj_pred, s):
        root = pos_pred[s][:, :, 0]
        l_shoulder = pos_pred[s][:, :, 18]
        r_shoulder = pos_pred[s][:, :, 25]
        facing = torch.cross(l_shoulder - root, r_shoulder - root, dim=-1)[..., [0, 2]]
        facing /= (torch.norm(facing, dim=-1)[..., None] + 1e-9)
        tangents = traj_pred[s][:, 1:] - traj_pred[s][:, :-1]
        tangent_norms = torch.norm(tangents, dim=-1)
        tangents /= (tangent_norms[..., None] + 1e-9)
        tangents = torch.cat((torch.zeros_like(tangents[:, 0:1]), tangents), dim=1)
        tangent_norms = torch.cat((torch.zeros_like(tangent_norms[:, 0:1]), tangent_norms), dim=1)
        axis_diff = torch.cross(torch.cat((facing[..., 0:1],
                                           torch.zeros_like(facing[..., 0:1]),
                                           facing[..., 1:]), dim=-1),
                                torch.cat((tangents[..., 0:1],
                                           torch.zeros_like(tangents[..., 0:1]),
                                           tangents[..., 1:]), dim=-1))
        axis_diff_norms = torch.norm(axis_diff, dim=-1)
        axis_diff /= (axis_diff_norms[..., None] + 1e-9)
        angle_diff = torch.acos(torch.einsum('ijk,ijk->ij', facing, tangents).clamp(min=-1., max=1.))
        angle_diff[tangent_norms < 1e-6] = 0.
        return axis_diff, angle_diff

    def rotate_gaits(self, orient_pred, quat_pred, quat_diff, head_tilt, l_shoulder_slouch, r_shoulder_slouch):
        quat_reshape = quat_pred.contiguous().view(quat_pred.shape[0], quat_pred.shape[1], -1, self.D).clone()
        quat_reshape[..., 14, :] = qmul(torch.from_numpy(head_tilt).cuda().float(),
                                        quat_reshape[..., 14, :])
        quat_reshape[..., 16, :] = qmul(torch.from_numpy(l_shoulder_slouch).cuda().float(),
                                        quat_reshape[..., 16, :])
        quat_reshape[..., 17, :] = qmul(torch.from_numpy(qinv(l_shoulder_slouch)).cuda().float(),
                                        quat_reshape[..., 17, :])
        quat_reshape[..., 23, :] = qmul(torch.from_numpy(r_shoulder_slouch).cuda().float(),
                                        quat_reshape[..., 23, :])
        quat_reshape[..., 24, :] = qmul(torch.from_numpy(qinv(r_shoulder_slouch)).cuda().float(),
                                        quat_reshape[..., 24, :])
        return qmul(quat_diff, orient_pred), quat_reshape.contiguous().view(quat_reshape.shape[0],
                                                                            quat_reshape.shape[1], -1)

    def generate_motion(self, load_saved_model=True, samples_to_generate=1534, max_steps=300, randomized=True):

        if load_saved_model:
            self.load_best_model()
        self.model.eval()
        test_loader = self.data_loader['test']

        pos, quat, orient, z_mean, z_dev, \
        root_speed, affs, spline, labels = self.return_batch([samples_to_generate], test_loader, randomized=randomized)
        pos = torch.from_numpy(pos).cuda()
        traj = pos[:, :, 0, [0, 2]].clone()
        orient = torch.from_numpy(orient).cuda()
        quat = torch.from_numpy(quat).cuda()
        z_mean = torch.from_numpy(z_mean).cuda()
        z_dev = torch.from_numpy(z_dev).cuda()
        root_speed = torch.from_numpy(root_speed).cuda()
        affs = torch.from_numpy(affs).cuda()
        spline = torch.from_numpy(spline).cuda()
        z_rs = torch.cat((z_dev, root_speed), dim=-1)
        quat_all = torch.cat((orient[:, self.prefix_length - 1:], quat[:, self.prefix_length - 1:]), dim=-1)
        labels = np.tile(labels, (1, max_steps + self.prefix_length, 1))

        # Begin for transition
        # traj[:, self.prefix_length - 2] = torch.tensor([-0.208, 4.8]).cuda().float()
        # traj[:, self.prefix_length - 1] = torch.tensor([-0.204, 5.1]).cuda().float()
        # final_emo_idx = int(max_steps/2)
        # labels[:, final_emo_idx:] = np.array([1., 0., 0., 0.])
        # labels[:, :final_emo_idx + 1] = np.linspace(labels[:, 0], labels[:, final_emo_idx],
        #                                             num=final_emo_idx + 1, axis=1)
        # End for transition
        labels = torch.from_numpy(labels).cuda()

        # traj_np = traj_markers.detach().cpu().numpy()
        # import matplotlib.pyplot as plt
        # plt.plot(traj_np[6, :, 0], traj_np[6, :, 1])
        # plt.show()

        happy_idx = [25, 295, 390, 667, 1196]
        sad_idx = [169, 184, 258, 948, 974]
        angry_idx = [89, 93, 96, 112, 289, 290, 978]
        neutral_idx = [72, 106, 143, 237, 532, 747, 1177]
        sample_idx = np.squeeze(np.concatenate((happy_idx, sad_idx, angry_idx, neutral_idx)))

        ## CHANGE HERE
        # scene_corners = torch.tensor([[149.862, 50.833],
        #                               [149.862, 36.81],
        #                               [161.599, 36.81],
        #                               [161.599, 50.833]]).cuda().float()
        # character_heights = torch.tensor([0.95, 0.88, 0.86, 0.90, 0.95, 0.82]).cuda().float()
        # num_characters_per_side = torch.tensor([2, 3, 2, 3]).cuda().int()
        # traj_markers, traj_offsets, character_scale =\
        #     generate_trajectories(scene_corners, z_mean, character_heights,
        #                           num_characters_per_side, traj[:, :self.prefix_length])
        # num_characters_per_side = torch.tensor([4, 0, 0, 0]).cuda().int()
        # traj_markers, traj_offsets, character_scale =\
        #     generate_simple_trajectories(scene_corners, z_mean[:4], z_mean[:4],
        #                                  num_characters_per_side, traj[sample_idx, :self.prefix_length])
        # traj_markers, traj_offsets, character_scale =\
        #     generate_rvo_trajectories(scene_corners, z_mean[:4], z_mean[:4],
        #                               num_characters_per_side, traj[sample_idx, :self.prefix_length])

        # traj[sample_idx, :self.prefix_length] += traj_offsets
        # pos_sampled = pos[sample_idx].clone()
        # pos_sampled[:, :self.prefix_length, :, [0, 2]] += traj_offsets.unsqueeze(2).repeat(1, 1, self.V, 1)
        # pos[sample_idx] = pos_sampled
        # traj_markers = self.generate_linear_trajectory(traj)

        pos_pred = self.copy_prefix(pos)
        traj_pred = self.copy_prefix(traj)
        orient_pred = self.copy_prefix(orient)
        quat_pred = self.copy_prefix(quat)
        z_rs_pred = self.copy_prefix(z_rs)
        affs_pred = self.copy_prefix(affs)
        spline_pred = self.copy_prefix(spline)
        labels_pred = self.copy_prefix(labels, prefix_length=max_steps + self.prefix_length)

        # forward
        elapsed_time = np.zeros(len(sample_idx))
        for counter, s in enumerate(sample_idx):  # range(samples_to_generate):
            start_time = time.time()
            orient_h_copy = self.orient_h.clone()
            quat_h_copy = self.quat_h.clone()
            ## CHANGE HERE
            num_markers = max_steps + self.prefix_length + 1
            # num_markers = traj_markers[s].shape[0]
            marker_idx = 0
            t = -1
            with torch.no_grad():
                while marker_idx < num_markers:
                    t += 1
                    if t > max_steps:
                        print('Sample: {}. Did not reach end in {} steps.'.format(s, max_steps), end='')
                        break
                    pos_pred[s] = torch.cat((pos_pred[s], torch.zeros_like(pos_pred[s][:, -1:])), dim=1)
                    traj_pred[s] = torch.cat((traj_pred[s], torch.zeros_like(traj_pred[s][:, -1:])), dim=1)
                    orient_pred[s] = torch.cat((orient_pred[s], torch.zeros_like(orient_pred[s][:, -1:])), dim=1)
                    quat_pred[s] = torch.cat((quat_pred[s], torch.zeros_like(quat_pred[s][:, -1:])), dim=1)
                    z_rs_pred[s] = torch.cat((z_rs_pred[s], torch.zeros_like(z_rs_pred[s][:, -1:])), dim=1)
                    affs_pred[s] = torch.cat((affs_pred[s], torch.zeros_like(affs_pred[s][:, -1:])), dim=1)
                    spline_pred[s] = torch.cat((spline_pred[s], torch.zeros_like(spline_pred[s][:, -1:])), dim=1)

                    orient_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1], \
                    quat_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1], \
                    z_rs_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1], \
                    orient_h_copy, quat_h_copy = \
                        self.model(
                            orient_pred[s][:, t:self.prefix_length + t],
                            quat_pred[s][:, t:self.prefix_length + t],
                            z_rs_pred[s][:, t:self.prefix_length + t],
                            affs_pred[s][:, t:self.prefix_length + t],
                            spline_pred[s][:, t:self.prefix_length + t],
                            labels_pred[s][:, t:self.prefix_length + t],
                            orient_h=None if t == 0 else orient_h_copy,
                            quat_h=None if t == 0 else quat_h_copy, return_prenorm=False)

                    traj_curr = traj_pred[s][:, self.prefix_length + t - 1:self.prefix_length + t].clone()
                    # root_speed = torch.norm(
                    #     pos_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1, 0] - \
                    #     pos_pred[s][:, self.prefix_length + t - 1:self.prefix_length + t, 0], dim=-1)

                    ## CHANGE HERE
                    # traj_next = \
                    #     self.compute_next_traj_point(
                    #         traj_curr,
                    #         traj_markers[s, marker_idx],
                    #         o_z_rs_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1, 2])
                    try:
                        traj_next = traj[s, self.prefix_length + t]
                    except IndexError:
                        traj_next = \
                            self.compute_next_traj_point_sans_markers(
                                pos_pred[s][:, self.prefix_length + t - 1:self.prefix_length + t],
                                quat_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1],
                                z_rs_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1, 0],
                                z_rs_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1, 1])

                    pos_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1], \
                    affs_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1], \
                    spline_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1] = \
                        self.mocap.get_predicted_features(
                            pos_pred[s][:, :self.prefix_length + t],
                            traj_next,
                            z_rs_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1, 0] + z_mean[s:s + 1],
                            orient_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1],
                            quat_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1])

                    # min_speed_pred = torch.min(torch.cat((lf_speed_pred.unsqueeze(-1),
                    #                                        rf_speed_pred.unsqueeze(-1)), dim=-1), dim=-1)[0]
                    # if min_speed_pred - diff_speeds_mean[s] - diff_speeds_std[s] < 0.:
                    #     root_speed_pred = 0.
                    # else:
                    #     root_speed_pred = o_z_rs_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1, 2]
                    #

                    ## CHANGE HERE
                    # traj_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1] = \
                    #     self.compute_next_traj_point(
                    #         traj_curr,
                    #         traj_markers[s, marker_idx],
                    #         root_speed_pred)
                    # if torch.norm(traj_next - traj_curr, dim=-1).squeeze() >= \
                    #         torch.norm(traj_markers[s, marker_idx] - traj_curr, dim=-1).squeeze():
                    #     marker_idx += 1
                    traj_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1] = traj_next
                    marker_idx += 1
                    pos_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1], \
                    affs_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1], \
                    spline_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1] = \
                        self.mocap.get_predicted_features(
                            pos_pred[s][:, :self.prefix_length + t],
                            pos_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1, 0, [0, 2]],
                            z_rs_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1, 0] + z_mean[s:s + 1],
                            orient_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1],
                            quat_pred[s][:, self.prefix_length + t:self.prefix_length + t + 1])
                    print('Sample: {}. Steps: {}'.format(s, t), end='\r')
            print()

            # shift = torch.zeros((1, scene_corners.shape[1] + 1)).cuda().float()
            # shift[..., [0, 2]] = scene_corners[0]
            # pos_pred[s] = (pos_pred[s] - shift) / character_scale + shift
            # pos_pred_np = pos_pred[s].contiguous().view(pos_pred[s].shape[0],
            #                                             pos_pred[s].shape[1], -1).permute(0, 2, 1).\
            #     detach().cpu().numpy()
            # display_animations(pos_pred_np, self.V, self.C, self.mocap.joint_parents, save=True,
            #                    dataset_name=self.dataset, subset_name='epoch_' + str(self.best_loss_epoch),
            #                    save_file_names=[str(s).zfill(6)],
            #                    overwrite=True)

            # plt.cla()
            # fig, (ax1, ax2) = plt.subplots(2, 1)
            # ax1.plot(root_speeds[s])
            # ax1.plot(lf_speeds[s])
            # ax1.plot(rf_speeds[s])
            # ax1.plot(min_speeds[s] - root_speeds[s])
            # ax1.legend(['root', 'left', 'right', 'diff'])
            # ax2.plot(root_speeds_pred)
            # ax2.plot(lf_speeds_pred)
            # ax2.plot(rf_speeds_pred)
            # ax2.plot(min_speeds_pred - root_speeds_pred)
            # ax2.legend(['root', 'left', 'right', 'diff'])
            # plt.show()

            head_tilt = np.tile(np.array([0., 0., 0.]), (1, quat_pred[s].shape[1], 1))
            l_shoulder_slouch = np.tile(np.array([0., 0., 0.]), (1, quat_pred[s].shape[1], 1))
            r_shoulder_slouch = np.tile(np.array([0., 0., 0.]), (1, quat_pred[s].shape[1], 1))
            change_steps = self.prefix_length
            # Begin for transition from sad
            # if s in sad_idx:
            #     head_tilt[:, :self.T, 0] = np.deg2rad(60.)
            #     head_tilt[:, self.T:self.T + change_steps, 0] = \
            #         np.deg2rad(np.linspace(60., 0., change_steps))
            #     l_shoulder_slouch[:, :self.T, 2] = np.deg2rad(-45.)
            #     l_shoulder_slouch[:, self.T:self.T + change_steps, 2] = \
            #         np.deg2rad(np.linspace(-45., 0., change_steps))
            #     r_shoulder_slouch[:, :self.T, 2] = np.deg2rad(75.)
            #     r_shoulder_slouch[:, self.T:self.T + change_steps, 2] = \
            #         np.deg2rad(np.linspace(75., 0., change_steps))
            # End for transition from sad
            # Begin for transition to sad
            # if s not in sad_idx:
            #     head_tilt[:, self.T:self.T + change_steps, 0] = \
            #         np.deg2rad(np.linspace(0., 60., change_steps))
            #     head_tilt[:, self.T + change_steps:, 0] = np.deg2rad(60.)
            #     l_shoulder_slouch[:, self.T:self.T + change_steps, 2] = \
            #         np.deg2rad(np.linspace(0., -45., change_steps))
            #     l_shoulder_slouch[:, self.T + change_steps:, 2] = np.deg2rad(-45.)
            #     r_shoulder_slouch[:, self.T:self.T + change_steps, 2] = \
            #         np.deg2rad(np.linspace(0., 75., change_steps))
            #     r_shoulder_slouch[:, self.T + change_steps:, 2] = np.deg2rad(75.)
            # End for transition to sad
            # Begin for maintaining sad
            if s in sad_idx:
                head_tilt[..., 0] = np.deg2rad(60.)
                l_shoulder_slouch[..., 2] = np.deg2rad(-45.)
                r_shoulder_slouch[..., 2] = np.deg2rad(75.)
            # End for maintaining sad
            head_tilt = Quaternions.from_euler(head_tilt, order='xyz').qs
            l_shoulder_slouch = Quaternions.from_euler(l_shoulder_slouch, order='xyz').qs
            r_shoulder_slouch = Quaternions.from_euler(r_shoulder_slouch, order='xyz').qs

            # Begin for aligning facing direction to trajectory
            axis_diff, angle_diff = self.get_diff_from_traj(pos_pred, traj_pred, s)
            angle_thres = 0.3
            # angle_thres = torch.max(angle_diff[:, 1:self.prefix_length])
            angle_diff[angle_diff <= angle_thres] = 0.
            angle_diff[:, self.prefix_length] = 0.
            # End for aligning facing direction to trajectory
            # pos_copy, quat_copy = self.rotate_gaits(pos_pred, quat_pred, quat_diff,
            #                                         head_tilt, l_shoulder_slouch, r_shoulder_slouch, s)
            # pos_pred[s] = pos_copy.clone()
            # angle_diff_intermediate = self.get_diff_from_traj(pos_pred, traj_pred, s)
            # if torch.max(angle_diff_intermediate[:, self.prefix_length:]) > np.pi / 2.:
            #     quat_diff = Quaternions.from_angle_axis(-angle_diff.cpu().numpy(), np.array([0, 1, 0])).qs
            #     pos_copy, quat_copy = self.rotate_gaits(pos_pred, quat_pred, quat_diff,
            #                                         head_tilt, l_shoulder_slouch, r_shoulder_slouch, s)
            # pos_pred[s] = pos_copy.clone()
            # axis_diff = torch.zeros_like(axis_diff)
            # axis_diff[..., 1] = 1.
            # angle_diff = torch.zeros_like(angle_diff)
            quat_diff = torch.from_numpy(Quaternions.from_angle_axis(
                angle_diff.cpu().numpy(), axis_diff.cpu().numpy()).qs).cuda().float()
            orient_pred[s], quat_pred[s] = self.rotate_gaits(orient_pred[s], quat_pred[s],
                                                             quat_diff, head_tilt,
                                                             l_shoulder_slouch, r_shoulder_slouch)

            if labels_pred[s][:, 0, 0] > 0.5:
                label_dir = 'happy'
            elif labels_pred[s][:, 0, 1] > 0.5:
                label_dir = 'sad'
            elif labels_pred[s][:, 0, 2] > 0.5:
                label_dir = 'angry'
            else:
                label_dir = 'neutral'

            ## CHANGE HERE
            # pos_pred[s] = pos_pred[s][:, self.prefix_length + 5:]
            # o_z_rs_pred[s] = o_z_rs_pred[s][:, self.prefix_length + 5:]
            # quat_pred[s] = quat_pred[s][:, self.prefix_length + 5:]

            traj_pred_np = pos_pred[s][0, :, 0].cpu().numpy()

            save_file_name = '{:06}_{:.2f}_{:.2f}_{:.2f}_{:.2f}'.format(s,
                                                                        labels_pred[s][0, 0, 0],
                                                                        labels_pred[s][0, 0, 1],
                                                                        labels_pred[s][0, 0, 2],
                                                                        labels_pred[s][0, 0, 3])

            animation_pred = {
                'joint_names': self.joint_names,
                'joint_offsets': torch.from_numpy(self.joint_offsets[1:]).float().unsqueeze(0).repeat(
                    len(pos_pred), 1, 1),
                'joint_parents': self.joint_parents,
                'positions': pos_pred[s],
                'rotations': torch.cat((orient_pred[s], quat_pred[s]), dim=-1)
            }
            self.mocap.save_as_bvh(animation_pred,
                                   dataset_name=self.dataset,
                                   # subset_name='epoch_' + str(self.best_loss_epoch),
                                   # save_file_names=[str(s).zfill(6)])
                                   subset_name=os.path.join('no_spline_epoch_' + str(self.best_loss_epoch),
                                                            str(counter).zfill(2) + '_' + label_dir),
                                   save_file_names=['root'])
            end_time = time.time()
            elapsed_time[counter] = end_time - start_time
            print('Elapsed Time: {}'.format(elapsed_time[counter]))

            # display_animations(pos_pred_np, self.V, self.C, self.mocap.joint_parents, save=True,
            #                    dataset_name=self.dataset,
            #                    # subset_name='epoch_' + str(self.best_loss_epoch),
            #                    # save_file_names=[str(s).zfill(6)],
            #                    subset_name=os.path.join('epoch_' + str(self.best_loss_epoch), label_dir),
            #                    save_file_names=[save_file_name],
            #                    overwrite=True)
        print('Mean Elapsed Time: {}'.format(np.mean(elapsed_time)))
