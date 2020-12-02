import math
import os
import torch.optim as optim
import torch.nn as nn
from net import quater_emonet

from torchlight.torchlight.io import IO
from utils.mocap_dataset import MocapDataset
from utils.visualizations import display_animations
from utils import losses
from utils.Quaternions_torch import *
from utils.spline import Spline

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
        return 0, np.inf
    loss_list = -1. * np.ones(len(all_models))
    acc_list = -1. * np.ones(len(all_models))
    for i, model in enumerate(all_models):
        loss_acc_val = str.split(model, '_')
        if len(loss_acc_val) > 1:
            loss_list[i] = float(loss_acc_val[3])
            acc_list[i] = float(loss_acc_val[5])
    if len(loss_list) < 3:
        best_model = all_models[np.argwhere(loss_list == min([n for n in loss_list if n > 0]))[0, 0]]
    else:
        loss_idx = np.argpartition(loss_list, 2)
        best_model = all_models[loss_idx[1]]
    all_underscores = list(find_all_substr(best_model, '_'))
    # return model name, best loss, best acc
    return best_model, int(best_model[all_underscores[0] + 1:all_underscores[1]]),\
           float(best_model[all_underscores[2] + 1:all_underscores[3]]),\
           float(best_model[all_underscores[4] + 1:all_underscores[5]])


class Processor(object):
    """
        Processor for gait generation
    """

    def __init__(self, args, dataset, data_loader, T, V, C, D, A, S,
                 joint_parents, num_labels, prefix_length, target_length,
                 min_train_epochs=-1, generate_while_train=False,
                 save_path=None, device='cuda:0'):

        self.args = args
        self.dataset = dataset
        self.mocap = MocapDataset(V, C, joint_parents)
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
        self.O = 1
        self.PRS = 2
        self.prefix_length = prefix_length
        self.target_length = target_length
        self.joint_parents = joint_parents
        self.model = quater_emonet.QuaterEmoNet(V, D, S, A, self.O, num_labels[0], self.PRS)
        self.model.cuda(device)
        self.quat_h = None
        self.p_rs_loss_func = nn.L1Loss()
        self.affs_loss_func = nn.L1Loss()
        self.best_loss = math.inf
        self.best_mean_ap = 0.
        self.loss_updated = False
        self.mean_ap_updated = False
        self.step_epochs = [math.ceil(float(self.args.num_epoch * x)) for x in self.args.step]
        self.best_loss_epoch = None
        self.best_acc_epoch = None
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
        if self.best_loss_epoch is None:
            model_name, self.best_loss_epoch, self.best_loss, self.best_mean_ap =\
                get_best_epoch_and_loss(self.args.work_dir)
            # load model
            # if self.best_loss_epoch > 0:
        loaded_vars = torch.load(os.path.join(self.args.work_dir, model_name))
        self.model.load_state_dict(loaded_vars['model_dict'])
        self.quat_h = loaded_vars['quat_h']

    def adjust_lr(self):
        self.lr = self.lr * self.args.lr_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def adjust_tf(self):
        if self.meta_info['epoch'] > 20:
            self.tf = self.tf * self.args.tf_decay

    def show_epoch_info(self):

        print_epochs = [self.best_loss_epoch if self.best_loss_epoch is not None else 0,
                        self.best_acc_epoch if self.best_acc_epoch is not None else 0,
                        self.best_acc_epoch if self.best_acc_epoch is not None else 0]
        best_metrics = [self.best_loss, 0, self.best_mean_ap]
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
        batch_affs = np.zeros((batch_size, self.T, self.A), dtype='float32')
        batch_spline = np.zeros((batch_size, self.T, self.S), dtype='float32')
        batch_phase_and_root_speed = np.zeros((batch_size, self.T, self.PRS), dtype='float32')
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
                pos = dataset[str(k)]['positions_world']
                quat = dataset[str(k)]['rotations']
                orient = dataset[str(k)]['orientations']
                affs = dataset[str(k)]['affective_features']
                spline, phase = Spline.extract_spline_features(dataset[str(k)]['spline'])
                root_speed = dataset[str(k)]['trans_and_controls'][:, -1].reshape(-1, 1)
                labels = dataset[str(k)]['labels'][:self.num_labels[0]]

                batch_pos[i] = pos
                batch_quat[i] = quat.reshape(self.T, -1)
                batch_orient[i] = orient.reshape(self.T, -1)
                batch_affs[i] = affs
                batch_spline[i] = spline
                batch_phase_and_root_speed[i] = np.concatenate((phase, root_speed), axis=-1)
                batch_labels[i] = np.expand_dims(labels, axis=0)
            yield batch_pos, batch_quat, batch_orient, batch_affs, batch_spline,\
                  batch_phase_and_root_speed / np.pi, batch_labels

    def return_batch(self, batch_size, dataset):
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
            rand_keys = np.random.choice(len(dataset), size=batch_size, replace=False, p=probs)

        batch_pos = np.zeros((batch_size, self.T, self.V, self.C), dtype='float32')
        batch_traj = np.zeros((batch_size, self.T, self.C), dtype='float32')
        batch_quat = np.zeros((batch_size, self.T, (self.V - 1) * self.D), dtype='float32')
        batch_orient = np.zeros((batch_size, self.T, self.O), dtype='float32')
        batch_affs = np.zeros((batch_size, self.T, self.A), dtype='float32')
        batch_spline = np.zeros((batch_size, self.T, self.S), dtype='float32')
        batch_phase_and_root_speed = np.zeros((batch_size, self.T, self.PRS), dtype='float32')
        batch_labels = np.zeros((batch_size, 1, self.num_labels[0]), dtype='float32')

        for i, k in enumerate(rand_keys):
            pos = dataset[str(k)]['positions_world']
            traj = dataset[str(k)]['trajectory']
            quat = dataset[str(k)]['rotations']
            orient = dataset[str(k)]['orientations']
            affs = dataset[str(k)]['affective_features']
            spline, phase = Spline.extract_spline_features(dataset[str(k)]['spline'])
            root_speed = dataset[str(k)]['trans_and_controls'][:, -1].reshape(-1, 1)
            labels = dataset[str(k)]['labels'][:self.num_labels[0]]

            batch_pos[i] = pos
            batch_traj[i] = traj
            batch_quat[i] = quat.reshape(self.T, -1)
            batch_orient[i] = orient.reshape(self.T, -1)
            batch_affs[i] = affs
            batch_spline[i] = spline
            batch_phase_and_root_speed[i] = np.concatenate((phase, root_speed), axis=-1)
            batch_labels[i] = np.expand_dims(labels, axis=0)

        return batch_pos, batch_traj, batch_quat, batch_orient, batch_affs, batch_spline,\
               batch_phase_and_root_speed, batch_labels

    def per_train(self):

        self.model.train()
        train_loader = self.data_loader['train']
        batch_loss = 0.
        N = 0.

        for pos, quat, orient, affs, spline, p_rs, labels in self.yield_batch(self.args.batch_size, train_loader):

            pos = torch.from_numpy(pos).cuda()
            quat = torch.from_numpy(quat).cuda()
            orient = torch.from_numpy(orient).cuda()
            affs = torch.from_numpy(affs).cuda()
            spline = torch.from_numpy(spline).cuda()
            p_rs = torch.from_numpy(p_rs).cuda()
            labels = torch.from_numpy(labels).cuda()

            pos_pred = pos.clone()
            quat_pred = quat.clone()
            p_rs_pred = p_rs.clone()
            affs_pred = affs.clone()
            pos_pred_all = pos.clone()
            quat_pred_all = quat.clone()
            p_rs_pred_all = p_rs.clone()
            affs_pred_all = affs.clone()
            prenorm_terms = torch.zeros_like(quat_pred)

            # forward
            self.optimizer.zero_grad()
            for t in range(self.target_length):
                quat_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1], \
                    p_rs_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1], \
                    self.quat_h, prenorm_terms[:, self.prefix_length + t: self.prefix_length + t + 1] = \
                    self.model(
                        quat_pred[:, t:self.prefix_length + t],
                        p_rs_pred[:, t:self.prefix_length + t],
                        affs_pred[:, t:self.prefix_length + t],
                        spline[:, t:self.prefix_length + t],
                        orient[:, t:self.prefix_length + t],
                        labels,
                        quat_h=None if t == 0 else self.quat_h, return_prenorm=True)
                pos_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1],\
                    affs_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1] = \
                    self.mocap.get_predicted_features(
                        pos_pred[:, self.prefix_length + t:self.prefix_length + t + 1, 0],
                        quat_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1],
                        orient[:, self.prefix_length + t:self.prefix_length + t + 1])
                if np.random.uniform(size=1)[0] > self.tf:
                    pos_pred[:, self.prefix_length + t:self.prefix_length + t + 1] = \
                        pos_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1]
                    quat_pred[:, self.prefix_length + t:self.prefix_length + t + 1] = \
                        quat_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1]
                    p_rs_pred[:, self.prefix_length + t:self.prefix_length + t + 1] = \
                        p_rs_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1]
                    affs_pred[:, self.prefix_length + t:self.prefix_length + t + 1] = \
                        affs_pred_all[:, self.prefix_length + t:self.prefix_length + t + 1]

            prenorm_terms = prenorm_terms.view(prenorm_terms.shape[0], prenorm_terms.shape[1], -1, self.D)
            quat_norm_loss = self.args.quat_norm_reg * torch.mean((torch.sum(prenorm_terms ** 2, dim=-1) - 1) ** 2)

            quat_loss, quat_derv_loss = losses.quat_angle_loss(quat_pred_all[:, self.prefix_length - 1:],
                                                               quat[:, self.prefix_length - 1:], self.V, self.D)
            quat_loss *= self.args.quat_reg

            p_rs_loss = self.p_rs_loss_func(p_rs_pred_all[:, self.prefix_length:],
                                            p_rs[:, self.prefix_length:])
            affs_loss = self.affs_loss_func(affs_pred_all[:, self.prefix_length:],
                                                affs[:, self.prefix_length:])
            # recons_loss = self.args.recons_reg *\
            #               (pos_pred_all[:, self.prefix_length:] - pos_pred_all[:, self.prefix_length:, 0:1] -
            #                 pos[:, self.prefix_length:] + pos[:, self.prefix_length:, 0:1]).norm()

            loss_total = quat_norm_loss + quat_loss + quat_derv_loss + p_rs_loss + affs_loss  # + recons_loss
            loss_total.backward()
            # nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
            self.optimizer.step()

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

        for pos, quat, orient, affs, spline, p_rs, labels in self.yield_batch(self.args.batch_size, test_loader):
            pos = torch.from_numpy(pos).cuda()
            quat = torch.from_numpy(quat).cuda()
            orient = torch.from_numpy(orient).cuda()
            affs = torch.from_numpy(affs).cuda()
            spline = torch.from_numpy(spline).cuda()
            p_rs = torch.from_numpy(p_rs).cuda()
            labels = torch.from_numpy(labels).cuda()

            pos_pred = pos.clone()
            quat_pred = quat.clone()
            p_rs_pred = p_rs.clone()
            affs_pred = affs.clone()
            prenorm_terms = torch.zeros_like(quat_pred)

            # forward
            self.optimizer.zero_grad()
            for t in range(self.target_length):
                quat_pred[:, self.prefix_length + t:self.prefix_length + t + 1], \
                    p_rs_pred[:, self.prefix_length + t:self.prefix_length + t + 1], \
                    self.quat_h, prenorm_terms[:, self.prefix_length + t: self.prefix_length + t + 1] = \
                    self.model(
                        quat_pred[:, t:self.prefix_length + t],
                        p_rs_pred[:, t:self.prefix_length + t],
                        affs_pred[:, t:self.prefix_length + t],
                        spline[:, t:self.prefix_length + t],
                        orient[:, t:self.prefix_length + t],
                        labels,
                        quat_h=None if t == 0 else self.quat_h, return_prenorm=True)
                pos_pred[:, self.prefix_length + t:self.prefix_length + t + 1], \
                affs_pred[:, self.prefix_length + t:self.prefix_length + t + 1] = \
                    self.mocap.get_predicted_features(
                        pos_pred[:, self.prefix_length + t:self.prefix_length + t + 1, 0],
                        quat_pred[:, self.prefix_length + t:self.prefix_length + t + 1],
                        orient[:, self.prefix_length + t:self.prefix_length + t + 1])

            prenorm_terms = prenorm_terms.view(prenorm_terms.shape[0], prenorm_terms.shape[1], -1, self.D)
            quat_norm_loss = self.args.quat_norm_reg * torch.mean((torch.sum(prenorm_terms ** 2, dim=-1) - 1) ** 2)

            quat_loss, quat_derv_loss = losses.quat_angle_loss(quat_pred[:, self.prefix_length - 1:],
                                                               quat[:, self.prefix_length - 1:], self.V, self.D)
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
        #     display_animations(pos_pred_np, self.V, self.C, self.joint_parents, save=True,
        #                        dataset_name=self.dataset, subset_name='epoch_' + str(self.best_loss_epoch),
        #                        overwrite=True)
        #     pos_in_np = pos_in.contiguous().view(pos_in.shape[0], pos_in.shape[1], -1).permute(0, 2, 1).\
        #         detach().cpu().numpy()
        #     display_animations(pos_in_np, self.V, self.C, self.joint_parents, save=True,
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
            self.load_best_model()
            self.args.start_epoch = self.best_loss_epoch
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
                            'quat_h': self.quat_h},
                           os.path.join(self.args.work_dir, 'epoch_{}_loss_{:.4f}_acc_{:.2f}_model.pth.tar'.
                                        format(epoch, self.best_loss, self.best_mean_ap * 100.)))

                if self.generate_while_train:
                    self.generate_motion(load_saved_model=False, samples_to_generate=1)

    def copy_prefix(self, var, target_length):
        shape = list(var.shape)
        shape[1] = self.prefix_length + target_length
        var_pred = torch.zeros(torch.Size(shape)).cuda().float()
        var_pred[:, :self.prefix_length] = var[:, :self.prefix_length]
        return var_pred

    def flip_trajectory(self, traj, target_length):
        traj_flipped = traj[:, -(target_length - self.target_length):].flip(dims=[1])
        orient_flipped = torch.zeros((traj_flipped.shape[0], traj_flipped.shape[1], 1)).cuda().float()
        # orient_flipped[:, 0] = np.pi
        # traj_diff = traj_flipped[:, 1:, [0, 2]] - traj_flipped[:, :-1, [0, 2]]
        # traj_diff /= torch.norm(traj_diff, dim=-1)[..., None]
        # orient_flipped[:, 1:, 0] = torch.atan2(traj_diff[:, :, 1], traj_diff[:, :, 0])
        return traj_flipped, orient_flipped

    def generate_motion(self, load_saved_model=True, target_length=100, samples_to_generate=10):

        if load_saved_model:
            self.load_best_model()
        self.model.eval()
        test_loader = self.data_loader['test']

        pos, traj, quat, orient, affs, spline, p_rs, labels = self.return_batch([samples_to_generate], test_loader)
        pos = torch.from_numpy(pos).cuda()
        traj = torch.from_numpy(traj).cuda()
        quat = torch.from_numpy(quat).cuda()
        orient = torch.from_numpy(orient).cuda()
        affs = torch.from_numpy(affs).cuda()
        spline = torch.from_numpy(spline).cuda()
        p_rs = torch.from_numpy(p_rs).cuda()
        labels = torch.from_numpy(labels).cuda()

        traj_flipped, orient_flipped = self.flip_trajectory(traj, target_length)
        traj = torch.cat((traj, traj_flipped), dim=1)
        orient = torch.cat((orient, orient_flipped), dim=1)

        pos_pred = self.copy_prefix(pos, target_length)
        quat_pred = self.copy_prefix(quat, target_length)
        p_rs_pred = self.copy_prefix(p_rs, target_length)
        affs_pred = self.copy_prefix(affs, target_length)
        spline_pred = self.copy_prefix(spline, target_length)

        # forward
        with torch.no_grad():
            for t in range(target_length):
                quat_pred[:, self.prefix_length + t:self.prefix_length + t + 1], \
                    p_rs_pred[:, self.prefix_length + t:self.prefix_length + t + 1], \
                    self.quat_h = \
                    self.model(
                        quat_pred[:, t:self.prefix_length + t],
                        p_rs_pred[:, t:self.prefix_length + t],
                        affs_pred[:, t:self.prefix_length + t],
                        spline_pred[:, t:self.prefix_length + t],
                        orient[:, t:self.prefix_length + t],
                        labels,
                        quat_h=None if t == 0 else self.quat_h, return_prenorm=False)
                data_pred = \
                    self.mocap.get_predicted_features(
                        pos_pred[:, :self.prefix_length + t],
                        orient[:, :self.prefix_length + t],
                        traj[:, self.prefix_length + t:self.prefix_length + t + 1],
                        quat_pred[:, self.prefix_length + t:self.prefix_length + t + 1],
                        orient[:, self.prefix_length + t:self.prefix_length + t + 1])
                pos_pred[:, self.prefix_length + t:self.prefix_length + t + 1] = data_pred['positions_world']
                affs_pred[:, self.prefix_length + t:self.prefix_length + t + 1] = data_pred['affective_features']
                spline_pred[:, self.prefix_length + t:self.prefix_length + t + 1] = data_pred['spline']


            recons_loss = self.args.recons_reg *\
                          (pos_pred[:, self.prefix_length:self.T] - pos_pred[:, self.prefix_length:self.T, 0:1] -
                           pos[:, self.prefix_length:self.T] + pos[:, self.prefix_length:self.T, 0:1]).norm()

        pos_pred_np = pos_pred.contiguous().view(pos_pred.shape[0], pos_pred.shape[1], -1).permute(0, 2, 1).\
            detach().cpu().numpy()
        pos_np = pos.contiguous().view(pos.shape[0], pos.shape[1], -1).permute(0, 2, 1).\
            detach().cpu().numpy()
        display_animations(pos_pred_np, self.V, self.C, self.joint_parents, save=True,
                           dataset_name=self.dataset, subset_name='epoch_' + str(self.best_loss_epoch),
                           overwrite=True)
        display_animations(pos_np, self.V, self.C, self.joint_parents, save=True,
                           dataset_name=self.dataset, subset_name='epoch_' + str(self.best_loss_epoch) +
                                                                  '_gt',
                           overwrite=True)
        self.mocap.save_as_bvh(traj.detach().cpu().numpy(), orient.detach().cpu().numpy(),
                               np.reshape(quat_pred.detach().cpu().numpy(),
                                          (quat_pred.shape[0], quat_pred.shape[1], -1, self.D)),
                               'render/bvh')
