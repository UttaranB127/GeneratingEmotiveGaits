import torch.nn as nn

from utils.Quaternions_torch import qeuler
from utils.common import *


def quat_angle_loss(quats_pred, quats_target, V, D):
    quats_pred = quats_pred.reshape(-1, quats_pred.shape[1], V, D)
    quats_target = quats_target.reshape(-1, quats_target.shape[1], V, D)
    euler_pred = qeuler(quats_pred.contiguous(), order='yzx', epsilon=1e-6)
    euler_target = qeuler(quats_target.contiguous(), order='yzx', epsilon=1e-6)
    # L1 loss on angle distance with 2pi wrap-around
    angle_distance = torch.remainder(euler_pred[:, 1:] - euler_target[:, 1:] + np.pi, 2 * np.pi) - np.pi
    angle_derv_distance = euler_pred[:, 1:] - euler_pred[:, :-1] - euler_target[:, 1:] + euler_target[:, :-1]
    return torch.mean(torch.abs(angle_distance)), torch.mean(torch.abs(angle_derv_distance))


def foot_speed_loss(pos_pred, pos_target):
    lf_speeds = torch.norm(pos_target[:, 1:, 4] - pos_target[:, :-1, 4], dim=-1)
    rf_speeds = torch.norm(pos_target[:, 1:, 9] - pos_target[:, :-1, 9], dim=-1)
    lt_speeds = torch.norm(pos_target[:, 1:, 5] - pos_target[:, :-1, 5], dim=-1)
    rt_speeds = torch.norm(pos_target[:, 1:, 10] - pos_target[:, :-1, 10], dim=-1)
    lf_speeds_pred = torch.norm(pos_pred[:, 1:, 4] - pos_pred[:, :-1, 4], dim=-1)[0]
    rf_speeds_pred = torch.norm(pos_pred[:, 1:, 9] - pos_pred[:, :-1, 9], dim=-1)[0]
    lt_speeds_pred = torch.norm(pos_pred[:, 1:, 5] - pos_pred[:, :-1, 5], dim=-1)[0]
    rt_speeds_pred = torch.norm(pos_pred[:, 1:, 10] - pos_pred[:, :-1, 10], dim=-1)[0]
    return torch.mean(torch.abs(lf_speeds - lf_speeds_pred)) + torch.mean(torch.abs(rf_speeds - rf_speeds_pred)) +\
        torch.mean(torch.abs(lt_speeds - lt_speeds_pred)) + torch.mean(torch.abs(rt_speeds - rt_speeds_pred))
