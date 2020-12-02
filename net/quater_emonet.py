import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

from torch.autograd import Variable
from utils.common import *
from utils.Quaternions_torch import qmul, qeuler, euler_to_quaternion

torch.manual_seed(1234)


class QuaterEmoNet(nn.Module):

    def __init__(self, V, D, S, A, O, Z, RS, L,
                 **kwargs):

        super().__init__()

        self.V = V
        self.D = D
        self.S = S
        self.A = A
        self.O = O
        self.Z = Z
        self.RS = RS
        self.L = L

        geom_fc1_size = 16
        geom_fc2_size = 16
        geom_fc3_size = 8
        self.geom_fc1 = nn.Linear(A + L, geom_fc1_size)
        self.geom_fc2 = nn.Linear(geom_fc1_size, geom_fc2_size)
        self.geom_fc3 = nn.Linear(geom_fc2_size, geom_fc3_size)

        spline_fc1_size = 8
        spline_fc2_size = 4
        self.spline_fc1 = nn.Linear(S + L, spline_fc1_size)
        self.spline_fc2 = nn.Linear(spline_fc1_size, spline_fc2_size)

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(0.05, inplace=True)
        self.elu = nn.ELU(0.05, inplace=True)

        orient_h_size = 32
        quat_h_size = 1024
        self.orient_rnn = nn.GRU(input_size=O + geom_fc3_size + spline_fc2_size + self.Z + self.RS,
                                 hidden_size=orient_h_size, num_layers=2,
                                 batch_first=True)
        self.quat_rnn = nn.GRU(input_size=(V - 1) * D + geom_fc3_size + spline_fc2_size + self.O + self.Z + self.RS,
                               hidden_size=quat_h_size, num_layers=2,
                               batch_first=True)
        self.orient_h0 = nn.Parameter(torch.zeros(self.orient_rnn.num_layers, 1, orient_h_size).normal_(std=0.01),
                                      requires_grad=True)
        self.quat_h0 = nn.Parameter(torch.zeros(self.quat_rnn.num_layers, 1, quat_h_size).normal_(std=0.01),
                                    requires_grad=True)

        orient_fc4_size = 8
        self.orient_fc4 = nn.Linear(orient_h_size, orient_fc4_size)
        self.orient_fc5 = nn.Linear(orient_fc4_size, O)
        quat_fc4_size = 128
        self.quat_fc4 = nn.Linear(quat_h_size, quat_fc4_size)
        self.quat_fc5 = nn.Linear(quat_fc4_size, (V - 1) * D)

        z_rs_fc4_size = 4
        self.z_rs_fc4 = nn.Linear(Z + RS + geom_fc3_size + spline_fc2_size, z_rs_fc4_size)
        self.z_fc5 = nn.Linear(z_rs_fc4_size, Z)
        self.rs_fc5 = nn.Linear(z_rs_fc4_size, RS)

    def forward(self, orient, quat, z_rs, affs, spline, labels,
                orient_h=None, quat_h=None,
                return_prenorm=False, return_all=False, teacher_steps=0):

        geom_controls = torch.cat((affs, labels), dim=-1)
        geom_controls = self.elu(self.geom_fc1(geom_controls))
        geom_controls = self.elu(self.geom_fc2(geom_controls))
        geom_controls = self.elu(self.geom_fc3(geom_controls))

        spline_controls = torch.cat((spline, labels), dim=-1)
        spline_controls = self.elu(self.spline_fc1(spline_controls))
        spline_controls = self.elu(self.spline_fc2(spline_controls))

        z_rs_combined = torch.cat((z_rs, geom_controls, spline_controls), dim=-1)

        orient_combined = torch.cat((orient, geom_controls, spline_controls, z_rs), dim=-1)

        quat_combined = torch.cat((quat[:, :, :self.V * self.D],
                                   geom_controls, spline_controls, orient, z_rs), dim=-1)

        if orient_h is None:
            orient_h = self.orient_h0.expand(-1, orient.shape[0], -1).contiguous()
        orient_combined, orient_h = self.orient_rnn(orient_combined, orient_h)

        if quat_h is None:
            quat_h = self.quat_h0.expand(-1, quat.shape[0], -1).contiguous()
        quat_combined, quat_h = self.quat_rnn(quat_combined, quat_h)

        orient = self.elu(self.orient_fc4(orient_combined))

        quat = self.elu(self.quat_fc4(quat_combined))

        z_rs = self.lrelu(self.z_rs_fc4(z_rs_combined))

        if return_all:
            orient = self.orient_fc5(orient)
            quat = self.quat_fc5(quat)
            z = self.z_fc5(z_rs)
            rs = torch.abs(self.rs_fc5(z_rs))
        else:
            orient = self.orient_fc5(orient[:, -1:])
            quat = self.quat_fc5(quat[:, -1:])
            z = self.z_fc5(z_rs[:, -1:])
            rs = torch.abs(self.rs_fc5(z_rs[:, -1:]))
        z_rs = torch.cat((z, rs), dim=-1)

        orient_pre_normalized = orient.contiguous()
        orient = orient_pre_normalized.view(-1, self.D)
        orient = F.normalize(orient, dim=1).view(orient_pre_normalized.shape)

        quat_pre_normalized = quat[:, :, :(self.V - 1) * self.D].contiguous()
        quat = quat_pre_normalized.view(-1, self.D)
        quat = F.normalize(quat, dim=1).view(quat_pre_normalized.shape)

        if return_prenorm:
            return orient, quat, z_rs, orient_h, quat_h, orient_pre_normalized, quat_pre_normalized
        else:
            return orient, quat, z_rs, orient_h, quat_h
