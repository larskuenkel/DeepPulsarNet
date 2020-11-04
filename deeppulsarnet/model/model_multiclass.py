import torch.nn as nn
from torch.nn import functional as F
import torch


class MultiClass(nn.Module):
    def __init__(self, number_classifiers, no_reg):
        super().__init__()
        self.number_classifiers = number_classifiers
        # self.net = nn.Linear(self.number_classifiers*2, 2)
        # self.non_lin = nn.LeakyReLU()
        self.no_reg = no_reg
        if self.no_reg:
            self.output_vals = 3
        else:
            self.output_vals = 3
        ini_para = torch.ones(self.number_classifiers)
        change_ini_para = 0
        if change_ini_para:
            ini_para[:-1] = 0.2
            ini_para[-1] = 1
        self.parameter = nn.Parameter(ini_para)

    def forward(self, x):
        # x_relu = x
        out_tensor = torch.zeros(x.shape[0], self.output_vals).cuda()
        for j in range(self.number_classifiers):
            out_tensor[:, :2] = out_tensor[:, :2] + x[:, j, :2] * self.parameter[j]
        # if not self.no_reg:
        #     total_weight = torch.sum(F.relu(x[:, :, 3])+0.0001, dim=1)
        #     for j in range(self.number_classifiers):
        #         mean_period_added = out_tensor[:, 0] + \
        #             F.relu(x[:, j, 2]) * F.relu(x[:, j, 3])
        #     out_tensor[:, 2] = mean_period_added / total_weight 
        #     out_tensor[:, 3] = total_weight
        softmax = F.softmax(x[:,:,:2],dim=2)
        total_weight = torch.sum(softmax[:,:,1], dim=1)+0.0001
        mean_period_added = torch.sum(softmax[:,:,1] * x[:,:,2], dim=1) / total_weight
        out_tensor[:, 2] = mean_period_added
        return out_tensor
