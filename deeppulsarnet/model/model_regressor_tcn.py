import torch.nn as nn
import torch
# from indrnn import indrnn
from model.model_tcn_multi import TemporalConvNet_single
from utils import calc_tcn_depth


class regressor_tcn(nn.Module):
    def __init__(self, input_channels, tcn_class, groups, dropout, no_reg):
        super().__init__()
        self.input_channels = input_channels
        self.layers = tcn_class[0]
        self.kernel = tcn_class[1]
        self.channels = [0, tcn_class[2]]
        self.pool_size = tcn_class[3]
        self.lin_channels = tcn_class[4:]
        self.groups = groups
        self.no_reg = no_reg
        if self.no_reg:
            self.final_output = 2
        else:
            self.final_output = 4

        # self.chan_layers = [self.channels,] * self.layers
        print('yooooo',self.kernel, self.input_channels)
        self.tcn = TemporalConvNet_single(self.input_channels, self.channels, self.layers, self.groups, kernel_size=self.kernel, dropout=dropout, acausal=0, 
            tcn_dilation=self.kernel)

        self.pool = nn.AdaptiveAvgPool1d(1)


        lin_layers = []
        lin_input = self.channels[0] + self.channels[1] * self.layers
        print(self.lin_channels)
        for lin_layer in self.lin_channels:
            lin_layers += [nn.Linear(lin_input, lin_layer), nn.LeakyReLU(),]
            lin_input = lin_layer
        lin_layers += [nn.Linear(lin_input, self.final_output)]

        self.lin_class = nn.Sequential(*lin_layers)

        tcn_range = calc_tcn_depth(self.kernel, self.layers, dilation_factor=self.kernel, factor=1)


    def forward(self, x):
        tcn_out = self.tcn(x)
        pooled = self.pool(tcn_out[:,:,-self.pool_size:])[:,:,0]
        output = self.lin_class(pooled)
        output = torch.cat((output, torch.zeros(output.shape[0], 1).cuda()), dim=1)

        return output
