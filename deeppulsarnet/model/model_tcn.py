import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F
import torch


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class Chomp1d_reverse(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d_reverse, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, self.chomp_size:].contiguous()


class Chomp1d_acausal(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d_acausal, self).__init__()
        self.chomp_size = int(chomp_size / 2)

    def forward(self, x):
        if self.chomp_size:
            return x[:, :, self.chomp_size:-self.chomp_size].contiguous()
        else:
            return x


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0, causal=0, groups=8, acausal=1, residual=False,
        final_norm=True):
        super(TemporalBlock, self).__init__()
        if acausal:
            chomp = Chomp1d_acausal
        else:
            chomp = Chomp1d

        padding = (kernel_size - 1) * dilation
        self.residual = residual
        print(self.residual)

        self.net = nn.Sequential(weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                       stride=stride, padding=padding, dilation=dilation, groups=groups[1])),
                                 chomp(padding),
                                 nn.LeakyReLU(),
                                 nn.GroupNorm(
                                     groups[0], n_outputs, affine=True),

                                 weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                       stride=stride, padding=padding, dilation=dilation, groups=groups[1])),
                                 chomp(padding),
                                 nn.LeakyReLU(),
                                 )
        if final_norm:
            self.net.add_module(str(len(self.net)+1), nn.GroupNorm(
                                     groups[0], n_outputs, affine=True))

        if self.residual:
            self.downsample = nn.Conv1d(
                n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            # self.res_factor = nn.Parameter(torch.rand(1))
            self.res_factor = 1  # nn.Parameter(torch.ones(1))

    def forward(self, x):
        out = self.net(x)
        if not self.residual:
            return out
        else:
            res = x if self.downsample is None else self.downsample(x)
            return out + self.res_factor * res


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, tcn_channels, tcn_layers, groups, output_chan, kernel_size=2, dropout=0, direction=[1, 0], acausal=1,
                 tcn_dilation=[2]):
        super(TemporalConvNet, self).__init__()

        self.ini_channels = tcn_channels[0]
        self.channels_added = tcn_channels[1]
        self.blocks = tcn_layers[0]
        self.groups = groups
        self.dilation = tcn_dilation[0]
        self.kernel_size = kernel_size
        #self.sub_levels = tcn_layers[2]
        self.ini_layers = tcn_layers[1]

        # ini_layers = []
        # for i in range(self.ini_layers):
        #     dil = 2 ** i

        #     ini_layers += [TemporalBlock(num_inputs, num_inputs, kernel_size[0], stride=1, dilation=dil,
        #                                  groups=groups, acausal=acausal)]

        # self.first_conv = nn.Sequential(*ini_layers)
        # if num_inputs != self.ini_channels:
        #     self.first_conv.add_module('downsample', nn.Conv1d(
        #         num_inputs, self.ini_channels, 1))
        #     self.first_conv.add_module('relu', nn.LeakyReLU())

        layers = []
        current_input = self.ini_channels
        for i in range(tcn_layers[0]):
            dilation_size = self.dilation ** i
            current_output = current_input + tcn_channels[1]
            layers += [TemporalBlock(current_input, current_output, kernel_size[2], stride=1, dilation=dilation_size,
                                     dropout=dropout,
                                     groups=groups, acausal=acausal)]
            current_input = current_output

        # layers_2 = []

        # input_chan = current_output
        # output_chan = int(input_chan / 4)
        # layers_2 += [nn.Dropout2d(dropout),
        #     TemporalBlock(input_chan, output_chan, kernel_size[3], stride=1, dilation=1,
        #                   groups=[1, 1]),
        #     nn.Conv1d(output_chan, 1, 1)]

        # layers_2 += [nn.Tanh()]
        self.output_chan = current_output

        self.network = nn.Sequential(*layers)
        # self.network_2 = nn.Sequential(*layers_2)

    def forward(self, x):
        # out_1 = self.first_conv(x)

        out_3 = self.network(x)
        return out_3
        # return self.network_2(out_3[:, :, :])
