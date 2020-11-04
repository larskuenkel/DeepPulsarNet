import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F
import torch
from model.model_decoder import pulsar_decoder
from model.model_encoder import pulsar_encoder
from model.shakedrop import ShakeDrop1d
from model.TemporalBlock import TemporalBlock
from model.TemporalBlock import TemporalBlock_2d



class Prep_2d(nn.Module):
    def forward(self, input):
        return input.unsqueeze(1)


class Post_2d(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1, input.shape[3])



class TemporalConvNet_single(nn.Module):
    def __init__(self, num_inputs, channels_added, blocks, groups, kernel_size=2, acausal=1,
                 tcn_dilation=2, dropout=0, residual=True):
        super().__init__()

        self.ini_channels = num_inputs
        self.channels_added = channels_added
        self.blocks = blocks
        self.groups = groups
        self.dilation = tcn_dilation
        self.kernel_size = kernel_size

        # self.ini_layers = tcn_layers[1]
        self.kernel_1d = kernel_size

        ini_layers = []

        layers = []
        current_input = self.ini_channels
        for i in range(self.blocks):
            dilation_size = self.dilation ** i
            current_output = current_input + channels_added if i > 0 else num_inputs
            layers += [TemporalBlock(current_input, current_output, self.kernel_1d, stride=1, dilation=dilation_size,
                                     groups=groups, acausal=acausal, dropout=dropout, residual=residual)]
            current_input = current_output

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out_1 = x
        return self.network(out_1)


class TemporalConvNet_multi(nn.Module):
    def __init__(self, num_inputs, tcn_channels_added, tcn_layers, groups, kernel_size=5, dropout=0, acausal=1,
                 dilation=5, stride=1, pool=1,
                 block_mode='cat',  levels=1, downsample_factor=4):
        super().__init__()

        self.ini_channels = num_inputs
        self.channels_added = tcn_channels_added
        self.blocks = tcn_layers
        self.groups = groups
        self.dilation = dilation
        self.kernel_1d = kernel_size
        self.levels = levels
        self.block_mode = block_mode
        self.downsample_factor = downsample_factor

        self.output_chan_block = self.ini_channels + self.channels_added * self.blocks

        if self.block_mode == 'cat':
            self.complete_channels = self.output_chan_block * self.levels
        elif self.block_mode == 'add':
            self.complete_channels = self.output_chan_block

        for i in range(self.levels):
            setattr(self, "tb%d" % i, TemporalConvNet_single(num_inputs, self.channels_added, self.blocks, groups,
                                                             kernel_size=self.kernel_1d, tcn_dilation=dilation, dropout=dropout))

            if i > 0:
                setattr(self, "down_%d" % i, nn.Sequential(nn.MaxPool1d(downsample_factor, padding=0),
                                                 nn.GroupNorm(groups, num_inputs)))
                decoding_blocks = [self.output_chan_block, ] * i
                setattr(self, "up_%d" % i, nn.Sequential(nn.Upsample(scale_factor=downsample_factor)))


        self.output_chan = self.complete_channels

    def forward(self, x):

        out_3 = torch.zeros(
            (x.shape[0], self.complete_channels, x.shape[2])).cuda()
        y = x

        for i in range(self.levels):
            if i == 0:
                block_out = getattr(self, "tb%d" % i)(x)
            else:
                down_1 = getattr(self, "down_%d" % i)(y)

                intermediate = getattr(self, "tb%d" % i)(down_1)
                block_out = getattr(self, "up_%d" % i)(intermediate)
                y = down_1

            if self.block_mode == 'cat':

                start_channel = i * self.output_chan_block
                out_3[:, start_channel:start_channel +
                      self.output_chan_block, :] = block_out

            elif self.block_mode == 'add':
                out_3 += block_out

        return out_3
