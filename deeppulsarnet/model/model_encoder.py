import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F
import torch
from model.TemporalBlock import TemporalBlock


class pulsar_encoder_block(nn.Module):
    # Single block of the pulsar encoder
    def __init__(self, n_inputs, n_outputs, kernel_size, stride=2, pool=1, conv_groups=1, norm_groups=4, no_pad=False):
        super().__init__()
        # Set padding in a way that the length is divided by the stride
        if no_pad:
            padding_1 = 0
            padding_2 = 0
        else:
            if (kernel_size - stride) % 2 == 0:
                padding_1 = int((kernel_size - stride) / 2)
                padding_2 = 0
            else:
                # Only works for ker=4,8 and stride=1, pool=4
                padding_1 = int((kernel_size - stride) / 2)
                # padding_1 = int(kernel_size / 2)
                padding_2 = int(kernel_size / pool)

        self.pool = pool
        self.kernel_size = kernel_size
        self.stride = stride

        # The net reduces the length input by stride * pool

        self.net = nn.Sequential(weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding_1, stride=stride, bias=True, groups=conv_groups)),
                                 nn.MaxPool1d(self.pool, padding=padding_2),
                                 nn.LeakyReLU(),
                                 nn.GroupNorm(norm_groups, n_outputs,
                                              affine=True),
                                 )

    def forward(self, x):
        out = self.net(x)
        # if hasattr(self, 'dm0'):
        #     if self.dm0=='concat':
        #         pooled = F.avg_pool2d(x.unsqueeze(1), (x.shape[1], self.stride*self.pool), stride=(1,self.stride*self.pool))
        #         out = torch.cat((out, pooled[:,0,:,:]), dim=1)
        # print(x.shape, out.shape)
        return out


class pool_block(nn.Module):
    # Single block of the pulsar encoder
    def __init__(self, kernel_size):
        super().__init__()
        # 2D pooling block
        self.pool = nn.MaxPool2d(kernel_size)

    def forward(self, x):
        out = self.pool(x.unsqueeze(1))
        return out.squeeze(1)


class pulsar_encoder(nn.Module):
    # Whole pulsar decoder using serveral encoder blocks
    # channel_list is a list of output channels which defines the amount of blocks
    def __init__(self, input_shape, model_para, no_pad=False):
        super().__init__()
        # layers = []

        channel_list = model_para.encoder_channels
        layers = [nn.Dropout(model_para.initial_dropout), ]
        levels = len(model_para.encoder_channels)
        self.input_channels = input_shape[0]
        # self.dm0 = dm0

        for i in range(levels):
            # if channel_list[i] > 0:
            in_channels = int(input_shape[0]) if i == 0 else int(
                channel_list[i - 1])
            out_channels = int(channel_list[i])

            layers += [pulsar_encoder_block(in_channels, out_channels, model_para.encoder_kernel,
                                            stride=model_para.encoder_stride, pool=model_para.encoder_pooling,
                                            conv_groups=model_para.encoder_conv_groups, norm_groups=model_para.encoder_norm_groups,
                                            no_pad=no_pad)]
        if model_para.tcn_1_layers != 0:
            for i in range(model_para.tcn_1_layers):
                dil = model_para.tcn_1_dilation ** i
                # if i ==0 and self.dm0 == 'concat':
                #     added_chan=1
                # else:
                #     added_chan=0

                layers += [TemporalBlock(out_channels, out_channels, model_para.tcn_1_kernel, stride=1, dilation=dil,
                                         groups=model_para.tcn_1_groups, residual=True, dropout=model_para.tcn_1_dropout)]
            if out_channels != model_para.tcn_2_channels:
                layers += [nn.Conv1d(out_channels, model_para.tcn_2_channels, 1),
                           nn.LeakyReLU()]
        else:
            if model_para.tcn_2_channels != model_para.encoder_channels[-1]:
                layers += [nn.Conv1d(model_para.encoder_channels[-1], model_para.tcn_2_channels, 1),
                           nn.LeakyReLU()]

        # if num_inputs != self.ini_channels:
        #     self.first_conv.add_module('downsample', nn.Conv1d(
        #         num_inputs, self.ini_channels, 1))
        #     self.first_conv.add_module('relu', nn.LeakyReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Input is expected to be of shape (batch_size, channel_in, time_steps)
        return self.network(x)
