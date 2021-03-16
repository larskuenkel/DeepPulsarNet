import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F
import torch


class Preprocess(nn.Module):
    # Preprocessing module
    def __init__(self, input_shape, norm, bias=0, clamp=[-1000, 1000], dm0_subtract=False, groups=1, cmask=False, rfimask=False):
        super().__init__()

        self.input_shape = input_shape
        self.bias = bias
        self.clamp = clamp
        self.use_norm = norm
        self.dm0_subtract = dm0_subtract
        self.cmask = cmask
        self.rfimask = rfimask
        if self.cmask:
            self.mask_layer = ChanMaskLayer(input_shape)

        if self.rfimask:
            self.rfi_layer = RFIMaskLayer(input_shape)

        self.input_chan = self.input_shape[0]
        # if self.dm0 == 'cat':
        #     self.input_chan += 1

        if self.use_norm:
            self.norm = nn.GroupNorm(groups, self.input_chan)


    def forward(self, x):
        # Input is expected to be of shape (batch_size, channel_in, time_steps)
        y = x - self.bias
        if self.clamp[0] or self.clamp[1]:
            y = y.clamp(*self.clamp)
        if hasattr(self, 'dm0_subtract'):
            if self.dm0_subtract:
                y = y - y.mean(dim=1, keepdim=True)
            # print('DM0 removed', y.shape)
            # if self.dm0 == 'cat':
            #     y = torch.cat((y, y.mean(dim=1).unsqueeze(1)), dim=1)

        if hasattr(self, 'cmask'):
            if self.cmask:
                channel_mask = self.mask_layer(y)
        if hasattr(self, 'rfimask'):
            if self.rfimask:
                rfi_mask = self.rfi_layer(y)

        if self.use_norm:
            y = self.norm(y)

        if hasattr(self, 'cmask'):
            if self.cmask:
                y = y * channel_mask[:, :, None]

        if hasattr(self, 'rfimask'):
            if self.rfimask:
                y = y.view(y.shape[0],y.shape[1],self.rfi_layer.chunk_size, -1) * rfi_mask[:, :, None,:]
                y = y.view(x.shape)

        output = y
        return output


class ChanMaskLayer(nn.Module):
    def __init__(self, input_shape):
        # Simple Masking Layer
        super().__init__()
        self.in_channels = input_shape[0]
        self.mode = 1
        if self.mode == 0:
            self.layers = nn.Sequential(nn.Conv1d(self.in_channels, 4 * self.in_channels, 2, groups=self.in_channels),
                                        nn.LeakyReLU(),
                                        nn.Conv1d(
                                            4 * self.in_channels, self.in_channels, 1, groups=self.in_channels),
                                        SteepSigmoid(4, 3))
        elif self.mode == 1:
            self.norm = nn.BatchNorm1d(self.in_channels)
            self.layers = nn.Sequential(nn.Conv1d(2, 4, 1),
                                        nn.LeakyReLU(),
                                        nn.Conv1d(4, 1, 1),
                                        TransposeLayer(),
                                        nn.Conv1d(self.in_channels, self.in_channels,
                                                  1, groups=self.in_channels),
                                        SteepSigmoid(4, 3))
        # self.non_lin = nn.Sigmoid()

    def forward(self, x):
        # Input is expected to be of shape (batch_size, channel_in, time_steps)
        if self.mode == 0:
            demeaned = x - x.mean()
            mean_vals = demeaned.mean(2)
            std_vals = demeaned.std(2)
            input_tensor = torch.stack((mean_vals, std_vals), dim=2)
            channel_mask = self.layers(input_tensor)[:, :, 0]
            return channel_mask
        elif self.mode == 1:
            normed = self.norm(x)
            mean_vals = normed.mean(2)
            std_vals = normed.std(2)
            input_tensor = torch.stack((mean_vals, std_vals), dim=1)
            channel_mask = self.layers(input_tensor)[:, :, 0]
            return channel_mask


class RFIMaskLayer(nn.Module):
    def __init__(self, input_shape, chunk_size=1000):
        # Simple Masking Layer
        super().__init__()
        self.in_channels = input_shape[0]
        # self.norm = nn.BatchNorm1d(self.in_channels)
        self.chunk_size = chunk_size
        self.layers = nn.Sequential(nn.Conv2d(5, 10, 1),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(10, 1, 1),
                                    # TransposeLayer(),
                                    # nn.Conv1d(self.in_channels,self.in_channels,1, groups=self.in_channels),
                                    SteepSigmoid(4, 0),
                                    # nn.Upsample(scale_factor=self.chunk_size, mode='nearest')
                                    )

        self.norm = nn.GroupNorm(5, 5)
        # self.non_lin = nn.Sigmoid()

    def forward(self, x):
        # Input is expected to be of shape (batch_size, channel_in, time_steps)
        # normed = self.norm(x)
        channel_means = x.mean(2)
        channel_std = x.std(2)
        #channel_max, _ = normed.max(2)

        chunked = x.view(x.shape[0],x.shape[1], -1, self.chunk_size)
        chunk_mean = chunked.mean(3)
        chunk_std = chunked.std(3)
        chunk_ptp = chunked.max(3)[0] - chunked.min(3)[0]

        scrunched_mean = chunk_mean.mean(1)
        scrunched_std = chunk_std.mean(1)

        mean_dev_chan = chunk_mean - channel_means[:,:,None]
        mean_dev_time = chunk_mean - scrunched_mean[:,None,:]

        std_dev_chan = chunk_std - channel_std[:,:,None]
        std_dev_time = chunk_std - scrunched_std[:,None,:]


        input_tensor = torch.stack((mean_dev_chan, mean_dev_time,
            std_dev_chan, std_dev_time, chunk_ptp), dim=1)
        input_tensor = self.norm(input_tensor)
        rfi_mask = self.layers(input_tensor)
        return rfi_mask[:,0,:,:]


class SteepSigmoid(nn.Module):
    def __init__(self, factor, bias):
        # Simple Masking Layer
        super().__init__()
        self.factor = nn.Parameter(torch.Tensor([factor]))
        self.bias = nn.Parameter(torch.Tensor([bias]))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input is expected to be of shape (batch_size, channel_in, time_steps)
        rescaled = self.factor * x + self.bias
        return self.sigmoid(rescaled)


class TransposeLayer(nn.Module):
    def __init__(self):
        # Transposes dim 1 and 2
        super().__init__()

    def forward(self, x):
        # Input is expected to be of shape (batch_size, channel_in, time_steps)
        return x.permute(0, 2, 1)
