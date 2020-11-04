import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Print_shape(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input


class regressor_fft(nn.Module):
    def __init__(self, fft_class, no_reg, coherent=False):
        super().__init__()

        self.coherent = coherent
        self.input_size = fft_class[1]
        self.fft_trunc = fft_class[2]
        self.pool_size = fft_class[3]


        if fft_class[4]>=0:
            self.use_rnn = fft_class[4]
            self.conv_kernel = 0
        else:
            self.use_rnn = 0
            self.conv_kernel = -fft_class[4]
        self.harmonics = np.abs(fft_class[5])
        if self.coherent:
            self.harmonic_mode = np.sign(-1)
        else:
            self.harmonic_mode = np.sign(fft_class[5])

        self.pool = nn.MaxPool1d(self.pool_size)

        self.fft_size_pooled = int(np.floor(self.fft_trunc / self.pool_size))
        #self.norm = nn.GroupNorm(1+self.harmonics, 1,eps=10, affine=False)
        #self.norm = nn.BatchNorm1d(1)

        self.scale_parameter = nn.Parameter(torch.ones(1))


        self.no_reg = no_reg
        if self.no_reg:
            self.final_output = 2
        else:
            self.final_output = 4

        if not self.use_rnn:

            self.lin_channels = fft_class[6:]

            lin_layers = []
            if self.conv_kernel:
                channel_number= (1+self.harmonics)
                padding = (self.conv_kernel -1) //2
                lin_layers += [nn.Conv1d((1+self.harmonics), channel_number, self.conv_kernel, stride=1, padding=padding),
                nn.MaxPool1d(self.conv_kernel, stride=self.conv_kernel),
                nn.LeakyReLU()]
                lin_input = int((np.floor((self.fft_size_pooled - self.conv_kernel)/self.conv_kernel) + 1) * channel_number)
            else:
                channel_number = (1+self.harmonics)
                lin_input = self.fft_size_pooled * channel_number
            glob_pool = 1
            if not glob_pool:
                lin_layers += [Flatten()]
            else:
                lin_layers += [nn.AdaptiveMaxPool1d(1),
                Flatten()]
                lin_input = channel_number

            for lin_layer in self.lin_channels:
                lin_layers += [nn.Dropout(0.2),
                               nn.Linear(lin_input, lin_layer), nn.LeakyReLU(),]
                lin_input = lin_layer
            lin_layers += [nn.Linear(lin_input, self.final_output)]

            self.lin_class = nn.Sequential(*lin_layers)
        else:
            self.hidden = fft_class[6]
            self.rnn = nn.LSTM(1, self.hidden, 1, batch_first=True)
            self.lin_class = nn.Linear(self.hidden, self.final_output)

    def forward(self, x):
        if not hasattr(self, 'coherent'):
            normed = self.compute_fft(x, harmonics=self.harmonics)
        else:
            if not self.coherent:
                normed = self.compute_fft(x, harmonics=self.harmonics)
            else:
                normed = self.compute_fft_coherent(x, harmonics=self.harmonics)


        if not self.use_rnn or not hasattr(self, 'use_rnn'):
            if hasattr(self, 'conv_kernel'):
                output = self.lin_class(normed)
            else:
                output = self.lin_class(normed.view(normed.size(0), -1))

        else:
            (out_rnn, hidden) = self.rnn(normed.unsqueeze(2), None)
            output = self.lin_class(out_rnn[:, -1, :])

        return output

    def compute_fft(self, x, scale=0, harmonics=0):
        x_mean = torch.mean(x, dim=2)
        x = x - x_mean[:, :, None]
        x = x[:, 0, :]
        length = x.shape[-1]
        l_pad = (self.input_size - length) // 2
        r_pad = self.input_size - length - l_pad
        x_padded = F.pad(x, (l_pad, r_pad))

        fft_out = torch.rfft(x_padded, 1)
        power_fft = fft_out[:, :, :1] ** 2 + fft_out[:, :, 1:] ** 2
        power_fft = power_fft.permute(0,2,1)


        if harmonics:
            output_harm = torch.zeros((power_fft.shape[0], self.harmonics+1, self.fft_trunc)).cuda()
            for harm in range(self.harmonics+1):
                if harm==0:
                    added = power_fft[:,0, :self.fft_trunc]
                else:
                    harm_number = harm +1
                    if not hasattr(self, 'harmonic_mode'):
                        downsampled = F.max_pool1d(power_fft, harm_number)
                    else:
                        if self.harmonic_mode>0:
                            downsampled = F.max_pool1d(power_fft, harm_number)
                        else:
                            downsampled = F.interpolate(power_fft, scale_factor=harm_number)
                    added = added[:,:] + downsampled[:,0,:self.fft_trunc]
                output_harm[:,harm, :added.shape[1]] = added / np.sqrt(harm+1)
        else:
            output_harm = power_fft[:, :,:self.fft_trunc]
        pooled = self.pool(output_harm)
        normed = pooled / 10000
        return normed


    def compute_fft_coherent(self, x, scale=0, harmonics=0):
        x_mean = torch.mean(x, dim=2)
        x = x - x_mean[:, :, None]
        x = x[:, 0, :]
        length = x.shape[-1]
        l_pad = (self.input_size - length) // 2
        r_pad = self.input_size - length - l_pad
        x_padded = F.pad(x, (l_pad, r_pad))

        fft_out = torch.rfft(x_padded, 1)
        fft_out = fft_out.permute(0,2,1)


        if harmonics:
            output_harm = torch.zeros((fft_out.shape[0], harmonics+1, self.fft_trunc)).cuda()
            for harm in range(harmonics+1):
                if harm==0:
                    added = fft_out[:,:2, :self.fft_trunc]
                    
                else:
                    harm_number = harm +1
                    downsampled = F.interpolate(fft_out, scale_factor=harm_number)
                    added = added[:,:] + downsampled[:,:,:self.fft_trunc]
                added_power = added[:, 0, :] ** 2 + added[:, 1, :] ** 2
                output_harm[:,harm, :added_power.shape[1]] = added_power / np.sqrt(harm+1)
        else:
            output_harm = fft_out[:, :1, :self.fft_trunc] ** 2 + fft_out[:, 1:, :self.fft_trunc] ** 2
        pooled = self.pool(output_harm)
        normed = pooled / 10000

        return normed
