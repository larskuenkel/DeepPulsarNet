import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class calc_max_std(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        max_val, _ = torch.max(x, dim=2)
        std_val = torch.std(x, dim=2)
        output = max_val / std_val
        return output.unsqueeze(1)


class simple_peak_detector(nn.Module):
    def __init__(self, tseries, chunks=10):
        super().__init__()
        self.chunks = chunks
        self.whole_chunks = int(self.chunks * tseries)
        self.norm = nn.GroupNorm(
            self.whole_chunks, self.whole_chunks, affine=False)

    def forward(self, x):
        x_reshaped = x.view(
            x.shape[0], self.whole_chunks, x.shape[2] // self.chunks)
        #x_norm = self.norm(x_reshaped)
        x_norm = x_reshaped
        peak_heights, _ = torch.max(x_norm, dim=2)
        medians, _ = torch.median(x_norm, dim=2)
        peak_heights = peak_heights - medians
        # print(peak_heights.shape)
        return peak_heights.unsqueeze(1)


class score_recompute(nn.Module):
    def __init__(self, ini_val):
        super().__init__()
        self.parameter = nn.Parameter(torch.ones(1) * ini_val)

    def forward(self, x):
        output = torch.zeros(x.shape[0], 2).cuda()
        output[:, :1] = self.parameter - x
        output[:, 1:] = x
        return output


class regressor_fft_new(nn.Module):
    def __init__(self, fft_class, no_reg, coherent=False):
        super().__init__()

        self.coherent = coherent
        self.mode = fft_class[0]
        # mode==0: One network looks at all harmonics at the same time
        # mode==1_ The same network looks at all harmonics eparately and uses highest

        self.input_size = fft_class[1]
        self.fft_trunc = fft_class[2]
        self.pool_size = fft_class[3]

        if fft_class[4] >= 0:
            self.use_rnn = fft_class[4]
            self.conv_kernel = 0
        else:
            self.use_rnn = 0
            self.conv_kernel = -fft_class[4]
        self.harmonics = np.abs(fft_class[5])
        self.total_blocks = int(self.harmonics + 1)
        # if self.coherent:
        #     self.harmonic_mode = np.sign(-1)
        # else:
        self.harmonic_mode = np.sign(fft_class[5])

        self.max_pool = nn.MaxPool1d(self.pool_size)
        # self.mean_pool = nn.AvgPool1d(self.pool_size)

        self.fft_size_pooled = int(np.floor(self.fft_trunc / self.pool_size))
        # self.norm = nn.GroupNorm(1+self.harmonics, 1,eps=10, affine=False)
        # self.norm = nn.GroupNorm(1, self.total_blocks, affine=False)
        #self.norm = nn.BatchNorm1d(1)

        self.scale_parameter = nn.Parameter(torch.ones(1))

        self.no_reg = no_reg
        if self.no_reg:
            if self.mode==1 and self.conv_kernel:
                self.final_output = 1
            else:
                self.final_output = 2
        else:
            self.final_output = 4

        if not self.use_rnn:
            if self.conv_kernel:
                self.conv_pool = fft_class[6]
                self.conv_chan = fft_class[7]
                self.conv_layers = fft_class[8]
                self.final_pool = fft_class[9]
                self.final_intermediate = fft_class[10]
                conv_layers = []
                if self.mode==0:
                    input_chan = 1 + self.harmonics
                elif self.mode==1:
                    input_chan = 1

                    final_layers = [nn.AdaptiveMaxPool1d(1),
                    Flatten(), 
                    score_recompute(1)
                                    ]
                    self.final_layers = nn.Sequential(*final_layers)
                for i in range(self.conv_layers):
                    output_chan = (i + 1) * self.conv_chan
                    conv_layers += [nn.Conv1d(input_chan, output_chan, self.conv_kernel),
                                    nn.MaxPool1d(self.conv_pool),
                                    nn.LeakyReLU()]
                    input_chan = output_chan

                conv_layers += [nn.AdaptiveMaxPool1d(self.final_pool),
                                Flatten()]

                input_chan = self.final_pool * output_chan
                if self.final_intermediate:
                    conv_layers += [nn.Linear(input_chan, self.final_intermediate),
                                    nn.LeakyReLU()]
                    input_chan = self.final_intermediate

                conv_layers += [nn.Linear(input_chan, self.final_output)]
                self.lin_class = nn.Sequential(*conv_layers)

            else:

                self.lin_channels = fft_class[6:]

                lin_layers = []
                # if self.conv_kernel:
                #     channel_number = (1 + self.harmonics)
                #     padding = (self.conv_kernel - 1) // 2
                #     lin_layers += [nn.Conv1d((1 + self.harmonics), channel_number, self.conv_kernel, stride=1, padding=padding),
                #                    nn.MaxPool1d(self.conv_kernel,
                #                                 stride=self.conv_kernel),
                #                    nn.LeakyReLU()]
                #     lin_input = int((np.floor(
                #         (self.fft_size_pooled - self.conv_kernel) / self.conv_kernel) + 1) * channel_number)
                # else:
                channel_number = 1
                if self.mode == 0:
                    channel_number *= (1 + self.harmonics)
                lin_input = self.fft_size_pooled * channel_number
                lin_layers += [Flatten()]
                for lin_layer in self.lin_channels:
                    lin_layers += [nn.Dropout(0.2),
                                   nn.Linear(lin_input, lin_layer), nn.LeakyReLU(), ]
                    lin_input = lin_layer
                # lin_layers += [nn.Linear(lin_input, self.final_output)]
                if self.mode == 0:
                    lin_layers += [nn.Linear(lin_layer, self.final_output)]

                self.lin_class = nn.Sequential(*lin_layers)
                if self.mode == 1:
                    final_channels = lin_layer * self.total_blocks

                    final_layers = [nn.AdaptiveMaxPool1d(1),
                                    Flatten(),
                                    # nn.Linear(final_channels, lin_layer),
                                    # nn.LeakyReLU(),
                                    # nn.Linear(lin_layer, 2),
                                    score_recompute(1)
                                    ]
                    self.final_layers = nn.Sequential(*final_layers)
            # #self.max_calc = calc_max_std()
            # self.peak_finder = simple_peak_detector(self.total_blocks, chunks=self.pool_size)
        else:
            self.hidden = fft_class[6]
            self.rnn = nn.LSTM(1, self.hidden, 1, batch_first=True)
            self.lin_class = nn.Linear(self.hidden, self.final_output)

    def forward(self, x):

        normed = self.compute_fft(x, harmonic_blocks=self.harmonics)

        # normed = self.norm(diff)

        if self.mode == 1:
            block_outputs = torch.zeros(
                (normed.shape[0], 1, self.total_blocks)).cuda()
            for j in range(normed.shape[1]):
                out_block = self.lin_class(normed[:, j:j+1, :])
                block_outputs[:, :, j] = out_block
        elif self.mode == 0:
            block_outputs = self.lin_class(normed)
            return block_outputs
        # print(block_outputs)

        # block_outputs = self.max_calc(normed)
        # block_outputs = self.peak_finder(normed)

        output = self.final_layers(block_outputs)

        return output

    def compute_fft(self, x, scale=0, harmonic_blocks=0):
        x_mean = torch.mean(x, dim=2)
        x = x - x_mean[:, :, None]
        x = x[:, 0, :]
        length = x.shape[-1]
        l_pad = (self.input_size - length) // 2
        r_pad = self.input_size - length - l_pad
        x_padded = F.pad(x, (l_pad, r_pad))

        fft_out = torch.rfft(x_padded, 1, normalized=True)
        power_fft = fft_out[:, :, :1] ** 2 + fft_out[:, :, 1:] ** 2
        power_fft = power_fft.permute(0, 2, 1)

        if harmonic_blocks:
            output_harm = torch.zeros(
                (power_fft.shape[0], harmonic_blocks + 1, self.fft_trunc)).cuda()
            added_harmonics = 0
            block = 0
            for harm_counter in range(2 ** harmonic_blocks):
                if harm_counter == 0:
                    added = power_fft[:, 0, :self.fft_trunc]
                else:
                    if self.harmonic_mode>0:
                        downsampled = F.max_pool1d(power_fft, harm_counter + 1)
                        # downsampled = F.avg_pool1d(power_fft, harm_counter + 1)
                        #print(torch.mean(downsampled, dim=2).shape)
                        # downsampled = downsampled - torch.mean(downsampled, dim=2, keepdim=True)
                    else:
                        downsampled = F.interpolate(power_fft, scale_factor=harm_counter+1)
                    if downsampled.shape[2] < added.shape[1]:
                        padding = (0, added.shape[1] - downsampled.shape[2])
                        downsampled = F.pad(downsampled, padding)
                    added = added[:, :] + downsampled[:, 0, :self.fft_trunc]
                added_harmonics += 1
                if added_harmonics == 2 ** block:
                    output_harm[:, block, :added.shape[1]
                                ] = added / np.sqrt(added_harmonics + 1)
                    block += 1
        else:
            output_harm = power_fft[:, :, :self.fft_trunc]
        max_pooled = self.max_pool(output_harm)
        # max_pooled = max_pooled / self.input_size
        # mean_pooled = self.mean_pool(output_harm)
        # diff = max_pooled - mean_pooled
        # diff = diff / 10000
        # return diff
        return max_pooled

    # def compute_fft_coherent(self, x, scale=0, harmonics=0):
    #     x_mean = torch.mean(x, dim=2)
    #     x = x - x_mean[:, :, None]
    #     x = x[:, 0, :]
    #     length = x.shape[-1]
    #     l_pad = (self.input_size - length) // 2
    #     r_pad = self.input_size - length - l_pad
    #     x_padded = F.pad(x, (l_pad, r_pad))

    #     fft_out = torch.rfft(x_padded, 1)
    #     fft_out = fft_out.permute(0, 2, 1)

    #     if harmonics:
    #         output_harm = torch.zeros(
    #             (fft_out.shape[0], harmonics + 1, self.fft_trunc)).cuda()
    #         for harm in range(harmonics + 1):
    #             if harm == 0:
    #                 added = fft_out[:, :2, :self.fft_trunc]

    #             else:
    #                 harm_number = harm + 1
    #                 downsampled = F.interpolate(
    #                     fft_out, scale_factor=harm_number)
    #                 added = added[:, :] + downsampled[:, :, :self.fft_trunc]
    #             added_power = added[:, 0, :] ** 2 + added[:, 1, :] ** 2
    #             output_harm[:, harm, :added_power.shape[1]
    #                         ] = added_power / np.sqrt(harm + 1)
    #     else:
    #         output_harm = fft_out[:, :1, :self.fft_trunc] ** 2 + \
    #             fft_out[:, 1:, :self.fft_trunc] ** 2
    #     pooled = self.pool(output_harm)
    #     normed = pooled / 10000

    #     return normed
