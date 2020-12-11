import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import optim


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Squeeze_Layer(nn.Module):
    def forward(self, input):
        return input[:, 0, 0, :, :]


class Permute_Layer(nn.Module):
    def forward(self, input):
        return input.permute(0, 2, 1, 3, 4)


class Height_pool(nn.Module):
    def forward(self, input):
        if input.shape[2] == 1:
            return input
        else:
            return F.adaptive_avg_pool3d(input, (1, input.shape[3], input.shape[4]))


class ChannelPool(nn.Module):
    def forward(self, input):
        return F.adaptive_avg_pool2d(input.permute(0, 2, 1, 3), (1, 1)).permute(0, 2, 1, 3)


class Tstep_combine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 3, (1, 1)),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(3, 1, (1, 1)),
                                  nn.LeakyReLU())
        # self.conv_2 = nn.Conv2d(channels, channels, (1, 1))
        # self.relu = nn.LeakyReLU()

    def forward(self, input):
        if input.shape[2] == 1:
            return input
        else:
            min_ten, _ = input.min(dim=2, keepdim=True)
            max_ten, _ = input.max(dim=2, keepdim=True)
            # std_ten = input.std(dim=2, keepdim=True)
            avg_ten = input.mean(dim=2, keepdim=True)

            new_tensor = torch.cat((min_ten, max_ten, avg_ten), dim=2)

            output = self.conv(new_tensor.permute(
                0, 2, 1, 3)).permute(0, 2, 1, 3)

            return output


class Height_conv(nn.Module):
    def __init__(self, height):
        super().__init__()
        if height > 1:
            # self.conv = nn.Sequential(nn.Conv1d(height, 10, 1),
            #     nn.LeakyReLU(),
            #     nn.Conv1d(10, 5, 1),
            #     nn.LeakyReLU(),
            #     nn.Conv1d(5, 1, 1))
            self.conv = nn.Sequential(nn.Conv2d(height, 2 * height, (1, 1)),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(2 * height, 2 *
                                                height, (1, 1)),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(2 * height, 1, (1, 1)))

    def forward(self, input):
        if input.shape[2] == 1:
            return input
        else:
            output = self.conv(input[:, 0, :, :]).unsqueeze(1)

            return output


def compute_stft(x, length=0, pool_size=0, crop=1000, hop_length=None, norm=0, harmonics=0, harmonic_downsample=False):
    if length == 0:
        length = x.shape[2]
    if hop_length == 0:
        hop_length = length
    x = x - x.mean(dim=2, keepdim=True)
    switch = 0
    for j in range(x.shape[1]):
        added_harmonics = 0
        stft_count = 0
        switch_harm = 0
        stft = torch.stft(x[:, j, :], length, hop_length=hop_length, win_length=None,
                          window=None, center=False, normalized=True, onesided=True)

        power_stft = stft[:, :crop, :, 0] ** 2 + \
            stft[:, :crop, :, 1] ** 2

        for harm_counter in range(2 ** harmonics):
            if harm_counter == 0:
                added = power_stft[:, :, :]
            else:
                if not harmonic_downsample:
                    upsampled = F.interpolate(power_stft.transpose(
                        2, 1), scale_factor=harm_counter + 1, mode='nearest').transpose(2, 1)
                    added = added[:, :, :] + upsampled[:, :crop, :]
                    # print(added, upsampled)
                else:
                    downsampled = F.max_pool1d(power_stft.transpose(
                        2, 1), kernel_size=(harm_counter + 1),stride=(harm_counter + 1)).transpose(2, 1)[:, :crop, :]
                    pad_val = added.shape[1] - downsampled.shape[1]
                    downsampled = F.pad(downsampled, (0,0,0,pad_val))
                    added = added[:, :, :] + downsampled
                    # print(added, downsampled, pad_val)
                # print(added.device, added.shape)
            added_harmonics += 1
            if added_harmonics == 2 ** stft_count:
                if pool_size:
                    single_out = F.adaptive_max_pool1d(
                        added.transpose(2, 1), pool_size).transpose(2, 1)
                else:
                    single_out = added
                if not switch_harm:
                    out_harm = single_out.unsqueeze(3)
                    switch_harm = 1
                else:
                    out_harm = torch.cat((out_harm, single_out.unsqueeze(
                        3) / np.sqrt(added_harmonics)), dim=3)
                stft_count += 1
        power_stft = out_harm
        # if pool_size:
        #     power_stft = F.adaptive_max_pool1d(power_stft.transpose(2, 1), pool_size)
        # else:
        power_stft = power_stft.transpose(2, 1)
        if not switch:
            out_stft = power_stft
            switch = 1
        else:
            out_stft = torch.cat((out_stft, power_stft), dim=3)
    if norm:
        std = out_stft.view(out_stft.shape[0], -1).std(dim=1)
        out_stft = out_stft / std[:, None, None]
    # if not harmonics:
    #     out_stft = out_stft[:, :, :, :]
    # print(out_stft.shape)

    return out_stft


class regressor_stft_comb(nn.Module):
    def __init__(self, input_length, input_resolution, height_dropout=0, norm=0, harmonics=4, nn_layers=2,
                 stft_count=1, dm0_class=False, crop_factor=0.8, channels=8,
                 kernel=11, name='', harmonic_downsample=False):
        super().__init__()
        self.input_length = input_length
        #self.crop = int(crop_factor * (self.input_length // 2))
        self.final_output = 2
        self.crop_factor = crop_factor
        self.channels = channels
        self.kernel = kernel
        self.norm = norm
        self.harmonics = harmonics
        self.nn_layers = nn_layers
        self.dm0_class = dm0_class
        self.input_resolution = input_resolution

        self.use_center = False

        self.stft_count = stft_count
        print(input_length)

        max_height = 2 ** (stft_count - 1)
        self.lengths = []
        self.total_height = 0
        self.heights = []

        self.name = name
        self.harmonic_downsample = harmonic_downsample

        for j in range(stft_count):
            current_in = 1
            current_out = 0
            layers = []
            height = 2 ** j
            self.total_height += height
            current_length = int(np.floor(self.input_length / height))
            self.min_length = current_length
            self.lengths.append(current_length)
            self.heights.append(height)
            pool = max_height // height
            setattr(self, f"pool_{current_length}", nn.AvgPool3d(
                (1, pool, 1), stride=(1, pool, 1), padding=(0, pool // 2, 0)))
            # print(height)

        current_in = 1
        if self.total_height >1:
            current_out = self.channels
        else:
            current_out = 0
        layers = []
        if height_dropout:
            layers +=[nn.Dropout2d(height_dropout)]

        if nn_layers >= 0:
            current_out += self.channels
            pool = max_height // height
            layers += [nn.Conv2d(self.total_height, current_out, (self.kernel, 1), stride=(
                1, 1), padding=(self.kernel // 2, 0)),
            ]
            for i in range(nn_layers):
                current_in = current_out
                current_out += self.channels
                dilation = 2 ** (i + 1)
                pad = (self.kernel // 2) * dilation

                layers += [nn.LeakyReLU(),
                           nn.Conv2d(current_in, current_out, (self.kernel, 1), stride=(
                               1, 1), padding=(pad, 0), dilation=(dilation, 1))]
            layers += [nn.LeakyReLU(),
                       nn.Conv2d(current_out, 1, (1, 1), stride=(
                           1, 1)),
                       ]
        self.conv = nn.Sequential(*layers)

        self.glob_pool = nn.Sequential(nn.AdaptiveMaxPool2d(
            (1, 1), return_indices=True),
        )
        final_channels = 1
#        self.conv = nn.Sequential(*layers)
        self.final = nn.Sequential(
            nn.Linear(final_channels, self.final_output))

        if self.harmonics:
            self.final_combine = nn.AdaptiveMaxPool1d(1, return_indices=True)

        ini_conv = 0
        if ini_conv:
            self.ini_conv(1, 0.1)
        self.ini_final()

        pretrain_conv = 0
        if pretrain_conv:
            self.pretrain_conv()

        self.fft_res = 1 / (self.input_resolution * self.min_length)

    def forward(self, x):

        combined_pool = torch.zeros(
            (x.shape[0], self.total_height, self.min_length // 2, x.shape[1] * (self.harmonics + 1))).to(x.device)
        j = 0
        k = 0
        for length in self.lengths:
            stft = compute_stft(x, length, hop_length=length, norm=self.norm, crop=int(length*self.crop_factor),
                                harmonics=self.harmonics, harmonic_downsample=self.harmonic_downsample)
            out_pool = getattr(self, f"pool_{length}")(stft.unsqueeze(1))
            out_pool = out_pool[:, 0, :, :self.min_length // 2, :]
            current_height = out_pool.shape[1]
            combined_pool[:, k:k + current_height, :, :] = out_pool[:, :, :, :]
            j += 1
            k += current_height

        out_conv = self.conv(combined_pool)

        if getattr(self, 'dm0_class', 0):
            total_harms = (self.harmonics + 1)
            total_channels = out_conv.shape[3] // (self.harmonics + 1) - 1
            out_conv = out_conv[:, :, :, :-total_harms] - out_conv[:,
                                                                   :, :, -total_harms:].repeat(1, 1, 1, total_channels)

        # if getattr(self, 'dm0_class', 0):
        #    out_conv_combined = out_conv_combined[:,:,:,:,:-1] - out_conv_combined[:,:,:,:,-1][:,:,:,:,None]

        #out_conv = self.conv(stft.unsqueeze(1))

        out_pool, max_pos = self.glob_pool(out_conv)
        #out_pool = out_pool[:, 0, :, :, :]
        print(max_pos, out_conv.shape, self.fft_res, 1 / (self.input_resolution * self.input_length))
        max_pos = max_pos[:, :1, 0, 0].float()

        max_pos_freq = max_pos // out_conv.shape[3] % out_conv.shape[2]
        # (max_pos % out_conv.shape[4]) % (self.harmonics + 1)
        max_pos_harm = 1
        # (max_pos % out_conv.shape[4]) // (self.harmonics + 1)
        max_pos_chan = 1

        out_pool_final = out_pool[:, :, 0, 0]

        output = self.final(out_pool_final)

        output_freq = 1 / (max_pos_freq * self.fft_res + 0.0000001)
        #output_freq = output_freq.clamp(0, 5)
        #output_freq = torch.ones((x.shape[0], 1)).to(x.device)
        output = torch.cat((output, output_freq), dim=1)
        # print(output_freq)
        return output

    def ini_conv(self, mean=0, std=1):
        for child in self.conv.modules():
            if isinstance(child, nn.Conv3d):
                torch.nn.init.normal_(child.weight, mean=mean, std=std)
                torch.nn.init.zeros_(child.bias)

    def ini_final(self, weight=0.5):

        final_para = torch.nn.Parameter(torch.Tensor([[-weight], [weight]]))
        self.final[0].weight = final_para
        # self.final[0].weight[0,:] = - weight
        # self.final[0].weight[1,:] = + weight

    def pretrain_conv(self):
        print('Pretraining stft classifier')
        batch = 1
        length = 100
        middle = 50
        steps = 3000
        optimizer = optim.Adam(self.conv.parameters(), lr=0.001)
        loss_function = nn.MSELoss()
        input_tensor = torch.zeros((batch, 1, self.total_height, length, 1))
        input_tensor[:, :, :, middle, :] = 1
        success = 0
        for i in range(steps):
            optimizer.zero_grad()
            input_slight_noise = input_tensor.clone() + torch.rand_like(input_tensor) * 0.05
            outputs = self.conv(input_slight_noise)
            loss = loss_function(outputs, input_tensor)
            loss.backward()
            optimizer.step()

            max_pos = torch.argmax(outputs)
            if np.abs(max_pos - middle) < 5 and i > 200:
                success = 1
                # plt.plot(input_slight_noise.detach().numpy()[0,0,0,:,0])
                # plt.plot(outputs.squeeze().detach().numpy())
                # plt.show()
                break
        if success:
            print(f'Pretraining stft classifier successfull after {i} steps')
        else:
            print(
                f'Pretraining stft classifier failed even after after {steps} steps')
