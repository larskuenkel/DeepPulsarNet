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


class regressor_stft_conv(nn.Module):
    def __init__(self, stft_para, no_reg, dropout=[0, 0], input_chan=1, norm=0, crop_augment=0., harmonics=0, layer_number=1, height_pooling=1,
                 input_length=0, dm0_class=False):
        super().__init__()
        self.no_reg = no_reg
        if self.no_reg:
            self.final_output = 2
        else:
            self.final_output = 2
        self.stft_length = stft_para[0]
        self.crop = int(stft_para[1] * (self.stft_length // 2))
        self.channels = stft_para[2]
        self.pool_size = stft_para[3]
        self.kernel = stft_para[4]
        self.hop_length = int(stft_para[5] * self.stft_length)
        self.input_chan = input_chan
        self.norm = norm
        self.crop_augment = crop_augment
        print(stft_para)
        self.harmonics = np.abs(harmonics)
        self.harmonic_mode = np.sign(harmonics)
        self.height_pooling = height_pooling
        self.layer_number = layer_number
        self.dm0_class = dm0_class

        self.use_center = False

        self.ini_dropout = nn.Dropout(dropout[0])

        print(input_length)

        if input_length == 0:
            self.height = 2
        else:
            self.height = int(
                np.floor((input_length - self.stft_length) / self.hop_length)) + 1
            print('height', self.height)

        # if self.pool_size:
        #     # self.pool = nn.AdaptiveMaxPool1d(
        #     # self.pool_size)
        #     self.rnn_input_size = self.pool_size * self.input_chan
        # else:
        #     self.rnn_input_size = self.crop * self.input_chan

        # self.conv = nn.Sequential(nn.Conv2d(1, self.channels, (1,self.kernel), stride=(1,1)),
        #     # nn.MaxPool2d((1,self.kernel)),
        #     # nn.LeakyReLU(),
        #     # nn.Conv2d(self.channels, self.channels * 2, (1,self.kernel), stride=(1,1)),
        #     nn.AvgPool2d((2,1)),
        #     nn.AdaptiveMaxPool2d((1,1)),
        #     nn.LeakyReLU())
        layers = []
        current_in = 1
        current_out = 0

        self.use_simple = 0

        if height_pooling != 3:
            for i in range(layer_number):
                current_out += self.channels
                layers += [nn.Conv3d(current_in, current_out, (1, self.kernel, 1), stride=(1, 1, 1), padding=(0, self.kernel // 2, 0)),
                           nn.LeakyReLU()]

                if i + 1 != layer_number:
                    layers += [nn.MaxPool3d((1, self.kernel, 1))]
                current_in = current_out

            layers += [nn.Dropout3d(dropout[1]),
                       # nn.Conv2d(current_out, current_out, (1, 1)),
                       # nn.LeakyReLU(),
                       # nn.Conv2d(current_out, np.min((10,current_out)), (1, 1)),
                       # nn.LeakyReLU(),
                       # nn.Conv2d(np.min((10,current_out)), 1, (1, 1)),
                       nn.Conv3d(current_out, 1, (1, 1, 1)),
                       # nn.Conv3d(current_out, 1, (1, self.kernel, 1), stride=(1, 1, 1), padding=(0,self.kernel//2, 0)),
                       # nn.LeakyReLU(),
                       ]

            if self.height_pooling == 1:
                layers += [Height_pool()]
            elif self.height_pooling == 2:
                if self.height > 1:
                    layers += [nn.LeakyReLU(),
                               Height_conv(self.height)]
            else:
                layers += [Tstep_combine(current_out)]

            # self.glob_pool = nn.AdaptiveMaxPool3d(
            #     (1, 1, 1), return_indices=True)
        else:
            if layer_number >= 0:
                current_out += self.channels
                layers += [
                    nn.Conv3d(1, current_out, (self.height, self.kernel, 1), stride=(
                        1, 1, 1), padding=(0, self.kernel // 2, 0)),
                    # nn.LeakyReLU(),
                    # nn.Conv3d(current_out, 1, (1, 1, 1), stride=(1,1,1)),
                    # Squeeze_Layer(),
                ]
                for i in range(layer_number):
                    current_in = current_out
                    current_out += self.channels
                    dilation = 2 ** (i+1)
                    pad = (self.kernel // 2 )*dilation

                    layers += [nn.LeakyReLU(),
                    nn.Conv3d(current_in, current_out, (1, self.kernel, 1), stride=(
                        1, 1, 1), padding=(0, pad, 0), dilation=(1,dilation,1))]
                layers += [nn.LeakyReLU(),
                    nn.Conv3d(current_out, 1, (1, 1, 1), stride=(
                        1, 1, 1)),
                    # nn.LeakyReLU()
                    ]
            else:
                self.use_simple = 1
                if layer_number ==-1:
                    self.use_simple_norm = 0
                else:
                    self.use_simple_norm = 1


        self.glob_pool = nn.Sequential(#Permute_Layer(),
            # Permute not needed when not height_pooling=3
                                       nn.AdaptiveMaxPool3d(
                                           (1, 1, 1), return_indices=True),
                                       )
        self.use_meanpool = 0
        if self.use_meanpool:
            self.mean_pool = nn.AdaptiveAvgPool3d(
                                           (1, 1, 1))
            final_channels = 2
        else:
            final_channels = 1
        # self.c_pool = ChannelPool()
        self.conv = nn.Sequential(*layers)
        self.final = nn.Sequential(nn.Linear(final_channels, self.final_output))

        if self.harmonics:
            # self.final_combine = nn.Linear((self.harmonics+1)*self.final_output, self.final_output)
            self.final_combine = nn.AdaptiveMaxPool1d(1, return_indices=True)

        ini_conv = 0
        if ini_conv:
            self.ini_conv(1, 0.1)
        self.ini_final()

        pretrain_conv = 1
        if pretrain_conv and not self.use_simple:
            self.pretrain_conv()

        self.fft_res = 1 / (0.00064 * 4 * self.stft_length)

    def forward(self, x):
        if hasattr(self, 'ini_dropout'):
            x = self.ini_dropout(x)

        if hasattr(self, 'crop_augment'):
            if self.training and self.crop_augment != 0.:
                used_crop = int(np.random.uniform(
                    self.crop - self.crop * self.crop_augment, self.crop + self.crop * self.crop_augment))
            else:
                used_crop = self.crop
        else:
            used_crop = self.crop

        if self.harmonic_mode >=0:
            stft = compute_stft(x, self.stft_length, pool_size=self.pool_size, hop_length=self.hop_length, norm=self.norm, crop=used_crop,
                            harmonics=self.harmonics)
        else:
            stft = compute_stft_custom(x, self.stft_length, pool_size=self.pool_size, hop_length=self.hop_length, norm=self.norm, crop=used_crop,
                            harmonics=self.harmonics)

        if not hasattr(self,'use_simple'):
            out_conv = self.conv(stft.unsqueeze(1))
        else:
            if not self.use_simple:
                out_conv = self.conv(stft.unsqueeze(1))
            else:
                if self.use_simple_norm:
                    stft = (stft - stft.mean(dim=2, keepdim=True)) / stft.std(dim=2, keepdim=True)
                stft = stft.unsqueeze(1)
                out_conv = F.adaptive_max_pool3d(stft,(1, stft.shape[3], stft.shape[4]))

        if getattr(self, 'dm0_class', 0):
            total_harms = self.harmonics + 1
            total_channels = out_conv.shape[4] // total_harms - 1
            out_conv = out_conv[:,:,:,:,:-total_harms] - out_conv[:,:,:,:,-total_harms:].repeat(1,1,1,1,total_channels)

        out_pool, max_pos = self.glob_pool(out_conv[:, :, :, :, :])
        #out_pool = out_pool[:, 0, :, :, :]
        max_pos = max_pos[:, :1, 0, 0, 0].float()

        if self.height_pooling != 3:
            max_pos_freq = max_pos // out_conv.shape[4]
            max_pos_harm = max_pos % out_conv.shape[4]
        else:
            # max_pos_chan = max_pos // (out_conv.shape[4] * out_conv.shape[3])
            max_pos_freq = max_pos // out_conv.shape[4] % out_conv.shape[3]
            max_pos_harm = (max_pos % out_conv.shape[4]) % (self.harmonics + 1)
            max_pos_chan = (max_pos % out_conv.shape[4]) // (self.harmonics + 1)

        # print(max_pos_freq, max_pos, max_pos_harm)

        if not self.use_meanpool:
            out_pool_final = out_pool[:, :, 0, 0, 0]
        else:
            if self.final[0].in_features == 2:
                out_pool_2 = self.mean_pool(out_conv)[:, :, 0, 0, 0]
                out_pool_final = torch.cat((out_pool[:, :, 0, 0, 0], out_pool_2), dim=1)
            elif self.final[0].in_features == 3:
                out_pool_min, _ = self.glob_pool(-out_conv[:, :, :, :, :])
                out_pool_final = torch.cat((out_pool[:, :, 0, 0, 0], out_pool_min[:, :, 0, 0, 0], out_pool_2), dim=1)

        output = self.final(out_pool_final)

        # if not self.harmonics:
        #     out_conv = self.conv(stft.unsqueeze(1))

        #     if hasattr(self, 'glob_pool'):
        #         out_conv, max_pos = self.glob_pool(out_conv)

        #     output = self.final(out_conv[:, :, 0, 0])

        # else:
        #     for k in range(self.harmonics + 1):
        #         out_conv, max_pos = self.conv(stft[:, :, :, k].unsqueeze(1))
        #         # output_lin = self.final(out_conv[:,:,0,0])
        #         if k == 0:
        #             out_blocks = out_conv[:, :, 0, :]
        #             max_poss = max_pos[:, :, 0, 0]
        #         else:
        #             out_blocks = torch.cat(
        #                 (out_blocks, out_conv[:, :, 0, :]), dim=2)
        #             max_poss = torch.cat(
        #                 (max_poss, max_pos[:, :, 0, 0]), dim=1)
        #     out_com, max_pos_harm = self.final_combine(out_blocks)
        # print(max_pos_harm, max_pos_harm[:,0,0].shape)
        # output = self.final(out_com[:, :, 0])

        # max_pos = max_poss[max_pos_harm]# / 2 ** max_pos_harm[:,0,0]

        output_freq = 1 / (max_pos_freq * self.fft_res / (2 ** max_pos_harm) + 0.00001)
        output_freq = output_freq.clamp(0, 5)
        # print(output_freq, max_pos_freq , self.fft_res , self.kernel , (self.layer_number-1), max_pos_harm, self.fft_res)
        # output_freq = max_pos.float() * self.fft_res * self.kernel ** (self.layer_number-1)

        # print(max_pos, output_freq, self.fft_res, self.kernel ** (self.layer_number-1))
        # if not self.no_reg:
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
        input_tensor = torch.zeros((batch, 1, self.height, length, 1))
        input_tensor[:,:,:,middle,:] = 1
        success = 0
        for i in range(steps):
            optimizer.zero_grad()
            input_slight_noise = input_tensor.clone() + torch.rand_like(input_tensor) * 0.05
            outputs = self.conv(input_slight_noise)
            loss = loss_function(outputs, input_tensor)
            loss.backward()
            optimizer.step()

            max_pos = torch.argmax(outputs)
            if np.abs(max_pos - middle) < 5 and i >200:
                success=1
                # plt.plot(input_slight_noise.detach().numpy()[0,0,0,:,0])
                # plt.plot(outputs.squeeze().detach().numpy())
                # plt.show()
                break
        if success:
            print(f'Pretraining stft classifier successfull after {i} steps')
        else:
            print(f'Pretraining stft classifier failed even after after {steps} steps')



def compute_stft(x, length=0, pool_size=0, crop=1000, hop_length=None, norm=0, harmonics=0):
    if length == 0:
        length = x.shape[2]
    if hop_length == 0:
        hop_length = length
    x = x - x.mean(dim=2, keepdim=True)
    switch = 0
    for j in range(x.shape[1]):
        added_harmonics = 0
        block = 0
        switch_harm = 0
        stft = torch.stft(x[:, j, :], length, hop_length=hop_length, win_length=None,
                          window=None, center=False, normalized=True, onesided=True)

        power_stft = stft[:, :crop, :, 0] ** 2 + \
            stft[:, :crop, :, 1] ** 2

        for harm_counter in range(2 ** harmonics):
            if harm_counter == 0:
                added = power_stft[:, :, :]
            else:
                downsampled = F.interpolate(power_stft.transpose(
                    2, 1), scale_factor=harm_counter + 1, mode='nearest').transpose(2, 1)
                added = added[:, :, :] + downsampled[:, :crop, :]
            added_harmonics += 1
            if added_harmonics == 2 ** block:
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
                block += 1
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
    def __init__(self, stft_para, no_reg, dropout=[0, 0], input_chan=1, norm=0, crop_augment=0., harmonics=0, layer_number=1,
                 input_length=0, blocks=1, added_height=1, dm0_class=False):
        super().__init__()
        self.no_reg = no_reg
        if self.no_reg:
            self.final_output = 2
        else:
            self.final_output = 2
        self.ini_length = stft_para[0]
        self.stft_length = stft_para[0]
        self.crop = int(stft_para[1] * (self.stft_length // 2))
        # print(self.crop)
        self.channels = stft_para[2]
        self.pool_size = stft_para[3]
        self.kernel = stft_para[4]
        self.hop_length = int(stft_para[5] * self.stft_length)
        self.input_chan = input_chan
        self.norm = norm
        self.crop_augment = crop_augment
        print(stft_para)
        self.harmonics = np.abs(harmonics)
        self.harmonic_mode = np.sign(harmonics)
        self.layer_number = layer_number
        self.dm0_class = dm0_class

        self.use_center = False

        self.ini_dropout = nn.Dropout(dropout[0])

        self.blocks = blocks

        print(input_length)

        if input_length == 0:
            self.height = 2
        else:
            self.height = int(
                np.floor((input_length - self.stft_length) / self.hop_length)) + 1
            print('height', self.height)

        if self.pool_size:
            # self.pool = nn.AdaptiveMaxPool1d(
            # self.pool_size)
            self.rnn_input_size = self.pool_size * self.input_chan
        else:
            self.rnn_input_size = self.crop * self.input_chan

        self.use_simple = 0

        #height = 1

        if blocks ==1:
            blocks= 3
        max_height = 2 ** (blocks-1)
        self.lengths = []
        self.total_height = 0
        self.heights = []

        for j in range(blocks):
            current_in = 1
            current_out = 0
            layers = []
            height = 2 ** j
            self.total_height += height 
            current_length = int(np.floor(self.stft_length/height))
            self.min_length = current_length
            self.lengths.append(current_length)
            self.heights.append(height)
            pool = max_height//height
            setattr(self, f"pool_{current_length}",nn.AvgPool3d((1, pool, 1), stride=(1,pool,1), padding=(0, pool // 2, 0)))
            print(height)

        current_in = 1
        current_out = self.channels
        layers = []

        if layer_number >= 0:
            current_out += self.channels
            pool = max_height//height
            layers += [nn.Conv2d(self.total_height, current_out, (self.kernel, 1), stride=(
                    1, 1), padding=(self.kernel // 2, 0)),
            ]
            for i in range(layer_number):
                current_in = current_out
                current_out += self.channels
                dilation = 2 ** (i+1)
                pad = (self.kernel // 2 )*dilation

                layers += [nn.LeakyReLU(),
                nn.Conv2d(current_in, current_out, (self.kernel, 1), stride=(
                    1, 1), padding=(pad, 0), dilation=(dilation,1))]
            layers += [nn.LeakyReLU(),
                nn.Conv2d(current_out, 1, (1, 1), stride=(
                    1, 1)),
                ]
        self.conv = nn.Sequential(*layers)

        self.harmpool = (self.harmonics+1) * self.input_chan

        self.glob_pool = nn.Sequential(nn.AdaptiveMaxPool2d(
                                           (1,1), return_indices=True),
                                       )
        self.use_meanpool = 0
        if self.use_meanpool:
            self.mean_pool = nn.AdaptiveAvgPool3d(
                                           (1, 1, 1))
            final_channels = 2
        else:
            final_channels = 1
#        self.conv = nn.Sequential(*layers)
        self.final = nn.Sequential(nn.Linear(final_channels, self.final_output))

        if self.harmonics:
            self.final_combine = nn.AdaptiveMaxPool1d(1, return_indices=True)

        ini_conv = 0
        if ini_conv:
            self.ini_conv(1, 0.1)
        self.ini_final()

        pretrain_conv = 0
        if pretrain_conv and not self.use_simple:
            self.pretrain_conv()

        self.fft_res = 1 / (0.00064 * 4 * self.stft_length)

    def forward(self, x):
        if hasattr(self, 'ini_dropout'):
            x = self.ini_dropout(x)

        combined_pool = torch.zeros((x.shape[0], self.total_height, self.min_length//2, x.shape[1]*(self.harmonics+1))).cuda()
        j = 0
        k = 0
        for length in self.lengths:
            stft = compute_stft(x, length, hop_length=length, norm=self.norm, crop=length//2,
                            harmonics=self.harmonics)
            out_pool = getattr(self, f"pool_{length}")(stft.unsqueeze(1))
            out_pool = out_pool[:,0,:,:self.min_length//2,:]
            current_height = out_pool.shape[1]
            combined_pool[:, k:k+current_height,:,:] = out_pool[:,:,:,:]
            j += 1
            k += current_height

        out_conv = self.conv(combined_pool)

        if getattr(self, 'dm0_class', 0):
            total_harms = (self.harmonics+1)
            total_channels = out_conv.shape[3] // (self.harmonics+1) - 1
            out_conv = out_conv[:,:,:,:-total_harms] - out_conv[:,:,:,-total_harms:].repeat(1,1,1,total_channels)

        # if getattr(self, 'dm0_class', 0):
        #    out_conv_combined = out_conv_combined[:,:,:,:,:-1] - out_conv_combined[:,:,:,:,-1][:,:,:,:,None]

        #out_conv = self.conv(stft.unsqueeze(1))

        out_pool, max_pos = self.glob_pool(out_conv)
        #out_pool = out_pool[:, 0, :, :, :]
        max_pos = 1#max_pos[:, :1, 0, 0, 0].float()

        max_pos_freq = 1#max_pos // out_conv.shape[4] % out_conv.shape[3]
        max_pos_harm = 1#(max_pos % out_conv.shape[4]) % (self.harmonics + 1)
        max_pos_chan = 1#(max_pos % out_conv.shape[4]) // (self.harmonics + 1)


        if not self.use_meanpool:
            out_pool_final = out_pool[:, :, 0, 0]
        else:
            if self.final[0].in_features == 2:
                out_pool_2 = self.mean_pool(out_conv)[:, :, 0, 0, 0]
                out_pool_final = torch.cat((out_pool[:, :, 0, 0, 0], out_pool_2), dim=1)
            elif self.final[0].in_features == 3:
                out_pool_min, _ = self.glob_pool(-out_conv[:, :, :, :, :])
                out_pool_final = torch.cat((out_pool[:, :, 0, 0, 0], out_pool_min[:, :, 0, 0, 0], out_pool_2), dim=1)

        output = self.final(out_pool_final)

        #output_freq = 1 / (max_pos_freq * self.fft_res / (2 ** max_pos_harm) + 0.00001)
        #output_freq = output_freq.clamp(0, 5)
        output_freq = torch.ones((x.shape[0],1)).cuda()
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
        input_tensor = torch.zeros((batch, 1, self.height, length, 1))
        input_tensor[:,:,:,middle,:] = 1
        success = 0
        for i in range(steps):
            optimizer.zero_grad()
            input_slight_noise = input_tensor.clone() + torch.rand_like(input_tensor) * 0.05
            outputs = self.conv(input_slight_noise)
            loss = loss_function(outputs, input_tensor)
            loss.backward()
            optimizer.step()

            max_pos = torch.argmax(outputs)
            if np.abs(max_pos - middle) < 5 and i >200:
                success=1
                # plt.plot(input_slight_noise.detach().numpy()[0,0,0,:,0])
                # plt.plot(outputs.squeeze().detach().numpy())
                # plt.show()
                break
        if success:
            print(f'Pretraining stft classifier successfull after {i} steps')
        else:
            print(f'Pretraining stft classifier failed even after after {steps} steps')