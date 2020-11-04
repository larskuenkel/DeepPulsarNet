import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from riptide import TimeSeries, ffa_search, peak_detection


def sigma_fit(stats, polydeg=2):
    x = stats['logpmid']
    y = stats['sigma']
    poly = np.poly1d( np.polyfit(x, y, polydeg) )
    def func(period):
        return poly(np.log(period))
    return func, poly.coefficients


def median_fit(stats, polydeg=2):
    x = stats['logpmid']
    y = stats['median']
    poly = np.poly1d( np.polyfit(x, y, polydeg) )
    def func(period):
        return poly(np.log(period))
    return func, poly.coefficients


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


class regressor_ffa(nn.Module):
    def __init__(self, ffa_para, input_chan=1, no_reg=True, input_length=90000, ffa_args=[0.07, 1.1, 15, 80, 1], dm0_class=False):
        super().__init__()
        self.no_reg = no_reg
        if self.no_reg:
            self.final_output = 2
        else:
            self.final_output = 4
        self.pool_size = ffa_para[0]
        self.num_layers = ffa_para[1]
        self.channels = ffa_para[2]
        self.kernel = ffa_para[3]
        self.norm = ffa_para[4]
        self.use_ampl = ffa_para[5]
        self.use_pytorch = ffa_para[6]
        self.ffa_args = ffa_args
        self.dm0_class = dm0_class

        if self.use_pytorch:
            dummy_array = np.zeros(input_length)
            tseries = TimeSeries.from_numpy_array(dummy_array,0.00064*4)
            ts, plan, pgram = ffa_search(tseries,period_min=0.07,period_max=1.1,bins_min=10, bins_max=12)
            self.ffa_module = pytorch_ffa(plan)
        self.input_chan = input_chan


        # if input_length == 0:
        #     self.height = 2
        # else:
        #     self.height = int(
        #         np.floor((input_length - self.stft_length) / self.hop_length)) + 1
        #     print('height', self.height)
        self.height = 1
        current_in = 1
        current_out = 0
        layers = []
        if not self.use_ampl:
            if self.pool_size:
                self.pool = nn.MaxPool3d((1, self.pool_size, 1))
            current_out += self.channels
            if self.ffa_args[4] == 4:
                ini_channels = 3
            else:
                ini_channels = 1
            layers += [
                nn.Conv3d(ini_channels, current_out, (self.height, self.kernel, 1), stride=(
                    1, 1, 1), padding=(0, self.kernel // 2, 0)),
                # nn.LeakyReLU(),
                # nn.Conv3d(current_out, 1, (1, 1, 1), stride=(1,1,1)),
                # Squeeze_Layer(),
            ]
            for i in range(self.num_layers - 1):
                current_in = current_out
                current_out += self.channels
                dilation = 2 ** (i + 1)
                pad = (self.kernel // 2) * dilation

                layers += [nn.LeakyReLU(),
                           nn.Conv3d(current_in, current_out, (1, self.kernel, 1), stride=(
                               1, 1, 1), padding=(0, pad, 0), dilation=(1, dilation, 1))]
            layers += [nn.LeakyReLU(),
                       nn.Conv3d(current_out, 1, (1, 1, 1), stride=(
                           1, 1, 1)),
                       # nn.LeakyReLU()
                       ]

            self.conv = nn.Sequential(*layers)
            pretrain_conv = 0
            if pretrain_conv:
                self.pretrain_conv()
        self.glob_pool = nn.AdaptiveMaxPool3d(
                (1, 1, 1), return_indices=True)
        self.final = nn.Sequential(nn.Linear(1, self.final_output))

        self.ini_final()

    def forward(self, x):
        if not hasattr(self, 'use_pytorch'):
            ffa_tensor = calc_ffa(x)
        else:
            if self.use_pytorch:
                periods, ffa_tensor = self.ffa_module(x)
                ffa_tensor = ffa_tensor.permute(0,2,1).unsqueeze(1).unsqueeze(1)
            else:
                if not hasattr(self, 'ffa_args'):
                    ffa_tensor, ffa_periods = calc_ffa(x)
                else:
                    if not self.ffa_args[0] == -1:
                        ffa_tensor, ffa_periods = calc_ffa(x, bin_min=int(self.ffa_args[2]), bin_max=int(self.ffa_args[3]),
                            min_period=self.ffa_args[0], max_period=self.ffa_args[1], renorm= False)#int(self.ffa_args[4]))
                    else:
                        ffa_tensor, ffa_periods = calc_ffa_piecewise(x, renorm=self.ffa_args[4])
            if getattr(self, 'ffa_args', [0,0,0,0,0])[4]==1:
                ffa_tensor = renorm_ffa_gpu(ffa_tensor)
        if not self.use_ampl:
            if self.pool_size:
                ffa_tensor = self.pool(ffa_tensor)
            out_conv = self.conv(ffa_tensor)
        else:
            out_conv = ffa_tensor

        if getattr(self, 'dm0_class', 0):
            out_conv = out_conv[:,:,:,:,:-1] - out_conv[:,:,:,:,-1][:,:,:,:,None]
        pooled, max_pos = self.glob_pool(out_conv)
        max_pos = max_pos[:, 0, 0, 0, 0]
        max_pos_period = max_pos // out_conv.shape[4] % out_conv.shape[3]
        max_pos_chan = max_pos % out_conv.shape[4]
        ffa_periods = torch.Tensor(ffa_periods).cuda()
        if self.pool_size:
            position_factor = self.pool_size
        else:
            position_factor = 1
        periods = ffa_periods[max_pos_period * position_factor]
        output = self.final(pooled[:, :, 0, 0, 0])
        output = torch.cat((output, periods.unsqueeze(1)), dim=1)
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
        print('Pretraining ffa classifier')
        batch = 1
        length = 1000
        middle = 500
        steps = 3000
        optimizer = optim.Adam(self.conv.parameters(), lr=0.001)
        loss_function = nn.MSELoss()
        input_tensor = torch.zeros((batch, 1, self.height, length, 1))
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
            print(f'Pretraining ffa classifier successfull after {i} steps')
        else:
            print(
                f'Pretraining ffa classifier failed even after after {steps} steps')


def calc_ffa(tensor, bin_min=10, bin_max=12, renorm=True, min_period=0.07, max_period=1.1):
    cpu_tensor = tensor.detach().cpu().numpy()
    ini_shape = cpu_tensor.shape
    switch = 0
    for i in range(ini_shape[0]):
        for j in range(ini_shape[1]):
            # for k in range(ini_shape[2]):
            tseries = TimeSeries.from_numpy_array(cpu_tensor[i,j,:], 0.00064 * 4)
            ts, pgram = ffa_search(
                tseries, period_min=min_period, period_max=max_period, bins_min=bin_min, bins_max=bin_max)
            snr = pgram.snrs.max(axis=1)
            if renorm:
                snr = renorm_ffa(snr)
            if not switch:
                new_tensor = np.zeros((ini_shape[0], 1, 1, len(snr), ini_shape[1]))
                switch = 1
            new_tensor[i,0,0,:,j] = snr
    gpu_tensor = torch.Tensor(new_tensor).cuda()
    return gpu_tensor, pgram.periods


def detrend_pgram(pgram):
    boundaries = peak_detection.segment(pgram.periods, pgram.tobs)
    snr = pgram.snrs.max(axis=1)
    stats = peak_detection.segment_stats(snr, boundaries)
    tfunc_m, polyco_m = median_fit(stats, polydeg=2)
    medians = tfunc_m(pgram.periods)
    tfunc_s, polyco_s = sigma_fit(stats, polydeg=2)
    sigmas = tfunc_s(pgram.periods)
    snrs = (snr - medians) / sigmas
    return snrs


def dethresh_pgram_old(pgram, snr_min=6.5, nsigma=6.5, polydeg=2, cat=False):
    boundaries = peak_detection.segment(pgram.periods, pgram.tobs)
    snr = pgram.snrs.max(axis=1)
    stats = peak_detection.segment_stats(snr, boundaries)
    tfunc_m, polyco_m = median_fit(stats, polydeg=polydeg)
    medians = tfunc_m(pgram.periods)
    tfunc_s, polyco_s = sigma_fit(stats, polydeg=polydeg)
    sigmas = tfunc_s(pgram.periods)
    if cat==False:
        threshold = np.maximum(medians+ nsigma*sigmas, snr_min)
        snrs = snr - threshold
    else:
        snrs = np.stack((snr, medians, sigmas), axis=0)
    return snrs

def dethresh_pgram(pgram, snr_min=6.5, n_sigma=6.5, polydeg=2):
    snr = pgram.snrs.max(axis=1)
    fc, smed, sstd = peak_detection.segment_stats(pgram.periods, snr, pgram.tobs)
    sc = smed + n_sigma * sstd
    coeffs = np.polyfit(np.log(fc), sc, polydeg)
    poly = np.poly1d(coeffs)
    dynthr = np.maximum(poly(np.log(pgram.periods)),snr_min)
    snrs = snr-dynthr
    return snrs


def calc_ffa_piecewise(tensor, renorm=0):
    cpu_tensor = tensor.detach().cpu().numpy()
    ini_shape = cpu_tensor.shape
    switch = 0
    for i in range(ini_shape[0]):
        for j in range(ini_shape[1]):
            # for k in range(ini_shape[2]):
            tseries = TimeSeries.from_numpy_array(cpu_tensor[i,j,:], 0.00064 * 4)
            ts, pgram1 = ffa_search(tseries,period_min=0.03,period_max=0.12,bins_min=10, bins_max=14)
            #print(plan)
            ts, pgram2 = ffa_search(tseries,period_min=0.12,period_max=0.48,bins_min=40, bins_max=44)
            #print(plan)
            ts, pgram3 = ffa_search(tseries,period_min=0.48,period_max=1.1,bins_min=160, bins_max=176)
            periods_combined = np.concatenate((pgram1.periods, pgram2.periods, pgram3.periods))
            if renorm==2:
                snr1 = detrend_pgram(pgram1)
                snr2 = detrend_pgram(pgram2)
                snr3 = detrend_pgram(pgram3)
                snr = np.concatenate((snr1, snr2, snr3))
            elif renorm==3:
                snr1= dethresh_pgram(pgram1)
                snr2 = dethresh_pgram(pgram2)
                snr3 = dethresh_pgram(pgram3)
                snr = np.concatenate((snr1, snr2, snr3))
            elif renorm==4:
                snr1 = dethresh_pgram(pgram1, cat=True)
                snr2 = dethresh_pgram(pgram2, cat=True)
                snr3 = dethresh_pgram(pgram3, cat=True)
                snr = np.concatenate((snr1, snr2, snr3),axis=1)
            else:
                snr = np.concatenate((pgram1.snrs.max(axis=1), pgram2.snrs.max(axis=1), pgram3.snrs.max(axis=1)))
             #   snr = renorm_ffa(snr)
            if not renorm==4:
                if not switch:
                    new_tensor = np.zeros((ini_shape[0], 1, 1, len(snr), ini_shape[1]))
                    switch = 1
                new_tensor[i,0,0,:,j] = snr
            else:
                if not switch:
                    new_tensor = np.zeros((ini_shape[0], 3, 1, snr.shape[1], ini_shape[1]))
                    switch = 1
                new_tensor[i,:,0,:,j] = snr
    gpu_tensor = torch.Tensor(new_tensor).cuda()
    return gpu_tensor, periods_combined


def renorm_ffa(snr, parts=20):
    length_seg = (snr.shape[0] // parts)
    trunc = snr.shape[0] % length_seg
    snr = snr[:-trunc]
    parts_altered = snr.shape[0] // length_seg
    snr_split = np.split(snr, parts_altered)
    new_snr = []
    for segment in snr_split:
        median = np.mean(segment)
        dev = (np.percentile(segment,75) - np.percentile(segment,25)) / 1.349
        new_segment = (segment-median) / dev
        new_snr.extend(new_segment)
    snr = np.asarray(new_snr)
    return snr

def renorm_ffa_gpu(ffa_tensor, parts=10):
    length_seg = (ffa_tensor.shape[3] // parts)
    trunc = ffa_tensor.shape[3] % length_seg
    ffa_trunc = ffa_tensor[:,:,:,:-trunc,:]
    reshaped = ffa_trunc.view(ffa_tensor.shape[0],ffa_tensor.shape[1],ffa_tensor.shape[2],parts,length_seg,ffa_tensor.shape[4])
    std = reshaped.std(4, keepdim=True)
    mean = reshaped.mean(4, keepdim=True)
    renormed = (reshaped - mean) / std
    out = renormed.view(ffa_tensor.shape[0],ffa_tensor.shape[1],ffa_tensor.shape[2],-1,ffa_tensor.shape[4])
    return out

class pytorch_ffa(nn.Module):
    def __init__(self, plan):
        super().__init__()
        self.plan = plan

    def forward(self, x):
        switch = 0
        device = x.device
        periods = []
        std_dev = x[:,:,:].std(2)
        for step in self.plan.steps.iterrows():
            down_factor = step[1].dsfactor
            bins_min = int(step[1].bins_min)
            bins_max = int(step[1].bins_max)
            down = self.downsample(x, scale=down_factor)
            for current_bins in range(bins_min, bins_max):
                trunc_val = down.shape[2] % current_bins
                if trunc_val:
                    padded_array = down[:,:,:-trunc_val]
                else:
                    padded_array = down
                reshaped_array = padded_array.reshape(padded_array.shape[0],padded_array.shape[1], -1,current_bins)
                ini_rows = reshaped_array.shape[2]
                k = np.ceil(np.log2(ini_rows)).astype(int)
                target_rows = int(2**k)
                added_rows = target_rows - ini_rows
                if added_rows:
                    added_shape = (reshaped_array.shape[0], reshaped_array.shape[1], added_rows, reshaped_array.shape[3])
                    out_array = torch.cat([reshaped_array, torch.zeros(added_shape).to(device)], 2)
                else:
                    out_array = reshaped_array
                ffa_out = self.ffa(out_array, down_factor, added_rows, std_dev)
                if not switch:
                    ffa_combined = ffa_out
                    switch=1
                else:
                    ffa_combined = torch.cat([ffa_combined, ffa_out], 2)
                p0 = step[1].tsamp * current_bins
                p1 = step[1].tsamp * (current_bins+1)
                p_step = step[1].tsamp / (out_array.shape[2]-1)
                p_vals = np.arange(p0, p1, p_step)
                p_vals = np.linspace(p0, p1, out_array.shape[2])
                periods.extend(p_vals.tolist())
        return periods, ffa_combined
    
    def ffa(self, x, down_factor, added_rows, std_dev):
        nstage = np.log2(x.shape[2]).astype(int)
        bins = x.shape[3]
        ini_shape = x.shape
        for stage in range(1, nstage+1):
            x = self.shiftadd(x, stage)
        x = x.reshape(ini_shape)
        snr = self.calc_snr(x, down_factor, added_rows, std_dev)
        return snr


    def downsample(self, in_data, target=None, scale=None):
        if target:
            downsampled = F.interpolate(in_data, size=target, mode='linear', align_corners=False)
        elif scale:
            downsampled = F.interpolate(in_data, scale_factor=1/scale, mode='linear', align_corners=False)
        return downsampled
        
    def shiftadd(self, x, stage):
        nBatch = x.shape[0]
        nChan = x.shape[1]
        nRow = x.shape[2]
        nCol = x.shape[3]
        nRowGroup = int(2 ** stage)
        nGroup = int(nRow//nRowGroup)
        y = torch.zeros_like(x)
        y = self.groupshiftadd(x, nRowGroup, nCol)
        y = y.reshape(nBatch, nChan, nRow, nCol)
        return y
    
    def groupshiftadd(self, x, nRowGroup, nCol):
        x = x.reshape(x.shape[0], x.shape[1], -1, nRowGroup, nCol)
        y = torch.zeros_like(x)
        half_nRowGroup = int(nRowGroup // 2)
        for i in range(nRowGroup):
            iA = int(i//2)
            iB = iA + half_nRowGroup
            Bs = int((i + 1) // 2)
            y[:,:,:,i,:] = x[:,:,:,iA,:] + x[:,:,:,iB,:].roll(Bs, dims=3)
        return y
    
    def calc_snr(self, profiles, down_factor, added_rows, std_dev):
        # Metric 1 from https://arxiv.org/pdf/1805.08247.pdf
        max_val, _ = profiles.max(dim=3)
        med_val, _ = profiles.median(dim=3)
        snr = (max_val - med_val) / (std_dev[:,:,None]) / (np.sqrt(profiles.shape[2] - added_rows))
        return snr
