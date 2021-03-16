import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class candidate_creator(nn.Module):
    def __init__(self, added_cands=0, psr_cands=False, candidate_threshold=0):
        super().__init__()
        self.added_cands = added_cands
        self.psr_cands = psr_cands
        self.candidate_threshold = candidate_threshold

        self.glob_pool = nn.AdaptiveMaxPool2d((1, 1), return_indices=True)

        # self.cand_pool = nn.AdaptiveMaxPool2d((1, 1), return_indices=True)

    def forward(self, class_data, final_layer, channel_correction, target=None):
        out_conv = class_data[0]
        periods = class_data[1]

        if self.added_cands > 0:
            candidates, cand_target = self.create_cands_global(
                out_conv, periods, final_layer, channel_correction, target=target, num_cands=self.added_cands, threshold=self.candidate_threshold)
        # else:
        #     cands_target = torch.empty(0, requires_grad=True)
        if target is not None and self.psr_cands:
            psr_in_batch = (target[:, 2].long() % 2) != 0
            if True in psr_in_batch:
                do_psr_cands = True
            else:
                do_psr_cands = False
        else:
            do_psr_cands = False

        if do_psr_cands:
            psr_candidates, psr_cand_target = self.create_cands_psr(
                out_conv, periods, final_layer, target=target)
            if self.added_cands > 0:
                candidates = torch.cat((candidates, psr_candidates), dim=0)
                cand_target = torch.cat((cand_target, psr_cand_target), dim=0)
            else:
                candidates = psr_candidates
                cand_target = psr_cand_target

        if self.added_cands == 0 and not do_psr_cands:
            candidates = torch.empty(0, requires_grad=True)
            cand_target = torch.empty(0, requires_grad=True)

        return candidates, cand_target

    def create_cands_global(self, x, periods, final_layer, channel_correction, target=None, num_cands=2, masked_area=20, threshold=0):
        x = x.permute(0, 1, 2, 3)
        x_repeated = x.repeat(num_cands, 1, 1, 1)
        if target is not None:
            target_repeated = target.repeat(num_cands, 1)
        else:
            target_repeated = torch.empty(0, requires_grad=True)

        ini_mask = torch.arange(x.shape[2]).reshape(1, 1, x.shape[2], 1).expand(
            x.shape[0], x.shape[1], -1, x.shape[3]).float().to(x.device)

        for i in range(num_cands):
            start = i * x.shape[0]
            end = (i + 1) * x.shape[0]
            out_pool, max_pos = self.glob_pool(x_repeated[start:end, :, :, :])
            # print(max_pos.shape)

            max_pos = max_pos[:, :, 0, 0].float()
            max_pos_freq = max_pos // x.shape[3] % x.shape[2]
            # print(max_pos_freq, max_pos)
            # #plt.imshow(x_repeated.cpu().detach().numpy()[start:end,0,:,0], aspect='auto', interpolation=None)
            # plt.plot(x_repeated.cpu().detach().numpy()[end-1,0,:,1])
            # #plt.colorbar()
            # plt.ylim(-5,1)
            # plt.show()
            if num_cands - i != 1:
                mask = (
                    ini_mask - max_pos_freq[:, :, None, None]).abs() < masked_area
                mask_repeated = mask.repeat(num_cands - i - 1, 1, 1, 1)
                # print(mask_repeated.shape)

                x_repeated[end:, :, :, :][mask_repeated] = -100

            # x[:, :, :, max_pos_chan.long(), max_pos_chan.long()] = -100

        # print(mask_repeated.shape, target_repeated.shape, ini_mask.shape, x_repeated.shape)
        # plt.imshow(x_repeated.cpu().detach().numpy()[:,0,:,1], aspect='auto', interpolation='nearest', vmin=-5)
        # plt.colorbar()
        # plt.show()
        out_pool, max_pos = self.glob_pool(x_repeated[:, :, :, :])
        max_pos = max_pos[:, 0, 0, 0] // x.shape[3] % x.shape[2]
        max_pos_per = max_pos.long()
        cands_periods = periods[max_pos_per]

        max_pos_chan_raw = (max_pos % x.shape[3]).long()
        # print(max_pos_chan_raw)
        # if not channel_correction is None:
        #     print(channel_correction)
        #     corrections = channel_correction[max_pos_chan_raw]
        #     print(cands_periods, corrections)
        #     cands_periods = cands_periods * corrections
        #     print(cands_periods)

        out_pool_reshape = out_pool[:, :, 0, 0]
        output = final_layer(out_pool_reshape)
        converted = F.softmax(output, 1)
        if self.candidate_threshold > 0:
            output = output[converted[:, 1] > self.candidate_threshold, :]
            cands_periods = cands_periods[converted[:, 1]
                                          > self.candidate_threshold, :]
            if target is not None:
                target_repeated = target_repeated[converted[:, 1]
                                                  > self.candidate_threshold, :]

        if target is not None:
            target_repeated = torch.cat(
                (target_repeated, target_repeated[:, 2:3]), 1)

        if output.shape[0] > 0 and target is not None:
            for i in range(output.shape[0]):
                out_period = cands_periods[i]
                target_periods = target_repeated[i, 0]
                # is_harmonic = check_harmonic(
                #     out_period.cpu(), target_periods.cpu())
                is_harmonic = check_harmonic_bins(
                    out_period.cpu().numpy(), target_periods.cpu().numpy(), periods.cpu().numpy())
                if is_harmonic:
                    target_repeated[i, 2] = 1
                else:
                    target_repeated[i, 2] = 0

        output = torch.cat((output, cands_periods.unsqueeze(1)), 1)

        return output, target_repeated

    def create_cands_psr(self, x, periods, final_layer, target, pool_range=3):

        # print(target)

        batch_mask = (target[:, 2].long() % 2) != 0
        psr_conv = x[batch_mask, :, :, :]
        #psr_target = target[target[:,0]==target[:,0],:]
        psr_target = target[batch_mask, :]
        # print(psr_conv.shape, psr_target.shape)
        periods_expanded = periods.unsqueeze(0).expand(psr_target.shape[0], -1)
        period_positions = torch.argmin(
            (periods_expanded - psr_target[:, :1]).abs(), dim=1)
        #ini_mask = torch.ones_like(x)
        ini_mask = torch.arange(x.shape[2]).reshape(1, 1, x.shape[2], 1).expand(
            psr_conv.shape[0], x.shape[1], -1, x.shape[3]).to(x.device).float()

        mask = (ini_mask - period_positions[:,
                                            None, None, None]).abs() > pool_range
        psr_conv[mask] = -100
        out_pool, max_pos = self.glob_pool(psr_conv[:, :, :, :])
        max_pos = max_pos[:, 0, 0, 0] // x.shape[3] % x.shape[2]
        max_pos_per = max_pos.long()
        cands_periods = periods[max_pos_per]

        out_pool_reshape = out_pool[:, :, 0, 0]
        output = final_layer(out_pool_reshape)

        output = torch.cat((output, cands_periods.unsqueeze(1)), 1)

        psr_target = torch.cat((psr_target, psr_target[:, 2:3]), 1)

        psr_target[:, 2] = 3

        return output, psr_target


def check_harmonic(period_1, period_2, harmonics=16, fraction_error=0.02):
    quot = np.max((period_1 / period_2, period_2 / period_1))
    rounded = np.min((np.round(quot), harmonics))
    deviation = np.abs(((quot - rounded) / rounded))
    if deviation < fraction_error:
        is_harmonic = True
    else:
        is_harmonic = False

    return is_harmonic


def check_harmonic_bins(period_1, period_2, period_array, harmonics=32, bins_distance=20):
    # We check what kind of harmonic the signal could be
    # Afterwards we rescale how close the signals are after rescaling it to the lower
    per_sort = sorted([period_1, period_2])
    max_per = per_sort[1]
    min_per = per_sort[0]
    quot = max_per / min_per
    rounded = np.min((np.round(quot), harmonics))
    # code for rescaling to target
    # if period_2 > period_1:
    #     pos_round = check_pos(period_array, period_1 * rounded)
    # else:
    #     pos_round = check_pos(period_array, period_1 / rounded)

    # pos_orig = check_pos(period_array, period_2)

    # code for rescaling to lower
    pos_round = check_pos(period_array, max_per/rounded)
    pos_orig = check_pos(period_array, min_per)
    deviation = np.abs(pos_round - pos_orig)
    if deviation < bins_distance:
        is_harmonic = True
    else:
        is_harmonic = False
    # print(deviation, period_1, period_2, rounded,
    #       max_per / min_per, is_harmonic)

    return is_harmonic


def check_pos(array, value):
    position = np.argmin(np.abs(array - value))
    return position
