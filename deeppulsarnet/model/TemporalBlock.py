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
        return x[:, :, self.chomp_size:-self.chomp_size].contiguous()


class Chomp1d_acausal_2d(nn.Module):
    def __init__(self, kernel, dilation):
        super().__init__()
        if kernel % 2 == 0:
            print('For now no even kernel until I understand them better.')
            sys.exit()
        self.complete_chomp = max(0, (kernel - 1) * dilation)
        if self.complete_chomp % 2 == 0:
            self.chomp_size_1 = int(self.complete_chomp / 2)
            self.chomp_size_2 = int(self.complete_chomp / 2)
        else:
            self.chomp_size_1 = int(np.ceil(self.complete_chomp / 2))
            self.chomp_size_2 = int(np.floor(self.complete_chomp / 2))

    def forward(self, x):
        return x[:, :, :, self.chomp_size_1:-self.chomp_size_2].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, causal=0, norm_groups=4, conv_groups=1, acausal=1, dropout=0, residual=True, final_norm=True):
        super().__init__()
        if acausal:
            chomp = Chomp1d_acausal
        else:
            chomp = Chomp1d

        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                       stride=stride, padding=padding, dilation=dilation, groups=conv_groups)),
                                 chomp(padding),
                                 nn.LeakyReLU(),
                                 nn.GroupNorm(
                                     norm_groups, n_outputs, affine=True),
                                 nn.Dropout(dropout),
                                 weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                       stride=stride, padding=padding, dilation=dilation, groups=conv_groups)),
                                 chomp(padding),
                                 nn.LeakyReLU(),
                                 # nn.GroupNorm(
                                 #     groups[0], n_outputs, affine=True),
                                 # nn.Dropout(dropout)
                                 )

        if final_norm:
            self.net.add_module(str(len(self.net)+1), nn.GroupNorm(
                                     norm_groups, n_outputs, affine=True))
        self.net.add_module(str(len(self.net)+1), nn.Dropout(dropout))


        self.residual = residual
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


class TemporalBlock_2d(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, causal=0, groups=8, acausal=1, dropout=0):
        super().__init__()
        chomp = Chomp1d_acausal_2d
        self.kernel = kernel_size

        padding_1 = max(0, kernel_size[0] - 2)
        padding_2 = max(0, (kernel_size[1] - 1) * dilation)
        padding = (padding_1, padding_2)

        self.net = nn.Sequential(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=(1, dilation), groups=groups),
                                 chomp(self.kernel[1], dilation),
                                 nn.LeakyReLU(),
                                 nn.GroupNorm(
                                     1, n_outputs, affine=True),
                                 nn.Dropout(dropout),
                                 nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=(1, dilation), groups=groups),
                                 chomp(self.kernel[1], dilation),
                                 nn.LeakyReLU(),
                                 nn.GroupNorm(
                                     1, n_outputs, affine=True),
                                 nn.Dropout(dropout)
                                 )

        self.downsample = nn.Conv2d(
            n_inputs, n_outputs, (1, 1)) if n_inputs != n_outputs else None
        # self.res_factor = nn.Parameter(torch.rand(1))
        self.res_factor = 1  # nn.Parameter(torch.ones(1))

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + self.res_factor * res