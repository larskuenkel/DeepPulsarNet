import torch
import torch.nn as nn
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import torch.nn.functional as F
# from torch.utils.checkpoint import checkpoint


class smooth(nn.Module):
    def __init__(self, binary=0, kernel=(1, 29), sigma=(0, 10)):
        super().__init__()
        self.kernel = kernel
        self.pad = (int((self.kernel[0] - 1) / 2), int((self.kernel[1] - 1) / 2))
        # dummy_array = np.zeros(kernel)
        # dummy_array[int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2)] = 1
        # self.gaussian_kernel = torch.cuda.FloatTensor(
        #     gaussian_filter(dummy_array, sigma)).unsqueeze(0).unsqueeze(1)
        # self.smooth = nn.Conv2d(1, 1, kernel, bias=False, padding=(0, int((kernel[1] - 1) / 2)))
        # self.smooth.weight.data.copy_(torch.from_numpy(self.gaussian_kernel))
        self.threshold = binary

    def forward(self, x):
        # out = self.smooth(x.unsqueeze(1)).squeeze()
        # out = F.conv2d(x.unsqueeze(1), self.gaussian_kernel, padding=(
        #     int((self.kernel[0] - 1) / 2), int((self.kernel[1] - 1) / 2))).squeeze()
        #out = F.max_pool1d(x, self.kernel, stride=1, padding=self.pad)
        out = F.max_pool2d(x, self.kernel, stride=1, padding=self.pad)
        max_vals, _ = torch.max(out.view(x.shape[0], -1), dim=1)
        max_vals[max_vals == 0] = 1
        if len(out.shape) < 3:
            out = out.unsqueeze(dim=0)
        out /= max_vals[:, None, None]
        if self.threshold:
            out = torch.where(out > self.threshold, torch.ones(
                1).cuda(), torch.zeros(1).cuda())

        return out.detach()


class smooth_input(nn.Module):
    def __init__(self, kernel=201):
        super().__init__()
        if kernel%2==0:
            kernel += 1
            print('Pre pool kernel willl be increased by 1.')
        self.kernel = kernel
        self.pad = int((self.kernel - 1) / 2)

    def forward(self, x):

        mean_vals = F.avg_pool1d(x, self.kernel, stride=1, padding=self.pad)
        combined = x.detach() - mean_vals.detach()

        return combined
