import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


class classifier_filter(nn.Module):
    def __init__(self, pulses=7, min_period=0.065, max_period=0.8, points=50, gauss_para=(111//4,15/4)):
        super().__init__()
        self.min_period = min_period
        self.max_period = max_period
        self.resolution = 0.00064*4
        self.gauss_para = gauss_para
        self.gauss = scipy.signal.gaussian(*gauss_para)

        self.corr =  np.max(np.convolve(self.gauss, self.gauss))


        length_tensor = int(self.max_period / (self.resolution)  * pulses)
        self.pulse_trains = np.zeros((points, length_tensor))

        self.start_per = self.min_period / self.resolution
        self.end_per = self.max_period / self.resolution
        self.periods = np.linspace(self.start_per, self.end_per, points, dtype=int)
        middle = length_tensor // 2
        i=0
        for per in self.periods:
            whole_range = per * (pulses-1)
            start = middle-whole_range//2
            end = start + whole_range
            pulse_index = np.linspace(start, end, pulses, dtype=int)
            #self.pulse_trains[i,pulse_index] = 1
            dummy = np.zeros(length_tensor)
            dummy[pulse_index] = 1
            dummy = np.convolve(dummy, self.gauss, mode='same')
            self.pulse_trains[i,:] = dummy

            i += 1

        # plt.imshow(self.pulse_trains, aspect='auto')
        # plt.show()
        self.pulse_trains = torch.Tensor(self.pulse_trains).cuda().unsqueeze(1)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.lin_classifier = nn.Sequential(
            nn.Linear(points, points//2),
            nn.LeakyReLU(),
            nn.Linear(points//2, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 2))




    def forward(self, x):
        convolved = F.conv1d(x, self.pulse_trains)
        pooled = self.pool(convolved).squeeze() - self.corr
        output = self.lin_classifier(pooled)


        return output
