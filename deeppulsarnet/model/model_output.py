import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F
import torch
from model.TemporalBlock import TemporalBlock


class OutputLayer(nn.Module):
    def __init__(self, input_channels, intermediate, final_nonlin,
                 dropout=0, kernel=19, residual=True, output_channels=1):
        super().__init__()
        layers = []
        layers_2 = []
        self.input_channels = input_channels
        self.output_channels = output_channels
        if intermediate > 1 and kernel != 0:
            layers += [nn.Dropout2d(dropout),
                       TemporalBlock(input_channels, intermediate, kernel, stride=1, dilation=1,
                                     groups=1, residual=residual, final_norm=False),
                       nn.Conv1d(intermediate, output_channels, 1)]
        elif intermediate > 1:
            layers += [nn.Dropout2d(dropout),
                       nn.Conv1d(input_channels, intermediate, 1),
                       nn.LeakyReLU(),
                       nn.Conv1d(intermediate, output_channels, 1)]
        else:
            layers += [nn.Dropout2d(dropout),
                       nn.Conv1d(input_channels, output_channels, 1)]

        layers_2 += [nn.LeakyReLU(),
                     # nn.Conv1d(1, 1, 1, bias=False)
                     ]

        if final_nonlin:
            layers_2 += [nn.Tanh()]

        self.network_1 = nn.Sequential(*layers)
        self.network_2 = nn.Sequential(*layers_2)

    def forward(self, x):
        out_1 = self.network_1(x)
        out_2 = self.network_2(out_1)
        return out_2
