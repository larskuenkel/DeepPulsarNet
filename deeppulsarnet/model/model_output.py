import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn import functional as F
import torch
from model.TemporalBlock import TemporalBlock


class OutputLayer(nn.Module):
    def __init__(self, input_channels, mode, intermediate, final_nonlin, channel_rnn, layer_rnn, direct_class, 
        dropout=0, kernel=19, residual=False, output_channels=1):
        super().__init__()
        self.mode = mode
        self.layer_rnn = layer_rnn
        layers = []
        layers_2 = []
        self.input_channels = input_channels
        self.use_direct_class = 0
        self.out_chan = channel_rnn
        self.output_channels = output_channels
        if mode == 0:
            if intermediate>1 and kernel!=0:
                layers += [nn.Dropout2d(dropout),
                           TemporalBlock(input_channels, intermediate, kernel, stride=1, dilation=1,
                                         groups=1, residual=residual, final_norm=False),
                           nn.Conv1d(intermediate, output_channels, 1)]
            elif intermediate>1:
                layers += [nn.Dropout2d(dropout),
                nn.Conv1d(input_channels, intermediate, 1),
                nn.LeakyReLU(),
                nn.Conv1d(intermediate, output_channels, 1)]
            else:
                layers += [nn.Dropout2d(dropout),
                nn.Conv1d(input_channels, output_channels, 1)]
        elif mode == 1 or mode == 2:
            if direct_class:
                self.use_direct_class = 1
                self.in_class_chan = mode * 2
                self.out_chan += 2
            # else:
            #     self.out_chan = 1
            if mode == 1:
                bi = False
            elif mode == 2:
                bi = True
            layers += [nn.Dropout2d(dropout)]
            if intermediate != 0:
                if not intermediate == input_channels:
                    layers += [nn.Conv1d(input_channels, intermediate, 1)]
            else:
                intermediate = input_channels
            self.rnn = nn.LSTM(intermediate, self.out_chan,
                               self.layer_rnn, bidirectional=bi)
            if bi:
                layers_2 += [nn.Conv1d(channel_rnn * 2, output_channels, 1)]
            else:
                layers_2 += [nn.Conv1d(channel_rnn, output_channels, 1)]
            if self.use_direct_class:
                if bi:
                    self.early_class = nn.Linear(self.in_class_chan, 2)

        layers_2 += [nn.LeakyReLU(),
                     # nn.Conv1d(1, 1, 1, bias=False)
                     ]

        # layers_2[-1].weight.data.fill_(1)
        if final_nonlin:
            layers_2 += [nn.Tanh()]

        self.network_1 = nn.Sequential(*layers)
        self.network_2 = nn.Sequential(*layers_2)

    def forward(self, x):
        out_1 = self.network_1(x)
        if self.mode:
            out_2, _ = self.rnn(out_1.permute(0, 2, 1), None)
            out_2 = out_2.permute(0, 2, 1)
            if not hasattr(self, 'use_direct_class'):
                out_3 = self.network_2(out_2)
                return out_3

            if not self.use_direct_class:
                out_3 = self.network_2(out_2)
                return out_3
            else:
                if self.mode == 2:
                    # in_series = torch.cat((out_2[:,:1,:],out_2[:,self.out_chan:self.out_chan+1,:]), dim=1)
                    out_3 = self.network_2(out_2[:, 2:-2, :])
                    in_class = torch.cat(
                        (out_2[:, 0:2, -1], out_2[:, -2:, 0]), dim=1)
                    out_class = self.early_class(in_class)
                    return out_3, out_class
                else:
                    out_3 = self.network_2(out_2[:, 2:, :])
                    out_class = self.early_class(out_2[:, :2, -1])
                    return out_3, out_class
        else:
            out_2 = out_1
            out_3 = self.network_2(out_2)
            return out_3

        # out_3 = self.network_2(out_2)
        # if self.use_direct_class:
        #     return out_3[:, :1, :], out_3[:, 1:, :]
        # else:
        #     return out_3, _
