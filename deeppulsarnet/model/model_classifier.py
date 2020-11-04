import torch.nn as nn
import torch


class classifier(nn.Module):
    # Simple multi layer perceptron that is used to analyze the decoded data.
    # List channels defines the hidden layers
    def __init__(self, n_inputs, n_outputs, channel_list, pool_lin=0):
        super().__init__()
        layers = []
        levels = len(channel_list)
        for i in range(levels):
            in_channels = n_inputs if i == 0 else channel_list[i - 1]
            out_channels = channel_list[i]
            layers += [nn.Linear(in_channels, out_channels), nn.ReLU()]
        # Final layer for regression
        layers += [nn.Linear(channel_list[-1], n_outputs)]
        if pool_lin:
            self.pool = nn.AdaptiveMaxPool1d(pool_lin)
        self.pool_max = nn.AdaptiveMaxPool1d(3)
        self.pool_mean = nn.AdaptiveAvgPool1d(3)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten input first
        if hasattr(self, 'pool'):
            x = self.pool(x)
        max_vals = self.pool_max(x).view(x.size(0), -1)
        mean_vals = self.pool_mean(x).view(x.size(0), -1)
        concat_tensor = torch.cat(
            (x.view(x.size(0), -1), max_vals, mean_vals), dim=1)
        return self.net(concat_tensor)
        # return (x.view(x.size(0), -1)
