import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class regressor_simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.lin = nn.Linear(1, 2)

    def forward(self, x):
        pooled = self.pool(x)[:,0,:]
        output = self.lin(pooled)

        return output
