import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class regressor_acf(nn.Module):
    def __init__(self, acf_class, no_reg):
        super().__init__()
        self.acf_pad = acf_class[1]
        self.acf_size = acf_class[1] + 1
        self.pool_size = acf_class[2]
        self.acf_size_pooled = int(np.floor(self.acf_size/self.pool_size))
        self.lin_channels = acf_class[3:]
        self.no_reg = no_reg
        if self.no_reg:
            self.final_output = 2
        else:
            self.final_output = 4

        self.pool = nn.MaxPool1d(self.pool_size)
        lin_layers = []
        lin_input = self.acf_size_pooled
        for lin_layer in self.lin_channels:
            lin_layers += [nn.Dropout(0.1),
            nn.Linear(lin_input, lin_layer), nn.LeakyReLU(),]
            lin_input = lin_layer
        lin_layers += [nn.Linear(lin_input, self.final_output)]


        self.lin_class = nn.Sequential(*lin_layers)



    def forward(self, x):
        # x_mean = torch.mean(x, dim=2)
        # x = x - x_mean[:,:,None]
        x_reshaped =x.permute(1, 0, 2)
        tensor_acf = F.conv1d(x_reshaped, x,
                              padding=self.acf_pad, groups=x.shape[0])
        tensor_acf_reshaped = tensor_acf.permute(1,0,2)[:,:,self.acf_pad:]

        tensor_acf_pooled = self.pool(tensor_acf_reshaped)[:,0,:]
        # tensor_acf_pooled = F.dropout(tensor_acf_pooled, 0.2)

        acf_max = torch.max(tensor_acf_pooled,dim=1, keepdim=True)[0]
        acf_norm = tensor_acf_pooled / acf_max
        output = self.lin_class(acf_norm)

        return output