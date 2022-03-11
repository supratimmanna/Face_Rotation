# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:56:40 2022

@author: supratim
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout),
                            nn.ReLU()
                            )
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return out
    
class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, cin, cout, kernel_size, stride, padding, *args, **kwargs):
        super(DownBlock2d, self).__init__()
        
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2, stride=2)
                            )

    def forward(self, x):
        out = self.conv_block(x)
        return out
    

class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, cin, cout, kernel_size, stride, padding):
        super(UpBlock2d, self).__init__()
        
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout),
                            nn.ReLU()
                            )

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv_block(out)
        return out