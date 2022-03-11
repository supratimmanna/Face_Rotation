# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 12:13:33 2021

@author: supratim
"""
import torch
import torch.nn as nn
import numpy as np
#from modelsummary import summary
from utils import *
#import utils

# IMG_WHT=128
# ETA = 0.001
######################### Image Generator ###############################
class Generator(nn.Module):
    def __init__(self, IMG_WHT, ETA):
        super(Generator, self).__init__()
        
        ##ENCODER##

        
        f = np.array([64, 64, 128, 256, 512]) #/ 2
        
        f = f.astype(int)
        #print(f[0],f[1],f[2],f[3],f[4])

        
        batch_norm = False
        self.conv0 = nn.Sequential(conv(5, f[0], 7, 1, calc_conv_pad(IMG_WHT, IMG_WHT, 7, 1), "xavier_normal", batch_norm, nn.ReLU(ETA)))#, # 128 x 128
                                   #ResidualBlock(f[0], activation = nn.ReLU(ETA))) 
        self.conv1 = nn.Sequential(conv(f[0], f[1], 5, 2, calc_conv_pad(IMG_WHT, IMG_WHT / 2, 5, 2), "xavier_normal",  batch_norm, nn.ReLU(ETA)))#, # 64 x 64
                                   #ResidualBlock(f[1], activation = nn.ReLU(ETA)))
        self.conv2 = nn.Sequential(conv(f[1], f[2], 3, 2, calc_conv_pad(IMG_WHT / 2, IMG_WHT / 4, 3, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)))#,  # 32 x 32
                                   #ResidualBlock(f[2], activation = nn.ReLU(ETA))) 
        self.conv3 = nn.Sequential(conv(f[2], f[3], 3, 2, calc_conv_pad(IMG_WHT / 4, IMG_WHT / 8, 3, 2), "xavier_normal",  batch_norm, nn.ReLU(ETA)))#, # 16 x 16
                                   #ResidualBlock(f[3], activation = nn.ReLU(ETA))) 
        self.conv4 = nn.Sequential(conv(f[3], f[4], 3, 2, calc_conv_pad(IMG_WHT / 8, IMG_WHT / 16, 3, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)))#, # 8 x 8
                                   #ResidualBlock(f[4], activation = nn.ReLU(ETA))) 
      
        self.fc1 = nn.Linear(f[4] * 8 * 8, f[4])
        self.relu = nn.ReLU(inplace=True)
        self.maxout = nn.MaxPool1d(2 )
        ##DECODER##
        
        self.fc2 = nn.Linear(f[3], f[1] * 8 * 8) 
        
        f = np.array([64, 32, 16, 8]) 
        f = f.astype(int)

        
        self.dc0_1 = deconv(f[0], f[1], 4, 4, calc_deconv_pad(IMG_WHT / 16, IMG_WHT / 4, 4, 4), "xavier_normal", batch_norm, activation=nn.ReLU(ETA)) # 32 x 32
        self.dc0_2 = deconv(f[1], f[2], 2, 2, calc_deconv_pad(IMG_WHT / 4, IMG_WHT / 2, 2, 2), "xavier_normal", batch_norm, activation=nn.ReLU(ETA)) # 64 x 64
        self.dc0_3 = deconv(f[2], f[3], 2, 2, calc_deconv_pad(IMG_WHT / 2, IMG_WHT, 2, 2), "xavier_normal", batch_norm, activation=nn.ReLU(ETA)) # 128 x 128

        f = np.array([512, 256, 128, 64, 32, 16, 8])
        f = f.astype(int)

        
        self.dc1 = nn.Sequential(deconv(f[0] + f[3], f[0], 2, 2, calc_deconv_pad(IMG_WHT / 16, IMG_WHT / 8, 2, 2),"xavier_normal", batch_norm, activation=nn.ReLU(ETA)), # 16 x 16
                                 ResidualBlock(f[0], activation = nn.ReLU(ETA)),
                                 ResidualBlock(f[0], activation = nn.ReLU(ETA)))
        self.dc2 = nn.Sequential(deconv(f[0] + f[1], f[1], 2, 2, calc_deconv_pad(IMG_WHT / 8, IMG_WHT / 4, 2, 2),"xavier_normal", batch_norm, activation=nn.ReLU(ETA)), # 32 x 32
                                 ResidualBlock(f[1], activation = nn.ReLU(ETA)),
                                 ResidualBlock(f[1], activation = nn.ReLU(ETA)))
        self.dc3 = nn.Sequential(deconv(f[2] + f[1] + 3 + f[4], f[2], 2, 2, calc_deconv_pad(IMG_WHT / 4, IMG_WHT / 2, 2, 2), "xavier_normal", batch_norm, activation=nn.ReLU(ETA)), # 64 x 64
                                 ResidualBlock(f[2], activation = nn.ReLU(ETA)),
                                 ResidualBlock(f[2], activation = nn.ReLU(ETA)))
        self.dc4 = nn.Sequential(deconv(f[2] + f[3] + 3 + f[5], f[3], 2, 2, calc_deconv_pad(IMG_WHT / 2, IMG_WHT, 2, 2), "xavier_normal", batch_norm, activation=nn.ReLU(ETA)), # 128 x 128
                                 ResidualBlock(f[3], activation = nn.ReLU(ETA)),
                                 ResidualBlock(f[3], activation = nn.ReLU(ETA)))

        #final convs
        
        self.conv5 = conv(f[1], 3, 3, 1, calc_conv_pad(IMG_WHT / 4, IMG_WHT / 4, 3, 1), "xavier_normal", batch_norm, nn.ReLU(ETA)) # 32 x 32
        self.conv6 = conv(f[2], 3, 3, 1, calc_conv_pad(IMG_WHT / 2, IMG_WHT / 2, 3, 1), "xavier_normal", batch_norm, nn.ReLU(ETA)) # 64 x 64
        self.conv7 = conv(f[3] + f[3] + 3 + f[6], f[3], 5, 1, calc_conv_pad(IMG_WHT, IMG_WHT, 5, 1), "xavier_normal", batch_norm, nn.ReLU(ETA)) # 128 x 128
        self.conv8 = conv(f[3], f[4], 3, 1, calc_conv_pad(IMG_WHT, IMG_WHT, 3, 1), "xavier_normal", batch_norm, nn.ReLU(ETA)) # 128 x 128
        self.conv9 = conv(f[4], 3, 3, 1, calc_conv_pad(IMG_WHT, IMG_WHT, 3, 1), "xavier_normal", batch_norm, nn.ReLU(ETA)) # 128 x 128
        
    def forward(self, picture, landmarks_real, landmarks_wanted): #img = x
        #Ecoder
        x = torch.cat([picture, landmarks_real, landmarks_wanted], dim = 1)
        c0 = self.conv0(x)
        c1 = self.conv1(c0)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        tmp = self.num_flat_features(c4)
        f1 = c4.view(x.size()[0], tmp)
        f1 = self.fc1(f1)
        f1 = self.relu(f1)
        f1 = f1.unsqueeze(0)
        maxout = self.maxout(f1)[0]
        
        #Decoder
        #1
        f2 = self.fc2(maxout)
        rsh = f2.reshape((x.size()[0], int(64), 8, 8))
        
        dc01 = self.dc0_1(rsh)
        
        dc02 = self.dc0_2(dc01)

        #2
        dc1r = self.dc1(torch.cat((rsh, c4), dim=1))
        dc2r = self.dc2(torch.cat((dc1r, c3), dim=1))
        pic_div_2 = nn.MaxPool2d(2)(picture)
        pic_div_4 = nn.MaxPool2d(2)(pic_div_2)
        dc3r = self.dc3(torch.cat((dc2r, c2, pic_div_4, dc01), dim=1))
        dc4r = self.dc4(torch.cat((dc3r, c1, pic_div_2, dc02), dim=1))

        #3
        c5 = self.conv5(dc2r)
        c6 = self.conv6(dc3r)
        c7 = self.conv7(torch.cat((dc4r, c0, picture, dc03), dim=1))
        c8 = self.conv8(c7)
        c9 = self.conv9(c8)
        
        return c5, c6, c9  #img_32, img_64, img_128
        #return picture, nn.MaxPool2d(2)(picture), nn.MaxPool2d(2)(nn.MaxPool2d(2)(picture))
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
##################### Face Discriminator #############################
class Discriminator_faces(nn.Module):
    def __init__(self, IMG_WHT, ETA):
        super(Discriminator_faces, self).__init__()
        batch_norm = False
        self.model = nn.Sequential(conv(6, 64, 4, 2, calc_conv_pad(IMG_WHT, IMG_WHT / 2, 4, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 64 x 64
                                   conv(64, 128, 4, 2, calc_conv_pad(IMG_WHT / 2, IMG_WHT / 4, 4, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 32 x 32
                                   conv(128, 256, 4, 2, calc_conv_pad(IMG_WHT / 4, IMG_WHT / 8, 4, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 16 x 16
                                   conv(256, 512, 4, 2, calc_conv_pad(IMG_WHT / 8, IMG_WHT / 16, 4, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 8 x 8
                                   conv(512, 512, 4, 1, calc_conv_pad(IMG_WHT / 16, 7, 4, 1), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 7 x 7
                                   conv(512, 1, 4, 1, calc_conv_pad(7, 6, 4, 1), "xavier_normal", False, None)) # 6 x 6
    def forward(self, x):
        return self.model(x)


##################### Face-Landmark Discriminator #############################
class Discriminator_marks(nn.Module):
    def __init__(self, IMG_WHT, ETA):
        super(Discriminator_marks, self).__init__()
        batch_norm = False
        self.model = nn.Sequential(conv(4, 64, 4, 2, calc_conv_pad(IMG_WHT, IMG_WHT / 2, 4, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 64 x 64
                                   conv(64, 128, 4, 2, calc_conv_pad(IMG_WHT / 2, IMG_WHT / 4, 4, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 32 x 32
                                   conv(128, 256, 4, 2, calc_conv_pad(IMG_WHT / 4, IMG_WHT / 8, 4, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 16 x 16
                                   conv(256, 512, 4, 2, calc_conv_pad(IMG_WHT / 8, IMG_WHT / 16, 4, 2), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 8 x 8
                                   conv(512, 512, 4, 1, calc_conv_pad(IMG_WHT / 16, 7, 4, 1), "xavier_normal", batch_norm, nn.ReLU(ETA)), # 7 x 7
                                   conv(512, 1, 4, 1, calc_conv_pad(7, 6, 4, 1), "xavier_normal", False, None)) # 6 x 6
    def forward(self, x):
        return self.model(x)