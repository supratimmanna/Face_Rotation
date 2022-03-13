# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 21:47:14 2022

@author: supratim
"""
import os
import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision.transforms.functional as F
from Light_CNN.light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2

model_path = 'Light_CNN/LightCNN_29Layers_V2_checkpoint.pth.tar'

model = LightCNN_29Layers_v2(num_classes=79077)

class Feature_Extractor:
    def __init__(self, device='cpu', model_path=model_path):
        super(Feature_Extractor, self).__init__()
        
        self.model_path = model_path
        self.model = LightCNN_29Layers_v2(num_classes=79077)
        self.model.eval()
        self.model = torch.nn.DataParallel(model).to(device)
        if os.path.exists(self.model_path):
            print('Loading light cnn model')
            #checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            if device == 'cpu':
              checkpoint = torch.load(model_path,map_location=torch.device('cpu'))
            else:
              checkpoint = torch.load(model_path)
            #new_state_dict = OrderedDict()
            #for k, v in checkpoint['state_dict'].items():
             # name = k[7:]
              #new_state_dict[name] = v
            #model.load_state_dict(new_state_dict, strict=False)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            print('no saved model for light cnn')
    
    def forward(self, img):
        img = F.rgb_to_grayscale(img)
        fc_feat, conv_feat = self.model(img)
        return fc_feat, conv_feat

