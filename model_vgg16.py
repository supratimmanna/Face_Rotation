# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:39:06 2022

@author: supratim
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from utils import  kp2gaussian, make_coordinate_grid
import face_alignment
from conv_block import Conv2d, DownBlock2d, UpBlock2d

################## Basic generator ################################
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.LAND_MARKS_INDEXES = [36, 39, 42, 45, 31, 35, 48, 51, 54, 57]
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=self.device)
        
        self.face_encoder = nn.ModuleList([ 
            nn.Sequential(Conv2d(cin=3, cout=64, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=64, cout=64, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=64, cout=128, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=128, cout=128, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=128, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
            )])
        
        self.lm_encoder = nn.ModuleList([ 
            nn.Sequential(Conv2d(cin=10, cout=64, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=64, cout=64, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=64, cout=128, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=128, cout=128, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=128, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
            )])
        
        self.face_decoder = nn.ModuleList([
            nn.Sequential(UpBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=128, kernel_size=3, stride=1, padding=1)),
            nn.Sequential(UpBlock2d(cin=128, cout=128, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=128, cout=64, kernel_size=3, stride=1, padding=1)),
            nn.Sequential(UpBlock2d(cin=64, cout=64, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=64, cout=3, kernel_size=3, stride=1, padding=1)
                )])
    
    
    def normalize(self, x):
      NewMin = -1
      NewMax = 1
      OldMin = 0
      OldMax = 127
      NewRange = (NewMax - NewMin)  
      OldRange = (OldMax - OldMin) 
      xnew = NewMin + ((x-OldMin)*NewRange)/OldRange

      return xnew
    
    
    def get_lm(self, img):
        
        batch_size = img.shape[0]
        lm = np.zeros((batch_size, 10, 2),dtype='f')
        
        for j, im in enumerate(img):
          #print('im:',im.shape)
          im = im.permute(1,2,0)
          im = im.cpu().numpy()
          #print(im.type())
          im = (255*im).astype(np.uint8)
          preds=self.fa.get_landmarks(im)
          #print(len(preds))
          if len(preds)>0:
            k=0
            for i,f in enumerate(preds[0]):
                  if i in self.LAND_MARKS_INDEXES:
                    x=f[0]
                    y=f[1]
                    lm[j,k,0] = x
                    lm[j,k,1] = y
                    k+=1
          else:
            print('face not detected in source image')
        #lm = torch.FloatTensor(lm)
        lm = self.normalize(lm)
        lm = transforms.ToTensor()(lm).permute([1, 2, 0])
        lm = lm.to( self.device)
        return lm, im
    
    def create_heatmap_representations(self, src_img, srl_lm, target_lm):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = src_img.shape[2:]
        src_lm_heatmap = kp2gaussian(srl_lm, spatial_size=spatial_size, kp_variance=0.1)
        target_lm_heatmap = kp2gaussian(target_lm, spatial_size=spatial_size, kp_variance=0.1)
        
        return src_lm_heatmap, target_lm_heatmap
        
    def forward(self, src_img, target_img):
        
        src_lm, im = self.get_lm(src_img)
        target_lm, _ = self.get_lm(target_img)
        print('lm shape:',src_lm.shape)
        
        src_heatmap, target_heatmap = self.create_heatmap_representations(src_img, src_lm, target_lm)
 
        ########### source image feature extraction #####
        img_feat = []
        for f in self.face_encoder:
            src_img = f(src_img)
        
        ###### source heat map feature extraction #######
        for f in self.lm_encoder:
            src_heatmap = f(src_heatmap)
            
        ###### target heat map feature extraction ########
        for f in self.lm_encoder:
            target_heatmap = f(target_heatmap)
        
        heatmap = target_heatmap - src_heatmap
        src_img_heatmap = src_img + heatmap

        recn_img = src_img_heatmap
        for f in self.face_decoder:
            recn_img = f(recn_img)

        return recn_img

######## Concatenation of the image and heatmap for reconstruction ##############
class generator_cat(nn.Module):
    def __init__(self):
        super(generator_cat, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.LAND_MARKS_INDEXES = [36, 39, 42, 45, 31, 35, 48, 51, 54, 57]
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=self.device)
        
        self.face_encoder = nn.ModuleList([ 
            nn.Sequential(Conv2d(cin=3, cout=64, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=64, cout=64, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=64, cout=128, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=128, cout=128, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=128, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
            )])
        
        self.lm_encoder = nn.ModuleList([ 
            nn.Sequential(Conv2d(cin=10, cout=64, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=64, cout=64, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=64, cout=128, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=128, cout=128, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=128, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
            )])
        
        
        self.image_hm_encoder = nn.ModuleList([
            nn.Sequential(Conv2d(cin=512, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1)
            )])
        
        
        self.face_decoder = nn.ModuleList([
            nn.Sequential(UpBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=128, kernel_size=3, stride=1, padding=1)),
            nn.Sequential(UpBlock2d(cin=128, cout=128, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=128, cout=64, kernel_size=3, stride=1, padding=1)),
            nn.Sequential(UpBlock2d(cin=64, cout=64, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=64, cout=3, kernel_size=3, stride=1, padding=1)
                )])
    
    
    def normalize(self, x):
      NewMin = -1
      NewMax = 1
      OldMin = 0
      OldMax = 127
      NewRange = (NewMax - NewMin)  
      OldRange = (OldMax - OldMin) 
      xnew = NewMin + ((x-OldMin)*NewRange)/OldRange

      return xnew
    
    
    def get_lm(self, img):
        
        batch_size = img.shape[0]
        lm = np.zeros((batch_size, 10, 2),dtype='f')
        
        for j, im in enumerate(img):
          #print('im:',im.shape)
          im = im.permute(1,2,0)
          im = im.cpu().numpy()
          #print(im.type())
          im = (255*im).astype(np.uint8)
          preds=self.fa.get_landmarks(im)
          #print(len(preds))
          if len(preds)>0:
            k=0
            for i,f in enumerate(preds[0]):
                  if i in self.LAND_MARKS_INDEXES:
                    x=f[0]
                    y=f[1]
                    lm[j,k,0] = x
                    lm[j,k,1] = y
                    k+=1
          else:
            print('face not detected in source image')
        #lm = torch.FloatTensor(lm)
        lm = self.normalize(lm)
        lm = transforms.ToTensor()(lm).permute([1, 2, 0])
        lm = lm.to( self.device)
        return lm, im
    
    def create_heatmap_representations(self, src_img, srl_lm, target_lm):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = src_img.shape[2:]
        src_lm_heatmap = kp2gaussian(srl_lm, spatial_size=spatial_size, kp_variance=0.1)
        target_lm_heatmap = kp2gaussian(target_lm, spatial_size=spatial_size, kp_variance=0.1)
        
        return src_lm_heatmap, target_lm_heatmap
        
    def forward(self, src_img, target_img):
        
        src_lm, im = self.get_lm(src_img)
        target_lm, _ = self.get_lm(target_img)
        print('lm shape:',src_lm.shape)
        
        src_heatmap, target_heatmap = self.create_heatmap_representations(src_img, src_lm, target_lm)
 
        ########### source image feature extraction #####
        img_feat = []
        for f in self.face_encoder:
            src_img = f(src_img)
        
        ###### source heat map feature extraction #######
        for f in self.lm_encoder:
            src_heatmap = f(src_heatmap)
            
        ###### target heat map feature extraction ########
        for f in self.lm_encoder:
            target_heatmap = f(target_heatmap)
        
        heatmap = target_heatmap - src_heatmap
        img_heatmap = torch.cat((src_img, heatmap), 1)
        
        for f in self.image_hm_encoder:
            img_heatmap = f(img_heatmap)

        
        recn_img = img_heatmap + src_img
        for f in self.face_decoder:
            recn_img = f(recn_img)

        return recn_img

######## Concatenation of the image and heatmap for reconstruction ##############
class generator_cat_reg(nn.Module):
    def __init__(self):
        super(generator_cat_reg, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.LAND_MARKS_INDEXES = [36, 39, 42, 45, 31, 35, 48, 51, 54, 57]
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=self.device)
        
        self.face_encoder = nn.ModuleList([ 
            nn.Sequential(Conv2d(cin=3, cout=64, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=64, cout=64, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=64, cout=128, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=128, cout=128, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=128, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
            )])
        
        self.lm_encoder = nn.ModuleList([ 
            nn.Sequential(Conv2d(cin=10, cout=64, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=64, cout=64, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=64, cout=128, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=128, cout=128, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=128, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
            )])
        
        
        self.image_hm_encoder = nn.ModuleList([
            nn.Sequential(Conv2d(cin=512, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1)
            )])
        
        
        self.face_decoder = nn.ModuleList([
            nn.Sequential(UpBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=128, kernel_size=3, stride=1, padding=1)),
            nn.Sequential(UpBlock2d(cin=128, cout=128, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=128, cout=64, kernel_size=3, stride=1, padding=1)),
            nn.Sequential(UpBlock2d(cin=64, cout=64, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=64, cout=3, kernel_size=3, stride=1, padding=1)
                )])
    
    
    def normalize(self, x):
      NewMin = -1
      NewMax = 1
      OldMin = 0
      OldMax = 127
      NewRange = (NewMax - NewMin)  
      OldRange = (OldMax - OldMin) 
      xnew = NewMin + ((x-OldMin)*NewRange)/OldRange

      return xnew
    
    
    def get_lm(self, img):
        
        batch_size = img.shape[0]
        lm = np.zeros((batch_size, 10, 2),dtype='f')
        
        for j, im in enumerate(img):
          #print('im:',im.shape)
          im = im.permute(1,2,0)
          im = im.cpu().numpy()
          #print(im.type())
          im = (255*im).astype(np.uint8)
          preds=self.fa.get_landmarks(im)
          #print(len(preds))
          if len(preds)>0:
            k=0
            for i,f in enumerate(preds[0]):
                  if i in self.LAND_MARKS_INDEXES:
                    x=f[0]
                    y=f[1]
                    lm[j,k,0] = x
                    lm[j,k,1] = y
                    k+=1
          else:
            print('face not detected in source image')
        #lm = torch.FloatTensor(lm)
        lm = self.normalize(lm)
        lm = transforms.ToTensor()(lm).permute([1, 2, 0])
        lm = lm.to( self.device)
        return lm, im
    
    def create_heatmap_representations(self, src_img, srl_lm, target_lm):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = src_img.shape[2:]
        src_lm_heatmap = kp2gaussian(srl_lm, spatial_size=spatial_size, kp_variance=0.1)
        target_lm_heatmap = kp2gaussian(target_lm, spatial_size=spatial_size, kp_variance=0.1)
        
        return src_lm_heatmap, target_lm_heatmap
        
    def forward(self, src_img, target_img):
        
        src_lm, im = self.get_lm(src_img)
        target_lm, _ = self.get_lm(target_img)
        print('lm shape:',src_lm.shape)
        
        src_heatmap, target_heatmap = self.create_heatmap_representations(src_img, src_lm, target_lm)
 
        ########### source image feature extraction #####
        for f in self.face_encoder:
            src_img = f(src_img)

        ########### source image feature extraction #####
        for f in self.face_encoder:
            target_img = f(target_img)
        
        ###### feature difference between source image and targte image for regularization purpose ########
        img_feat_diff =  target_img - src_img

        ###### source heat map feature extraction #######
        for f in self.lm_encoder:
            src_heatmap = f(src_heatmap)
            
        ###### target heat map feature extraction ########
        for f in self.lm_encoder:
            target_heatmap = f(target_heatmap)
        
        heatmap = target_heatmap - src_heatmap
        img_heatmap = torch.cat((src_img, heatmap), 1)
        
        for f in self.image_hm_encoder:
            img_heatmap = f(img_heatmap)

        
        recn_img = img_heatmap + src_img
        for f in self.face_decoder:
            recn_img = f(recn_img)

        return recn_img, img_feat_diff, img_heatmap

############# generator with residual connection with the original image ###############
class generator_image_residual(nn.Module):
    def __init__(self):
        super(generator_image_residual, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.LAND_MARKS_INDEXES = [36, 39, 42, 45, 31, 35, 48, 51, 54, 57]
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=self.device)
        
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        # self.layer1_lm = vgg_conv_block([10,64], [64,64], [3,3], [1,1], 2, 2)
        # self.layer1 = vgg_conv_block([3,64], [64,64], [3,3], [1,1], 2, 2)
        # self.layer2 = vgg_conv_block([64,128], [128,128], [3,3], [1,1], 2, 2)
        # self.layer3 = vgg_conv_block([128,256,256], [256,256,256], [3,3,3], [1,1,1], 2, 2)
        
        self.face_encoder = nn.ModuleList([ 
            nn.Sequential(Conv2d(cin=3, cout=64, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=64, cout=64, kernel_size=3, stride=1, padding=1)), ## (1 X 64 X 64X 64)
            nn.Sequential(Conv2d(cin=64, cout=128, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=128, cout=128, kernel_size=3, stride=1, padding=1)), ## (1 X 128 X 32 32)
            nn.Sequential(Conv2d(cin=128, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1), ## (1 X 256 X 16 X 16)
            )])
        
        self.lm_encoder = nn.ModuleList([ 
            nn.Sequential(Conv2d(cin=10, cout=64, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=64, cout=64, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=64, cout=128, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=128, cout=128, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=128, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
            )])
        
        self.face_decoder = nn.ModuleList([
            nn.Sequential(UpBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1), ## (1 X 256 X 32 X 32)
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=128, kernel_size=3, stride=1, padding=1)), ## (1 X 128 X 32 X 32)
            nn.Sequential(UpBlock2d(cin=256, cout=128, kernel_size=3, stride=1, padding=1), ## (1 X 128 X 64X 64)
                          Conv2d(cin=128, cout=64, kernel_size=3, stride=1, padding=1)),  ## (1 X 64 X 64X 64)
            nn.Sequential(UpBlock2d(cin=128, cout=64, kernel_size=3, stride=1, padding=1), ## (1 X 64 X 128X 128)
                          Conv2d(cin=64, cout=3, kernel_size=3, stride=1, padding=1)  ## (1 X 3 X 128 X 128)
                )])
    
    
    def normalize(self, x):
      NewMin = -1
      NewMax = 1
      OldMin = 0
      OldMax = 127
      NewRange = (NewMax - NewMin)  
      OldRange = (OldMax - OldMin) 
      xnew = NewMin + ((x-OldMin)*NewRange)/OldRange

      return xnew
    
    
    def get_lm(self, img):
        
        batch_size = img.shape[0]
        lm = np.zeros((batch_size, 10, 2),dtype='f')
        
        for j, im in enumerate(img):
          #print('im:',im.shape)
          im = im.permute(1,2,0)
          im = im.cpu().numpy()
          #print(im.type())
          im = (255*im).astype(np.uint8)
          preds=self.fa.get_landmarks(im)
          #print(len(preds))
          if len(preds)>0:
            k=0
            for i,f in enumerate(preds[0]):
                  if i in self.LAND_MARKS_INDEXES:
                    x=f[0]
                    y=f[1]
                    lm[j,k,0] = x
                    lm[j,k,1] = y
                    k+=1
          else:
            print('face not detected in source image')
        #lm = torch.FloatTensor(lm)
        lm = self.normalize(lm)
        lm = transforms.ToTensor()(lm).permute([1, 2, 0])
        lm = lm.to( self.device)
        return lm, im
    
    def create_heatmap_representations(self, src_img, srl_lm, target_lm):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = src_img.shape[2:]
        src_lm_heatmap = kp2gaussian(srl_lm, spatial_size=spatial_size, kp_variance=0.1)
        target_lm_heatmap = kp2gaussian(target_lm, spatial_size=spatial_size, kp_variance=0.1)
        
        return src_lm_heatmap, target_lm_heatmap
        
    def forward(self, src_img, target_img):
        
        src_lm, im = self.get_lm(src_img)
        target_lm, _ = self.get_lm(target_img)
        
        src_heatmap, target_heatmap = self.create_heatmap_representations(src_img, src_lm, target_lm)
 
        ########### source image feature extraction #################
        img_feat = []
        for f in self.face_encoder:
            src_img = f(src_img)
            img_feat.append(src_img)
        
        ########## source heat map feature extraction ###############
        for f in self.lm_encoder:
            src_heatmap = f(src_heatmap)

        ########### target heat map feature extraction ###############
        
        for f in self.lm_encoder:
            target_heatmap = f(target_heatmap)
        
        heatmap = target_heatmap - src_heatmap
        src_img_heatmap = src_img + heatmap

        res = img_feat.pop()
        recn_img = src_img_heatmap
        for i,f in enumerate(self.face_decoder):
          if i!=2:
            print(i)
            res = img_feat.pop()
            recn_img = f(recn_img)
            recn_img = torch.cat((recn_img, res),1)
          else:
            recn_img = f(recn_img)

        return recn_img

#### Another type of generator residual connection with (src_img+heatmap_diff) ################################
class generator_heatmap_residual(nn.Module):
    def __init__(self):
        super(generator_heatmap_residual, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.LAND_MARKS_INDEXES = [36, 39, 42, 45, 31, 35, 48, 51, 54, 57]
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=self.device)

        
        self.face_encoder = nn.ModuleList([ 
            nn.Sequential(Conv2d(cin=3, cout=64, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=64, cout=64, kernel_size=3, stride=1, padding=1)), ## (1 X 64 X 64X 64)
            nn.Sequential(Conv2d(cin=64, cout=128, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=128, cout=128, kernel_size=3, stride=1, padding=1)), ## (1 X 128 X 32 32)
            nn.Sequential(Conv2d(cin=128, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1), ## (1 X 256 X 16 X 16)
            )])
        
        self.lm_encoder = nn.ModuleList([ 
            nn.Sequential(Conv2d(cin=10, cout=64, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=64, cout=64, kernel_size=3, stride=1, padding=1)), ## (1 X 64 X 64X 64)
            nn.Sequential(Conv2d(cin=64, cout=128, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=128, cout=128, kernel_size=3, stride=1, padding=1)), ## (1 X 128 X 32 32)
            nn.Sequential(Conv2d(cin=128, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1), ## (1 X 256 X 16 X 16)
            )])
        
        self.face_decoder = nn.ModuleList([
            nn.Sequential(UpBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1), ## (1 X 256 X 32 X 32)
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=128, kernel_size=3, stride=1, padding=1)), ## (1 X 128 X 32 X 32)
            nn.Sequential(UpBlock2d(cin=256, cout=128, kernel_size=3, stride=1, padding=1), ## (1 X 128 X 64X 64)
                          Conv2d(cin=128, cout=64, kernel_size=3, stride=1, padding=1)),  ## (1 X 64 X 64X 64)
            nn.Sequential(UpBlock2d(cin=128, cout=64, kernel_size=3, stride=1, padding=1), ## (1 X 64 X 128X 128)
                          Conv2d(cin=64, cout=3, kernel_size=3, stride=1, padding=1)  ## (1 X 3 X 128 X 128)
                )])
    
    
    def normalize(self, x):
      NewMin = -1
      NewMax = 1
      OldMin = 0
      OldMax = 127
      NewRange = (NewMax - NewMin)  
      OldRange = (OldMax - OldMin) 
      xnew = NewMin + ((x-OldMin)*NewRange)/OldRange

      return xnew
    
    
    def get_lm(self, img):
        
        batch_size = img.shape[0]
        target_lm = np.zeros((batch_size, 10, 2))
        
        for j, im in enumerate(img):
          im = im.squeeze(0)
          im = im.permute(1,2,0)
          im = im.cpu().numpy()
          im = (255 * im).astype(np.uint8)
          preds=self.fa.get_landmarks(im)
          if len(preds)>0:
            k=0
            for i,f in enumerate(preds[0]):
                  if i in self.LAND_MARKS_INDEXES:
                    x=f[0]
                    y=f[1]
                    target_lm[j,k,0] = x
                    target_lm[j,k,1] = y
                    k+=1
          else:
            print('face not detected in source image')
        target_lm = self.normalize(target_lm)
        target_lm = torch.FloatTensor(target_lm)
        target_lm = target_lm.to( self.device)
        return target_lm, im
    
    def create_heatmap_representations(self, src_img, srl_lm, target_lm):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = src_img.shape[2:]
        src_lm_heatmap = kp2gaussian(srl_lm, spatial_size=spatial_size, kp_variance=0.1)
        target_lm_heatmap = kp2gaussian(target_lm, spatial_size=spatial_size, kp_variance=0.1)
        
        return src_lm_heatmap, target_lm_heatmap
        
    def forward(self, src_img, target_img):
        
        src_lm, im = self.get_lm(src_img)
        target_lm, _ = self.get_lm(target_img)
        src_heatmap, target_heatmap = self.create_heatmap_representations(src_img, src_lm, target_lm)
 
        ########### source image feature extraction #################
        img_feat = []
        for f in self.face_encoder:
            src_img = f(src_img)
            img_feat.append(src_img)
        ########## source heat map feature extraction ###############
        src_lm_feat = []
        for f in self.lm_encoder:
            src_heatmap = f(src_heatmap)
            src_lm_feat.append(src_heatmap)

        ########### target heat map feature extraction ###############
        target_lm_feat = []
        for f in self.lm_encoder:
            target_heatmap = f(target_heatmap)
            target_lm_feat.append(target_heatmap)

        heatmap_diff=[]
        for i in range(len(target_lm_feat)):
          heatmap_diff.append(target_lm_feat[i] - src_lm_feat[i] )#+ img_feat[i])
        

        recn_img1 = heatmap_diff.pop()
        recn_img = img_feat.pop()
        for i,f in enumerate(self.face_decoder):
          if i!=2:
            print(i)
            res = heatmap_diff.pop()
            recn_img = f(recn_img)
            recn_img = torch.cat((recn_img, res),1)
            print(recn_img.shape)
          else:
            recn_img = f(recn_img)

        return recn_img


#######################################################################################
class generator1(nn.Module):
    def __init__(self):
        super(generator1, self).__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.LAND_MARKS_INDEXES = [36, 39, 42, 45, 31, 35, 48, 51, 54, 57]
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=self.device)
        
        self.face_encoder = nn.ModuleList([ 
            nn.Sequential(Conv2d(cin=3, cout=64, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=64, cout=64, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=64, cout=128, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=128, cout=128, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=128, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
            )])
        
        self.lm_encoder = nn.ModuleList([ 
            nn.Sequential(Conv2d(cin=11, cout=64, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=64, cout=64, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=64, cout=128, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=128, cout=128, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=128, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          DownBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
            )])
        
        self.face_decoder = nn.ModuleList([
            nn.Sequential(UpBlock2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=256, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=256, cout=128, kernel_size=3, stride=1, padding=1)),
            nn.Sequential(UpBlock2d(cin=128, cout=128, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=128, cout=64, kernel_size=3, stride=1, padding=1)),
            nn.Sequential(UpBlock2d(cin=64, cout=64, kernel_size=3, stride=1, padding=1),
                          Conv2d(cin=64, cout=3, kernel_size=3, stride=1, padding=1)
                )])
    
    
    def normalize(self, x):
      NewMin = -1
      NewMax = 1
      OldMin = 0
      OldMax = 127
      NewRange = (NewMax - NewMin)  
      OldRange = (OldMax - OldMin) 
      xnew = NewMin + ((x-OldMin)*NewRange)/OldRange

      return xnew
    
    
    def get_lm(self, img):
        
        batch_size = img.shape[0]
        lm = np.zeros((batch_size, 10, 2),dtype='f')
        
        for j, im in enumerate(img):
          #print('im:',im.shape)
          im = im.permute(1,2,0)
          im = im.cpu().numpy()
          #print(im.type())
          im = (255*im).astype(np.uint8)
          preds=self.fa.get_landmarks(im)
          #print(len(preds))
          if len(preds)>0:
            k=0
            for i,f in enumerate(preds[0]):
                  if i in self.LAND_MARKS_INDEXES:
                    x=f[0]
                    y=f[1]
                    lm[j,k,0] = x
                    lm[j,k,1] = y
                    k+=1
          else:
            print('face not detected in source image')
        #lm = torch.FloatTensor(lm)
        lm = self.normalize(lm)
        lm = transforms.ToTensor()(lm).permute([1, 2, 0])
        lm = lm.to( self.device)
        return lm, im
    
    def create_heatmap_representations(self, src_img, srl_lm, target_lm):
        """
        Eq 6. in the paper H_k(z)
        """
        spatial_size = src_img.shape[2:]
        src_lm_heatmap = kp2gaussian(srl_lm, spatial_size=spatial_size, kp_variance=0.1)
        target_lm_heatmap = kp2gaussian(target_lm, spatial_size=spatial_size, kp_variance=0.1)
        
        return src_lm_heatmap, target_lm_heatmap
        
    def forward(self, src_img, target_img):

        spatial_size = src_img.shape[2:]
        bs = src_img.shape[0]

        src_lm, im = self.get_lm(src_img)
        target_lm, _ = self.get_lm(target_img)
        #print('lm shape:',src_lm.shape)
        
        src_heatmap, target_heatmap = self.create_heatmap_representations(src_img, src_lm, target_lm)
 
        ########### source image feature extraction #####
        img_feat = []
        for f in self.face_encoder:
            src_img = f(src_img)
        
        ###### difference heat map feature extraction #######
        diff_heatmap = target_heatmap - src_heatmap
        zeros = torch.zeros(bs, 1, spatial_size[0], spatial_size[1]).type(diff_heatmap.type())
        diff_heatmap = torch.cat((diff_heatmap,zeros),1)
        print('heatmap shape:', diff_heatmap.shape)
        for f in self.lm_encoder:
            diff_heatmap = f(diff_heatmap)
            
        ###### target heat map feature extraction ########
        #for f in self.lm_encoder:
         #   target_heatmap = f(target_heatmap)
        
        #heatmap = target_heatmap - src_heatmap
        src_img_heatmap = src_img + diff_heatmap
        #src_img_heatmap = torch.cat((src_img,diff_heatmap),1)

        recn_img = src_img_heatmap
        for f in self.face_decoder:
            recn_img = f(recn_img)

        return recn_img