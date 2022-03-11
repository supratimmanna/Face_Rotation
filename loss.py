# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 20:21:10 2021

@author: supratim
"""

import numpy as np
import torch
import torch.nn as nn
import os
from models import Generator, Discriminator_faces, Discriminator_marks
from dataset import TrainDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from logger import Logger
import cv2
import dlib
from imutils import face_utils
import face_alignment
from face_features.inception_resnet_v1 import InceptionResnetV1
from Light_CNN.feature_extractor import Feature_Extractor
from utils import ImagePyramide, Vgg19


class Loss_Fns():
    def __init__(self, device):
        self.device = device
        self.L1 = torch.nn.L1Loss().to(device)
        self.cross_entropy = torch.nn.CrossEntropyLoss().to(device)
        self.mse = torch.nn.MSELoss().to(device)
        self.face_feature = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        #self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
        #self.detector = dlib.get_frontal_face_detector()
        #self.predictor = dlib.shape_predictor("Landmark_Detection_Model\\shape_predictor_68_face_landmarks.dat")
        self.light_cnn = Feature_Extractor(device=self.device)
        self.LAND_MARKS_INDEXES = [36, 39, 42, 45, 31, 35, 48, 51, 54, 57]
        
        self.perceptual_scales = [1, 0.5, 0.25]
        self.pyramid = ImagePyramide(scales=self.perceptual_scales, num_channels=3).to(device)
        self.vgg = Vgg19().to(device)
        
        self.loss_weights = {
            'pix':10,
            'adv1':0.1,
            'adv2':0.1,
            'ip'  :0.02,
            'reg' :1e-4
        }
    
    
    def discriminator_loss(self, D, real, fake):
        D_real = D(real)
        L_D_real = torch.mean(torch.nn.BCEWithLogitsLoss()(D_real, torch.ones_like(D_real) * 0.986))
        D_fake = D(fake)
        L_D_fake = torch.mean(torch.nn.BCEWithLogitsLoss()(D_fake, torch.zeros_like(D_fake)))
        L_D1 = L_D_real + L_D_fake
        
        return L_D1
    

    def gradient_penalty_loss(self, D, real, fake):
        D1_loss_no_gp = - torch.mean(D(real)) + torch.mean(D(fake))
        alpha_1 = torch.rand( real.shape[0] , 1 , 1 , 1 ).expand_as(real).pin_memory().cuda()
        interpolated_x_1 = torch.autograd.Variable( alpha_1 * fake   + (1.0 - alpha_1) * real , requires_grad = True)
        out_1 = D(interpolated_x_1)
        dxdD_1 = torch.autograd.grad( outputs = out_1 , inputs = interpolated_x_1 , grad_outputs = torch.ones(out_1.size()).cuda() , retain_graph = True , create_graph = True , only_inputs = True  )[0].view(out_1.shape[0],-1)
        gp_loss_1 = torch.mean( ( torch.norm( dxdD_1 , p = 2 ) - 1 )**2 )
        L_D1 = D1_loss_no_gp + 10 * gp_loss_1
        
        return L_D1
    
    
    def pixelwise_loss(self, img, img_32_gen, img_64_gen, img_128_gen, weight):
        img_128 = img
        img_64 = nn.MaxPool2d(2)(img_128)
        img_32 = nn.MaxPool2d(2)(img_64)
        
        pixelwise_128_loss = self.L1(img_128, img_128_gen)
        pixelwise_64_loss = self.L1(img_64, img_64_gen)
        pixelwise_32_loss = self.L1(img_32, img_32_gen)
        total_pixel_loss = weight[0]*pixelwise_128_loss + weight[1]*pixelwise_64_loss + weight[2]*pixelwise_32_loss
        
        return total_pixel_loss
    

    def pixelwise_ll1_oss(self, img, img_128_gen):
        
        img_128 = img
        pixelwise_128_loss = self.L1(img_128, img_128_gen)
        
        return pixelwise_128_loss
    

    def image_mse_loss(self, img, img_128_gen):
        loss = self.mse(img, img_128_gen)
        
        return loss


    def image_feature_loss(self, img, img_128_gen):
        
        real_ft = self.face_feature(img)
        fake_ft = self.face_feature(img_128_gen)
        
        loss = self.mse(real_ft, fake_ft)
        
        return loss
    

    def light_cnn_feature_loss(self, img, img_128_gen):

        real_feat_fc, real_feat_conv = self.light_cnn.forward(img)
        b,c,h,w = real_feat_conv.shape
        real_feat_conv = real_feat_conv.reshape(b, c*h*w)
        
        fake_feat_fc, fake_feat_conv = self.light_cnn.forward(img_128_gen)
        b,c,h,w = fake_feat_conv.shape
        fake_feat_conv = fake_feat_conv.reshape(b, c*h*w)
        
        #fc_loss = torch.mean((real_feat_fc - fake_feat_fc).pow(2))
        fc_loss = self.L1(real_feat_fc, fake_feat_fc)
        #conv_loss = torch.mean((real_feat_conv - fake_feat_conv).pow(2))
        conv_loss = self.L1(real_feat_conv, fake_feat_conv)
        
        loss = fc_loss + conv_loss
        
        return loss

        
    def image_perceptual_loss(self, img, img_128_gen, loss_weights_perceptual=[10,10,10,10,10]):
        #print(img.type())
        pyramide_real = self.pyramid(img)
        pyramide_generated = self.pyramid(img_128_gen)
        
        loss = 0
        
        for scale in self.perceptual_scales:
            x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
            y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])
            
            for i, weight in enumerate(loss_weights_perceptual):
                value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                loss += loss_weights_perceptual[i] * value
                
        return loss
        

    def G_loss_adv(self,D1, D2, D1_fake_inp, D2_fake_inp):
        D1_fake_op = D1(D1_fake_inp) 
        adv1 = torch.mean(torch.nn.BCEWithLogitsLoss()(D1_fake_op, 0.986*torch.ones_like(D1_fake_op))) 
        D2_fake_op = D2(D2_fake_inp)
        adv2 = torch.mean(torch.nn.BCEWithLogitsLoss()(D2_fake_op, 0.986*torch.ones_like(D2_fake_op))) 
        
        adv = adv1 + adv2
        return adv
    
    
    
    def lm_loss(self, img, img_128_gen):
        
        batch_size = img.shape[0]
        target_lm = np.zeros((batch_size, 10, 2))
        generated_lm = np.zeros((batch_size, 10, 2))
        
        ##### for original images ################
        for j, img in enumerate(img):
          img = img.permute(1,2,0)
          img = img.cpu().numpy()
          #print(type(img))
          img = (255 * img).astype(np.uint8)
          preds=self.fa.get_landmarks(img)
          print(len(preds))
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
        target_lm = torch.tensor(target_lm)
        #########################################
        images = img.cpu().numpy()
        images = np.transpose(images,[0,2,3,1])
        for j, img in enumerate(images):
            img = (255 * img).astype(np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            if len(rects)>0:
                print('Detect face in source image')
                for rect in rects:
                    shape = self.predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                k = 0
                for (i, (x, y)) in enumerate(shape):
                    if ((i + 1) in self.LAND_MARKS_INDEXES):
                        target_lm[j,k,0] = x
                        target_lm[j,k,1] = y
                        k+=1
            else:
                print('face not detected in source image')
        target_lm = torch.tensor(target_lm)
        
        ##### for generated images ################
        for j, img in enumerate(img_128_gen):
          img = img.permute(1,2,0)
          img = img.cpu().numpy()
          #print(type(img))
          img = (255 * img).astype(np.uint8)
          preds=self.fa.get_landmarks(img)
          print(len(preds))
          if len(preds)>0:
            k=0
            for i,f in enumerate(preds[0]):
                  if i in self.LAND_MARKS_INDEXES:
                    x=f[0]
                    y=f[1]
                    generated_lm[j,k,0] = x
                    generated_lm[j,k,1] = y
                    k+=1
          else:
            print('face not detected in source image')
        
        ############################################
        gen_images = img_128_gen.cpu().numpy()
        gen_images = np.transpose(gen_images,[0,2,3,1])
        for j, img in enumerate(gen_images):
            img = (255 * img).astype(np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)
            if len(rects)>0:
                print('Detect face in generated image')
                for rect in rects:
                    shape = self.predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                k = 0
                for (i, (x, y)) in enumerate(shape):
                    if ((i + 1) in self.LAND_MARKS_INDEXES):
                        generated_lm[j,k,0] = x
                        generated_lm[j,k,1] = y
                        k+=1
            else:
                print('face not detected in generated image')
        generated_lm = torch.tensor(generated_lm)
        
        lm_mae_loss = self.L1(target_lm, generated_lm)
        
        return lm_mae_loss
    

######################################################################
# def discriminator_loss(D, real, fake):
#     D1_real = D(real)
#     L_D1_real = torch.mean(torch.nn.BCEWithLogitsLoss()(D1_real, torch.ones_like(D1_real) * 0.9))
#     D1_fake = D(fake)
#     L_D1_fake = torch.mean(torch.nn.BCEWithLogitsLoss()(D1_fake, torch.zeros_like(D1_fake)))
#     L_D1 = L_D1_real + L_D1_fake
    
#     return L_D1

# def gradient_penalty_loss(D, real, fake):
#     D1_loss_no_gp = - torch.mean(D(real)) + torch.mean(D(fake))
#     alpha_1 = torch.rand( real.shape[0] , 1 , 1 , 1 ).expand_as(real).pin_memory().cuda()
#     interpolated_x_1 = torch.autograd.Variable( alpha_1 * fake   + (1.0 - alpha_1) * real , requires_grad = True)
#     out_1 = D(interpolated_x_1)
#     dxdD_1 = torch.autograd.grad( outputs = out_1 , inputs = interpolated_x_1 , grad_outputs = torch.ones(out_1.size()).cuda() , retain_graph = True , create_graph = True , only_inputs = True  )[0].view(out_1.shape[0],-1)
#     gp_loss_1 = torch.mean( ( torch.norm( dxdD_1 , p = 2 ) - 1 )**2 )
#     L_D1 = D1_loss_no_gp + 10 * gp_loss_1
    
#     return L_D1

# def pixelwise_loss(img, img_32_gen, img_64_gen, img_128_gen, weight):
#     L1 = torch.nn.L1Loss().to(device)
#     img_128 = img
#     img_64 = nn.MaxPool2d(2)(img_128)
#     img_32 = nn.MaxPool2d(2)(img_64)
    
#     pixelwise_128_loss = L1(img_128, img_128_gen)
#     pixelwise_64_loss = L1(img_64, img_64_gen)
#     pixelwise_32_loss = L1(img_32, img_32_gen)
#     total_pixel_loss = weight[0]*pixelwise_128_loss + weight[1]*pixelwise_64_loss + weight[2]*pixelwise_32_loss
    
#     return total_pixel_loss

# def lm_loss(img, img_128_gen):
#     L1 = torch.nn.L1Loss().to(device)
    
#     target_lm = np.zeros((10,2))
#     generated_lm = np.zeros((10,2))
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     rects = detector(gray, 0)
#     for rect in rects:
#         shape = predictor(gray, rect)
#         shape = face_utils.shape_to_np(shape)
#         for (i, (x, y)) in enumerate(shape):
#             if ((i + 1) in LAND_MARKS_INDEXES):
#                 target_lm[i,0] = x
#                 target_lm[i,1] = y
    
#     gray = cv2.cvtColor(img_128_gen, cv2.COLOR_BGR2GRAY)
#     rects = detector(gray, 0)
#     if len(rects)>0:
#         for rect in rects:
#             shape = predictor(gray, rect)
#             shape = face_utils.shape_to_np(shape)
#             for (i, (x, y)) in enumerate(shape):
#                 if ((i + 1) in LAND_MARKS_INDEXES):
#                     generated_lm[i,0] = x
#                     generated_lm[i,1] = y
    
#     lm_mae_loss = L1(target_lm, generated_lm)
    
#     return lm_mae_loss