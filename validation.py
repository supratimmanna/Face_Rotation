# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 15:26:48 2022

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
from loss import Loss_Fns
import tqdm


#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("Landmark_Detection_Model/shape_predictor_68_face_landmarks.dat")

LAND_MARKS_INDEXES = [37, 40, 43, 46, 32, 36, 49, 52, 55, 58]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_fns = Loss_Fns(device)

IMG_WHT=128
ETA = 0.001
pixel_loss_weight = [1,1,1]

RESUME_MODEL=False
LAST_EPOCH=0
LEARNING_RATE = 0.0002
TOTAL_EPOCHS = 100
batch_size = 16

root_dir = 'preprocessed'

checkpoint = 'log/final_test_1-checkpoint.pth.tar'#'log/00004499-checkpoint.pth.tar'

test_data = TrainDataset(root_dir, is_train=False)

test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

gen = Generator(IMG_WHT, ETA)
d_f = Discriminator_faces(IMG_WHT, ETA)
d_lm = Discriminator_marks(IMG_WHT, ETA)

G = nn.DataParallel(gen).to(device)
D1 = nn.DataParallel(d_f).to(device) 
D2 = nn.DataParallel(d_lm).to(device)

if os.path.exists(checkpoint):
    print('loading checkpoints')
    start_epoch = Logger.load_cpk(checkpoint, G, D1, D2)
else:
    print('No checkpoint for validation')

G.eval()        
log_dir ='log'
start_epoch=0   
with Logger(log_dir=log_dir,  checkpoint_freq=50, device=device) as logger:   
    for epoch in range(start_epoch+1, TOTAL_EPOCHS):
        print('#####################################')
        print("## EPOCH", epoch,"##")
        print('Validation')
        for step1, batch1 in enumerate(test_dataloader):
            with torch.no_grad():
                print(batch1['source_img'].shape)
                val_loss={}
                for k in  batch1:
                    batch1[k] =  torch.autograd.Variable( batch1[k].to(device) , requires_grad = False)
                    
                img_32_gen_val, img_64_gen_val, img_128_gen_val = G(batch1['source_img'], batch1['source_lm'], batch1['target_lm'])
                
                total_pixel_loss_val = loss_fns.pixelwise_loss(batch1['target_img'].data, img_32_gen_val, img_64_gen_val, img_128_gen_val, pixel_loss_weight)
                val_loss['pixel_loss'] = total_pixel_loss_val
                
                total_variation_loss_val = torch.mean(torch.abs(img_128_gen_val[:,:,:-1,:] - img_128_gen_val[:,:,1:,:]))
                val_loss['variation_loss'] = total_variation_loss_val
                
                val_total_loss = total_variation_loss_val + total_pixel_loss_val
                val_loss['total_loss'] = val_total_loss
                
                all_val_loss = {key: value.mean().detach().data.cpu().numpy() for key, value in val_loss.items()}
                logger.log_iter_val(losses=all_val_loss)
                
                logger.log_epoch_val(epoch, batch1['source_img'], batch1['target_img'], img_128_gen_val)