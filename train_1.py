# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 20:53:36 2022

@author: supratim
"""

import numpy as np
import torch
import torch.nn as nn
import os
from model_vgg16 import generator, generator_image_residual, generator_heatmap_residual
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
#predictor = dlib.shape_predictor("Landmark_Detection_Model\\shape_predictor_68_face_landmarks.dat")

LAND_MARKS_INDEXES = [37, 40, 43, 46, 32, 36, 49, 52, 55, 58]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_fns = Loss_Fns(device)

IMG_WHT=128
ETA = 0.001
pixel_loss_weight = [1,1,1]

RESUME_MODEL=False
LAST_EPOCH=0
LEARNING_RATE = 0.0002
TOTAL_EPOCHS = 3000
batch_size = 16

root_dir = 'preprocessed'

checkpoint = 'log\\00000000-checkpoint.pth.tar'

train_data = TrainDataset(root_dir, is_train=True)
test_data = TrainDataset(root_dir, is_train=False)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

def set_requires_grad(module , b ):
    for parm in module.parameters():
        parm.requires_grad = b


gen = generator()
#gen = generator_image_residual()
#gen = generator_heatmap_residual()
#d_f = Discriminator_faces(IMG_WHT, ETA)
#d_lm = Discriminator_marks(IMG_WHT, ETA)

G = nn.DataParallel(gen).to(device)
#D1 = nn.DataParallel(d_f).to(device) 
#D2 = nn.DataParallel(d_lm).to(device)



optimizer_G = torch.optim.Adam(G.parameters(), lr = LEARNING_RATE)
#optimizer_D1 = torch.optim.Adam(D1.parameters(), lr = LEARNING_RATE) 
#optimizer_D2 = torch.optim.Adam(D2.parameters(), lr = LEARNING_RATE) 

# if os.path.exists(checkpoint):
#         start_epoch = Logger.load_cpk(checkpoint, G, D1, D2,
#                                       optimizer_G, optimizer_D1, optimizer_D2)
#         print('starting epoch:',start_epoch)
# else:
#         start_epoch = 0

start_epoch = 0

mse = torch.nn.MSELoss().to(device)

cross_entropy = torch.nn.CrossEntropyLoss().to(device)

#epoch = LAST_EPOCH
def evaluation(G, test_dataloader, logger):
    G.eval()        
    #start_epoch=0      
    print('Validation')
    for step1, batch1 in enumerate(test_dataloader):
        with torch.no_grad():
            print('val:',batch1['source_img'].type())
            print('val shape:',batch1['source_img'].shape)
            val_loss={}
            for k in  batch1:
                batch1[k] =  torch.autograd.Variable( batch1[k].to(device) , requires_grad = False)
                
            img_128_gen_val = G(batch1['source_img'], batch1['target_img'])
            
            total_pixel_loss_val = loss_fns.pixelwise_ll1_oss(batch['target_img'].data, img_128_gen_val)
            val_loss['pixel_loss'] = total_pixel_loss_val
            
            total_variation_loss_val = torch.mean(torch.abs(img_128_gen_val[:,:,:-1,:] - img_128_gen_val[:,:,1:,:]))
            val_loss['variation_loss'] = total_variation_loss_val
            
            #lm_loss = loss_fns.lm_loss(batch1['target_img'].data, img_128_gen_val.data)
            #val_loss['lm_loss'] = lm_loss
            
            val_total_loss = total_variation_loss_val + total_pixel_loss_val #+ lm_loss
            val_loss['total_loss'] = val_total_loss
            
            all_val_loss = {key: value.mean().detach().data.cpu().numpy() for key, value in val_loss.items()}
            logger.log_iter_val(losses=all_val_loss)
            
            logger.log_epoch_val(epoch, batch1['source_img'], batch1['target_img'], img_128_gen_val)
            
            return img_128_gen_val

log_dir ='log'   
with Logger(log_dir=log_dir,  checkpoint_freq=500, val_freq=20) as logger:   
    for epoch in range(start_epoch+1, TOTAL_EPOCHS):
        print('#####################################')
        print("## EPOCH", epoch,"##")
        for step, batch in enumerate(train_dataloader):
            losses = {}
            #batch = batch.to(device)
            for k in  batch:
                batch[k] =  torch.autograd.Variable( batch[k].to(device) , requires_grad = False) 
                
            #print(batch['source_lm'].shape)
            src=batch['source_img']
            #print(src[0,0,:3,:3])
            #print(src[0,0,:3,:3]*255)
            #b=batch['source_img'].to(device)
            #print(b.type())
            
            img_128_gen = G(batch['source_img'], batch['target_img'])
            #print(img_128_gen.shape)
            ###### Train Discriminator 1 (D1) ###################
            # set_requires_grad( D1 , True )
            # D1_real_input = torch.cat([batch['target_img'].data, batch['source_img'].data], dim = 1)
            # D1_fake_input = torch.cat([img_128_gen.detach().data, batch['source_img'].data], dim = 1)
            # L_D1 = loss_fns.discriminator_loss(D1, D1_real_input, D1_fake_input)
            # #L_D1 = gradient_penalty_loss(D1, D1_real_input, D1_fake_input)
            # losses['D1_loss'] = L_D1
            # optimizer_D1.zero_grad()
            # L_D1.backward()
            # optimizer_D1.step()
            # #print('L_D1:', L_D1)
            # set_requires_grad( D1 , False )
            
            # ###### Train Discriminator 2 (D2) ###################
            # set_requires_grad( D2 , True )
            # D2_real_input =  torch.cat([batch['target_img'].data, batch['target_lm'].data], dim = 1)
            # D2_fake_input = torch.cat([img_128_gen.detach().data, batch['target_lm'].data], dim = 1)
            # L_D2 = loss_fns.discriminator_loss(D2, D2_real_input, D2_fake_input)
            # #L_D2 = gradient_penalty_loss(D2, D2_real_input, D2_fake_input)
            # losses['D2_loss'] = L_D2
            # optimizer_D2.zero_grad()
            # L_D2.backward()
            # optimizer_D2.step()
            # #print('L_D2:', L_D2)
            # set_requires_grad( D2 , False )
            
            ################# Pixelwise Loss #######################
            total_pixel_loss = loss_fns.pixelwise_ll1_oss(batch['target_img'].data, img_128_gen)
            #total_pixel_loss = loss_fns.image_mse_loss(batch['target_img'].data, img_128_gen)
            losses['pixel_loss'] = total_pixel_loss
            
            ######### variational loss ###########################
            total_variation_loss = torch.mean(torch.abs(img_128_gen[:,:,:-1,:] - img_128_gen[:,:,1:,:]))  + torch.mean(torch.abs(img_128_gen[:,:,:,:-1] - img_128_gen[:,:,:,1:]))
            losses['variation_loss'] = total_variation_loss
            ##################################################
            feature_loss = loss_fns.light_cnn_feature_loss(batch['target_img'].data, img_128_gen)
            #print('feat:', feature_loss)
            losses['feature_loss'] = feature_loss
            ######## Generator adv loss #############################
            #adv_loss = loss_fns.G_loss_adv(D1, D2, D1_fake_input, D2_fake_input)
            #losses['gen_adv_loss'] = adv_loss
            
            ########## land marks positional loss ###################
            ## lm_loss = loss_fns.lm_loss(batch['target_img'].data, img_128_gen.data)
            ## losses['lm_loss'] = lm_loss
            # if epoch > 8:
            #     lm_loss = loss_fns.lm_loss(batch['target_img'].data, img_128_gen.data)
            #     losses['lm_loss'] = lm_loss
            # else:
            #     lm_loss = 0.0
            #     losses['lm_loss'] = torch.tensor(lm_loss, dtype=torch.float64)
                
            
            
            ###### Total generator loss ############################
            L_final = 10*total_pixel_loss + 1e-4 *(total_variation_loss) #+ 0.01*feature_loss# + 0.1*adv_loss +  #+ 0.1*lm_loss
            losses['total_gen_loss'] = L_final
            
            optimizer_G.zero_grad()
            L_final.backward()
            optimizer_G.step()
            
            all_loss = {key: value.mean().detach().data.cpu().numpy() for key, value in losses.items()}
            logger.log_iter(losses=all_loss)
            
            logger.log_epoch(epoch, {'generator': G,
                                      'optimizer_generator': optimizer_G},batch['source_img'],batch['target_img'], img_128_gen)
        print('Training finished')   
        print('#######################')
        
        #if epoch%20==0:
        #fg = evaluation(G, test_dataloader, logger)
        # print('Validation')
        # for step1, batch1 in enumerate(test_dataloader):
        #     #print(batch1['source_img'].shape)
        #     val_loss={}
        #     for k in  batch1:
        #         batch1[k] =  torch.autograd.Variable( batch1[k].to(device) , requires_grad = False)
                
        #     img_32_gen_val, img_64_gen_val, img_128_gen_val = G(batch1['source_img'], batch1['source_lm'], batch1['target_lm'])
            
        #     total_pixel_loss_val = loss_fns.pixelwise_loss(batch1['target_img'].data, img_32_gen_val, img_64_gen_val, img_128_gen_val, pixel_loss_weight)
        #     val_loss['pixel_loss'] = total_pixel_loss_val
            
        #     total_variation_loss_val = torch.mean(torch.abs(img_128_gen_val[:,:,:-1,:] - img_128_gen_val[:,:,1:,:]))
        #     val_loss['variation_loss'] = total_variation_loss_val
            
        #     val_total_loss = total_variation_loss_val + total_pixel_loss_val
        #     val_loss['total_loss'] = val_total_loss
            
        #     all_val_loss = {key: value.mean().detach().data.cpu().numpy() for key, value in val_loss.items()}
        #     logger.log_iter_val(losses=all_val_loss)
            
        #     logger.log_epoch_val(epoch, batch1['source_img'], batch1['target_img'], img_128_gen_val)