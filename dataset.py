# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 12:27:59 2021

@author: supratim
"""
import numpy as np
import shutil
import os
import cv2
import torch
from imutils import face_utils
from torch.utils.data import Dataset
import dlib
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from augmentation import AllAugmentationTransform


    

class TrainDataset(Dataset):
    def __init__( self , root_dir, is_train=True, aug=False, augmentation_params=None, random_seed=0 ):
        super(type(self),self).__init__()
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.aug = aug
        
        if os.path.exists(os.path.join(root_dir, 'train')):
            #print(os.path.join(root_dir, 'train'))
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            train_videos = os.listdir(os.path.join(root_dir, 'train', 'images'))
            train_lms = os.listdir(os.path.join(root_dir, 'train', 'landmarks'))
            #print(train_videos)
            
            test_videos = os.listdir(os.path.join(root_dir, 'test', 'images'))
            test_lms = os.listdir(os.path.join(root_dir, 'test', 'landmarks'))
            
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
            #print(self.root_dir)
        else:
            print("Use random train-test split.") 
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)
            
        if is_train:
            print('Training Data')
            self.videos = train_videos
            #print(self.videos)
            self.lms = train_lms
        else:
            print('Testing Data')
            self.videos = test_videos
            self.lms = test_lms

        self.is_train = is_train

        if self.is_train and self.aug:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None
            
            
    def __len__( self ):
        return len(self.videos)
    
    
    def __getitem__( self , idx ):
        #example - 001_01_01_010_05_crop_128
        name = self.videos[idx]
        path = os.path.join(self.root_dir, 'images', name)
        path_lm = os.path.join(self.root_dir,'landmarks', name)
        video_name = os.path.basename(path)
        #print(path)
        
        #if self.is_train and os.path.isdir(path):
        if os.path.isdir(path):
            frames = os.listdir(path)
            frames_lm = os.listdir(path_lm)
            num_frames = len(frames)
            frame_idx = np.random.choice(range(1,num_frames), replace=True, size=1)
            
            src_frame = os.path.join(path,frames[0])
            #print('source:', src_frame)
            src_lm = os.path.join(path_lm,frames_lm[0])
            #print(src_lm)
            source_img = cv2.imread(src_frame)
            source_img = transforms.ToTensor()(source_img)
            #source_img =  torch.FloatTensor(source_img.transpose([2,0,1])).unsqueeze(0)
            #print('source:',src_frame)
            source_lm = cv2.imread(src_lm, cv2.IMREAD_GRAYSCALE)
            source_lm = transforms.ToTensor()(source_lm)
            #source_lm =  torch.FloatTensor(source_lm.transpose([2,0,1])).unsqueeze(0)
            #print('source lm:',source_lm.shape)
            
            target_frame = os.path.join(path,frames[frame_idx[0]])
            #print('target:', target_frame)
            target_lm = os.path.join(path_lm,frames_lm[frame_idx[0]])
            target_img = cv2.imread(target_frame)
            target_img = transforms.ToTensor()(target_img)
            #target_img =  torch.FloatTensor(target_img.transpose([2,0,1])).unsqueeze(0)

            target_lm = cv2.imread(target_lm, cv2.IMREAD_GRAYSCALE)
            target_lm = transforms.ToTensor()(target_lm)
            #target_lm =  torch.FloatTensor(target_lm.transpose([2,0,1])).unsqueeze(0)
            #print('target:',target_lm.shape)

        batch = {}
        #if self.is_train:
        batch['source_img'] = source_img
        batch['source_lm'] = source_lm
        batch['target_img'] = target_img
        batch['target_lm'] = target_lm
        
        
        return batch
    


# root_dir = 'preprocessed'    
# train = TrainDataset(root_dir)
# i=0
# for batch in train:
#     print(i)
#     i=i+1
#     im = batch['source_img']

