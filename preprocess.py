# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 18:17:38 2021

@author: supratim
"""

import numpy as np
import shutil
import os
from os import path
from glob import glob
import cv2
from imutils import face_utils
from torch.utils.data import Dataset
import dlib
import torchvision.transforms as transforms
from PIL import Image
import face_detection
import torch
import time
import face_alignment
#from face_detection import *

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Landmark_Detection_Model/shape_predictor_68_face_landmarks.dat")

#LAND_MARKS_INDEXES = [37, 40, 43, 46, 32, 36, 49, 52, 55, 58]
LAND_MARKS_INDEXES = [36, 39, 42, 45, 31, 35, 48, 51, 54, 57]
IMG_DIM = (128,128)

pady1, pady2, padx1, padx2 = 0, 0, 0, 0 #10, 20, 20, 20

data_root = 'original_data/train'
vfilelist = glob(os.path.join(data_root, '*.mp4'))

processed_dir = 'preprocessed'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
fa_lm = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
################# Face Detection ########################
def process_video_file_face_main(vfile, processed_dir, batch_size):
     video_stream = cv2.VideoCapture(vfile)
     
     frames = []
     f = -1
     while 1:
          still_reading, frame = video_stream.read()
          if not still_reading:
              video_stream.release()
              break
          else:
            f=f+1
            if f%5 == 0:
              frames.append(frame)
     
     #frames=frames[:8]
     vidname = os.path.basename(vfile).split('.')[0]
     dirname = vfile.split('/')[-2]

     fulldir = path.join(processed_dir, dirname, 'images', vidname)
     os.makedirs(fulldir, exist_ok=True)
     fulldir_lm = path.join(processed_dir, dirname, 'landmarks', vidname)
     os.makedirs(fulldir_lm, exist_ok=True)

     batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
     print('Batch Length',len(batches))
     i = -1
     for fb in batches:
          preds = fa.get_detections_for_batch(np.asarray(fb))
          for j, f in enumerate(preds):
              i += 1
              #print(i)
              if f is None:
                  continue
              #x1, y1, x2, y2 = f
              y1 = max(0, f[1] - pady1)
              y2 = min(fb[0].shape[0], f[3] + pady2)
              x1 = max(0, f[0] - padx1)
              x2 = min(fb[0].shape[1], f[2] + padx2)
              fb_crop = fb[j][y1:y2, x1:x2]
              fb_crop=cv2.resize(fb_crop, IMG_DIM, interpolation = cv2.INTER_AREA)
              #cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), fb[j][y1:y2, x1:x2])
              cv2.imwrite(path.join(fulldir, '{}.jpg'.format(i)), fb_crop)
              img_1 = get_landmarks_image(fb_crop)    
              cv2.imwrite(path.join(fulldir_lm, '{}.jpg'.format(i)), img_1)
              
##############################################################
def get_landmarks_image(img):
        img_1 = np.zeros([128,128,1],dtype=np.uint8)
        img_1.fill(0)
        preds = fa_lm.get_landmarks(img)
        if len(preds)==0:
          print('Landmark not detected')
        else:
          for i,f in enumerate(preds[0]):
            if i in LAND_MARKS_INDEXES:
              x=f[0]
              y=f[1]
              img_1 =cv2.circle(img_1, (x,y), 3, (255,0,0), -1)
        return img_1              
############### Landmark Detection ##########################
def get_landmarks(img):
        X=[]
        Y=[]
        #img = cv2.imread(link)
        #img = cv2.resize(img, IMG_DIM, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # The square where we place landmarks
        #black = cv2.imread("black_square160.png")
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            for (i, (x, y)) in enumerate(shape):
                if ((i + 1) in LAND_MARKS_INDEXES):
                    X.append(x)
                    Y.append(y)
        return X,Y
        #cv2.imwrite(path.join(dirr, '{}.jpg'.format(j)), black_image)
        #return transforms.ToTensor()(Image.fromarray(black))[1].unsqueeze(0)
    
    
#############################################################################

for vfile in vfilelist:
    print(vfile)
    process_video_file_face_main(vfile=vfile, processed_dir=processed_dir, batch_size=8)
    
#lm_processed_dir = 'preprocessed\\train'
#lm_processed_dir_img = 
#img_filelist = glob(os.path.join(lm_processed_dir, '*.jpg'))
