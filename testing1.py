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
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Landmark_Detection_Model\\shape_predictor_68_face_landmarks.dat")
LAND_MARKS_INDEXES = [37, 40, 43, 46, 32, 36, 49, 52, 55, 58]
IMG_DIM = (128,128)
pady1, pady2, padx1, padx2 = 10, 20, 20, 20

fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device='cpu')

img = cv2.imread('0.jpg')
batch=[]
batch.append(img)
batch.append(img)
#preds = fa.get_detections_for_batch(np.asarray(batch))
img = cv2.resize(img, IMG_DIM, interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 0)
for r in rects:
    x1=r.left()
    y1=r.top()
    x2=r.right()
    y2=r.bottom()
    
#image = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)
X=[]
Y=[]
image=img
img_1 = np.zeros([128,128,1],dtype=np.uint8)
img_1=1.0*np.zeros_like(gray)
#img_1.fill(0)
for rect in rects:
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    for (i, (x, y)) in enumerate(shape):
        if ((i + 1) in LAND_MARKS_INDEXES):
            X.append(x)
            Y.append(y)
            #img_1=cv2.circle(img_1, (x, y), 5, (0, 255, 0), -1)

def get_landmarks(img):
        X=[]
        Y=[]
        img = cv2.imread(img)
        img = cv2.resize(img, IMG_DIM, interpolation = cv2.INTER_AREA)
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
    
X,Y=get_landmarks('0.jpg')
img_1 = np.zeros([128,128,1],dtype=np.uint8)
img_1.fill(0)

for i in range(len(X)):
    x=X[i]
    y=Y[i]
    img_1=cv2.circle(img_1, (x,y), 2, (255,0,0), -1)
    
print(img_1.shape)
cv2.imwrite('11.jpg',img_1)

# # print(x,y) 

# for i in range(len(X)):
#     x=X[i]
#     y=Y[i]
#     gray=cv2.circle(gray, (x,y), 2, (255,0,0), -1)
    

    