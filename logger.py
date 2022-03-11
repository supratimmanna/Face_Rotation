# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:04:43 2021

@author: supratim
"""

import numpy as np
import torch
import torch.nn.functional as F
import imageio
import cv2

import os
from skimage.draw import circle

import matplotlib.pyplot as plt
import collections


class Logger:
    def __init__(self, log_dir, checkpoint_freq=750, val_freq=10, device='cpu', visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.loss_list_val = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.visualizations_dir_val = os.path.join(log_dir, 'val-vis')
        if not os.path.exists(self.visualizations_dir_val):
            os.makedirs(self.visualizations_dir_val)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.log_file_val = open(os.path.join(log_dir, 'log_val.txt'), 'a')
        self.zfill_num = zfill_num
        #self.devie = device
 #       self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.val_freq = val_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None
        self.names_val = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()
        
    def log_scores_val(self, loss_names):
        loss_mean = np.array(self.loss_list_val).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file_val)
        self.loss_list_val = []
        self.log_file_val.flush()

    def visualize_rec(self, src, inp, out):
        #image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        s, t, g = self.visualize(src, inp, out)
        Horizontal=np.hstack([s,t,g])
        #imageio.imsave
        cv2.imwrite(os.path.join(self.visualizations_dir, "%s-.png" % str(self.epoch).zfill(self.zfill_num)), Horizontal)
        #cv2.imwrite(os.path.join(self.visualizations_dir, "%s-generated.png" % str(self.epoch).zfill(self.zfill_num)), g)
    
    def visualize_rec_val(self, src, inp, out):
        #image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        s, t, g = self.visualize(src, inp, out)
        Horizontal=np.hstack([s,t,g])
        #imageio.imsave
        cv2.imwrite(os.path.join(self.visualizations_dir_val, "%s-.png" % str(self.epoch).zfill(self.zfill_num)), Horizontal)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, discriminator_1=None, discriminator_2=None,
                 optimizer_generator=None, optimizer_discriminator_1=None, optimizer_discriminator_2=None):
        checkpoint = torch.load(checkpoint_path)
        if generator is not None:
            print('load checkpoints for generator')
            generator.load_state_dict(checkpoint['generator'])
        if discriminator_1 is not None:
            try:
                discriminator_1.load_state_dict(checkpoint['discriminator_1'])
            except:
               print ('No discriminator 1 in the state-dict. Dicriminator 1 will be randomly initialized')
        if discriminator_2 is not None:
            try:
               discriminator_2.load_state_dict(checkpoint['discriminator_2'])
            except:
               print ('No discriminator 2 in the state-dict. Dicriminator 2 will be randomly initialized')
        if optimizer_generator is not None:
            optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
        if optimizer_discriminator_1 is not None:
            try:
                optimizer_discriminator_1.load_state_dict(checkpoint['optimizer_discriminator_1'])
            except RuntimeError as e:
                print ('No discriminator 1 optimizer in the state-dict. Optimizer will be not initialized')
        if optimizer_discriminator_2 is not None:
            try:
                optimizer_discriminator_2.load_state_dict(checkpoint['optimizer_discriminator_2'])
            except RuntimeError as e:
                print ('No discriminator 1 optimizer in the state-dict. Optimizer will be not initialized')

        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))
    
    def log_iter_val(self, losses):
        losses_val = collections.OrderedDict(losses.items())
        if self.names_val is None:
            self.names_val = list(losses_val.keys())
        self.loss_list_val.append(list(losses_val.values()))
        #print( self.names_val)

    def log_epoch(self, epoch, models, src, inp, out):
        self.epoch = epoch 
        self.models = models
        if (self.epoch) % self.checkpoint_freq == 0:
            print('Saving Checkpoints')
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(src, inp, out)
        
    def log_epoch_val(self, epoch, src, inp, out):
        self.epoch = epoch 
        self.log_scores_val(self.names_val)
        if (self.epoch) % self.val_freq == 0:
            print('save val images')
            #self.log_scores_val(self.names_val)
            self.visualize_rec_val(src, inp, out)
    
    def visualize(self, src, inp, out):
        images = []
        b = src.shape[0]
        a=np.random.choice(range(0,b))
        #a=np.random.choice([0,b])
        ## Need to change to this: a=np.random.choice(range(0,b)) 
        source = src.data.cpu().numpy()
        source = np.transpose(source, [0, 2, 3, 1])
        s = source[a,:,:,:]
        s = (255 * s).astype(np.uint8)
        # Source image with keypoints
        target = inp.data.cpu().numpy()
        target = np.transpose(target, [0, 2, 3, 1])
        t = target[a,:,:,:]
        t = (255 * t).astype(np.uint8)
        generated = out.data.cpu().numpy()
        generated = np.transpose(generated, [0, 2, 3, 1])
        g = generated[a,:,:,:]
        g = (255 * g).astype(np.uint8)
        
        return s, t, g
        
    def create_image_grid(self, *args):
        out = []
        for arg in args:
            out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)
    
    def create_image_column(self, images):
        images = np.copy(images)
        images[:, :, [0, -1]] = (1, 1, 1)
        images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)