from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from .util import *
import argparse
import os 
import os.path as osp
from .darknet import Darknet
from .preprocess import prep_image, inp_to_image
import random 
import pickle as pkl
import itertools


class DetectModel(object):
    def __init__(self,model_path, config_path,names_path,cuda=False,nms=0.4,reso=640,confidence=0.5,num_classes=80):
        self.cuda_mode = False
        self.model = Darknet(config_path)
        self.model.load_weights(model_path)
        self.classes = load_classes(names_path) 
        self.model.net_info["height"] = reso
        self.inp_dim = reso
        self.confidence = confidence
        self.num_classes = num_classes
        self.nms = nms
        assert self.inp_dim % 32 == 0 
        assert self.inp_dim > 32
        
        if torch.cuda.is_available() and cuda:
            self.cuda_mode = True
            self.model = self.model.cuda()
        
        self.model.eval()
        
    def predict(self, images):

        image_trans = [prep_image(img,self.inp_dim) for img in images]

        image_batch = [img[0] for img in image_trans]
       
        im_dim_list = [img[2] for img in image_trans]
       
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

        image_batch = torch.cat(image_batch)
      
        inp_dim = self.inp_dim
       
        if self.cuda_mode:
            image_batch = image_batch.cuda()
        
        
        with torch.no_grad():
            prediction = self.model(Variable(image_batch), self.cuda_mode)
       
        output = write_results(prediction, self.confidence, self.num_classes, nms = True, nms_conf =self.nms,cuda=self.cuda_mode)
        

        if isinstance(output,int):
            return []

        output = output.cpu()

        im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    
        scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
    
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    
        output[:,1:5] /= scaling_factor
    
        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])

        batch_output = []

        for x in output:
            conf = x[6]
           
            batch_len = len(batch_output)
            id = int(x[0].item())
            if id == batch_len:
                batch_output.append([x])

            else:
                batch_output[batch_len-1].append(x)
        
        return batch_output

