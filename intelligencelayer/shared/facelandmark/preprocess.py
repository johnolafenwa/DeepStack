import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
#from skimage import io, transform
#from tqdm.auto import tqdm

import torch
#import torchvision
#import torch.nn as nn
#import torch.optim as optim
import random
from math import *
import imutils

import xml.etree.ElementTree as ET

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torchsummary import summary
import torchvision.transforms.functional as TF
from __future__ import print_function, division
import pandas as pd
import time
import copy



''' Creating transforms'''

class Transforms():
    def __init__(self):
        pass
    
    def rotate(self, image, landmarks, angle):
        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor([
            [+cos(radians(angle)), -sin(radians(angle))], 
            [+sin(radians(angle)), +cos(radians(angle))]
        ])

        image = imutils.rotate(np.array(image), angle)

        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return Image.fromarray(image), new_landmarks

    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks

    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(brightness=0.3, 
                                              contrast=0.3,
                                              saturation=0.3, 
                                              hue=0.1)
        image = color_jitter(image)
        return image, landmarks

    def crop_face(self, image, landmarks, crops):
        left = int(crops['left'])
        top = int(crops['top'])
        width = int(crops['width'])
        height = int(crops['height'])

        image = TF.crop(image, top, left, height, width)

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        image, landmarks = self.color_jitter(image, landmarks)
        image, landmarks = self.rotate(image, landmarks, angle=10)
        
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image, landmarks

''' Create face landmark dataset'''
class FaceLandmarkDataset(Dataset):
    def __init__(self, transform=None):
        tree=ET.parse('path/to/annotations.xml')
        root=tree.getroot()

        self.image_filenames=[]
        self.landmarks=[]
        self.crops=[]
        self.transform=transform
        self.root_dir='path/to/images/'

        for filename in root[2]:
            self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))
            self.crops.append(filename[0].attrib)

            landmark=[]

            for num in range(68):
                x_coord=int(filename[0][num].attrib['x'])
                y_coord=int(filename[0][num].attrib['y'])

                landmark.append([x_coord, y_coord])
            self.landmarks.append(landmark)

        self.landmarks=np.array(self.landmarks).astype('float32')

        assert len(self.image_filenames)==len(self.landmarks)
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image=cv2.imread(self.image_filenames[idx],0)
        landmarks=self.landmarks[idx]

        if self.transform:
            image, landmarks=self.transform(image, landmarks, self.crops[idx])

        landmarks=landmarks - 0.5
        return image, landmarks