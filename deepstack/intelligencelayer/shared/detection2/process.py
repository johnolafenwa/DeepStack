import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox

class ObjectDetector2(object):
    def __init__(self, model_path: str, reso: int = 640, cuda: bool = False):

        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.half = self.device.type != 'cpu'

        self.reso = (reso,reso)
        self.cuda = cuda
        self.model = attempt_load(model_path, map_location=self.device)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        if self.half:
            self.model.half()

    def predict(self, img_path: str, confidence: float=0.4):

        img0 = cv2.imread(img_path)
        img = letterbox(img0, new_shape=self.reso)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, confidence, 0.5, classes=None, agnostic=False)[0]

        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()

        return pred
