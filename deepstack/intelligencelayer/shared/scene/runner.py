
import torch
import time
import json
import io
import _thread as thread
from multiprocessing import Process
from PIL import Image
import cv2
import torch.nn.functional as F
import ast
import sqlite3
import numpy as np
import warnings
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../"))

from process import SceneModel
from shared import SharedOptions


import torchvision.transforms as transforms


def scenerecognition(thread_name,delay):

    classes = list()
    with open(os.path.join(SharedOptions.SHARED_APP_DIR,"categories_places365.txt")) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    
    placesnames = tuple(classes)

    IMAGE_QUEUE = "scene_queue"
    classifier = SceneModel(os.path.join(SharedOptions.SHARED_APP_DIR,"scene.model"),SharedOptions.CUDA_MODE)

    while True:
        queue = SharedOptions.db.lrange(IMAGE_QUEUE,0,0)
        
        if len(queue) > 0:

            SharedOptions.db.ltrim(IMAGE_QUEUE,len(queue), -1)

            for req_data in queue:
                req_data = json.JSONDecoder().decode(req_data)
                img_id = req_data["imgid"]
                req_id = req_data["reqid"]
                req_type = req_data["reqtype"]
                try:
                    
                   
                    img = Image.open(SharedOptions.TEMP_PATH+img_id).convert("RGB")

                    trans = transforms.Compose([
                        transforms.Resize((256,256)),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    img = trans(img)
                    img = img.numpy()
                    img = np.expand_dims(img,0).astype(np.float32)
                  
                    os.remove( SharedOptions.TEMP_PATH + img_id)

                    cl , conf = classifier.predict(img)

                    cl = placesnames[cl]

                    conf = float(conf)

                    output = {"success":True, "label":cl, "confidence":conf}

                    SharedOptions.db.set(req_id,json.dumps(output))

                except Exception as e:
                   
                    output = {"success":False, "error":"invalid image","code":400}
                    SharedOptions.db.set(req_id,json.dumps(output))
                    if os.path.exists(SharedOptions.TEMP_PATH + img_id):
                        os.remove( SharedOptions.TEMP_PATH + img_id)

        time.sleep(delay)


p = Process(target=scenerecognition,args=("",SharedOptions.SLEEP_TIME))
p.start()

