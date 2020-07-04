from redis import StrictRedis, RedisError
import torch
import time
import json
import io
import os
import _thread as thread
from multiprocessing import Process
from PIL import Image
import cv2
import torch.nn.functional as F
import ast
import sqlite3
import numpy as np
import warnings
import tensorflow as tf
import sys
from intelligencelayer.shared.detection import ObjectDetector
from intelligencelayer.shared.face.detection import FaceModel
"""
sys.stdout = open(os.devnull, 'w')
"""
tf.logging.set_verbosity(tf.logging.ERROR)

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CUDA_MODE = os.getenv("CUDA_MODE","False")

APPDIR = os.getenv("APPDIR","..")

TEMP_PATH = "/deeptemp/"
SLEEP_TIME = 0.01
SHARED_APP_DIR = os.path.join(APPDIR,"sharedfiles")
GPU_APP_DIR = os.path.join(APPDIR,"gpufiles")
CPU_APP_DIR = os.path.join(APPDIR,"cpufiles")
DATA_DIR = "/datastore"


if CUDA_MODE == "True":
    APP_DIR = GPU_APP_DIR
    CUDA_MODE = True
else:
    APP_DIR = CPU_APP_DIR
    CUDA_MODE = False

MODE = "Medium"

if "MODE" in os.environ:
     MODE = os.environ["MODE"]

db = StrictRedis(host="localhost",db=0,decode_responses=True)


def objectdetection(thread_name: str, delay: float):

    CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
    ]

    IMAGE_QUEUE = "detection_queue"
    
    reso = 600
   
    if MODE == "High":

        reso = 800   

    elif MODE == "Medium":
        
        reso = 600
    elif MODE == "Low":

        reso = 360

    detector = ObjectDetector(os.path.join(SHARED_APP_DIR,"detr.pth"),reso,cuda=CUDA_MODE)
    while True:
        queue = db.lrange(IMAGE_QUEUE,0,0)

        db.ltrim(IMAGE_QUEUE,len(queue), -1)
        
        if len(queue) > 0:

            for req_data in queue:
                
                req_data = json.JSONDecoder().decode(req_data)


                img_id = req_data["imgid"]
                req_id = req_data["reqid"]
                req_type = req_data["reqtype"]
                threshold = float(req_data["minconfidence"])
               
                try:

                    img = os.path.join(TEMP_PATH,img_id)
            
                    prob,boxes = detector.predict(img,threshold)
                      
                    outputs = []
                    for p,(x_min, y_min, x_max, y_max) in zip(prob,boxes.tolist()):
                        
                        cl = p.argmax()
                        conf = p.max()
                        label = CLASSES[cl]
                        score = conf.item()

                        detection = {"confidence":score,"label":label, "x_min":int(x_min), "y_min":int(y_min),"x_max":int(x_max), "y_max":int(y_max)}
                
                        outputs.append(detection)

                    response = {"success":True,"predictions":outputs}

                    db.set(req_id,json.dumps(response)) 
                    os.remove(img)
                    
                except Exception as e:
 
                    output = {"success":False, "error":"invalid image","code":400}
                    db.set(req_id,json.dumps(output))
                    if os.path.exists(TEMP_PATH + img_id):
                        os.remove(img)
                    continue

                    
        time.sleep(delay)

def face(thread_name,delay):

    
    facedetector = FaceModel(os.path.join(SHARED_APP_DIR,"facedetector-high.model"),CUDA_MODE)
   
    IMAGE_QUEUE = "face_queue"

    while True:

        queue = db.lrange(IMAGE_QUEUE,0,0)

        db.ltrim(IMAGE_QUEUE,len(queue), -1)

        img_size = 400
        detect_size = 1600
        skip = 0
        
        if MODE == "High":
            img_size = 400
            skip = 0
            
        elif MODE == "Low":
            img_size = 260
            skip = 1
            
        else:
            img_size = 360
            skip = 1

        if len(queue) > 0:

            for req_data in queue:

                req_data = json.JSONDecoder().decode(req_data)

                task_type = req_data["reqtype"]
                req_id = req_data["reqid"]

                if task_type == "detect":

                    try:
                        img_id = req_data["imgid"]
                        threshold = float(req_data["minconfidence"])

                        img = TEMP_PATH+img_id
                        
                        image = cv2.imread(img)

                        os.remove(img)
                    
                        bboxs = facedetector.predict(image, img_size=img_size, threshold=threshold)

                        pad_x = max(image.shape[0] - image.shape[1], 0) * (img_size / max(image.shape))
                        pad_y = max(image.shape[1] - image.shape[0], 0) * (img_size / max(image.shape))
                        unpad_h = img_size - pad_y
                        unpad_w = img_size - pad_x 

                        outputs = []
                        for face in bboxs:

                            x_min = int(face[0])
                            
                            y_min = int(face[1])
                            
                            x_max = int(face[2])
                            
                            y_max = int(face[3])

                            box_h = ((y_max - y_min) / unpad_h) * image.shape[0]
                            box_w = ((x_max - x_min) / unpad_w) * image.shape[1]
                            y_min = int(((y_min - pad_y // 2) / unpad_h) * image.shape[0])
                            x_min = int(((x_min - pad_x // 2) / unpad_w) * image.shape[1])

                            y_max = int(y_min + box_h)
                            x_max = int(x_min + box_w)

                            detection = {"confidence":float(face[4]), "x_min":x_min, "y_min":y_min,"x_max":x_max, "y_max":y_max}
                
                            outputs.append(detection)  

                        output = {"success": True, "predictions": outputs}
                        
                        db.set(req_id,json.dumps(output))

                    except Exception as e:
                      
                        output = {"success":False, "error":"invalid image","code":400}
                        db.set(req_id,json.dumps(output))
                        if os.path.exists(img):
                            os.remove(img)


        time.sleep(delay)


if "VISION-DETECTION" in os.environ:

    activate = os.environ["VISION-DETECTION"]

    if activate == "True":
        p = Process(target=objectdetection,args=("",SLEEP_TIME))
        p.start()

if "VISION-FACE" in os.environ:

    activate = os.environ["VISION-FACE"]

    if activate == "True":
        p = Process(target=face,args=("",SLEEP_TIME))
        p.start()