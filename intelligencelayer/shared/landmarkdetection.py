from intelligencelayer.shared.facelandmark.utils import landmarks_draw, transform_img
from intelligencelayer.shared.superresolution import SHARED_APP_DIR
from shared import SharedOptions
from PIL import Image
from io import BytesIO
import numpy as np
import sys
import os
import torch
from multiprocessing import process
import base64
import time
from threading import Thread
from queue import Queue
import cv2
from imutils import face_utils
import dlib
import json

from facelandmark.utils import load_model

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "."))
if SharedOptions.PROFILE == "windows_native":
    sys.path.append(os.path.join(SharedOptions.APP_DIR, "windows_packages"))

def get_network(model_path):
    from facelandmark.config import config
    from facelandmark.network import XceptionNet

    # an ugly operation
    if 'KERNEL_PATH' in config.MODEL:
        config.MODEL.KERNEL_PATH = config.MODEL.KERNEL_PATH.replace('../', '')

    return config, XceptionNet(config)

SHARED_APP_DIR = SharedOptions.SHARED_APP_DIR
CUDA_MODE = SharedOptions.CUDA_MODE
db = SharedOptions.db
TEMP_PATH = SharedOptions.TEMP_PATH

model_path=os.path.join(
    SHARED_APP_DIR, SharedOptions.SETTINGS.FACELANDMARK_MODEL)

use_cuda= SharedOptions.CUDA_MODE
use_cpu=False if use_cuda else True

config, model=get_network(model_path)
if use_cuda:
    device=torch.device('cuda')
else:
    device=torch.device('cpu')

model=model.to(device)
load_model(model, model_path, strict=True, cpu=use_cpu)

scale=config.MODEL.SCALE

detector=dlib.get_frontal_face_detector()

def run_task(q):
    while True:
        req_data=q.get()

        try:
            with torch.no_grad():
                img_id=req_data['imgid']
                req_id=req_data['reqid']
                req_type=req_data['reqtype']
                img_path=os.path.join(TEMP_PATH, img_id)


                img=Image.open(img_path)

                def inference(frame):
                    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces=detector(gray,1)
                    outputs=[]
                    for (i,face) in enumerate(faces):
                        (x,y,w,h)=face_utils.rect_to_bb(face)
                        crop_img=gray[y:y+h, x:x+w]
                        transformed_img=transform_img(crop_img)
                        landmarks_predictions=model(transformed_img.cpu())
                        outputs.append(landmarks_predictions.cpu(),(x,y,w,h))
                        return landmarks_draw(frame, outputs)
                output=inference(img)
                output_pil=Image.fromarray(output)
                buffered = BytesIO()
                output_pil.save(buffered, format="JPEG")
                base64_img = base64.b64encode(buffered.getvalue())
                output_response={
                    "Sucess":True,
                    "base64_img":base64_img.decode('utf-8'),
                }
        except Exception as e:
            output_response={
                    "Sucess":False,
                    "error":"An error occured while processing the request",
                    "code": 500,
                }
        finally:
            db.set(req_id, json.dumps(output_response))
            if os.path.exists(img_path):
                os.remove(img_path)
def landmarkdetection2x(delay:float):
    q = Queue(maxsize=0)
    t = Thread(target=run_task, args=(q,))
    t.daemon = True
    t.start()
    while True:
        try:
            req_data=db.get(db.keys()[0])
            if req_data:
                q.put(json.loads(req_data))
            time.sleep(delay)
        except Exception as e:
            print(e)
            time.sleep(delay)

if __name__ == "__main__":
    landmarkdetection2x(SharedOptions.SLEEP_TIME)


            

