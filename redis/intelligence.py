import sys
import argparse
sys.path.append(r"C:\DeepStack\interpreter\packages")

from redis import StrictRedis, RedisError
import torch
import time
import json
import io
import os
import _thread as thread
from multiprocessing import Process
from sharedintelligence.commons import preprocess
import torchvision.transforms as transform
from PIL import Image
import cv2
from sharedintelligence import TrafficModel, SceneModel, GenderModel, FaceModel,DetectModel3, FaceRecognitionModel, compute_distance, NudityModel, TFCustomModel,OnnxCustomModel,KerasCustomModel
import torch.nn.functional as F
from sharedintelligence.commons.commons import preprocess_image
import ast
import sqlite3
import numpy as np
import warnings
import tensorflow as tf
import sys


parser = argparse.ArgumentParser()
parser.add_argument("-MODE",default="Low")
parser.add_argument("-VFACE",default=False)
parser.add_argument("-VSCENE",default=False)
parser.add_argument("-VDETECTION",default=False)

args = parser.parse_args()

if bool(args.VFACE) == True:

    os.environ["VISION-FACE"] = "True"

if bool(args.VSCENE) == True:

    os.environ["VISION-SCENE"] = "True"

if bool(args.VDETECTION) == True:

    os.environ["VISION-DETECTION"] = "True"

"""
sys.stdout = open(os.devnull, 'w')
"""

tf.logging.set_verbosity(tf.logging.ERROR)

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BATCH_SIZE = 8
SLEEP_TIME = 0.01

#APP_DIR = r"C:\Users\John Olafenwa\Documents\AI Commons\DeepStack Core 4.0\\"
APP_DIR = r"C:\DeepStack\\"
user = os.environ["USERPROFILE"]
DATA_DIR = os.path.join(user,".deepstack/datastore")
TEMP_PATH = os.path.join(user,".deepstack/deeptemp/")


CUDA_MODE = False

if CUDA_MODE == True:
    torch.backends.cudnn.enabled = True

"""
if "BATCH_SIZE" in os.environ:
     BATCH_SIZE = int(os.environ["BATCH_SIZE"])

if "SLEEP_TIME" in os.environ:
    SLEEP_TIME = float(os.environ["SLEEP_TIME"])
"""

MODE = "Medium"
if "MODE" in os.environ:
     MODE = os.environ["MODE"]

db = StrictRedis(host="localhost",db=0,decode_responses=True)

def pad_image(img,img_size):

     pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
     pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
     unpad_h = img_size - pad_y
     unpad_w = img_size - pad_x 

     return pad_x,pad_y,unpad_w,unpad_h

def convert_boxes(img,x_min, y_min, x_max,y_max,unpad_w,unpad_h,pad_x,pad_y):

    box_h = ((y_max - y_min) / unpad_h) * img.shape[0]
    box_w = ((x_max - x_min) / unpad_w) * img.shape[1]
    y_min = int(((y_min - pad_y // 2) / unpad_h) * img.shape[0])
    x_min = int(((x_min - pad_x // 2) / unpad_w) * img.shape[1])

    y_max = int(y_min + box_h)
    x_max = int(x_min + box_w)

    return x_min, y_min, x_max, y_max

def load_faces():

    master_face_map = {"map": {}}

    SELECT_FACE = "SELECT * FROM TB_EMBEDDINGS"
    conn = sqlite3.connect(DATA_DIR+"/faceembedding.db")
    cursor = conn.cursor()
    embeddings = cursor.execute(SELECT_FACE)
    embedding_arr = []
    
    i = 0
    for row in embeddings:

        embedding = row[1]
        user_id = row[0]
        embedding = ast.literal_eval(embedding)
        embedding_arr.append(embedding)
        master_face_map["map"][i] = user_id
        i += 1

    master_face_map["tensors"] = embedding_arr
    facemap = repr(master_face_map)
    db.set("facemap",facemap)

    conn.close()

fp = open(APP_DIR+"coco.names", "r")
coconames = fp.read().split("\n")[:-1]

classes = list()
with open(APP_DIR+"categories_places365.txt") as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
   
placesnames = tuple(classes)

def face(thread_name,delay):

    faceclassifier = FaceRecognitionModel(APP_DIR+"model_ir_se50.pth",cuda=CUDA_MODE)
   
    facedetector = FaceModel(APP_DIR+"facedetector.onnx",CUDA_MODE)
    load_faces()

    label_map = {0:"female",1:"male"}
    
    IMAGE_QUEUE = "face_register_queue"
    ADD_FACE = "INSERT INTO TB_EMBEDDINGS(userid,embedding) VALUES(?,?)"
    UPDATE_FACE = "UPDATE TB_EMBEDDINGS SET embedding = ? where userid = ?"
    SELECT_FACE = "SELECT * FROM TB_EMBEDDINGS where userid = ? "

    IMAGE_QUEUE = "face_queue"

    trans = transform.Compose([
        transform.Resize((112,112)),
        transform.ToTensor(),
        transform.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ])

    face_transforms = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ])

    while True:

        queue = db.lrange(IMAGE_QUEUE,0,BATCH_SIZE - 1)

        db.ltrim(IMAGE_QUEUE,len(queue), -1)

        img_size = 512
        skip = 1
        match_skip = 1

        match_size = 360

        if MODE == "High":
            img_size = 400
            match_size = 360
            skip = 0
            match_skip = 1
        elif MODE == "Low":
            img_size = 512
            match_size = 200
            match_skip = 2
            skip = 2
        else:
            img_size = 460
            match_size = 260
            skip = 1
            match_skip = 2

        if len(queue) > 0:

            batch = None
            imageids = []

            for req_data in queue:

                req_data = json.JSONDecoder().decode(req_data)

                task_type = req_data["reqtype"]

                if task_type == "detect":

                    try:
                        img_id = req_data["imgid"]
                        threshold = float(req_data["minconfidence"])
                        img = cv2.imread(TEMP_PATH+img_id)
                       
                        os.remove(TEMP_PATH+img_id)
                   
                      
                        bboxs = facedetector.predict(img,img_size=img_size, threshold=threshold,skip=skip)
                        
                        
                        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
                        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
                        unpad_h = img_size - pad_y
                        unpad_w = img_size - pad_x 

                        outputs = []
                        batch = []
                        for face in bboxs:
                            

                            x_min = face[0]
                            
                            y_min = face[1]
                            
                            x_max = face[2]
                            
                            y_max = face[3]

                            box_h = ((y_max - y_min) / unpad_h) * img.shape[0]
                            box_w = ((x_max - x_min) / unpad_w) * img.shape[1]
                            y_min = int(((y_min - pad_y // 2) / unpad_h) * img.shape[0])
                            x_min = int(((x_min - pad_x // 2) / unpad_w) * img.shape[1])

                            y_max = int(y_min + box_h)
                            x_max = int(x_min + box_w)

                            detection = {"confidence":float(face[4]), "x_min":x_min, "y_min":y_min,"x_max":x_max, "y_max":y_max}
                   
                            outputs.append(detection) 

                        
                        output = {"success": True, "predictions": outputs}

                        db.set(img_id,json.dumps(output))

                    except Exception as e:

                        output = {"success":False, "error":"invalid image"}
                        db.set(img_id,json.dumps(output))
                        if os.path.exists(TEMP_PATH+img_id):
                            os.remove(TEMP_PATH+img_id)

                elif task_type == "recognize":

                    try:

                        master_face_map = db.get("facemap")
                        master_face_map = ast.literal_eval(master_face_map)
           
                        facemap = master_face_map["map"]
                     
                        face_array = master_face_map["tensors"]

                        if len(face_array) > 0:

                            face_array_tensors = [torch.tensor(emb).unsqueeze(0) for emb in face_array]
                            face_tensors = torch.cat(face_array_tensors)

                        if CUDA_MODE:
                            face_tensors = face_tensors.cuda()

                        img_id = req_data["imgid"]
                        threshold = float(req_data["minconfidence"])
               
                        img = cv2.imread(TEMP_PATH+img_id)

                        pad_x,pad_y,unpad_w,unpad_h = pad_image(img,img_size)

                        pil_image = Image.fromarray(img).convert("RGB")

                        os.remove(TEMP_PATH+img_id)
                    
                        bboxs = facedetector.predict(img,img_size,skip=skip)
                    
                        faces = []
                        detections = []
                        for face in bboxs:
                            found_face = True
                            x_min, y_min, x_max, y_max = convert_boxes(img,face[0],face[1],face[2],face[3],unpad_w,unpad_h,pad_x,pad_y)
                   
                            new_img = pil_image.crop((x_min,y_min,x_max,y_max))
                       
                            faces.append(trans(new_img).unsqueeze(0))
                            detections.append((x_min,y_min,x_max,y_max))

                        if len(faces) < 1:

                            output = {"success":True, "predictions":[]}
                            db.set(img_id,json.dumps(output))
                    
                        elif len(facemap) == 0:

                            predictions = []

                            for face in detections:

                                x_min = int(face[0])
                                if x_min < 0:
                                    x_min = 0
                                _min = int(face[1])
                                if y_min < 0:
                                    y_min = 0
                                x_max = int(face[2])
                                if x_max < 0:
                                    x_max = 0
                                y_max = int(face[3])
                                if y_max < 0:
                                    y_max = 0

                                user_data = {"confidence":0,"userid":"unknown", "x_min":x_min, "y_min":y_min,"x_max":x_max, "y_max":y_max}

                                predictions.append(user_data)

                            output = {"success":True, "predictions":predictions}
                            db.set(img_id,json.dumps(output))

                        else:
                       
                            faces = torch.cat(faces)
                        
                            embeddings = faceclassifier.predict(faces)

                            predictions = []
                        
                            for embedding,face in zip(embeddings,detections):
                            
                                embedding = embedding.unsqueeze(0)
                            
                                embedding_proj = torch.cat([embedding for i in range(face_tensors.size(0))])

                                similarity = F.cosine_similarity(embedding_proj,face_tensors)

                                user_index = similarity.argmax().item()
                                max_similarity = (similarity.max().item() + 1)/2

                                if max_similarity < threshold:
                                    confidence = 0
                                    user_id = "unknown"
                                else:
                                    confidence = max_similarity
                                    user_id = facemap[user_index]


                                x_min = int(face[0])
                                if x_min < 0:
                                    x_min = 0
                                y_min = int(face[1])
                                if y_min < 0:
                                    y_min = 0
                                x_max = int(face[2])
                                if x_max < 0:
                                    x_max = 0
                                y_max = int(face[3])
                                if y_max < 0:
                                    y_max = 0
                            
                                user_data = {"confidence":confidence,"userid":user_id, "x_min":x_min, "y_min":y_min,"x_max":x_max, "y_max":y_max}

                                predictions.append(user_data)
                       
                            output = {"success":True, "predictions":predictions}
                            db.set(img_id,json.dumps(output))

                    except Exception as e:


                        output = {"success":False, "error":"invalid image"}
                        db.set(img_id,json.dumps(output))

                        if os.path.exists(TEMP_PATH+img_id):
                            os.remove(TEMP_PATH+img_id)

                elif task_type == "register":

                    try:
                        
                        user_id = req_data["userid"]
                        req_id = req_data["reqid"]

                        user_images = req_data["images"]

                        conn = sqlite3.connect(DATA_DIR+"/faceembedding.db")

                        batch = None

                        for img_id in user_images:
                        
                            img = cv2.imread(TEMP_PATH+img_id)

                            pad_x,pad_y,unpad_w,unpad_h = pad_image(img,img_size)

                            pil_image = Image.fromarray(img).convert("RGB")
              
                            os.remove(TEMP_PATH+img_id)
                         
                            bboxs = facedetector.predict(img,img_size)
                        
                            new_img = None

                            if len(bboxs) > 0:
                                face = bboxs[0]
                                x_min, y_min, x_max, y_max = convert_boxes(img,face[0],face[1],face[2],face[3],unpad_w,unpad_h,pad_x,pad_y)
                   
                                new_img = pil_image.crop((x_min,y_min,x_max,y_max))
                        
                            if new_img is not None:
                
                                img = trans(new_img).unsqueeze(0)

                                if batch is None:
                                    batch = img
                                else:
                                    batch = torch.cat([batch,img],0)

                        if batch is None:

                            output = {"success":False, "error":"no face detected"}
                            db.set(req_id,json.dumps(output))
                            continue

                        img_embeddings = faceclassifier.predict(batch).cpu()
                    
                        img_embeddings = torch.mean(img_embeddings,0)
                        
                        cursor = conn.cursor()
                
                        emb = img_embeddings.tolist()
                        emb = repr(emb)

                        exist_emb = cursor.execute(SELECT_FACE,(user_id,))

                        user_exist = False

                        for row in exist_emb:
                            user_exist = True
                            break
                
                        if user_exist:

                            cursor.execute(UPDATE_FACE,(emb,user_id))
                            message = "face updated"
                        else:
                            cursor.execute(ADD_FACE,(user_id,emb))
                            message = "face added"
                
                        conn.commit()

                        output = {"success":True, "message":message}
                        db.set(req_id,json.dumps(output))
                
                        conn.close()

                    except Exception as e:
                   
                        output = {"success":False, "error":"invalid image"}
                        db.set(req_id,json.dumps(output))

                        for img_id in user_images:
                            if os.path.exist(TEMP_PATH+img_id):
                                os.remove(TEMP_PATH+img_id)

                elif task_type == "match":

                    try:
                   
                        req_id = req_data["reqid"]

                        user_images = req_data["images"]

                        img1 = user_images[0]
                        img2 = user_images[1]

                        cv_img1 = cv2.imread(TEMP_PATH+img1)
                        image1 = Image.fromarray(cv_img1).convert("RGB")
                        cv_img2 = cv2.imread(TEMP_PATH+img2)
                        image2 = Image.fromarray(cv_img2).convert("RGB")

                        img1_pad_x,img1_pad_y,img1_unpad_w,img1_unpad_h = pad_image(cv_img1,match_size)
                        img2_pad_x,img2_pad_y,img2_unpad_w,img2_unpad_h = pad_image(cv_img2 ,match_size)


                        os.remove(TEMP_PATH+img1)
                        os.remove(TEMP_PATH+img2)
                    
                        bboxs1 = facedetector.predict(cv_img1,match_size,skip=match_skip)
                        bboxs2 = facedetector.predict(cv_img2,match_size,skip=match_skip)

                        if len(bboxs1) == 0 or len(bboxs2) == 0:

                            output = {"success":False, "error":"no face found"}
                            db.set(req_id,json.dumps(output)) 
                            continue

                        face1 = bboxs1[0]
                        face2 = bboxs2[0]
                        img1_x_min, img1_y_min, img1_x_max, img1_y_max = convert_boxes(cv_img1,face1[0],face1[1],face1[2],face1[3],img1_unpad_w,img1_unpad_h,img1_pad_x,img1_pad_y)
                        img2_x_min, img2_y_min, img2_x_max, img2_y_max = convert_boxes(cv_img2,face2[0],face2[1],face2[2],face2[3],img2_unpad_w,img2_unpad_h,img2_pad_x,img2_pad_y)
                        face1 = trans(image1.crop((img1_x_min,img1_y_min,img1_x_max,img1_y_max))).unsqueeze(0)
                        face2 = trans(image2.crop((img2_x_min,img2_y_min,img2_x_max,img2_y_max))).unsqueeze(0)

                        faces = torch.cat([face1,face2],dim=0)

                        embeddings = faceclassifier.predict(faces)

                        embed1 = embeddings[0,:].unsqueeze(0)
                        embed2 = embeddings[1,:].unsqueeze(0)
                   
                        similarity = (F.cosine_similarity(embed1,embed2).item() + 1)/2

                        output = {"success":True, "similarity":similarity}
                        db.set(req_id,json.dumps(output))           

                    except Exception as e:
                        
                        output = {"success":False, "error":"invalid image"}
                        db.set(req_id,json.dumps(output))

        time.sleep(delay)


def scene(thread_name,delay):
    IMAGE_QUEUE = "scene_queue"
    classifier = SceneModel(APP_DIR+"scene.onnx",CUDA_MODE)

    trans = trans = transform.Compose([
        transform.Resize((256,256)),
        transform.CenterCrop(224),
        transform.ToTensor(),
        transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    i = 0
    while True:
        queue = db.lrange(IMAGE_QUEUE,0,BATCH_SIZE - 1)
        
        if len(queue) > 0:

            db.ltrim(IMAGE_QUEUE,len(queue), -1)

            batch = None
            imageids = []

            for img_id in queue:
              
                try:

                    img = Image.open(TEMP_PATH+img_id).convert("RGB")
                    img = img.resize((256,256))
                    img = numpy.array(img)
                    img = preprocess(img)
                    img = numpy.transpose(img,(2,0,1))

                    img = numpy.expand_dims(img,0).astype(numpy.float32)
                    os.remove( TEMP_PATH + img_id)

                    if batch is None:
                        batch = img
                   
                    else:
                        batch = torch.cat([batch,img],0)

                    imageids.append(img_id)

                except:

                    output = {"success":False, "error":"invalid image"}
                    db.set(img_id,json.dumps(output))
                    if os.path.exists(TEMP_PATH + img_id):
                        os.remove( TEMP_PATH + img_id)

            if batch is not None:
                class_preds, confs = classifier.predict(batch)

                for img_id, cl, conf in zip(imageids,class_preds,confs):
                
                    cl = placesnames[cl.item()]
                    conf = conf.item()

                    output = {"success":True, "label":cl, "confidence":conf}
                    db.set(img_id,json.dumps(output))
            
            
        
        time.sleep(delay)


def scene2(thread_name,delay):
    IMAGE_QUEUE = "scene_queue"
    classifier = SceneModel(APP_DIR+"scene.onnx",CUDA_MODE)

    while True:
        queue = db.lrange(IMAGE_QUEUE,0,BATCH_SIZE - 1)
        
        if len(queue) > 0:

            db.ltrim(IMAGE_QUEUE,len(queue), -1)

            for img_id in queue:
              
                try:

                    img = Image.open(TEMP_PATH+img_id).convert("RGB")

                    """
                    
                    img = img.resize((224,224))
                    img = np.array(img)
                    img = preprocess(img)
                    img = np.transpose(img,(2,0,1))
                    """

                    trans = trans = transform.Compose([
                        transform.Resize((256,256)),
                        transform.CenterCrop(224),
                        transform.ToTensor(),
                        transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    img = trans(img)
                    img = img.numpy()
                    img = np.expand_dims(img,0).astype(np.float32)
                    os.remove( TEMP_PATH + img_id)

                    cl , conf = classifier.predict(img)

                    cl = placesnames[cl]

                    conf = float(conf)

                    output = {"success":True, "label":cl, "confidence":conf}
                    db.set(img_id,json.dumps(output))

                except Exception as e:

                    output = {"success":False, "error":"invalid image"}
                    db.set(img_id,json.dumps(output))
                    if os.path.exists(TEMP_PATH + img_id):
                        os.remove( TEMP_PATH + img_id)

        time.sleep(delay)

def detection(thread_name,delay):


    labels_map = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                                 6: 'train',
                                 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                                 12: 'parking meter',
                                 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
                                 20: 'elephant',
                                 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
                                 27: 'tie',
                                 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball',
                                 33: 'kite',
                                 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
                                 38: 'tennis racket',
                                 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
                                 45: 'bowl',
                                 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                                 52: 'hot dog',
                                 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                                 59: 'bed',
                                 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                                 66: 'keyboard',
                                 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                                 72: 'refrigerator',
                                 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                                 78: 'hair dryer',
                                 79: 'toothbrush'}

    IMAGE_QUEUE = "detection_queue"
    

    nms = 0.45
    reso = 640
    model_path = ""

    if MODE == "High":

        reso = 640
        model_path = "yolov3_cpu_nms.pb"
        

    elif MODE == "Medium":
        
        reso = 416
        model_path = "yolov3_cpu_nms_416.pb"

    else:

        reso = 320
        model_path = "yolov3_cpu_nms_320.pb"

    detector = DetectModel3(model_path=APP_DIR+model_path,reso=reso)
    while True:
        queue = db.lrange(IMAGE_QUEUE,0,BATCH_SIZE - 1)

        db.ltrim(IMAGE_QUEUE,len(queue), -1)
        
        if len(queue) > 0:

            for req_data in queue:
                
                req_data = json.JSONDecoder().decode(req_data)


                img_id = req_data["imgid"]
                threshold = float(req_data["minconfidence"])
               
                if True:

                    try:

                        img = Image.open(TEMP_PATH+img_id).convert("RGB")
                        detector.confidence = threshold
                
                        boxes, scores, labels = detector.predict(img)

                        os.remove( TEMP_PATH + img_id)
                        
                    except Exception as e:

                        output = {"success":False, "error":"invalid image"}
                        db.set(img_id,json.dumps(output))
                        if os.path.exists(TEMP_PATH + img_id):
                            os.remove(TEMP_PATH + img_id)
                        continue

                    outputs = []

                   
                    try:
                        for i in range(len(labels)):

                            label = labels[i]
                            bbox = boxes[i]
                            confidence = float(scores[i])

                            detection_size, original_size = np.array([reso,reso]), np.array(img.size)
                            ratio = original_size / detection_size
                            obj = list((bbox.reshape(2,2) * ratio).reshape(-1))

                            x_min = int(obj[0])
                            if x_min < 0:
                                x_min = 0
                            y_min = int(obj[1])
                            if y_min < 0:
                                y_min = 0
                            x_max = int(obj[2])
                            if x_max < 0:
                                x_max = 0
                            y_max = int(obj[3])
                            if y_max < 0:
                                y_max = 0

                            detection = {"confidence":confidence,"label":labels_map[label], "x_min":x_min, "y_min":y_min,"x_max":x_max, "y_max":y_max}
                   
                            outputs.append(detection) 
                            
                    except Exception as e:
                            pass

                    response = {"success":True,"predictions":outputs}

                    db.set(img_id,json.dumps(response))   
                else:
                    output = {"success":False, "error":"invalid image"}
                    db.set(img_id,json.dumps(output))

        time.sleep(delay)

def custom(model_name,delay):
    IMAGE_QUEUE = model_name+"_queue"
    
    with open(DATA_DIR + "/models/vision/"+model_name + "/config.json") as f:
        config = json.load(f)
    
    label_map = config["map"]
    mean = config["mean"]
    std = config["std"]
    framework = config["framework"]
    width = config["width"]
    height = config["height"]

    if framework == "TF":

        input_name = config["input_name"]
        output_name = config["output_name"]

        classifier = TFCustomModel(DATA_DIR + "/models/vision/"+model_name + "/model.pb",input_name=input_name,output_name=output_name,cuda=CUDA_MODE)
    
    elif framework == "ONNX":
        classifier = OnnxCustomModel(DATA_DIR + "/models/vision/"+model_name + "/model.pb")

    elif framework == "KERAS":
        classifier = KerasCustomModel(DATA_DIR + "/models/vision/"+model_name + "/model.pb")
    else:
        return

    while True:
        queue = db.lrange(IMAGE_QUEUE,0,BATCH_SIZE - 1)

       
        if len(queue) > 0:

            db.ltrim(IMAGE_QUEUE,len(queue), -1)

            for img_id in queue:

                try:

                    img = Image.open(TEMP_PATH+img_id).convert("RGB")
                    img = img.resize((width,height))
                    img = np.array(img)
                    img = preprocess(img,mean=mean,scale=std)

                    if framework == "ONNX":
                        img = np.transpose(img,(2,0,1))

                    img = np.expand_dims(img,0).astype(np.float32)
                  
                    os.remove( TEMP_PATH + img_id)
               
                    class_idx, confidence = classifier.predict(img)

                    cl = label_map[str(class_idx)]

                    output = {"success":True, "label":cl, "confidence":float(confidence)}
                    db.set(img_id,json.dumps(output))  

                except Exception as e:


                    output = {"success":False, "error":"invalid image"}
                    db.set(img_id,json.dumps(output))
            
        time.sleep(delay)


def nudity(model_name,delay):
    IMAGE_QUEUE = "nudity_queue"
    classifier = NudityModel(APP_DIR + "nudity.pth",CUDA_MODE)
    
    with open(APP_DIR+ "nudity-info.json") as f:
        label_map = json.load(f)["map"]
  
    trans = transform.Compose([transform.CenterCrop((224,224)),transform.ToTensor(),transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    while True:
        queue = db.lrange(IMAGE_QUEUE,0,BATCH_SIZE - 1)
       
        if len(queue) > 0:

            db.ltrim(IMAGE_QUEUE,len(queue), -1)

            batch = None
            imageids = []

            for img_id in queue:
             
                try:

                    img = Image.open(TEMP_PATH+img_id).convert("RGB")
                    img = trans(img).unsqueeze(0)
                    os.remove( TEMP_PATH + img_id)
               
                    if batch is None:
                        batch = img
                   
                    else:
                        batch = torch.cat([batch,img],0)

                    imageids.append(img_id)

                except Exception as e:

                    output = {"success":False, "error":"invalid image"}
                    db.set(img_id,json.dumps(output))
                    if os.path.exists(TEMP_PATH + img_id):
                        os.remove( TEMP_PATH + img_id)
            
            if batch is not None:
                class_preds, confs = classifier.predict(batch)

                for img_id, cl, conf in zip(imageids,class_preds,confs):
                
                    cl = label_map[str(cl.item())]
                    conf = conf.item()

                    output = {"success":True, "label":cl, "confidence":conf}
                    db.set(img_id,json.dumps(output))      
            
        time.sleep(delay)

def update_faces(thread_name,delay):

    while True:

        load_faces()

        time.sleep(delay)

p1 = Process(target=update_faces,args=("",1))
p1.start()

if os.path.exists(DATA_DIR + "/models/vision"):

    for model in os.listdir(DATA_DIR + "/models/vision"):

        p = Process(target=custom,args=(model,SLEEP_TIME))
        p.start()

if "VISION-DETECTION" in os.environ:

    activate = bool(os.environ["VISION-DETECTION"])

    if activate:
        p = Process(target=detection,args=("",SLEEP_TIME))
        p.start()

if "VISION-SCENE" in os.environ:

    activate = bool(os.environ["VISION-SCENE"])

    if activate:
        p = Process(target=scene2,args=("",SLEEP_TIME))
        p.start()

if "VISION-FACE" in os.environ:

    activate = bool(os.environ["VISION-FACE"])

    if activate:

        p = Process(target=face,args=("",SLEEP_TIME))
        p.start()

        

