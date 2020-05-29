from redis import StrictRedis, RedisError
import torch
import time
import json
import io
import os
import _thread as thread
from multiprocessing import Process
import sharedintelligence.shared.commons.transforms as transform
from PIL import Image
import cv2
import torch.nn.functional as F
import ast
import sqlite3
import numpy as np
import warnings
import tensorflow as tf
import sys
from sharedintelligence.shared.commons import preprocess
from sharedintelligence.shared.commons.commons import preprocess_image
from sharedintelligence.shared import KerasCustomModel,OnnxCustomModel,TFCustomModel,SceneModel
from sharedintelligence.shared.commons import compute_distance
from sharedintelligence.shared.detection import SharedDetectionModel
"""
sys.stdout = open(os.devnull, 'w')
"""
tf.logging.set_verbosity(tf.logging.ERROR)

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if "CUDA_MODE" in os.environ:
    CUDA_MODE = os.environ["CUDA_MODE"]
else:
    CUDA_MODE = "False"

if "APPDIR" in os.environ:
    APPDIR = os.environ["APPDIR"]
else:
    APPDIR = "."

TEMP_PATH = "/deeptemp/"
BATCH_SIZE = 8
SLEEP_TIME = 0.01
SHARED_APP_DIR = APPDIR+"/sharedfiles/"
GPU_APP_DIR = APPDIR+"/gpufiles/"
CPU_APP_DIR = APPDIR+"/cpufiles/"
DATA_DIR = "/datastore"


if CUDA_MODE == "True":
    APP_DIR = GPU_APP_DIR
    CUDA_MODE = True
    from sharedintelligence.gpu.face import FaceModel, FaceModel2, FaceRecognitionModel
    from sharedintelligence.gpu.detection import DetectModel
else:
    APP_DIR = CPU_APP_DIR
    CUDA_MODE = False
    from sharedintelligence.cpu.face import FaceModel,FaceModel2, FaceRecognitionModel
    from sharedintelligence.cpu.detection3 import DetectModel3

   
"""
if "BATCH_SIZE" in os.environ:
     BATCH_SIZE = int(os.environ["BATCH_SIZE"])

if "SLEEP_TIME" in os.environ:
    SLEEP_TIME = float(os.environ["SLEEP_TIME"])
"""

MODE = "Medium"

if "MODE" in os.environ:
     MODE = os.environ["MODE"]

TB_EMBEDDINGS = "TB_EMBEDDINGS"

if "VISION-FACE2" in os.environ:
    TB_EMBEDDINGS = "TB_EMBEDDINGS2"

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
    
    SELECT_FACE = "SELECT * FROM {}".format(TB_EMBEDDINGS)
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

fp = open(SHARED_APP_DIR+"coco.names", "r")
coconames = fp.read().split("\n")[:-1]

classes = list()
with open(SHARED_APP_DIR+"categories_places365.txt") as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
   
placesnames = tuple(classes)

def face(thread_name,delay):

    faceclassifier = FaceRecognitionModel(SHARED_APP_DIR+"facerec-high.model",cuda=CUDA_MODE)
   
    facedetector2 = FaceModel2(APP_DIR+"facebox.model",CUDA_MODE)
    facedetector = FaceModel(APP_DIR+"facedetector-high.model",CUDA_MODE)
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

        queue = db.lrange(IMAGE_QUEUE,0,0)

        db.ltrim(IMAGE_QUEUE,len(queue), -1)

        img_size = 400
        detect_size = 1600
        skip = 0
        

        if MODE == "High":
            img_size = 400
            detect_size = 1600
            skip = 0
            
        elif MODE == "Low":
            img_size = 260
            detect_size = 1000
            skip = 1
            
        else:
            img_size = 360
            detect_size = 1200
            skip = 1

        if len(queue) > 0:


            batch = None
            imageids = []

            for req_data in queue:

                req_data = json.JSONDecoder().decode(req_data)

                task_type = req_data["reqtype"]
                req_id = req_data["reqid"]

                if task_type == "detect":

                    try:
                        img_id = req_data["imgid"]
                        threshold = float(req_data["minconfidence"])

                        img = TEMP_PATH+img_id
                        
                        bboxs = facedetector2.predict(img,img_size=detect_size, threshold=threshold)

                        os.remove(TEMP_PATH+img_id)

                        outputs = []
                        batch = []

                        for face in bboxs:

                            confidence = face[len(face)-1]
                            if confidence < threshold:
                                continue
                            
                            x_min = int(face[0])
                            
                            y_min = int(face[1])
                            
                            x_max = int(face[2])
                            
                            y_max = int(face[3])

                            detection = {"confidence":float(confidence), "x_min":x_min, "y_min":y_min,"x_max":x_max, "y_max":y_max}
                    
                            outputs.append(detection) 

                        
                        output = {"success": True, "predictions": outputs}
                        
                        db.set(req_id,json.dumps(output))

                    except Exception as e:
                      
                        output = {"success":False, "error":"invalid image","code":400}
                        db.set(req_id,json.dumps(output))
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

                        if CUDA_MODE and len(face_array) > 0:
                            face_tensors = face_tensors.cuda()

                        img_id = req_data["imgid"]
                        threshold = float(req_data["minconfidence"])
               
                        img = cv2.imread(TEMP_PATH+img_id)

                        pad_x,pad_y,unpad_w,unpad_h = pad_image(img,img_size)

                        pil_image = Image.fromarray(img).convert("RGB")

                        os.remove(TEMP_PATH+img_id)
                    
                        bboxs = facedetector.predict(img,img_size)
                    
                        faces = [[]]
                        detections = []

                        found_face = False
                        
                        for face in bboxs:
                            found_face = True
                            x_min, y_min, x_max, y_max = convert_boxes(img,face[0],face[1],face[2],face[3],unpad_w,unpad_h,pad_x,pad_y)
                   
                            new_img = pil_image.crop((x_min,y_min,x_max,y_max))

                            img_tensor = trans(new_img).unsqueeze(0)
                            
                            if len(faces[-1]) % 10 == 0 and len(faces[-1]) > 0:
                                faces.append([img_tensor])
                               
                            else:
                                faces[-1].append(img_tensor)
                                
                       
                            detections.append((x_min,y_min,x_max,y_max))

                        if found_face == False:

                            output = {"success":True, "predictions":[]}
                            db.set(req_id,json.dumps(output))
                    
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
                            db.set(req_id,json.dumps(output))

                        else:
                            
                            embeddings = []
                            for face_list in faces:
                               
                                embedding = faceclassifier.predict(torch.cat(face_list))
                                embeddings.append(embedding)
                                
                            embeddings = torch.cat(embeddings)
                       
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
                            db.set(req_id,json.dumps(output))

                    except Exception as e:


                        output = {"success":False, "error":"invalid image","code":400}
                        db.set(req_id,json.dumps(output))

                        if os.path.exists(TEMP_PATH+img_id):
                            os.remove(TEMP_PATH+img_id)

                elif task_type == "register":

                    try:
                        
                        user_id = req_data["userid"]

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

                            output = {"success":False, "error":"no face detected","code":400}
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
                   
                        output = {"success":False, "error":"invalid image","code":400}
                        db.set(req_id,json.dumps(output))

                        for img_id in user_images:
                            if os.path.exist(TEMP_PATH+img_id):
                                os.remove(TEMP_PATH+img_id)

                elif task_type == "match":

                    try:

                        user_images = req_data["images"]

                        img1 = user_images[0]
                        img2 = user_images[1]

                        cv_img1 = cv2.imread(TEMP_PATH+img1)
                        image1 = Image.fromarray(cv_img1).convert("RGB")
                        cv_img2 = cv2.imread(TEMP_PATH+img2)
                        image2 = Image.fromarray(cv_img2).convert("RGB")

                        img1_pad_x,img1_pad_y,img1_unpad_w,img1_unpad_h = pad_image(cv_img1,img_size)
                        img2_pad_x,img2_pad_y,img2_unpad_w,img2_unpad_h = pad_image(cv_img2 ,img_size)


                        os.remove(TEMP_PATH+img1)
                        os.remove(TEMP_PATH+img2)
                    
                        bboxs1 = facedetector.predict(cv_img1,img_size)
                        bboxs2 = facedetector.predict(cv_img2,img_size)

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
                        
                        output = {"success":False, "error":"invalid image","code":400}
                        db.set(req_id,json.dumps(output))


        time.sleep(delay)


def face2(thread_name,delay):

    faceclassifier = FaceRecognitionModel(SHARED_APP_DIR+"facerec.model",cuda=CUDA_MODE)
   
    facedetector2 = FaceModel2(APP_DIR+"facebox.model",CUDA_MODE)
    load_faces()

    label_map = {0:"female",1:"male"}
    
    IMAGE_QUEUE = "face_register_queue"
    ADD_FACE = "INSERT INTO TB_EMBEDDINGS2(userid,embedding) VALUES(?,?)"
    UPDATE_FACE = "UPDATE TB_EMBEDDINGS2 SET embedding = ? where userid = ?"
    SELECT_FACE = "SELECT * FROM TB_EMBEDDINGS2 where userid = ? "

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

        img_size = 400
        detect_size = 1600 

        if MODE == "High":
            detect_size = 1600
           
        elif MODE == "Low":
            detect_size = 1000
            
        else:
            detect_size = 1200

        if len(queue) > 0:

            batch = None
            imageids = []

            

            for req_data in queue:

                req_data = json.JSONDecoder().decode(req_data)

                task_type = req_data["reqtype"]
                req_id = req_data["reqid"]

                if task_type == "detect":

                    try:
                        img_id = req_data["imgid"]
                        threshold = float(req_data["minconfidence"])

                        img = TEMP_PATH+img_id
                       
                        bboxs = facedetector2.predict(img,img_size=detect_size, threshold=threshold)

                        os.remove(TEMP_PATH+img_id)

                        outputs = []
                        batch = []

                        for face in bboxs:

                            confidence = face[len(face)-1]
                            if confidence < threshold:
                                continue
                            
                            x_min = int(face[0])
                            
                            y_min = int(face[1])
                            
                            x_max = int(face[2])
                            
                            y_max = int(face[3])

                            detection = {"confidence":float(confidence), "x_min":x_min, "y_min":y_min,"x_max":x_max, "y_max":y_max}
                   
                            outputs.append(detection) 

                        
                        output = {"success": True, "predictions": outputs}

                        db.set(req_id,json.dumps(output))

                    except Exception as e:

                        output = {"success":False, "error":"invalid image","code":400}
                        db.set(req_id,json.dumps(output))
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

                        if CUDA_MODE and len(face_array) > 0:
                            face_tensors = face_tensors.cuda()

                        img_id = req_data["imgid"]
                        threshold = float(req_data["minconfidence"])
               
                        bboxs = facedetector2.predict(TEMP_PATH+img_id,img_size=detect_size, threshold=0.75)

                        pil_image = Image.open(TEMP_PATH+img_id).convert("RGB")

                        os.remove(TEMP_PATH+img_id)
                    
                        faces = [[]]
                        detections = []

                        found_face = False
                        
                        for face in bboxs:
                            
                            confidence = face[len(face)-1]
                            if confidence < threshold:
                                continue
                            
                            x_min = int(face[0])
                            
                            y_min = int(face[1])
                            
                            x_max = int(face[2])
                            
                            y_max = int(face[3])

                            found_face = True
                   
                            new_img = pil_image.crop((x_min,y_min,x_max,y_max))

                            img_tensor = trans(new_img).unsqueeze(0)
                            
                            if len(faces[-1]) % 10 == 0 and len(faces[-1]) > 0:
                                faces.append([img_tensor])
                               
                            else:
                                faces[-1].append(img_tensor)
                                
                       
                            detections.append((x_min,y_min,x_max,y_max))

                        if found_face == False:

                            output = {"success":True, "predictions":[]}
                            db.set(req_id,json.dumps(output))
                    
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
                            db.set(req_id,json.dumps(output))

                        else:
                            
                            embeddings = []
                            for face_list in faces:
                               
                                embedding = faceclassifier.predict(torch.cat(face_list))
                                embeddings.append(embedding)
                                
                            embeddings = torch.cat(embeddings)
                       
                            predictions = []
                        
                            for embedding,face in zip(embeddings,detections):
                            
                                embedding = embedding.unsqueeze(0)
                            
                                embedding_proj = torch.cat([embedding for i in range(face_tensors.size(0))])

                                similarity = F.cosine_similarity(embedding_proj,face_tensors)

                                user_index = similarity.argmax().item()
                                max_similarity = min((similarity.max().item() + 1)/2,1)

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
                            db.set(req_id,json.dumps(output))

                    except Exception as e:

                     

                        output = {"success":False, "error":"invalid image","code":400}
                        db.set(req_id,json.dumps(output))

                        if os.path.exists(TEMP_PATH+img_id):
                            os.remove(TEMP_PATH+img_id)

                elif task_type == "register":

                    try:
                        
                        user_id = req_data["userid"]
                       
                        user_images = req_data["images"]

                        conn = sqlite3.connect(DATA_DIR+"/faceembedding.db")

                        batch = None

                        for img_id in user_images:
                        
                            pil_image = Image.open(TEMP_PATH+img_id).convert("RGB")
                         
                            bboxs = facedetector2.predict(TEMP_PATH+img_id,img_size=detect_size, threshold=0.75)

                            os.remove(TEMP_PATH+img_id)
                        
                            new_img = None

                            for face in bboxs:
                            
                                confidence = face[len(face)-1]
                                if confidence < 0.75:
                                    continue
                            
                                x_min = int(face[0])
                            
                                y_min = int(face[1])
                            
                                x_max = int(face[2])
                            
                                y_max = int(face[3])
                   
                                new_img = pil_image.crop((x_min,y_min,x_max,y_max))
                                break

                            if new_img is not None:
                
                                img = trans(new_img).unsqueeze(0)

                                if batch is None:
                                    batch = img
                                else:
                                    batch = torch.cat([batch,img],0)

                        if batch is None:

                            output = {"success":False, "error":"no face detected","code":400}
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
                   
                        output = {"success":False, "error":"invalid image","code":400}
                        db.set(req_id,json.dumps(output))

                        for img_id in user_images:
                            if os.path.exist(TEMP_PATH+img_id):
                                os.remove(TEMP_PATH+img_id)

                elif task_type == "match":

                    try:
                   
                        user_images = req_data["images"]

                        img1 = user_images[0]
                        img2 = user_images[1]

                        image1 = Image.open(TEMP_PATH+img1).convert("RGB")
                        
                        image2 = Image.open(TEMP_PATH+img2).convert("RGB")

                        bboxs1 = facedetector2.predict(TEMP_PATH+img1,img_size=detect_size, threshold=0.75)
                        bboxs2 = facedetector2.predict(TEMP_PATH+img2,img_size=detect_size, threshold=0.75)

                        os.remove(TEMP_PATH+img1)
                        os.remove(TEMP_PATH+img2)

                        face1 = None
                        face2 = None

                        for face in bboxs1:
                            
                                confidence = face[len(face)-1]
                                if confidence < 0.75:
                                    continue
                            
                                x_min = int(face[0])
                            
                                y_min = int(face[1])
                            
                                x_max = int(face[2])
                            
                                y_max = int(face[3])
                   
                                new_img = image1.crop((x_min,y_min,x_max,y_max))
                                face1 = trans(new_img).unsqueeze(0)
                                break

                        for face in bboxs2:
                            
                                confidence = face[len(face)-1]
                                if confidence < 0.75:
                                    continue
                            
                                x_min = int(face[0])
                            
                                y_min = int(face[1])
                            
                                x_max = int(face[2])
                            
                                y_max = int(face[3])
                   
                                new_img = image2.crop((x_min,y_min,x_max,y_max))
                                face2 = trans(new_img).unsqueeze(0)
                                break

                        if face1 is None or face2 is None:

                            output = {"success":False, "error":"no face found","code":400}
                            db.set(req_id,json.dumps(output)) 
                            continue

                        faces = torch.cat([face1,face2],dim=0)

                        embeddings = faceclassifier.predict(faces)

                        embed1 = embeddings[0,:].unsqueeze(0)
                        embed2 = embeddings[1,:].unsqueeze(0)
                   
                        similarity = min((F.cosine_similarity(embed1,embed2).item() + 1)/2,1)

                        output = {"success":True, "similarity":similarity}
                        db.set(req_id,json.dumps(output))           

                    except Exception as e:
                        print(e)
                        
                        output = {"success":False, "error":"invalid image","code":400}
                        db.set(req_id,json.dumps(output))

        time.sleep(delay)

def scene2(thread_name,delay):
    IMAGE_QUEUE = "scene_queue"
    classifier = SceneModel(SHARED_APP_DIR+"scene.model",CUDA_MODE)

    while True:
        queue = db.lrange(IMAGE_QUEUE,0,BATCH_SIZE - 1)
        
        if len(queue) > 0:

            db.ltrim(IMAGE_QUEUE,len(queue), -1)

            for req_data in queue:
                req_data = json.JSONDecoder().decode(req_data)
                img_id = req_data["imgid"]
                req_id = req_data["reqid"]
                req_type = req_data["reqtype"]
                try:
                    
                   
                    img = Image.open(TEMP_PATH+img_id).convert("RGB")

                    """
                    
                    img = img.resize((224,224))
                    img = np.array(img)
                    img = preprocess(img)
                    img = np.transpose(img,(2,0,1))
                    """

                    trans = transform.Compose([
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

                    db.set(req_id,json.dumps(output))

                except Exception as e:
                   
                    output = {"success":False, "error":"invalid image","code":400}
                    db.set(req_id,json.dumps(output))
                    if os.path.exists(TEMP_PATH + img_id):
                        os.remove( TEMP_PATH + img_id)

        time.sleep(delay)




def detectiongpu(thread_name,delay):
    IMAGE_QUEUE = "detection_queue"

    nms = 0.45
    reso = 640

    if MODE == "High":
        reso = 640

    elif MODE == "Medium":
        reso = 416

    else:
        reso = 320

    detector = DetectModel(model_path=APP_DIR+"yolov3.weights",config_path=APP_DIR+"yolov3.cfg",names_path=APP_DIR+"coco.names",cuda=CUDA_MODE,nms=nms,reso=reso)
               
    detector.nms = 0.3
    detector.inp_dim = reso
    detector.model.net_info["height"] = reso

    while True:

        queue = db.lrange(IMAGE_QUEUE,0,BATCH_SIZE - 1)

        db.ltrim(IMAGE_QUEUE,len(queue), -1)
        
        if len(queue) > 0:

            batch = []
            nms_batch = []
            imageids = []
            reqids = []
            
            for req_data in queue:
                
                req_data = json.JSONDecoder().decode(req_data)

                img_id = req_data["imgid"]
                req_id = req_data["reqid"]
                req_type = req_data["reqtype"]
                threshold = float(req_data["minconfidence"])
               
                try:
                   
                    img = np.asarray(Image.open(TEMP_PATH+img_id).convert("RGB"))
                    os.remove( TEMP_PATH + img_id)
                    batch.append(img)
                    nms_batch.append(threshold)
                    imageids.append(img_id)
                    reqids.append(req_id)

                except Exception as e:
                    print(e)
                    output = {"success":False, "error":"invalid image","code":400}
                    db.set(req_id,json.dumps(output))

                    if os.path.exists(TEMP_PATH + img_id):
                        os.remove( TEMP_PATH + img_id)
                    
                    continue

           
            outputs_ = detector.predict(batch)
            

            if len(outputs_) == 0:
                response = {"success":True,"predictions":[]}
                db.set(img_id,json.dumps(response))   


            for i in range(len(outputs_)):

                output = outputs_[i]
                nms = nms_batch[i]
                img_id = imageids[i]
                req_id = reqids[i]

                outputs = []
                for obj in output:

                    if obj[6] < nms:
                        continue

                    x_min = int(obj[1])
                    if x_min < 0:
                        x_min = 0
                    y_min = int(obj[2])
                    if y_min < 0:
                        y_min = 0
                    x_max = int(obj[3])
                    if x_max < 0:
                        x_max = 0
                    y_max = int(obj[4])
                    if y_max < 0:
                        y_max = 0

                    detection = {"confidence":obj[6].item(),"label":coconames[int(obj[-1])], "x_min":x_min, "y_min":y_min,"x_max":x_max, "y_max":y_max}
                   
                    outputs.append(detection) 

                response = {"success":True,"predictions":outputs}
                db.set(req_id,json.dumps(response))   

        time.sleep(delay)


def detectioncpu(thread_name,delay):


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
                req_id = req_data["reqid"]
                req_type = req_data["reqtype"]
                threshold = float(req_data["minconfidence"])
               
                if True:

                    try:

                        img = Image.open(TEMP_PATH+img_id).convert("RGB")
                        detector.confidence = threshold
                
                        boxes, scores, labels = detector.predict(img)

                        os.remove( TEMP_PATH + img_id)
                        
                    except Exception as e:

                        print(e)
                        output = {"success":False, "error":"invalid image","code":400}
                        db.set(req_id,json.dumps(output))
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

                    db.set(req_id,json.dumps(response))   
                else:
                    output = {"success":False, "error":"invalid image","code":400}
                    db.set(req_id,json.dumps(output))

        time.sleep(delay)



def sharedetection(thread_name,delay):


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
       

    elif MODE == "Medium":
        
        reso = 416
    else:

        reso = 320

    detector = SharedDetectionModel(model_path=SHARED_APP_DIR+"yolov3.onnx",reso=reso,cuda=CUDA_MODE)
    while True:
        queue = db.lrange(IMAGE_QUEUE,0,BATCH_SIZE - 1)

        db.ltrim(IMAGE_QUEUE,len(queue), -1)
        
        if len(queue) > 0:

            for req_data in queue:
                
                req_data = json.JSONDecoder().decode(req_data)


                img_id = req_data["imgid"]
                req_id = req_data["reqid"]
                req_type = req_data["reqtype"]
                threshold = float(req_data["minconfidence"])
               
                if True:

                    try:

                        img = Image.open(TEMP_PATH + img_id).convert("RGB")     
                        os.remove( TEMP_PATH + img_id)
                
                        boxes, scores, labels = detector.predict(img,threshold)


                        outputs = []
                        for box,score,label in zip(boxes,scores,labels):

                            y_min,x_min,y_max,x_max = box
                            y_min = int(y_min)
                            x_min = int(x_min)
                            y_max = int(y_max)
                            x_max = int(x_max)
                            label = labels_map[label]
                            score = float(score)

                            detection = {"confidence":score,"label":label, "x_min":x_min, "y_min":y_min,"x_max":x_max, "y_max":y_max}
                    
                            outputs.append(detection)

                        response = {"success":True,"predictions":outputs}

                        db.set(req_id,json.dumps(response)) 
                        
                    except Exception as e:

                       
                        output = {"success":False, "error":"invalid image","code":400}
                        db.set(req_id,json.dumps(output))
                        if os.path.exists(TEMP_PATH + img_id):
                            os.remove(TEMP_PATH + img_id)
                        continue

                    

        time.sleep(delay)


def custom(model_name,delay):
   
    IMAGE_QUEUE = model_name+"_queue"
    
    with open(DATA_DIR + "/models/vision/"+model_name + "/config.json") as f:
        config = json.load(f)
    
    sys_version = config["sys-version"]
    
    label_map = config["map"]
    mean = config["mean"]
    std = config["std"]
    framework = config["framework"]
    width = config["width"]
    height = config["height"]
    grayscale = False
    if "grayscale" in config:
       
        grayscale = True

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

            for req_data in queue:
                req_data = json.JSONDecoder().decode(req_data)
                req_type = req_data["reqtype"]
                req_id = req_data["reqid"]
                img_id = req_data["imgid"]

                try:

                    img = Image.open(TEMP_PATH+img_id)
                    if grayscale:
                        img = img.convert("L")
                    else:
                        img = img.convert("RGB")

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
                    db.set(req_id,json.dumps(output))  

                except Exception as e:

                    output = {"success":False, "error":"invalid image","code":400}
                    db.set(req_id,json.dumps(output))
            
        time.sleep(delay)
   
def update_faces(thread_name,delay):

    while True:

        load_faces()

        time.sleep(delay)

p1 = Process(target=update_faces,args=("",1))
p1.start()

if "VISION-CUSTOM" in os.environ:
    activate = os.environ["VISION-CUSTOM"]
    if activate == "True":
        if os.path.exists(DATA_DIR + "/models/vision"):

            for model in os.listdir(DATA_DIR + "/models/vision"):
               
                p = Process(target=custom,args=(model,SLEEP_TIME))
                p.start()

if "VISION-DETECTION" in os.environ:

    activate = os.environ["VISION-DETECTION"]

    if activate == "True":
        p = Process(target=detectiongpu if CUDA_MODE else detectioncpu,args=("",SLEEP_TIME))
        p.start()

if "VISION-SCENE" in os.environ:

    activate = os.environ["VISION-SCENE"]

    if activate == "True":
        p = Process(target=scene2,args=("",SLEEP_TIME))
        p.start()

if "VISION-FACE" in os.environ:

    activate = os.environ["VISION-FACE"]

    if activate == "True":

        p = Process(target=face,args=("",SLEEP_TIME))
        p.start()

if "VISION-FACE2" in os.environ:

    activate = os.environ["VISION-FACE2"]

    if activate == "True":

        p = Process(target=face2,args=("",SLEEP_TIME))
        p.start()
