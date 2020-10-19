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
import sys
from detection import FaceModel
from recognition import FaceRecognitionModel
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../"))

from shared import SharedOptions

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
    
    SELECT_FACE = "SELECT * FROM {}".format(SharedOptions.TB_EMBEDDINGS)
    conn = sqlite3.connect(SharedOptions.DATA_DIR+"/faceembedding.db")
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
    SharedOptions.db.set("facemap",facemap)

    conn.close()

def face(thread_name,delay):

    faceclassifier = FaceRecognitionModel(os.path.join(SharedOptions.SHARED_APP_DIR,"facerec-high.model"),cuda=SharedOptions.CUDA_MODE)
   
    facedetector = FaceModel(os.path.join(SharedOptions.SHARED_APP_DIR,"facedetector-high.model"),SharedOptions.CUDA_MODE)

    load_faces()

    ADD_FACE = "INSERT INTO TB_EMBEDDINGS(userid,embedding) VALUES(?,?)"
    UPDATE_FACE = "UPDATE TB_EMBEDDINGS SET embedding = ? where userid = ?"
    SELECT_FACE = "SELECT * FROM TB_EMBEDDINGS where userid = ? "

    trans = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ])
   
    IMAGE_QUEUE = "face_queue"

    while True:

        queue = SharedOptions.db.lrange(IMAGE_QUEUE,0,0)

        SharedOptions.db.ltrim(IMAGE_QUEUE,len(queue), -1)

        img_size = 400
        detect_size = 1600
        skip = 0
        
        if SharedOptions.MODE == "High":
            img_size = 400
            skip = 0
            
        elif SharedOptions.MODE == "Low":
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

                        img = SharedOptions.TEMP_PATH+img_id
                        
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
                        
                        SharedOptions.db.set(req_id,json.dumps(output))

                    except Exception as e:
                      
                        output = {"success":False, "error":"invalid image","code":400}
                        SharedOptions.db.set(req_id,json.dumps(output))
                        if os.path.exists(img):
                            os.remove(img)

                elif task_type == "register":

                    try:
                        
                        user_id = req_data["userid"]

                        user_images = req_data["images"]

                        conn = sqlite3.connect(SharedOptions.DATA_DIR+"/faceembedding.db")

                        batch = None

                        for img_id in user_images:
                        
                            img = cv2.imread(SharedOptions.TEMP_PATH+img_id)

                            pad_x,pad_y,unpad_w,unpad_h = pad_image(img,img_size)

                            pil_image = Image.fromarray(img).convert("RGB")
              
                            os.remove(SharedOptions.TEMP_PATH+img_id)
                         
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
                            SharedOptions.db.set(req_id,json.dumps(output))
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
                        SharedOptions.db.set(req_id,json.dumps(output))
                
                        conn.close()

                    except Exception as e:
                   
                        output = {"success":False, "error":"invalid image","code":400}
                        SharedOptions.db.set(req_id,json.dumps(output))

                        for img_id in user_images:
                            if os.path.exists(SharedOptions.TEMP_PATH+img_id):
                                os.remove(SharedOptions.TEMP_PATH+img_id)

                elif task_type == "recognize":

                    try:

                        master_face_map = SharedOptions.db.get("facemap")
                        master_face_map = ast.literal_eval(master_face_map)
           
                        facemap = master_face_map["map"]
                     
                        face_array = master_face_map["tensors"]

                        if len(face_array) > 0:

                            face_array_tensors = [torch.tensor(emb).unsqueeze(0) for emb in face_array]
                            face_tensors = torch.cat(face_array_tensors)

                        if SharedOptions.CUDA_MODE and len(face_array) > 0:
                            face_tensors = face_tensors.cuda()

                        img_id = req_data["imgid"]
                        threshold = float(req_data["minconfidence"])
               
                        img = cv2.imread(SharedOptions.TEMP_PATH+img_id)

                        pad_x,pad_y,unpad_w,unpad_h = pad_image(img,img_size)

                        pil_image = Image.fromarray(img).convert("RGB")

                        os.remove(SharedOptions.TEMP_PATH+img_id)
                    
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
                            SharedOptions.db.set(req_id,json.dumps(output))
                    
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
                            SharedOptions.db.set(req_id,json.dumps(output))

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
                            SharedOptions.db.set(req_id,json.dumps(output))

                    except Exception as e:


                        output = {"success":False, "error":"invalid image","code":400}
                        SharedOptions.db.set(req_id,json.dumps(output))

                        if os.path.exists(SharedOptions.TEMP_PATH+img_id):
                            os.remove(SharedOptions.TEMP_PATH+img_id)

                elif task_type == "match":

                    try:

                        user_images = req_data["images"]

                        img1 = user_images[0]
                        img2 = user_images[1]

                        cv_img1 = cv2.imread(SharedOptions.TEMP_PATH+img1)
                        image1 = Image.fromarray(cv_img1).convert("RGB")
                        cv_img2 = cv2.imread(SharedOptions.TEMP_PATH+img2)
                        image2 = Image.fromarray(cv_img2).convert("RGB")

                        img1_pad_x,img1_pad_y,img1_unpad_w,img1_unpad_h = pad_image(cv_img1,img_size)
                        img2_pad_x,img2_pad_y,img2_unpad_w,img2_unpad_h = pad_image(cv_img2 ,img_size)


                        os.remove(SharedOptions.TEMP_PATH+img1)
                        os.remove(SharedOptions.TEMP_PATH+img2)
                    
                        bboxs1 = facedetector.predict(cv_img1,img_size)
                        bboxs2 = facedetector.predict(cv_img2,img_size)

                        if len(bboxs1) == 0 or len(bboxs2) == 0:

                            output = {"success":False, "error":"no face found"}
                            SharedOptions.db.set(req_id,json.dumps(output)) 
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
                        SharedOptions.db.set(req_id,json.dumps(output))           

                    except Exception as e:
                        
                        output = {"success":False, "error":"invalid image","code":400}
                        SharedOptions.db.set(req_id,json.dumps(output))


        time.sleep(delay)


def update_faces(thread_name,delay):

    while True:

        load_faces()

        time.sleep(delay)

p1 = Process(target=update_faces,args=("",1))
p1.start()

p = Process(target=face,args=("",SharedOptions.SLEEP_TIME))
p.start()

