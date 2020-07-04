from redis import StrictRedis, RedisError
import torch
import time
import json
import io
import os
import _thread as thread
from multiprocessing import Process
import torchvision.transforms as transform
from PIL import Image
import cv2
from sharedintelligence import TrafficModel, SceneModel, GenderModel, FaceModel, DetectModel, FaceRecognitionModel, compute_distance, CustomModel, NudityModel
import torch.nn.functional as F
from sharedintelligence.commons.commons import preprocess_image
import ast
import sqlite3
import numpy as np

TEMP_PATH = "/deeptemp/"
BATCH_SIZE = 8
SLEEP_TIME = 0.001
APP_DIR = ""
DATA_DIR = "/datastore"

CUDA_MODE = False

faceclassifier = None
facedetector = None
"""
if "BATCH_SIZE" in os.environ:
     BATCH_SIZE = int(os.environ["BATCH_SIZE"])

if "SLEEP_TIME" in os.environ:
    SLEEP_TIME = float(os.environ["SLEEP_TIME"])
"""

MODE = "High"
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

def recognize_face(thread_name,delay):
    IMAGE_QUEUE = "face_recog_queue"

    load_faces()

    trans = transform.Compose([
        transform.Resize((112,112)),
        transform.ToTensor(),
        transform.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ])

    face_transforms = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ])

    img_size = 320
   
    while True:
        
        queue = db.lrange(IMAGE_QUEUE,0,BATCH_SIZE - 1)
  
        if len(queue) > 0:

            db.ltrim(IMAGE_QUEUE,len(queue), -1)

            batch = None
            imageids = []
            user_ids = []

            master_face_map = db.get("facemap")
            master_face_map = ast.literal_eval(master_face_map)
           
            facemap = master_face_map["map"]
            face_array = master_face_map["tensors"]
            
            if len(face_array) > 0:

                face_array_tensors = [torch.tensor(emb).unsqueeze(0) for emb in face_array]
                face_tensors = torch.cat(face_array_tensors)
                if CUDA_MODE:
                    face_tensors = face_tensors.cuda()
              
            for img_id in queue:
                
                try:

                    
                    threshold = float(db.get(img_id+"_min_confidence"))
                    db.delete(img_id+"_min_confidence")
               
                    img = cv2.imread(TEMP_PATH+img_id)

                    pad_x,pad_y,unpad_w,unpad_h = pad_image(img,img_size)

                    pil_image = Image.fromarray(img).convert("RGB")

                    os.remove(TEMP_PATH+img_id)
                    
                    bboxs = facedetector.predict(img,img_size)
                    
                    faces = []
                    for face in bboxs:
                        found_face = True
                        x_min, y_min, x_max, y_max = convert_boxes(img,face[0],face[1],face[2],face[3],unpad_w,unpad_h,pad_x,pad_y)
                   
                        new_img = pil_image.crop((x_min,y_min,x_max,y_max))
                       
                        faces.append(trans(new_img).unsqueeze(0))

                    if len(faces) < 1:

                        output = {"success":True, "predictions":[]}
                        db.set(img_id,json.dumps(output))
                    
                    elif len(facemap) == 0:

                        predictions = []

                        for face in bboxs:

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

                            user_data = {"confidence":0,"userid":"unknown", "x_min":x_min, "y_min":y_min,"x_max":x_max, "y_max":y_max}

                            predictions.append(user_data)

                        output = {"success":True, "predictions":predictions}
                        db.set(img_id,json.dumps(output))

                    else:
                       
                        faces = torch.cat(faces)
                        
                        embeddings = faceclassifier.predict(faces)

                        predictions = []
                        
                        for embedding,face in zip(embeddings,bboxs):
                            
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

                    os.remove(TEMP_PATH+img_id)

        time.sleep(delay)

def match_face(thread_name,delay):
    IMAGE_QUEUE = "face_match_queue"
   
    trans = transform.Compose([
        transform.Resize((112,112)),
        transform.ToTensor(),
        transform.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ])

    face_transforms = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ])

    img_size = 320
   
    while True:
        
        queue = db.lrange(IMAGE_QUEUE,0,BATCH_SIZE - 1)
  
        if len(queue) > 0:

            db.ltrim(IMAGE_QUEUE,len(queue), -1)

            batch = None
            imageids = []
            user_ids = []
              
            for req_data in queue:

                req_data = json.JSONDecoder().decode(req_data)
                   
                req_id = req_data["reqid"]

                user_images = req_data["images"]

                try:
                    

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

                    print(e)

                    output = {"success":False, "error":"invalid image"}
                    db.set(req_id,json.dumps(output))
        
        time.sleep(delay)


def register_face(thread_name,delay):
    IMAGE_QUEUE = "face_register_queue"
    ADD_FACE = "INSERT INTO TB_EMBEDDINGS(userid,embedding) VALUES(?,?)"
    UPDATE_FACE = "UPDATE TB_EMBEDDINGS SET embedding = ? where userid = ?"
    SELECT_FACE = "SELECT * FROM TB_EMBEDDINGS where userid = ? "

    trans = transform.Compose([
        transform.Resize((112,112)),
        transform.ToTensor(),
        transform.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ])

    face_transforms = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ])

    img_size = 320
   
    while True:

        queue = db.lrange(IMAGE_QUEUE,0,BATCH_SIZE - 1)
        
        if len(queue) > 0:

            db.ltrim(IMAGE_QUEUE,len(queue), -1)

            conn = sqlite3.connect(DATA_DIR+"/faceembedding.db")

            user_ids = []
            embeddings = []
            req_ids = []

            for req_data in queue:
  
                try:
                    req_data = json.JSONDecoder().decode(req_data)
                   
                    user_id = req_data["userid"]
                    req_id = req_data["reqid"]

                    user_images = req_data["images"]

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
   
                    user_ids.append(user_id)
                    req_ids.append(req_id)
                    embeddings.append(img_embeddings)
           
                    
                except Exception as e:
                   
                    output = {"success":False, "error":"invalid image"}
                    db.set(req_id,json.dumps(output))

                    for img_id in user_images:
                        os.remove(TEMP_PATH+img_id)

            if len(user_ids) == 0:
                
                continue
                
            for embedding, user_id, req_id in zip(embeddings,user_ids,req_ids):


                cursor = conn.cursor()

                
                emb = embedding.tolist()
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

            load_faces()
        
        time.sleep(delay)


def scene(thread_name,delay):
    IMAGE_QUEUE = "scene_queue"
    classifier = SceneModel(APP_DIR+"scenemodel.pth.tar",CUDA_MODE)

    

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
                    img = trans(img).unsqueeze(0)
                    os.remove( TEMP_PATH + img_id)

                    if batch is None:
                        batch = img
                   
                    else:
                        batch = torch.cat([batch,img],0)

                    imageids.append(img_id)

                except:

                    output = {"success":False, "error":"invalid image"}
                    db.set(img_id,json.dumps(output))
                    os.remove( TEMP_PATH + img_id)

            if batch is not None:
                class_preds, confs = classifier.predict(batch)

                for img_id, cl, conf in zip(imageids,class_preds,confs):
                
                    cl = placesnames[cl.item()]
                    conf = conf.item()

                    output = {"success":True, "label":cl, "confidence":conf}
                    db.set(img_id,json.dumps(output))
            
            
        
        time.sleep(delay)


def traffic(thread_name,delay):
    IMAGE_QUEUE = "traffic_queue"
    classifier = TrafficModel(APP_DIR+"trafficmodel.pth",CUDA_MODE)

    label_map = {3: "sparse", 0:"accident", 2:"fire", 1:"dense"}

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

                    os.remove( TEMP_PATH + img_id)
            
            if batch is not None:
                class_preds, confs = classifier.predict(batch)

                for img_id, cl, conf in zip(imageids,class_preds,confs):
                
                    cl = label_map[cl.item()]
                    conf = conf.item()

                    output = {"success":True, "label":cl, "confidence":conf}
                    db.set(img_id,json.dumps(output))      
            
        time.sleep(delay)


def detection(thread_name,delay):
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
    
    while True:
        queue = db.lrange(IMAGE_QUEUE,0,BATCH_SIZE - 1)

        db.ltrim(IMAGE_QUEUE,len(queue), -1)
        
        if len(queue) > 0:

            batch = None
            imageids = []

            for img_id in queue:
                

                threshold = float(db.get(img_id+"_nms"))
               
                db.delete(img_id+"_nms")
                detector.nms = (1-threshold)
                detector.inp_dim = reso
                detector.model.net_info["height"] = reso
                
                #img = cv2.imread(TEMP_PATH+img_id)
                img = np.asarray(Image.open(TEMP_PATH+img_id).convert("RGB"))
                os.remove( TEMP_PATH + img_id)
                
                if img is not None:

                    try:
                        output = detector.predict([img])
                    except:

                        output = {"success":True, "predictions":[]}
                        db.set(img_id,json.dumps(output))
                        continue

                    imageids.append(img_id)
                    outputs = []
                    for obj in output[0]:

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

                    db.set(img_id,json.dumps(response))   
                else:
                    output = {"success":False, "error":"invalid image"}
                    db.set(img_id,json.dumps(output))

        time.sleep(delay)

def face_detection(thread_name,delay):
    IMAGE_QUEUE = "face_queue"
    
    i = 0

    img_size = 416

    if MODE == "High":
        img_size = 420
    elif MODE == "Low":
        img_size = 260

    else:
        img_size = 320

    while True:
        queue = db.lrange(IMAGE_QUEUE,0,BATCH_SIZE - 1)

        db.ltrim(IMAGE_QUEUE,len(queue), -1)
        
        if len(queue) > 0:

            batch = None
            imageids = []

            for img_id in queue:

                try:

                    img = cv2.imread(TEMP_PATH+img_id)

                    os.remove(TEMP_PATH+img_id)
                   
                    bboxs = facedetector.predict(img,img_size=img_size)

                    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
                    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
                    unpad_h = img_size - pad_y
                    unpad_w = img_size - pad_x 

                    outputs = []
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
                        


                    output = {"Success": True, "predictions": outputs}

                    db.set(img_id,json.dumps(output))

                except:

                    output = {"success":False, "error":"invalid image"}
                    db.set(img_id,json.dumps(output))
                    if os.path.exists(TEMP_PATH+img_id):

                        os.remove(TEMP_PATH+img_id)

        time.sleep(delay)

def custom(model_name,delay):
    IMAGE_QUEUE = model_name+"_queue"
    classifier = CustomModel(DATA_DIR + "/models/vision/"+model_name + "/model.pb",CUDA_MODE)
    
    with open(DATA_DIR + "/models/vision/"+model_name + "/config.json") as f:
        config = json.load(f)
    
    label_map = config["map"]
    mean = config["mean"]
    std = config["std"]
    
    while True:
        queue = db.lrange(IMAGE_QUEUE,0,BATCH_SIZE - 1)
        
        if len(queue) > 0:

            db.ltrim(IMAGE_QUEUE,len(queue), -1)

            for img_id in queue:
             
                try:
                    
                    image = cv2.imread(TEMP_PATH+img_id)
                    image = cv2.resize(image,(224,224))
                    image = image.astype(float)
                    image = np.expand_dims(image,0)
                    image = preprocess_image(image,mean,std)
                  
                    os.remove( TEMP_PATH + img_id)
               
                    class_idx, confidence = classifier.predict(image)


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
        update_faces = db.get("update-faces")
        if update_faces is not None:
            db.delete("update-faces")
            load_faces()
    
        time.sleep(delay)

for model in os.listdir(DATA_DIR + "/models/vision"):

    p = Process(target=custom,args=(model,SLEEP_TIME))
    p.start()

os.environ["VISION-FACE"] = "True"

if "VISION-DETECTION" in os.environ:

    activate = bool(os.environ["VISION-DETECTION"])

    if activate:
        p = Process(target=detection,args=("",SLEEP_TIME))
        p.start()

if "VISION-SCENE" in os.environ:

    activate = bool(os.environ["VISION-SCENE"])

    if activate:
        p = Process(target=scene,args=("",SLEEP_TIME))
        p.start()

if "VISION-TRAFFIC" in os.environ:

    activate = bool(os.environ["VISION-TRAFFIC"])

    if activate:
        p = Process(target=traffic,args=("",SLEEP_TIME))
        p.start()

if "VISION-FACE" in os.environ:

    faceclassifier = FaceRecognitionModel(APP_DIR+"model_ir_se50.pth",cuda=CUDA_MODE)
    facedetector = FaceModel(APP_DIR+"s3fd_convert.pth",CUDA_MODE)

    activate = bool(os.environ["VISION-FACE"])

    if activate:

        p = Process(target=update_faces,args=("",SLEEP_TIME))
        p.start()

        p = Process(target=face_detection,args=("",SLEEP_TIME))
        p.start()

        p = Process(target=register_face,args=("",SLEEP_TIME))
        p.start()

        p = Process(target=recognize_face,args=("",SLEEP_TIME))
        p.start()

        p = Process(target=match_face,args=("",SLEEP_TIME))
        p.start()

if "VISION-NUDITY" in os.environ:

    activate = bool(os.environ["VISION-NUDITY"])

    if activate:
        p = Process(target=nudity,args=("",SLEEP_TIME))
        p.start()

while 1:
    pass