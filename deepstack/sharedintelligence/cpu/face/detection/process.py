import cv2
from ..detection.networks import s3fd, detect
from ....shared.commons import load_model
from .bbox import nms,decode, decode2
import torch
import time
from ....shared.commons import Compose,Resize,Pad
import torch.nn.functional as F
import onnxruntime as rt
from PIL import Image
import numpy as np

import numpy as np

def softmax(x):
    mx = np.amax(x,axis=1,keepdims = True)
    x_exp = np.exp(x - mx)
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    res = x_exp / x_sum
    return res

class FaceModel(object):
    def __init__(self,model_path,cuda=False):

        self.sess = rt.InferenceSession(model_path)
        self.input_name = self.sess.get_inputs()[0].name
        
    def predict(self, img,img_size=416,threshold=0.35):
        img = Image.fromarray(img)
        ratio = min(img_size/img.size[0], img_size/img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms=Compose([Resize((imh,imw)),
         Pad((max(int((imh-imw)/2),0), 
              max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0)), (128,128,128))
         ])

        img = img_transforms(img)
        img = np.asarray(img)
        img = img - np.array([104,117,123])
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img,0).astype(np.float32)


        olist = self.sess.run(None,{self.input_name: img})

        bboxlist = []
        for i in range(len(olist)//2):
             l = torch.from_numpy(np.array(olist[i*2]))
             olist[i*2] = F.softmax(l,dim=1).numpy()        
    
        for i in range(len(olist)//2):
            
            ocls,oreg = olist[i*2],olist[i*2+1]
            FB,FC,FH,FW = ocls.shape # feature map size
            
            stride = 2**(i+2)    # 4,8,16,32,64,128
            anchor = stride*4
            
            for Findex in range(FH*FW):
                windex,hindex = Findex%FW,Findex//FW
                axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
                score = ocls[0,1,hindex,windex]
                loc = oreg[0,:,hindex,windex].reshape(1,4)
                if score<0.05: continue
                priors = np.array([[axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]])
                variances = [0.1,0.2]
                box = decode2(loc,priors,variances)
                
                x1,y1,x2,y2 = box[0]*1.0
                # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
                if score < threshold: continue
                #print(x1,y1,x2,y2,score)
                bboxlist.append([x1,y1,x2,y2,score])

        bboxlist = np.array(bboxlist)
        if 0==len(bboxlist): bboxlist = []

        if bboxlist == []:
            return bboxlist

        keep = nms(bboxlist,0.3)
        bboxlist = bboxlist[keep,:]

        return bboxlist