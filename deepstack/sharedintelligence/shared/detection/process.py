import onnxruntime as rt
import numpy as np
from PIL import Image

from .utils import preprocess,cpu_nms
import time

class SharedDetectionModel(object):
    def __init__(self,model_path,reso,cuda=False):

        self.sess = rt.InferenceSession(model_path)
        self.reso = reso

        self.input_name = self.sess.get_inputs()[0].name
        self.input_name2 = self.sess.get_inputs()[1].name

    def predict(self, image,nms):

        image_size = np.array([image.size[1], image.size[0]], dtype=np.int32).reshape(1, 2)
        image_data = preprocess(image,self.reso)
        
        t1 = time.time()
        indices = self.sess.run(None,{self.input_name: image_data, self.input_name2: image_size})
        t2 = time.time()

        print((t2-t1)*1000)

        out_boxes, out_scores, out_classes = [], [], []

        out_boxes = None
        scores = None

        i = 0
        for idx in indices:
            idx = np.array(idx)

            if i == 0:
                out_boxes = idx
            elif i == 1:
                scores = idx.transpose((0,2,1))
            else:
                break 
            i += 1
        
        boxes, scores, labels = cpu_nms(out_boxes,scores,80,score_thresh=nms)
        
        return boxes, scores, labels