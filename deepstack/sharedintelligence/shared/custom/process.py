import tensorflow as tf
import os
from tensorflow.python.platform import gfile

try:
    from tensorflow.keras.models import load_model
except:
    from keras.models import load_model

from PIL import Image
import numpy as np 
import scipy
from scipy import misc
import cv2
import time 

import torch
import onnxruntime as rt

class TFCustomModel(object):

    def __init__(self,model_path,input_name="input_1:0", output_name="output_1:0",cuda=False):

        self.cuda_mode = cuda
        
        self.sess = None
        self.input_node = None
        self.output_node = None

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Graph().as_default() as graph:

            self.sess = tf.Session(config=config)

            with gfile.FastGFile(model_path,"rb") as f:

                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                self.sess.graph.as_default()

                tf.import_graph_def(
                    graph_def,
                    name=""
                )
                
                self.input_node = graph.get_tensor_by_name(input_name)
                self.output_node = graph.get_tensor_by_name(output_name)

                tf.global_variables_initializer()

    def predict(self,images):

        if self.cuda_mode == True:
         
            output = self.sess.run(self.output_node, feed_dict={self.input_node: images})
        else:
            
            output = self.sess.run(self.output_node, feed_dict={self.input_node: images})

        return output.argmax(),output.max()


def preprocess_image(x, mean=127.5, std=1):

    x -= mean
    x /= std
    return x


class OnnxCustomModel(object):
    def __init__(self,model_path,cuda=False):

        self.sess = rt.InferenceSession(model_path)
        self.input_name = self.sess.get_inputs()[0].name
        
    def predict(self, image_tensors):
        
        out = self.sess.run(None,{self.input_name: image_tensors})

        out = np.array(out)
        torch_out = torch.from_numpy(out).squeeze(1)
        torch_out = torch.softmax(torch_out,1)
        
        return out.argmax(),torch_out.max().item()


class KerasCustomModel(object):
    def __init__(self,model_path,cuda=False):

        self.model = load_model(model_path)
        
    def predict(self, image_tensors):
        
        out = self.model.predict(image_tensors)
        
        
        return out.argmax(),out.max()