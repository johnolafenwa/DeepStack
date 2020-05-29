from .utils import read_pb_return_tensors,cpu_nms
import tensorflow as tf
import numpy as np



class DetectModel3(object):
    def __init__(self,model_path,reso=640,confidence=0.5):

        self.confidence = confidence

        self.reso = reso
        
        self.numbers_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
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

        self.cpu_nms_graph = tf.Graph()

        self.input_tensor, self.output_tensors = read_pb_return_tensors(self.cpu_nms_graph, model_path,
                                           ["Placeholder:0", "concat_9:0", "mul_6:0"])

        self.sess = tf.Session(graph=self.cpu_nms_graph)
        
    def predict(self, img):

        img_resized = np.array(img.resize(size=(self.reso, self.reso)), dtype=np.float32)
        img_resized = img_resized / 255.
        
        boxes, scores = self.sess.run(self.output_tensors, feed_dict={self.input_tensor: np.expand_dims(img_resized, axis=0)})
        boxes, scores, labels = cpu_nms(boxes, scores, 80, score_thresh=self.confidence, iou_thresh=0.5)

        return boxes, scores, labels
            
