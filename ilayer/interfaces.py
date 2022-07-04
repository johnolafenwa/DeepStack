import numpy as np 
import cv2
from detection_utils import multiclass_nms
import time

class IDetector():
    def __init__(self, model_path, reso, executor) -> None:
        
        self.input_size = (reso, reso)
        self.executor = executor
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        
        raise NotImplementedError()
    
    def inference(self, image):
        
        raise NotImplementedError()
    
    def preprocess(self, image):
        
        swap=(2, 0, 1)
        
        if len(image.shape) == 3:
            padded_img = np.ones((self.input_size[0], self.input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.input_size, dtype=np.uint8) * 114

        r = min(self.input_size[0] / image.shape[0], self.input_size[1] / image.shape[1])
        resized_img = cv2.resize(
            image,
            (int(image.shape[1] * r), int(image.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(image.shape[0] * r), : int(image.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r
    
    def postprocess(self, ratio, outputs, nms_threshold = 0.45, score_threshold = 0.1):
    
        grids = []
        expanded_strides = []

        strides = [8, 16, 32]
        
        hsizes = [self.input_size[0] // stride for stride in strides]
        wsizes = [self.input_size[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        
        outputs = outputs[0]
        
        boxes = outputs[:, :4]
        scores = outputs[:, 4:5] * outputs[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=nms_threshold, score_thr=score_threshold)
        
        return dets
    
    def detect(self, image, nms_threshold = 0.45, score_threshold = 0.1):
        
        t1 = time.time()
        img, ratio = self.preprocess(image)
        
        pre_duration = time.time() - t1
        
        t1 = time.time()
        output = self.inference(img)
        infer_duration = time.time() - t1
        
        t1 = time.time()
        output = self.postprocess(ratio, output, nms_threshold, score_threshold)
        post_duration = time.time() - t1
        
        print(f"Pre: {pre_duration*1000}, Infer: {infer_duration*1000}, Post: {post_duration*1000}")
        
        if output is not None:
            final_boxes, final_scores, final_cls_inds = output[:, :4], output[:, 4], output[:, 5]
            
            return final_boxes, final_scores, final_cls_inds
        
        else:
            return None
        
class OnnxRuntimeDetector(IDetector):
    
    def load_model(self, model_path):
        
        import onnxruntime
        
        providers = ['CPUExecutionProvider']
        
        if self.executor == "CUDA":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif self.executor == "ONNXTRT":
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            
        model = onnxruntime.InferenceSession(model_path, providers=providers)
    
        return model
    
    def inference(self, img):
        
        ort_inputs = {self.model.get_inputs()[0].name: img[None, :, :, :]}
    
        output = self.model.run(None, ort_inputs)
        
        return output[0]
        
    
    