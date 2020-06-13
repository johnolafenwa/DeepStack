from PIL import Image
import torch 
import torch.nn as nn 
import torchvision.transforms as T 
from .detr import DETR
torch.set_grad_enabled(False)

class ObjectDetector(object):
    def __init__(self,model_path,reso=600,cuda=False):
        self.reso = reso
        self.cuda = cuda
        self.model = DETR(num_classes=91)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        if self.cuda == True:
            self.model = self.model.cuda()

        self.transforms = T.Compose([
            T.Resize(self.reso),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def preprocess(self,img):
        return self.transforms(img).unsqueeze(0)

    def postprocess(self,out_bbox,size):

        img_w, img_h = size
        
        x_c, y_c, w, h = out_bbox.unbind(1)

        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
    
        b = torch.stack(b, dim=1)

        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

        return b

    def predict(self,img_path,confidence=0.75):

        im = Image.open(img_path).convert("RGB")

        img = self.preprocess(im)

        if self.cuda == True:
            img = img.cuda()

        pred_logits,pred_boxes = self.model(img)
        pred_logits = pred_logits.cpu()
        pred_boxes = pred_boxes.cpu()

        probas = pred_logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > confidence

        bboxes_scaled = self.postprocess(pred_boxes[0, keep], im.size)
 
        return probas[keep], bboxes_scaled


