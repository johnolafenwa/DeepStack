
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torchvision.transforms.functional as TF
import cv2
def load_model(model,model_path,strict=True,cpu=False):
    if isinstance(model,DataParallel) or isinstance(model,DistributedDataParallel):
        model=model.module
    if cpu:
        loaded_model=torch.load(model_path,map_location='cpu')
    else:
        loaded_model=torch.load(model_path)
    model.load_state_dict(loaded_model,strict=strict)

def transform_img(image):
  image=TF.to_pil_image(image)
  image = TF.resize(image, (224, 224))
  image = TF.to_tensor(image)
  image = (image - image.min())/(image.max() - image.min())
  image = (2 * image) - 1
  return image.unsqueeze(0)


def landmarks_draw(image,img_landmarks):
  image=image.copy()
  for landmarks, (left, top,height,width) in img_landmarks:
    landmarks=landmarks.view(-1,2)
    landmarks=(landmarks+0.5)
    landmarks=landmarks.numpy()

    for i, (x,y) in enumerate(landmarks, 1):
      try:
        cv2.circle(image, (int((x * width) + left), int((y * height) + top)), 2, [40, 117, 255], -1)
      except:

        pass
  return image