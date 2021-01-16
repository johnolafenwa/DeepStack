import numpy as np
import onnxruntime as rt
import torch


class SceneModel(object):
    def __init__(self, model_path, cuda=False):

        self.sess = rt.InferenceSession(model_path)
        self.input_name = self.sess.get_inputs()[0].name

    def predict(self, image_tensors):

        out = self.sess.run(None, {self.input_name: image_tensors})

        out = np.array(out)
        torch_out = torch.from_numpy(out).squeeze(1)
        torch_out = torch.softmax(torch_out, 1)

        return out.argmax(), torch_out.max().item()
