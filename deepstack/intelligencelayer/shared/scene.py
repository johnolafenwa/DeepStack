import _thread as thread
import ast
import io
import json
import os
import sqlite3
import sys
import time
import warnings
from multiprocessing import Process

import numpy as np
import onnxruntime as rt
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../"))
import traceback

import torchvision.transforms as transforms
from shared import SharedOptions


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


def scenerecognition(thread_name, delay):

    classes = list()
    with open(
        os.path.join(SharedOptions.SHARED_APP_DIR, "categories_places365.txt")
    ) as class_file:
        for line in class_file:
            classes.append(line.strip().split(" ")[0][3:])

    placesnames = tuple(classes)

    IMAGE_QUEUE = "scene_queue"
    classifier = SceneModel(
        os.path.join(SharedOptions.SHARED_APP_DIR, "scene.model"),
        SharedOptions.CUDA_MODE,
    )

    while True:
        queue = SharedOptions.db.lrange(IMAGE_QUEUE, 0, 0)

        if len(queue) > 0:

            SharedOptions.db.ltrim(IMAGE_QUEUE, len(queue), -1)

            for req_data in queue:
                req_data = json.JSONDecoder().decode(req_data)
                img_id = req_data["imgid"]
                req_id = req_data["reqid"]
                req_type = req_data["reqtype"]
                try:

                    img = Image.open(SharedOptions.TEMP_PATH + img_id).convert("RGB")

                    trans = transforms.Compose(
                        [
                            transforms.Resize((256, 256)),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                            ),
                        ]
                    )
                    img = trans(img)
                    img = img.numpy()
                    img = np.expand_dims(img, 0).astype(np.float32)

                    os.remove(SharedOptions.TEMP_PATH + img_id)

                    cl, conf = classifier.predict(img)

                    cl = placesnames[cl]

                    conf = float(conf)

                    output = {"success": True, "label": cl, "confidence": conf}

                except UnidentifiedImageError:
                    err_trace = traceback.format_exc()
                    print(err_trace, file=sys.stderr, flush=True)

                    output = {
                        "success": False,
                        "error": "error occured on the server",
                        "code": 400,
                    }

                except Exception:

                    err_trace = traceback.format_exc()
                    print(err_trace, file=sys.stderr, flush=True)

                    output = {"success": False, "error": "invalid image", "code": 500}

                finally:
                    SharedOptions.db.set(req_id, json.dumps(output))
                    if os.path.exists(SharedOptions.TEMP_PATH + img_id):
                        os.remove(SharedOptions.TEMP_PATH + img_id)

        time.sleep(delay)


p = Process(target=scenerecognition, args=("", SharedOptions.SLEEP_TIME))
p.start()
