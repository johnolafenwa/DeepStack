from shared import SharedOptions
import argparse
from PIL import Image
from io import BytesIO
import numpy as np
import os
import sys
import torch
import json
from multiprocessing import Process
import base64
import time
from threading import Thread
from queue import Queue

from superresolution.utils import load_model
from superresolution.common import tensor2img

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "."))

if SharedOptions.PROFILE == "windows_native":
    sys.path.append(os.path.join(SharedOptions.APP_DIR, "windows_packages"))


def get_network(model_path):
    from superresolution.config import config
    from superresolution.network import Network

    # an ugly operation
    if 'KERNEL_PATH' in config.MODEL:
        config.MODEL.KERNEL_PATH = config.MODEL.KERNEL_PATH.replace('../', '')

    return config, Network(config)


SHARED_APP_DIR = SharedOptions.SHARED_APP_DIR
CUDA_MODE = SharedOptions.CUDA_MODE
db = SharedOptions.db
TEMP_PATH = SharedOptions.TEMP_PATH

IMAGE_QUEUE = "superresolution_queue"

model_path = os.path.join(
    SHARED_APP_DIR, SharedOptions.SETTINGS.SUPERRESOLUTION_MODEL
)

use_cuda = SharedOptions.CUDA_MODE
use_cpu = False if use_cuda else True

print('Loading Network ...')
config, model = get_network(model_path)
if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = model.to(device)
load_model(model, model_path, strict=True, cpu=use_cpu)

down = config.MODEL.DOWN
scale = config.MODEL.SCALE


def run_task(q):
    while True:
        req_data = q.get()

        try:
            with torch.no_grad():
                img_id = req_data["imgid"]
                req_id = req_data["reqid"]
                req_type = req_data["reqtype"]
                img_path = os.path.join(TEMP_PATH, img_id)

                lr_img = np.array(Image.open(img_path).convert("RGB"))
                lr_img = np.transpose(
                    lr_img[:, :, ::-1], (2, 0, 1)).astype(np.float32) / 255.0
                lr_img = torch.from_numpy(
                    lr_img).float().to(device).unsqueeze(0)

                _, C, H, W = lr_img.size()

                need_pad = False
                if H % down != 0 or W % down != 0:
                    need_pad = True
                    pad_y_t = (down - H % down) % down // 2
                    pad_y_b = (down - H % down) % down - pad_y_t
                    pad_x_l = (down - W % down) % down // 2
                    pad_x_r = (down - W % down) % down - pad_x_l
                    lr_img = torch.nn.functional.pad(lr_img, pad=(
                        pad_x_l, pad_x_r, pad_y_t, pad_y_b), mode='replicate')

                output = model(lr_img)

                if need_pad:
                    y_end = -pad_y_b * \
                        scale if pad_y_b != 0 else output.size(2)
                    x_end = -pad_x_r * \
                        scale if pad_x_r != 0 else output.size(3)
                    output = output[:, :, pad_y_t *
                                    scale: y_end, pad_x_l * scale: x_end]

                output = tensor2img(output)
                img_h, img_w, _ = output.shape

                output_pil = Image.fromarray(output, 'RGB')

                buffered = BytesIO()
                output_pil.save(buffered, format="JPEG")
                base64_img = base64.b64encode(buffered.getvalue())

                output_response = {"success": True, "base64": base64_img.decode(
                    "utf-8"), "width": img_w, "height": img_h}
        except Exception as e:
            output_response = {"success": False,
                               "error": "An error occured",
                               "code": 500,
                               }
        finally:
            db.set(req_id, json.dumps(output_response))
            if os.path.exists(img_path):
                os.remove(img_path)


def superresolution4x(delay: float):

    q = Queue(maxsize=0)

    for _ in range(SharedOptions.THREADCOUNT):
        worker = Thread(target=run_task, args=(q,))
        worker.setDaemon(True)
        worker.start()

    while True:
        queue = db.lrange(IMAGE_QUEUE, 0, 0)

        db.ltrim(IMAGE_QUEUE, len(queue), -1)

        if len(queue) > 0:

            for req_data in queue:

                req_data = json.JSONDecoder().decode(req_data)
                q.put(req_data)

        time.sleep(delay)


if __name__ == "__main__":
    superresolution4x(SharedOptions.SLEEP_TIME)
