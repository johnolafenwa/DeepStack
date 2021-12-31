import argparse
import cv2
import numpy as np
import os
import sys
import torch
import json
from multiprocessing import Process
import base64
import time


from superresolution.utils import load_model
from superresolution.common import tensor2img, calculate_psnr, calculate_ssim, bgr2ycbcr


sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "."))


from shared import SharedOptions
if SharedOptions.PROFILE == "windows_native":
    sys.path.append(os.path.join(SharedOptions.APP_DIR,"windows_packages"))


def get_network(model_path):
    from superresolution.config import config
    from superresolution.network import Network

    # an ugly operation
    if 'KERNEL_PATH' in config.MODEL:
        config.MODEL.KERNEL_PATH = config.MODEL.KERNEL_PATH.replace('../', '')

    return config, Network(config)

def superresolution4x(thread_name: str, delay: float):
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


    while True:
        queue = db.lrange(IMAGE_QUEUE, 0, 0)

        db.ltrim(IMAGE_QUEUE, len(queue), -1)

        if len(queue) > 0:

            for req_data in queue:
                try:
                    with torch.no_grad():
                        req_data = json.JSONDecoder().decode(req_data)
                        img_id = req_data["imgid"]
                        req_id = req_data["reqid"]
                        req_type = req_data["reqtype"]
                        img_path = os.path.join(TEMP_PATH, img_id)

                        lr_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                        lr_img = np.transpose(lr_img[:, :, ::-1], (2, 0, 1)).astype(np.float32) / 255.0
                        lr_img = torch.from_numpy(lr_img).float().to(device).unsqueeze(0)

                        _, C, H, W = lr_img.size()

                        need_pad = False
                        if H % down != 0 or W % down != 0:
                            need_pad = True
                            pad_y_t = (down - H % down) % down // 2
                            pad_y_b = (down - H % down) % down - pad_y_t
                            pad_x_l = (down - W % down) % down // 2
                            pad_x_r = (down - W % down) % down - pad_x_l
                            lr_img = torch.nn.functional.pad(lr_img, pad=(pad_x_l, pad_x_r, pad_y_t, pad_y_b), mode='replicate')

                        output = model(lr_img)

                        if need_pad:
                            y_end = -pad_y_b * scale if pad_y_b != 0 else output.size(2)
                            x_end = -pad_x_r * scale if pad_x_r != 0 else output.size(3)
                            output = output[:, :, pad_y_t * scale: y_end, pad_x_l * scale: x_end]

                        output = tensor2img(output)

                        img_h, img_w, _ = output.shape

                        retval, buffer = cv2.imencode('.jpg', output)
                        base64_img = base64.b64encode(buffer)

                        output_response = {"success": True, "base64": base64_img.decode("utf-8"), "width": img_w, "height": img_h}
                except Exception as e:
                    output_response = {"success": False,
                            "error": str(e),
                            "code": 500,
                            }
                finally:
                    db.set(req_id, json.dumps(output_response))
                    if os.path.exists(img_path):
                        os.remove(img_path)
        time.sleep(delay)

if __name__ == "__main__":     
    p = Process(target=superresolution4x, args=("", SharedOptions.SLEEP_TIME))
    p.start()

                   
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sr_type', type=str, default='SISR')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--gt_path', type=str, default=None)
    args = parser.parse_args()

    if args.output_path and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    use_cuda = False
    use_cpu = False if use_cuda else True

    print('Loading Network ...')
    config, model = get_network(args.model_path)
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    load_model(model, args.model_path, strict=True, cpu=use_cpu)

    down = config.MODEL.DOWN
    scale = config.MODEL.SCALE

    print('Reading Images ...')
    ipath_l = []
    for f in sorted(os.listdir(args.input_path)):
        if f.endswith('png') or f.endswith('jpg'):
            ipath_l.append(os.path.join(args.input_path, f))

    with torch.no_grad():
        for i, f in enumerate(ipath_l):
            img_name = f.split('/')[-1]
            print('Processing: %s' % img_name)
            lr_img = cv2.imread(f, cv2.IMREAD_COLOR)
            lr_img = np.transpose(lr_img[:, :, ::-1], (2, 0, 1)).astype(np.float32) / 255.0
            lr_img = torch.from_numpy(lr_img).float().to(device).unsqueeze(0)

            _, C, H, W = lr_img.size()

            need_pad = False
            if H % down != 0 or W % down != 0:
                need_pad = True
                pad_y_t = (down - H % down) % down // 2
                pad_y_b = (down - H % down) % down - pad_y_t
                pad_x_l = (down - W % down) % down // 2
                pad_x_r = (down - W % down) % down - pad_x_l
                lr_img = torch.nn.functional.pad(lr_img, pad=(pad_x_l, pad_x_r, pad_y_t, pad_y_b), mode='replicate')

            output = model(lr_img)

            if need_pad:
                y_end = -pad_y_b * scale if pad_y_b != 0 else output.size(2)
                x_end = -pad_x_r * scale if pad_x_r != 0 else output.size(3)
                output = output[:, :, pad_y_t * scale: y_end, pad_x_l * scale: x_end]

            output = tensor2img(output)
            if args.output_path:
                print(f"Saving {os.path.join(args.output_path, img_name)}")
                output_path = os.path.join(args.output_path, os.path.basename(img_name))
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                cv2.imwrite(output_path, output)
    print('Finished!')
"""
