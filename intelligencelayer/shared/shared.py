import os
import sys
from enum import Enum


def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


class Settings:
    def __init__(
        self,
        DETECTION_HIGH,
        DETECTION_MEDIUM,
        DETECTION_LOW,
        DETECTION_MODEL,
        FACE_HIGH,
        FACE_MEDIUM,
        FACE_LOW,
        FACE_MODEL,
        SUPERRESOLUTION_MODEL,
    ):
        self.DETECTION_HIGH = DETECTION_HIGH
        self.DETECTION_MEDIUM = DETECTION_MEDIUM
        self.DETECTION_LOW = DETECTION_LOW
        self.DETECTION_MODEL = DETECTION_MODEL
        self.FACE_HIGH = FACE_HIGH
        self.FACE_MEDIUM = FACE_MEDIUM
        self.FACE_LOW = FACE_LOW
        self.FACE_MODEL = FACE_MODEL
        self.SUPERRESOLUTION_MODEL = SUPERRESOLUTION_MODEL


class SharedOptions:

    CUDA_MODE = os.getenv("CUDA_MODE", "False")

    APPDIR = os.getenv("APPDIR", ".")
    PROFILE = os.getenv("PROFILE", "desktop_cpu")

    if PROFILE == "windows_native":
        sys.path.append(os.path.join(APPDIR, "windows_packages"))

    THREADCOUNT = int(os.getenv("THREADCOUNT", "5"))

    from redis import RedisError, StrictRedis

    TEMP_PATH = os.getenv("TEMP_PATH", "/deeptemp/")
    SLEEP_TIME = 0.01
    SHARED_APP_DIR = os.path.join(APPDIR, "sharedfiles")
    GPU_APP_DIR = os.path.join(APPDIR, "gpufiles")
    CPU_APP_DIR = os.path.join(APPDIR, "cpufiles")
    DATA_DIR = os.getenv("DATA_DIR", "/datastore")

    if CUDA_MODE == "True":
        APP_DIR = GPU_APP_DIR
        CUDA_MODE = True
    else:
        APP_DIR = CPU_APP_DIR
        CUDA_MODE = False

    MODE = "Medium"

    if "MODE" in os.environ:
        MODE = os.environ["MODE"]

    db = StrictRedis(host="localhost", db=0, decode_responses=True)
    TB_EMBEDDINGS = "TB_EMBEDDINGS"

    PROFILE_SETTINGS = {
        "desktop_cpu": Settings(
            DETECTION_HIGH=640,
            DETECTION_MEDIUM=416,
            DETECTION_LOW=256,
            DETECTION_MODEL="yolov5m.pt",
            FACE_HIGH=416,
            FACE_MEDIUM=320,
            FACE_LOW=256,
            FACE_MODEL="face.pt",
            SUPERRESOLUTION_MODEL="bebygan_x4.pth",
        ),
        "desktop_gpu": Settings(
            DETECTION_HIGH=640,
            DETECTION_MEDIUM=416,
            DETECTION_LOW=256,
            DETECTION_MODEL="yolov5m.pt",
            FACE_HIGH=416,
            FACE_MEDIUM=320,
            FACE_LOW=256,
            FACE_MODEL="face.pt",
            SUPERRESOLUTION_MODEL="bebygan_x4.pth",
        ),
        "jetson": Settings(
            DETECTION_HIGH=416,
            DETECTION_MEDIUM=320,
            DETECTION_LOW=256,
            DETECTION_MODEL="yolov5s.pt",
            FACE_HIGH=384,
            FACE_MEDIUM=256,
            FACE_LOW=192,
            FACE_MODEL="face_lite.pt",
            SUPERRESOLUTION_MODEL="bebygan_x4.pth",
        ),
        "arm64_cpu": Settings(
            DETECTION_HIGH=640,
            DETECTION_MEDIUM=416,
            DETECTION_LOW=256,
            DETECTION_MODEL="yolov5m.pt",
            FACE_HIGH=416,
            FACE_MEDIUM=320,
            FACE_LOW=256,
            FACE_MODEL="face.pt",
            SUPERRESOLUTION_MODEL="bebygan_x4.pth",
        ),
        "arm64_rpi": Settings(
            DETECTION_HIGH=416,
            DETECTION_MEDIUM=320,
            DETECTION_LOW=256,
            DETECTION_MODEL="yolov5s.pt",
            FACE_HIGH=384,
            FACE_MEDIUM=256,
            FACE_LOW=192,
            FACE_MODEL="face_lite.pt",
            SUPERRESOLUTION_MODEL="bebygan_x4.pth",
        ),
        "windows_native": Settings(
            DETECTION_HIGH=640,
            DETECTION_MEDIUM=416,
            DETECTION_LOW=256,
            DETECTION_MODEL="yolov5m.pt",
            FACE_HIGH=416,
            FACE_MEDIUM=320,
            FACE_LOW=256,
            FACE_MODEL="face.pt",
            SUPERRESOLUTION_MODEL="bebygan_x4.pth",
        ),
    }

    SETTINGS = PROFILE_SETTINGS[PROFILE]
