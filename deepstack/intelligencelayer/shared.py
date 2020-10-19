from redis import StrictRedis, RedisError

import os
import sys

class SharedOptions:

    CUDA_MODE = os.getenv("CUDA_MODE","False")

    APPDIR = os.getenv("APPDIR",".")

    TEMP_PATH = "/deeptemp/"
    SLEEP_TIME = 0.01
    #SHARED_APP_DIR = os.path.join(APPDIR,"sharedfiles")
    SHARED_APP_DIR="/home/johnolafenwa/Documents/Source/DeepStack/deepstack/sharedfiles"
    GPU_APP_DIR = os.path.join(APPDIR,"gpufiles")
    CPU_APP_DIR = os.path.join(APPDIR,"cpufiles")
    DATA_DIR = "/datastore"

    if CUDA_MODE == "True":
        APP_DIR = GPU_APP_DIR
        CUDA_MODE = True
    else:
        APP_DIR = CPU_APP_DIR
        CUDA_MODE = False

    MODE = "Medium"

    if "MODE" in os.environ:
        MODE = os.environ["MODE"]

    db = StrictRedis(host="localhost",db=0,decode_responses=True)
    TB_EMBEDDINGS = "TB_EMBEDDINGS"