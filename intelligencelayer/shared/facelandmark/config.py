from easydict import EasyDict as edict

class Config:
    MODEL=edict()
    MODEL.SCALE=4
    MODEL.IN_CHANNEL=3
    MODEL.OUT_CHANNEL=3
    MODEL.N_FEATURE=64
    MODEL.N_BLOCK=6
    MODEL.DEVICE='cuda'







config=Config()
