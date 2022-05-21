from easydict import EasyDict as edict


class Config:
    # dataset

    # dataloader

    # model
    MODEL = edict()
    MODEL.SCALE = 4
    MODEL.IN_CHANNEL = 3
    MODEL.OUT_CHANNEL = 3
    MODEL.N_FEATURE = 64
    MODEL.N_BLOCK = 23
    MODEL.GROWTH_CHANNEL = 32
    MODEL.DOWN = 1
    MODEL.DEVICE = 'cuda'

    # solver

    # initialization

    # log and save

    # validation


config = Config()
