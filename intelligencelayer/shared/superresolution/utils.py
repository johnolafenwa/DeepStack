import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel


def load_model(model, model_path, strict=True, cpu=False):
    if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
        model = model.module
    if cpu:
        loaded_model = torch.load(model_path, map_location='cpu')
    else:
        loaded_model = torch.load(model_path)
    model.load_state_dict(loaded_model, strict=strict)
