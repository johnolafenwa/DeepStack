import torch

def load_model(model,path):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

    try:
        model.load_state_dict(checkpoint)
        
    except:
        copy = dict()
        for x, y in zip(model.state_dict(), checkpoint):
            new_name = y[y.index(x):]
            copy[new_name] = checkpoint[y]
        model.load_state_dict(copy)