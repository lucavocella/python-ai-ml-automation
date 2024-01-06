import torch
from torchvision.models import efficientnet_v2_s

def get_model(path_to_model=None, device='cpu'):
    model = efficientnet_v2_s(weights='IMAGENET1K_V1')
    model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=1)
    if path_to_model:
        model.load_state_dict(torch.load(path_to_model, map_location=device))
    model = model.to(device)
    return model