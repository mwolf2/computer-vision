import torch

def deit_base(num_classes):
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    model.head = torch.nn.Linear(model.head.in_features, num_classes)

    return model