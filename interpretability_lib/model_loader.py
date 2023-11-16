import torch
from torchvision import models


def load_resnet50(pretrained=True):
    """Load a pretrained ResNet-50 model."""
    model = models.resnet50(pretrained=pretrained)
    model.eval()
    return model
