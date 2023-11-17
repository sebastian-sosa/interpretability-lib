from torchvision import models


def load_resnet50(pretrained=True):
    """Load a pretrained ResNet-50 model."""
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.eval()
    return model


def load_vgg16(pretrained=True):
    """Load a pretrained VGG16 model."""
    model = models.vgg16(weights='IMAGENET1K_V1')
    model.eval()
    return model

