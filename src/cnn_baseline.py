import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):

    model = models.mobilenet_v3_small(weights='DEFAULT')
    
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    model = get_model(num_classes=10)
    print("CNN Baseline (MobileNetV3) initialized.")