import torch
import torch.nn as nn
import timm

def get_model(num_classes):
   
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    
    model.head = nn.Linear(model.head.in_features, num_classes)
    
    return model

if __name__ == "__main__":
    model = get_model(num_classes=10)
    print("ViT (Vision Transformer) initialized via timm.")