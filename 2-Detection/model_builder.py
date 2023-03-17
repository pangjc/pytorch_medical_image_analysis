"""
Contains PyTorch model code to instantiate a resnet model.
"""
import torch
import torchvision
from torch import nn

class CardiacDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=512 ,out_features=4)
        
    def forward(self, data):
        return self.model(data)

  
