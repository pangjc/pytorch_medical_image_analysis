
"""
Contains PyTorch model code to instantiate a resnet model.
"""
import torch
import torchvision
from torch import nn

class PneumoniaModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = torchvision.models.resnet18()
        # change conv1 from 3 to 1 input channels
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # change out_feature of the last fully connected layer (called fc in resnet18) from 1000 to 1
        self.model.fc = torch.nn.Linear(in_features=512, out_features=2)
        
    def forward(self, data):
        pred = self.model(data)
        return pred
