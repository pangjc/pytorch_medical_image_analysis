"""
Trains a PyTorch image detection model using device-agnostic code.
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import imgaug.augmenters as iaa
from torchvision import transforms
import data_setup, engine, model_builder, utils
from data_setup import CardiacDataset


# Setup hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

# Setup directories
"""
train_root_path = "/home/pangjc/Dropbox/Pytorch_MIA/Data/Processed-Heart-Detection/train/"
train_subjects = "/home/pangjc/Dropbox/Pytorch_MIA/Data/Processed-Heart-Detection/train_subjects_det.npy"
val_root_path = "/home/pangjc/Dropbox/Pytorch_MIA/Data/Processed-Heart-Detection/val/"
val_subjects = "/home/pangjc/Dropbox/Pytorch_MIA/Data/Processed-Heart-Detection/val_subjects_det.npy"
label_csv_file = "/home/pangjc/Dropbox/Pytorch_MIA/Data/Processed-Heart-Detection/rsna_heart_detection.csv"
"""

train_root_path = "/Users/pangjc/Dropbox/Pytorch_MIA/Data/Processed-Heart-Detection/train/"
train_subjects = "/Users/pangjc/Dropbox/Pytorch_MIA/Data/Processed-Heart-Detection/train_subjects_det.npy"
val_root_path = "/Users/pangjc/Dropbox/Pytorch_MIA/Data/Processed-Heart-Detection/val/"
val_subjects = "/Users/pangjc/Dropbox/Pytorch_MIA/Data/Processed-Heart-Detection/val_subjects_det.npy"
label_csv_file = "/Users/pangjc/Dropbox/Pytorch_MIA/Data/Processed-Heart-Detection/rsna_heart_detection.csv"

# Setup target device: cuda, apple silicon, cpu
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
if torch.backends.mps.is_available():
    device = "mps"
print(f"Training on device {device}.")

# Create transforms
train_transforms = iaa.Sequential([
                                iaa.GammaContrast(),
                                iaa.Affine(
                                    scale=(0.8, 1.2),
                                    rotate=(-10, 10),
                                    translate_px=(-10, 10)
                                )
                            ])

# Create DataLoaders with help from data_setup.py
train_dataset = CardiacDataset(label_csv_file, train_subjects, train_root_path, train_transforms)
val_dataset = CardiacDataset(label_csv_file, val_subjects, val_root_path, None)

print(f"There are {len(train_dataset)} train images and {len(val_dataset)} val images")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Create model with help from model_builder.py
model = model_builder.CardiacDetectionModel().to(device=device)  # Instanciate the model

# Set loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss()

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_loader,
             test_dataloader=val_loader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="CardiacDetectionModel_rtx3090.pth")
