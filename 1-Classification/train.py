"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
from torchvision import transforms
import data_setup, engine, model_builder, utils

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

# Setup directories
train_dir = "/Users/pangjc/git_pangjc_datasets/pytorch_mia/Classification/train"
test_dir = "/Users/pangjc/git_pangjc_datasets/pytorch_mia/Classification/val"

# Setup target device: cuda, apple silicon, cpu
device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
if torch.backends.mps.is_available():
    device = "mps"
print(f"Training on device {device}.")

# Create transforms
train_transforms = transforms.Compose([
                                    transforms.ToTensor(),  # Convert numpy array to tensor
                                    transforms.Normalize(0.49, 0.248),  # Use mean and std from preprocessing notebook
                                    transforms.RandomAffine( # Data Augmentation
                                        degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
                                        transforms.RandomResizedCrop((224, 224), scale=(0.35, 1))

])

test_transforms = transforms.Compose([
                                    transforms.ToTensor(),  # Convert numpy array to tensor
                                    transforms.Normalize([0.49], [0.248]),  # Use mean and std from preprocessing notebook
])


# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    train_transform=train_transforms,
    test_transform=test_transforms,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.PneumoniaModel().to(device=device)  # Instanciate the model

# Set loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="PneumoniaModel.pth")
