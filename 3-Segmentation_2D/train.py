from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import imgaug.augmenters as iaa
import numpy as np

from data_setup import CardiacDataset
import model_builder, engine, utils 

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

# Create the dataset objects
train_path = Path("/Users/pangjc/Dropbox/Pytorch_MIA/Data/Task2/Preprocessed/train/")
val_path = Path("/Users/pangjc/Dropbox/Pytorch_MIA/Data/Task2/Preprocessed/val")

##train_path = Path("/Users/pangjc/Dropbox/Pytorch_MIA/Data/Task2/Preprocessed/train/")
##val_path = Path("/Users/pangjc/Dropbox/Pytorch_MIA/Data/Task2/Preprocessed/val")

seq = iaa.Sequential([
    iaa.Affine(scale=(0.85, 1.15),
              rotate=(-45, 45)),
    iaa.ElasticTransformation()
])


train_dataset = CardiacDataset(train_path, seq)
val_dataset = CardiacDataset(val_path, None)

print(f"There are {len(train_dataset)} train images and {len(val_dataset)} val images")

# Create model with help from model_builder.py
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = "mps"
print(device)

model = model_builder.AtriumSegmentation().to(device=device)  # Instanciate the model

# Set loss and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = engine.DiceLoss()

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
                 model_name="AtriumSegmentation_rtx3090.pth")
