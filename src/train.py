import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import OilSpillDataset
from model import UNet


# -----------------------------
# 1. DEVICE CONFIGURATION
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# -----------------------------
# 2. DATASET & DATALOADER
# -----------------------------
train_dataset = OilSpillDataset(
    image_dir="C:\\Users\\USER\\Desktop\\Oil-Spill-Detection\\data\\raw\\images\\train",
    mask_dir="C:\\Users\\USER\\Desktop\\Oil-Spill-Detection\\data\\raw\\masks\\train",
    augment=True
)
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)



# -----------------------------
# 3. MODEL INITIALIZATION
# -----------------------------
model = UNet(in_channels=4).to(device)


# -----------------------------
# 4. LOSS FUNCTION & OPTIMIZER
# -----------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# -----------------------------
# 5. TRAINING LOOP
# -----------------------------
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}")


# -----------------------------
# 6. SAVE MODEL
# -----------------------------
os.makedirs("results/models", exist_ok=True)
torch.save(model.state_dict(), "results/models/unet_baseline.pth")

print("Training completed and model saved.")
