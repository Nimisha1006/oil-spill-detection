import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import OilSpillDataset
from attention_unet import AttentionUNet


# ----------------------------
# 1. DEVICE
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# 2. DATASET
# ----------------------------
train_dataset = OilSpillDataset(
    image_dir="/content/drive/MyDrive/oil-spill-data/images/train",
    mask_dir="/content/drive/MyDrive/oil-spill-data/masks/train",
    augment=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

# ----------------------------
# 3. MODEL
# ----------------------------
model = AttentionUNet().to(device)

# ----------------------------
# 4. LOSS & OPTIMIZER
# ----------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ----------------------------
# 5. TRAINING LOOP
# ----------------------------
epochs = 5

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss / len(train_loader):.4f}")

# ----------------------------
# 6. SAVE MODEL
# ----------------------------
os.makedirs("results/models", exist_ok=True)
torch.save(model.state_dict(), "results/models/attention_unet.pth")

print("Training completed.")

