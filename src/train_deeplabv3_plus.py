import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import OilSpillDataset
from deeplabv3_plus import DeepLabV3Plus


# ----------------------------
# DEVICE
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ----------------------------
# DATASET
# ----------------------------
train_dataset = OilSpillDataset(
    image_dir="/content/data/images/train",
    mask_dir="/content/data/masks/train",
    augment=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)


# ----------------------------
# MODEL
# ----------------------------
model = DeepLabV3Plus().to(device)


# ----------------------------
# LOSS & OPTIMIZER
# ----------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# ----------------------------
# TRAINING LOOP
# ----------------------------
epochs = 30

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
# SAVE MODEL
# ----------------------------
os.makedirs("results/models", exist_ok=True)
torch.save(model.state_dict(), "results/models/deeplabv3_plus.pth")

print("DeepLabV3+ training completed.")

