import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from Unet import UNet
from dataset import OilSpillDataset

# -----------------------------
# 1. DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 2. LOAD MODEL
# -----------------------------
model = UNet(in_channels=1).to(device)
model.load_state_dict(torch.load("models/unet_baseline_30.pth", map_location=device))
model.eval()

# -----------------------------
# 3. LOAD VALIDATION DATA
# -----------------------------
val_dataset = OilSpillDataset(
    image_dir="data/raw/images/val",
    mask_dir="data/raw/masks/val",
    augment=False
)

# Pick ONE sample
image, mask = val_dataset[0]

image = image.unsqueeze(0).to(device)  # [1, 1, H, W]
mask = mask.squeeze().cpu().numpy()

# -----------------------------
# 4. INFERENCE
# -----------------------------
with torch.no_grad():
    output = model(image)
    pred = torch.sigmoid(output)
    pred = (pred > 0.5).float()

pred = pred.squeeze().cpu().numpy()
image = image.squeeze().cpu().numpy()

# -----------------------------
# 5. VISUALIZATION
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("SAR Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Ground Truth Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Predicted Mask")
plt.imshow(pred, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

