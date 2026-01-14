import torch
import matplotlib.pyplot as plt

from dataset import OilSpillDataset
from model import AttentionUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset (validation)
val_dataset = OilSpillDataset(
    image_dir="/content/drive/MyDrive/oil-spill-data/images/val",
    mask_dir="/content/drive/MyDrive/oil-spill-data/masks/val",
    augment=False
)

# Load model
model = AttentionUNet().to(device)
model.load_state_dict(
    torch.load("results/models/attention_unet.pth", map_location=device)
)
model.eval()

# Take one sample
image, mask = val_dataset[0]
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    pred = torch.sigmoid(output)
    pred = (pred > 0.5).float()

# Plot
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("SAR Image")
plt.imshow(image[0,0].cpu(), cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Ground Truth Mask")
plt.imshow(mask[0], cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Predicted Mask")
plt.imshow(pred[0,0].cpu(), cmap="gray")
plt.axis("off")

plt.show()
