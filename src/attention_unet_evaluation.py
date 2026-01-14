import torch
from torch.utils.data import DataLoader

from dataset import OilSpillDataset
from attention_unet import AttentionUNet
from utils import dice_score, iou_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset (VALIDATION)
val_dataset = OilSpillDataset(
    image_dir="/content/drive/MyDrive/oil-spill-data/images/val",
    mask_dir="/content/drive/MyDrive/oil-spill-data/masks/val",
    augment=False
)

val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Load model
model = AttentionUNet().to(device)

model.load_state_dict(
    torch.load("models/attention_unet.pth", map_location=device)
)


model.eval()
dice_total = 0.0
iou_total = 0.0

with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

        dice_total += dice_score(preds, masks).item()
        iou_total += iou_score(preds, masks).item()

dice_avg = dice_total / len(val_loader)
iou_avg = iou_total / len(val_loader)

print(f"Average Dice Score: {dice_avg:.4f}")
print(f"Average IoU Score : {iou_avg:.4f}")
