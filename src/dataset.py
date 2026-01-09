#dataset loader -Whenever training code asks for an image, load it from disk and return a normalized (0–1) version
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

class OilSpillDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = sorted(os.listdir(image_dir))
        self.augment = augment

        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        # Load image
        image = cv2.imread(
            os.path.join(self.image_dir, image_name),
            cv2.IMREAD_GRAYSCALE
        )
        image = image.astype(np.float32) / 255.0

        # Load mask
        mask = cv2.imread(
            os.path.join(self.mask_dir, image_name),
            cv2.IMREAD_GRAYSCALE
        )
        mask = (mask > 0).astype(np.float32)

        # Augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensors + add channel dim
        image = torch.from_numpy(image).unsqueeze(0)  # [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)    # [1, H, W]

        return image, mask
