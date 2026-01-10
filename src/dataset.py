#dataset loader -Whenever training code asks for an image, load it from disk and return a normalized (0–1) version
from torch.utils.data import Dataset

class OilSpillDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
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

        image_path = os.path.join(self.image_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image = image.astype(np.float32) / 255.0

        mask_path = os.path.join(self.mask_dir, image_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
