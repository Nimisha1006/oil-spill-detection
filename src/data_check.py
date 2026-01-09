import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---- paths (change only if your folder names differ) ----
IMAGE_DIR = "C:\\Users\\USER\\Desktop\\Oil-Spill-Detection\\data\\raw\\images\\train"
MASK_DIR = "C:\\Users\\USER\\Desktop\\Oil-Spill-Detection\\data\\raw\\masks\\train"

# ---- pick ONE image file ----
image_name = os.listdir(IMAGE_DIR)[0]

image_path = os.path.join(IMAGE_DIR, image_name)
mask_path = os.path.join(MASK_DIR, image_name)

# ---- load image and mask ----
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# ---- resize to 256x256 ----
image = cv2.resize(image, (256, 256))
mask = cv2.resize(mask, (256, 256))

# ---- print basic info ----
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Unique values in mask:", np.unique(mask))

# ---- show image & mask ----
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("SAR Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.show()
