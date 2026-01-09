
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "C:\\Users\\USER\\Desktop\\Oil-Spill-Detection\data\\raw\\images\\train\\palsar_1.png"  #
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

print("Original image:")
print("Shape:", image.shape)
print("Data type:", image.dtype)
print("Min pixel:", image.min())
print("Max pixel:", image.max())

image_normalized = image.astype(np.float32) / 255.0

print("\nAfter normalization:")
print("Min pixel:", image_normalized.min())
print("Max pixel:", image_normalized.max())

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Normalized Image")
plt.imshow(image_normalized, cmap="gray")
plt.axis("off")

plt.show()
