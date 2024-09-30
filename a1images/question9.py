import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the flower image
image = cv2.imread('a1images/daisy.jpg')  # Replace with the actual image path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 1: Initialize the mask (0s for background, 1s for probable background, 2s for probable foreground, and 3s for foreground)
mask = np.zeros(image.shape[:2], np.uint8)

# Step 2: Create background and foreground models (used by GrabCut internally)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# Step 3: Define a bounding box around the flower (foreground region)
# For demonstration, we assume the flower is in the center of the image.
rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)

# Step 4: Apply GrabCut
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

# Post-processing the mask
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')  # Convert the mask to binary (0 for background, 1 for foreground)

# Extract the foreground
foreground = image * mask2[:, :, np.newaxis]

# Extract the background
background = image * (1 - mask2[:, :, np.newaxis])

# Display the segmentation results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(mask2, cmap='gray')
plt.title('Segmentation Mask')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB))
plt.title('Foreground Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
plt.title('Background Image')
plt.axis('off')

plt.show()

# Step 1: Blur the background using GaussianBlur
blurred_background = cv2.GaussianBlur(image, (25, 25), 0)

# Step 2: Combine the blurred background with the original foreground
enhanced_image = blurred_background * (1 - mask2[:, :, np.newaxis]) + foreground

# Display the original and the enhanced image side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(enhanced_image.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.title('Enhanced Image with Blurred Background')
plt.axis('off')

plt.show()
