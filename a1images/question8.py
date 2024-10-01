import numpy as np
import cv2
import matplotlib.pyplot as plt

def zoom_image(image, s, method='nearest'):

    if method == 'nearest':
        zoomed_image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    elif method == 'bilinear':
        zoomed_image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError("Method must be 'nearest' or 'bilinear'.")
    
    return zoomed_image

def normalized_ssd(image1, image2):

    # Ensure images are the same shape
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions.")
    
    # Compute the SSD
    diff = image1.astype(float) - image2.astype(float)
    ssd = np.sum(diff**2) / np.prod(image1.shape)  # Normalization by number of pixels
    
    return ssd

# Load original and zoomed-out images
original_image_large = cv2.imread('a1images/a1q5images/im02.png')  # Replace with the path to your large image
small_image = cv2.imread('a1images/a1q5images/im02small.png')  # Replace with the path to your small image

# Convert images to RGB
original_image_large = cv2.cvtColor(original_image_large, cv2.COLOR_BGR2RGB)
small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)

# Zoom the small image by a factor of 4 using both methods
zoom_factor = 4
zoomed_nearest = zoom_image(small_image, zoom_factor, method='nearest')
zoomed_bilinear = zoom_image(small_image, zoom_factor, method='bilinear')

# Compare with the original large image
ssd_nearest = normalized_ssd(original_image_large, zoomed_nearest)
ssd_bilinear = normalized_ssd(original_image_large, zoomed_bilinear)

# Display results
print(f"Normalized SSD (Nearest Neighbor): {ssd_nearest:.4f}")
print(f"Normalized SSD (Bilinear): {ssd_bilinear:.4f}")

# Display images
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.title('Original Large Image')
plt.imshow(original_image_large)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Zoomed Image (Nearest Neighbor)')
plt.imshow(zoomed_nearest)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Zoomed Image (Bilinear)')
plt.imshow(zoomed_bilinear)
plt.axis('off')

plt.show()
