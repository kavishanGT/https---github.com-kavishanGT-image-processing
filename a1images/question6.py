import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step (a): Load the image and convert to HSV
image = cv2.imread('a1images/jeniffer.jpg')  # Replace with your image path
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Split the image into Hue, Saturation, and Value planes
hue, saturation, value = cv2.split(hsv_image)

# Display the hue, saturation, and value planes
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(hue, cmap='gray')
plt.title('Hue Plane')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(saturation, cmap='gray')
plt.title('Saturation Plane')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(value, cmap='gray')
plt.title('Value Plane')
plt.axis('off')
plt.show()

# Threshold the value plane to extract foreground
_, mask = cv2.threshold(value, 120, 255, cv2.THRESH_BINARY)

# Display the binary mask
plt.imshow(mask, cmap='gray')
plt.title('Foreground Mask')
plt.axis('off')
plt.show()

# Use mask to extract the foreground
foreground = cv2.bitwise_and(value, value, mask=mask)

# Compute the histogram of the foreground
foreground_hist = cv2.calcHist([foreground], [0], mask, [256], [0, 256])

# Plot the histogram of the foreground
plt.plot(foreground_hist)
plt.title('Foreground Histogram')
plt.show()

#Compute the cumulative sum of the histogram
cdf = np.cumsum(foreground_hist)
cdf_normalized = cdf * (foreground_hist.max() / cdf.max())

# Plot the cumulative distribution function (CDF)
plt.plot(cdf_normalized, color='b')
plt.title('Cumulative Sum of Foreground Histogram')
plt.show()

#Normalize the CDF and apply histogram equalization
cdf_m = np.ma.masked_equal(cdf, 0)  # Mask the zeros in CDF
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # Normalize CDF
cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')  # Fill the masked areas

# Apply the CDF to the foreground
foreground_equalized = cdf_final[foreground]

# Display the equalized foreground
plt.imshow(foreground_equalized, cmap='gray')
plt.title('Equalized Foreground')
plt.axis('off')
plt.show()

#Extract the background
background = cv2.bitwise_and(value, value, mask=cv2.bitwise_not(mask))

# Combine the equalized foreground and background
result = cv2.add(foreground_equalized, background)

# Merge back into the HSV image
hsv_image_equalized = cv2.merge([hue, saturation, result])

# Convert back to BGR for display
image_equalized = cv2.cvtColor(hsv_image_equalized, cv2.COLOR_HSV2BGR)

# Display the original and the final result
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_equalized, cv2.COLOR_BGR2RGB))
plt.title('Result with Histogram-Equalized Foreground')
plt.axis('off')

plt.show()

