import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image):
    """
    Perform histogram equalization on a grayscale image.
    """
    # Get the histogram of the original image
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Masking and normalizing the CDF (ignores zero values)
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')

    # Map the original image pixels based on the CDF
    image_equalized = cdf_final[image]

    return image_equalized, hist, cdf_final

# Load image as grayscale
image = cv2.imread('a1images/shells.tif')  # Replace with your image path

# Perform histogram equalization
image_equalized, original_hist, cdf_final = histogram_equalization(image)


# Plot Histograms Before and After Equalization
plt.figure(figsize=(12, 6))

# Original image and its histogram
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.hist(image.flatten(), 256, [0, 256], color='blue')
plt.title("Original Histogram")

# Equalized image and its histogram
plt.subplot(2, 2, 3)
plt.imshow(image_equalized, cmap='gray')
plt.title("Equalized Image")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.hist(image_equalized.flatten(), 256, [0, 256], color='green')
plt.title("Equalized Histogram")

plt.tight_layout()
plt.show()
