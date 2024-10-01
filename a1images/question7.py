import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image (convert to grayscale for simplicity)
image = cv2.imread('a1images/einstein.png', cv2.IMREAD_GRAYSCALE)
# Define the Sobel kernels for X and Y directions
sobel_x = np.array([[1, 0, -1], 
                    [2, 0, -2], 
                    [1, 0, -1]])

sobel_y = np.array([[1, 2, 1], 
                    [0, 0, 0], 
                    [-1, -2, -1]])

# Apply Sobel filter using filter2D for both X and Y directions
sobel_x_filtered = cv2.filter2D(image, -1, sobel_x)
sobel_y_filtered = cv2.filter2D(image, -1, sobel_y)

# Compute the gradient magnitude
sobel_combined = np.sqrt(sobel_x_filtered**2 + sobel_y_filtered**2)
sobel_combined = np.uint8(sobel_combined)

# Display the Sobel-filtered image
plt.subplot(1, 3, 1)
plt.imshow(sobel_x_filtered, cmap='gray')
plt.title('Sobel X')

plt.subplot(1, 3, 2)
plt.imshow(sobel_y_filtered, cmap='gray')
plt.title('Sobel Y')

plt.subplot(1, 3, 3)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Combined')

plt.show()


def custom_sobel_filter(image, kernel):
    """
    Custom implementation of 2D convolution using a Sobel kernel.
    """
    # Get image dimensions
    rows, cols = image.shape
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2

    # Pad the image to handle borders
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    
    # Initialize output image
    output = np.zeros_like(image)

    # Perform 2D convolution
    for i in range(rows):
        for j in range(cols):
            # Extract the region of interest
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            # Perform element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)
    
    return output

# Apply custom Sobel filter for X and Y directions
sobel_x_custom = custom_sobel_filter(image, sobel_x)
sobel_y_custom = custom_sobel_filter(image, sobel_y)

# Compute the gradient magnitude
sobel_custom_combined = np.sqrt(sobel_x_custom**2 + sobel_y_custom**2)
sobel_custom_combined = np.uint8(sobel_custom_combined)

# Display the results of custom Sobel filtering
plt.subplot(1, 3, 1)
plt.imshow(sobel_x_custom, cmap='gray')
plt.title('Custom Sobel X')

plt.subplot(1, 3, 2)
plt.imshow(sobel_y_custom, cmap='gray')
plt.title('Custom Sobel Y')

plt.subplot(1, 3, 3)
plt.imshow(sobel_custom_combined, cmap='gray')
plt.title('Custom Sobel Combined')

plt.show()

# Step 1: Apply the vertical filter
vertical_filter = np.array([[1], [2], [1]])
horizontal_filter = np.array([[1, 0, -1]])

# Apply the vertical filter first (for both X and Y directions)
vertical_filtered = cv2.filter2D(image, -1, vertical_filter)

# Step 2: Apply the horizontal filter
factorized_sobel_x = cv2.filter2D(vertical_filtered, -1, horizontal_filter)
factorized_sobel_y = cv2.filter2D(vertical_filtered, -1, horizontal_filter.T)

# Compute the gradient magnitude for the factorized Sobel
sobel_factorized_combined = np.sqrt(factorized_sobel_x**2 + factorized_sobel_y**2)
sobel_factorized_combined = np.uint8(sobel_factorized_combined)

# Display the results of factorized Sobel filtering
plt.subplot(1, 3, 1)
plt.imshow(factorized_sobel_x, cmap='gray')
plt.title('Factorized Sobel X')

plt.subplot(1, 3, 2)
plt.imshow(factorized_sobel_y, cmap='gray')
plt.title('Factorized Sobel Y')

plt.subplot(1, 3, 3)
plt.imshow(sobel_factorized_combined, cmap='gray')
plt.title('Factorized Sobel Combined')

plt.show()

