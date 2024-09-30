import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('a1images/emma.jpg', 0)  # Load in grayscale


# Original Image Histogram
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')


# Define the intensity transformation function based on the given plot
def intensity_transform(value):
    if value < 50:
        return value  # Linear for [0, 50)
    elif value < 150:
        return (1.55*value +22.5) # Linear for [50, 150)
    elif value < 255:
        return value    # Mapping 100-150 to 255


# Apply the transformation to each pixel in the image
transformed_image = np.vectorize(intensity_transform)(image)

plt.subplot(2, 2, 2)
plt.title("Transformed Image")
plt.imshow(transformed_image, cmap='gray')

# Define input intensity levels
input_intensity = [0, 50, 50, 150, 150, 255]

# Define corresponding output intensity levels
output_intensity = [0, 50, 100, 255, 150, 255]

# Create the plot
plt.subplot(2,2,3)
plt.plot(input_intensity, output_intensity, color='blue')

# Add labels
plt.xlabel('Input intensity')
plt.ylabel('Output intensity')

# Set axis limits
plt.xlim(0, 255)
plt.ylim(0, 255)

# Show the plot
plt.grid(False)
plt.show()

