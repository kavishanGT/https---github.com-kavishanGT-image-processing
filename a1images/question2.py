import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the brain image (proton density image)
img = cv2.imread('a1images/brain_proton_density_slice.png', 0)  # Assuming grayscale

# Define intensity transformation for white matter accentuation
def gray_matter(x):
    #sigmoid function to map the intensity values for gray matter
    aplpha = 0.1
    beta = 180
    curve = 255/(1+np.exp(-aplpha*(x-beta)))
    return curve


# Define intensity transformation for gray matter accentuation
def white_matter(x):
    #inverse sigmoid function to map the intensity values for gray matter
    aplpha = 0.1
    beta = 180
    curve = 255/(1+np.exp(aplpha*(x-beta)))
    return curve


# Apply transformations
white_matter_accent = white_matter(img)
gray_matter_accent = gray_matter(img)

# Plot the transformation functions
intensity_range = np.arange(256)
white_matter_trans = white_matter(intensity_range)
gray_matter_trans = gray_matter(intensity_range)

plt.figure()
plt.plot(intensity_range, white_matter_trans, label='White Matter')
plt.plot(intensity_range, gray_matter_trans, label='Gray Matter')
plt.xlabel('Original Intensity')
plt.ylabel('Transformed Intensity')
plt.legend()
plt.title('Intensity Transformation for White and Gray Matter')
plt.show()

# Display the original image and transformed images
plt.figure(figsize=(10,5))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(white_matter_accent, cmap='gray')
plt.title('White Matter Accentuated')

plt.subplot(1, 3, 3)
plt.imshow(gray_matter_accent, cmap='gray')
plt.title('Gray Matter Accentuated')

plt.show()
