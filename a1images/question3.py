import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image3 = cv.imread('a1images/highlights_and_shadows.jpg')
cvt_img = cv.cvtColor(image3, cv.COLOR_BGR2Lab)
fig, ax = plt.subplots(1,4, figsize = (18,4))
L,a,b = cv.split(cvt_img)
# Apply gamma correction to the L channel
gamma = 4  # You can modify the gamma value
L_float = L / 255.0  # Normalize to [0, 1]
L_gamma_corrected = np.power(L_float, gamma) * 255  # Gamma correction
L_gamma_corrected = np.uint8(L_gamma_corrected)

# Merge the corrected L with original a, b channels
lab_gamma_corrected = cv.merge([L_gamma_corrected, a, b])

# Convert back to BGR color space
corrected_image = cv.cvtColor(lab_gamma_corrected, cv.COLOR_LAB2BGR)

#plot histogram for original image
hist1, bins = np.histogram(image3.ravel(), 256, [0, 256])

#plot histogram for corrected image
hist2, bins = np.histogram(corrected_image.ravel(), 256, [0, 256])
plt.subplot(2,2,1)
plt.imshow(cv.cvtColor(image3, cv.COLOR_BGR2RGB))
#ax[1].imshow(corrected_image)
plt.subplot(2,2,2)
plt.imshow(cv.cvtColor(corrected_image, cv.COLOR_BGR2RGB))

plt.subplot(2,2,3)
plt.plot(hist1)
plt.title('Histogram for original image')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')

plt.subplot(2,2,4)
plt.plot(hist2)
plt.title('Histogram for corrected image')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.show()