import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

a = 0.5
sigma = 70

#load the image
image4 = cv.imread('a1images/spider.png')

#convert image to hsv type
hsv_image = cv.cvtColor(image4, cv.COLOR_BGR2HSV)
#split the image
hue, saturation, value = cv.split(hsv_image)
saturation_transform = np.minimum(saturation + a* 128* np.exp(-((saturation- 128)**2)/(2* sigma**2)), 255)

# Convert the result back to an 8-bit format
saturation_transformed = np.uint8(saturation_transform)

hsv_transform = cv.merge([hue, saturation_transformed, value])
# Convert back to BGR color space for display
image_transformed = cv.cvtColor(hsv_transform, cv.COLOR_HSV2BGR)

# Original image
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(image4, cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# Vibrance-enhanced image
plt.subplot(1, 3, 2)
plt.imshow(cv.cvtColor(image_transformed, cv.COLOR_BGR2RGB))
plt.title("Vibrance-Enhanced Image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.plot(saturation, saturation_transform, color = 'green')
plt.show()