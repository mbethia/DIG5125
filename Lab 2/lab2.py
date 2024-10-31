import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('ye.jpg')

# Convert to Grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('image', gray_image)

# Convert to Binary
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Acessing Pixels, Rows, Columns, and Channels

pixel_value = image[50, 50]
row_values = image[50, :]
red_channel = image[:, :, 0]
blue_channel = image[0, :, :]

image[:, :, 0] += 50
image[0, :, :] -+ 25

# Visualise the image as plots and subplots

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(binary_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Thresholded Image')

plt.show()

