import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read the grayscale image
image = cv2.imread('pout.jpg')

# Convert BGR to RGB (OpenCV loads images in BGR format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Generate X values (0-255 for grayscale)
x_values = np.arange(256)

# Calculate histograms for each channel
hist_r = cv2.calcHist([image_rgb], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([image_rgb], [1], None, [256], [0, 256])
hist_b = cv2.calcHist([image_rgb], [2], None, [256], [0, 256])
hist_gray = cv2.calcHist([image], [0], None, [256], [0, 256])

# Create a 2x2 subplot
fig, axs = plt.subplots(2, 2, figsize=(10,8))

# Plot Histograms
axs[0, 0].bar(x_values, hist_gray.ravel(), color='gray', alpha=0.5)
axs[0, 0].set_title("Grayscale Histogram")
axs[0, 0].set_xlabel("Pixel Value")
axs[0, 0].set_ylabel("Frequency")

axs[0, 1].bar(x_values, hist_r.ravel(), color='red', alpha=0.5)
axs[0, 1].set_title("Red Channel Histogram")
axs[0, 1].set_xlabel("Pixel Value")
axs[0, 1].set_ylabel("Frequency")

axs[1, 0].bar(x_values, hist_g.ravel(), color='green', alpha=0.5)
axs[1, 0].set_title("Red Channel Histogram")
axs[1, 0].set_xlabel("Pixel Value")
axs[1, 0].set_ylabel("Frequency")

axs[1, 1].bar(x_values, hist_b.ravel(), color='blue', alpha=0.5)
axs[1, 1].set_title("Blue Channel Histogram")
axs[1, 1].set_xlabel("Pixel Value")
axs[1, 1].set_ylabel("Frequency")

# Adjust layout
plt.tight_layout()
plt.show()

# Create a 2D array
array_2D = np.array([[1, 2, 3],
                    [4, 5, 6]])

# Use ravel to flatten the 2D array to a 1D array
array_1d = array_2D.ravel()

array_1d = np.zeros(5)

array_2D = np.zeros((3, 4))

array_3D = np.zeros((2, 3, 4))

array_2d_int = np.zeros((3, 4), dtype=int)

print("Original 2D array:")
print(array_2D)

print("Flattened 1D array:")
print(array_1d)

# Create an empty grayscale image of 512x512 pixels
empty_image = np.zeros((512, 512), dtype=np.uint8)

# Create an empty RGB image of 512x512 pixels
empty_rgb_image = np.zeros((512, 512, 3), dtype=np.uint8)

# Read the image
def grayscale_histogram(image):
    hist = np.zeros(256, dtype=int)
    for pixel in image.ravel():
        hist[pixel] += 1
    return hist

image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
hist = grayscale_histogram(image)
plt.plot(hist)
plt.show()

# Histogram Equalisation
image2 = cv2.imread('pout.jpg', cv2.IMREAD_GRAYSCALE)
equalized_image = cv2.equalizeHist(image2)

plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))
plt.show()

# Histogram Normalisation
image3 = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
normalized_image = cv2.normalize(image3, None, 0, 255, cv2.NORM_MINMAX)

plt.imshow(cv2.CvtColor(normalized_image, cv2.COLOR_BGR2RGB))
plt.show()