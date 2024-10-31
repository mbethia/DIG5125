import numpy as np
import cv2

# Load the image using OpenCV
ImageA = cv2.imread('lena.tiff', cv2.IMREAD_UNCHANGED)
ImageB = cv2.imread('mandrill2.jpg', cv2.IMREAD_UNCHANGED)
# IMREAD_UNCHANGED ensures that the image is loaded as is
#(included alpha channel if it exists)

# Check the dimensions of ImageA and convert if needed
if ImageA is None:
    print("Error: Could not read the image.")
    exit()

dims = np.shape(ImageA)

if len(dims) == 3 and dims[2] == 3: # RGB image
    Imagea1 = cv2.cvtColor(ImageA, cv2.COLOR_BGR2GRAY)
    print('Image A is an RGB image, it is now converted ot grayscale')

# Check for images with channels > 3 (e.g., RGBA)
elif len(dims) == 3 and dims[2] > 3:
    print('Image A is not an RGB image')
    exit()

else: # Grayscale image
    print('Image A is a grayscale image, no conversion needed')


# Load the RGB images
ImageA_RGB = cv2.imread('lena.tiff', cv2.IMREAD_COLOR)
ImageB_RGB = cv2.imread('mandrill2.jpg', cv2.IMREAD_COLOR)

# Check if the images were successfully loaded
if ImageA_RGB is None or ImageB_RGB is None:
    print("Error: Could not read one of both images.")
    exit()

# Convert the RGB images to grayscale for further processing
ImageA1 = cv2.cvtColor(ImageA_RGB, cv2.COLOR_BGR2GRAY)
ImageB1 = cv2.cvtColor(ImageB_RGB, cv2.COLOR_BGR2GRAY)

# Get the sizes of the images
sizeA = ImageA1.shape
sizeB = ImageB1.shape

if sizeA != sizeB:
    #  Resize based on the width and the height of ImageA1
    print("The images are different sizes. Resizing ImageB1 to match ImageA1.")
    ImageB1 = cv2.resize(ImageB1, (sizeA[1], sizeA[0]))

else:
    print("The images are the same size, therefore I can continue")

# Now, irrespective of whether they were originally of the same size of not,
# you can proceed to the next steps:
# Threshold the grayscale images to create a binary result
# 127 is the threshold value. You can adjust it if needed.
_, ImageA2 = cv2.threshold(ImageA1, 127, 255, cv2.THRESH_BINARY)
_, ImageB2 = cv2.threshold(ImageB1, 127, 255, cv2.THRESH_BINARY)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# Assuming ImageA2 and ImageB2 are already loaded and processed as in the previous code
ImageC = cv2.bitwise_and(ImageA2, ImageB2)

fig = plt.figure()
gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 2, 2], height_ratios=[2, 1])

ax1 = plt.subplot(gs[0])
ax1.imshow(cv2.cvtColor(ImageA, cv2.COLOR_BGR2RGB))
ax1.set_title('original A')
ax1.axis('off') # To turn off axis numbers

ax2 = plt.subplot(gs[1])
ax2.imshow(ImageA1, cmap='gray')
ax2.set_title('grayscale')
ax2.axis('off')

ax3 = plt.subplot(gs[2])
ax3.imshow(ImageA2, cmap='gray')
ax3.set_title('Binary')
ax3.axis('off')

ax4 = plt.subplot(gs[5])
ax4.imshow(cv2.cvtColor(ImageB, cv2.COLOR_BGR2RGB))
ax4.set_title('original B')
ax4.axis('off')

ax5 = plt.subplot(gs[6])
ax5.imshow(ImageB1, cmap ='gray')
ax5.set_title('grayscale')
ax5.axis('off')

ax6 = plt.subplot(gs[7])
ax6.imshow(ImageB2, cmap='gray')
ax6.set_title('BW')
ax6.axis('off')

# Spanning over multiple positions for the "And image"
ax7 = plt.subplot(gs[3:5])
ax7.imshow(ImageC, cmap='gray')
ax7.set_title('And Image')
ax7.axis('off')

MyImageBW = np.copy(ImageA2)
# Assuming MyImageBW is already loaded as a grayscale image
# If you have a color image, convert it to a grayscale using:
# MyImageBW = cv2.cvtColor(MyImage, cv2.COLOR_BGR2GRAY) 

# Create the structuring element (disk with radius 5)
MyStrel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

# Dilation
MyDilation = cv2.dilate(MyImageBW, MyStrel, iterations=1)

# Erosion
MyErosion = cv2.erode(MyImageBW, MyStrel, iterations=1)

# Plotting the images
images = [cv2.cvtColor(ImageA, cv2.COLOR_BGR2RGB), MyImageBW, MyDilation, MyErosion, MyStrel]
titles = ['Original', 'BW Image', 'Dilation', 'Erosion', 'My Strel']

plt.figure()

for i, (img, title) in enumerate(zip(images, titles), 1):
    plt.subplot(1, 5, i)
    if i == 1: # If it's the original color image
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()

import cv2
import numpy as np

# Read the structuring element image
my_strel_image = cv2.imread('mario.jpg', cv2.IMREAD_GRAYSCALE)

# Convert the image data into logical 0-1 data
_, my_strel_image = cv2.threshold(my_strel_image, 127, 255, cv2.THRESH_BINARY)

# Read the main image you want to dilate 
my_image_bw = cv2.imread('mandrill2.jpg', cv2.IMREAD_GRAYSCALE)

# Perform dilation using the structuring element image
my_dilation = cv2.dilate(my_image_bw, my_strel_image, iterations=1)

# Perform dilation using the structuring element image 
my_dilation = cv2.dilate(my_image_bw, my_strel_image, iterations=1)

# Display the result
cv2.imshow('Dilation Image', my_dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

