from scipy.signal import convolve2d
import numpy as np
import cv2

def simple_1d_convolution(input_array, conv_mask):
    return np.convolve(input_array, conv_mask, mode='full')

def simple_2d_convolution(input_array, conv_mask):
    return convolve2d(input_array, conv_mask, mode='full')

input_array = np.array([1, 2, 3, 4, 5, 6])
conv_mask = np.array([1, 1, 1, 1, 1])

convolution_result = simple_1d_convolution(input_array, conv_mask)
print("Convolution Result:", convolution_result)

input_array2 = np.array([[1, 2, 3, 2], [2, 1, 1, 2], [50, 55, 50, 0], [55, 50, 50, 55]])
conv_mask2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

convolution_result2 = simple_2d_convolution(input_array2, conv_mask2)
print("2D Convolutoion Result:", convolution_result2)

def my_loop_convolution(my_input_array, my_conv_mask):
    my_input_array = np.array(my_input_array)
    my_conv_mask = np.array(my_conv_mask)

    my_result = np.zeros(len(my_input_array) + len(my_conv_mask) - 1)

    for i in range(len(my_input_array)):
        for j in range(len(my_conv_mask)):
            if i + j < len(my_result):
                my_result[i + j] += my_input_array[i] * my_conv_mask[j]
    print("Convolution Result:", my_result)

    return my_result

def apply_averaging_blur(image_path, kernel_size):
    image = cv2.imread('ye.jpg')
    blurred_image = cv2.blur(image, (kernel_size, kernel_size))

    cv2.imshow("Original", image)
    cv2.imshow|("Averaging Blur", blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()