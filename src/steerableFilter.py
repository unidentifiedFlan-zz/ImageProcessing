from PIL import Image
import numpy as np
import math

def convolve_separable(image, kernel_h):
    width, height = image.shape
    image_cpy = image.copy()
    for i in range(width):
        cpy_arr = image_cpy[i, :]
        image_cpy[i, :] = np.convolve(cpy_arr, kernel_h, 'same')

    for j in range(height):
        cpy_arr = image_cpy[:, j]
        image_cpy[:, j] = np.convolve(cpy_arr, kernel_h, 'same')

    return image_cpy

def oriented_convolution(image, kernel_h, theta):
    deriv_kernel = 0.5*np.array([-1, 0, 1])

    intermed_image = image.copy()
    intermed_image = convolve_separable(intermed_image, kernel_h)
    width, height = intermed_image.shape

    conv_h = intermed_image.copy()
    for i in range(width):
        cpy_arr = conv_h[i, :]
        conv_h[i, :] = np.convolve(cpy_arr, deriv_kernel, 'same')

    conv_v = intermed_image.copy()
    for j in range(height):
        cpy_arr = conv_v[:, j]
        conv_v[:, j] = np.convolve(cpy_arr, deriv_kernel, 'same')

    final_image = math.cos(theta)*conv_h + math.sin(theta)*conv_v
    return final_image


def main():
    file = "ComputerVision/steerable/cat.jpg"
    save_dir = "ComputerVision/steerable/"

    im = Image.open(file).convert('L')
    im_arr = np.asarray(im)

    gaussian_kernel_h = 0.0625*np.array([1, 4, 6, 4, 1])

    for x in range(8):
        theta = x*math.pi/8
        conv_arr = oriented_convolution(im_arr, gaussian_kernel_h, theta)
        conv_img = Image.fromarray(conv_arr).convert('L')
        conv_img.save(save_dir + "output" + str(theta) + ".png")


if __name__ == 'main':
    main()