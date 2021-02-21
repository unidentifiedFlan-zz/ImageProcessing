from PIL import Image, ImageFilter
import math, random
import numpy as np

def convolve_seperable(image, kernel_h):
    width, height = image.shape

    image_cpy = image.copy()
    for i in range(width):
        cpy_arr = image_cpy[i, :]
        image_cpy[i, :] = np.convolve(cpy_arr, kernel_h, 'same')

    for j in range(height):
        cpy_arr = image_cpy[:, j]
        convolved_arr = np.convolve(cpy_arr, kernel_h, 'same')
        image_cpy[:, j] = convolved_arr

    return image_cpy


def median_filter(image, size):
    width, height = image.shape
    half_size = size//2
    denoised_arr = image.copy()
    median_pos = size*size // 2
    for x in range(width):
        for y in range(height):
            if width > x + half_size and x - half_size > 0 and height > y + half_size and y - half_size > 0 :
                pxls = [denoised_arr[x, y]]
                for i in range(size):
                   for j in range(size):
                       pxls.append(denoised_arr[x + i - half_size, y + j - half_size])

                sorted_values = sorted(pxls)
                denoised_arr[x, y] = sorted_values[median_pos]

    return denoised_arr


def sp_noise(image, prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


noisy_file = "C:/Users/scday/Documents/coding/ComputerVision/Denoising/noisygradient.png"
save_dir = "C:/Users/scday/Documents/coding/ComputerVision/Denoising/"

noisy_im = Image.open(noisy_file).convert('L')
width, height = noisy_im.size

file = "C:/Users/scday/Documents/coding/ComputerVision/Denoising/IMG_3379.jpg"
clean_img = Image.open(file).convert('L')
clean_arr = np.asarray(clean_img)
noisy_arr = sp_noise(clean_arr, 0.04)
noisy_im = Image.fromarray(noisy_arr)

gaussian_kernel_h = 0.0625*np.array([1, 4, 6, 4, 1])
bilinear_kernel_h = 0.25*np.array([1, 2, 1])
box_kernel_h = 1/3 * np.array([1, 1, 1])
identity_kernel_h = np.array([1])

#noisy_arr = np.array(noisy_im)

gaussian_denoised = convolve_seperable(noisy_arr, gaussian_kernel_h)
bilinear_denoised = convolve_seperable(noisy_arr, bilinear_kernel_h)
box_denoised = convolve_seperable(noisy_arr, box_kernel_h)
median_denoised = median_filter(noisy_arr, 5)

identity = convolve_seperable(noisy_arr, identity_kernel_h)

gauss_denoised_im = Image.fromarray(gaussian_denoised, mode='L')
gauss_denoised_im.save(save_dir + "gauss_convolved.png")

bilinear_denoised_im = Image.fromarray(bilinear_denoised, mode='L')
bilinear_denoised_im.save(save_dir + "bilinear_convolved.png")

box_denoised_im = Image.fromarray(box_denoised,  mode='L')
box_denoised_im.save(save_dir + "box_convolved.png")

median_denoised_im = Image.fromarray(median_denoised, mode='L')
median_denoised_im.save(save_dir + "median_filter.png")

median_off = noisy_im.filter(ImageFilter.MedianFilter)
median_off.save(save_dir + "median_official.png")

identity_im = Image.fromarray(identity, mode='L')
identity_im.save(save_dir + "identity.png")