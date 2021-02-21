import numpy as np
from PIL import Image
from sys import exit


def create_upscaled_base_image(image):
    factor = int(2)

    i_width, i_height, i_channels = image.shape
    o_width = factor * i_width
    o_height = factor * i_height
    new_img = np.zeros((o_width, o_height, 4))

    for x in range(i_width):
        for y in range(i_height):
            new_img[factor*x][factor*y] = image[x][y]

#    new_img[:, :, 3] = 255*np.ones((o_width, o_height))
    return new_img


def create_decimated_image(image):
    factor = int(2)
    i_width, i_height, i_channels = image.shape
    o_width = int(1/factor * i_width)
    o_height = int(1/factor * i_height)
    new_img = np.zeros((o_width, o_height, 4))

    for x in range(o_width):
        for y in range(o_height):
            new_img[x][y] = image[factor * x][factor * y]

#    new_img[:, :, 3] = 255*np.ones((o_width, o_height))
    return new_img


def convolve_image(image, kernel):
    width, height, channels = image.shape

    image_cpy = image.copy()
    for c in range(channels):
        for i in range(width):
            cpy_arr = image_cpy[i, :, c]
            image_cpy[i, :, c] = np.convolve(cpy_arr, kernel, 'same')

        for j in range(height):
            cpy_arr = image_cpy[:, j, c]
            image_cpy[:, j, c] = np.convolve(cpy_arr, kernel, 'same')

    return image_cpy


def create_gaussian_pyramid(image):
    width, height, channels = image.shape
    if width < height:
        min_dim = width
    else:
        min_dim = height

    gaussian_list = [image]

    decimated_arr = image
    i = min_dim

    kernel_width = binomial_kernel.size
    while i > kernel_width:
        blurred_arr = convolve_image(decimated_arr, binomial_kernel)
        print(i)
        print(blurred_arr[:, :, 3])
        decimated_arr = create_decimated_image(blurred_arr)
        print("decimated")
        print(decimated_arr[:, :, 3])
        gaussian_list.append(decimated_arr)
        i = i/2


    return gaussian_list

def create_laplacian_pyramid(image):

    laplacian_list = []

    gaussian_list = create_gaussian_pyramid(image)
    list_length = len(gaussian_list)

    for j in range(list_length-1, 0, -1):
        upscaled_arr = create_upscaled_base_image(gaussian_list[j])
        upscaled_arr = 4*convolve_image(upscaled_arr, binomial_kernel)
        laplacian_arr = gaussian_list[j-1] - upscaled_arr
        laplacian_list.append(laplacian_arr)

    return laplacian_list

def reconstruct_from_laplacian_pyramid(pyramid):

    n = len(pyramid)
    current_img = pyramid[0]
    for i in range(1, n):
        upscaled_img = create_upscaled_base_image(current_img)
        upscaled_img = 4*convolve_image(upscaled_img, binomial_kernel)
        current_img = upscaled_img + pyramid[i]

    return current_img


bilinear_kernel = 0.25*np.array([1, 2, 1])
binomial_kernel = 1/16 * np.array([1, 4, 6, 4, 1])
bicubic_kernel = 1/16 * np.array([-1, 0, 5, 8, 5, 0, -1])
windowed_sinc_kernel = np.array([0.0, -0.0153, 0.0, 0.2684, 0.4939, 0.2684, 0.0, -0.0153, 0.0])

dir = "C:/Users/scday/Documents/coding/ComputerVision/image-blending/"
apple = dir + "apple.jpg"
orange = dir + "orange.jpg"
mask = dir + "vertical-mask.jpg"

apple_im = Image.open(apple).convert('RGBA')
apple_arr = np.asarray(apple_im)
laplacian_apple_list = np.asarray(create_laplacian_pyramid(apple_arr))


orange_im = Image.open(orange).convert('RGBA')
orange_arr = np.asarray(orange_im)
laplacian_orange_list = np.asarray(create_laplacian_pyramid(orange_arr))


mask_arr = np.array(Image.open(mask).convert('RGBA'))
mask_arr[:, :, 3] = mask_arr[:, :, 0]
width, height, channels = mask_arr.shape
for i in range(3):
    mask_arr[:, :, i] = np.ones((width, height))
print("Mask")
gaussian_mask_list = create_gaussian_pyramid(mask_arr)

inv_mask_arr = np.array(mask_arr)
inv_mask_arr[:, :, 3] = 255 - mask_arr[:, :, 3]
gaussian_inv_mask_list = create_gaussian_pyramid(inv_mask_arr)

mod_gaussian_mask_list = []
mod_gaussian_inv_mask_list = []
for i in range(len(gaussian_mask_list)-2, -1, -1):
    mod_gaussian_mask_list.append(gaussian_mask_list[i])
    mod_gaussian_inv_mask_list.append(gaussian_inv_mask_list[i])

gaussian_mask_list = np.asarray(mod_gaussian_mask_list)
gaussian_inv_mask_list = np.array(mod_gaussian_inv_mask_list)

combined_laplacian_pyramid = np.array([])

if len(laplacian_apple_list) == len(laplacian_orange_list) and len(laplacian_apple_list) == len(gaussian_mask_list):
    combined_laplacian_pyramid = gaussian_mask_list*laplacian_apple_list + gaussian_inv_mask_list*laplacian_orange_list
else:
    print("lists are not of the same length")
    exit()

l = len(combined_laplacian_pyramid)
debug_im = Image.fromarray(laplacian_apple_list[l-2].astype('uint8')).convert('RGB')
debug_im.save(dir + "debug.jpg")

l = len(combined_laplacian_pyramid)
debug_im = Image.fromarray(laplacian_orange_list[l-2].astype('uint8')).convert('RGB')
debug_im.save(dir + "debugo.jpg")

l = len(combined_laplacian_pyramid)
debug_im = Image.fromarray(combined_laplacian_pyramid[l-2].astype('uint8')).convert('RGB')
debug_im.save(dir + "debugc.jpg")

final_arr = reconstruct_from_laplacian_pyramid(combined_laplacian_pyramid)
final_im = Image.fromarray(final_arr.astype('uint8')).convert('RGB')
final_im.save(dir + "output.jpg")
