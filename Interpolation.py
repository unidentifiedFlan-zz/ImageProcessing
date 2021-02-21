import numpy as np
from PIL import Image


def create_upscaled_base_image(image):
    factor = int(2)

    i_width, i_height = image.shape
    o_width = factor * i_width
    o_height = factor * i_height
    new_img = np.zeros((o_width, o_height))

    for x in range(i_width):
        for y in range(i_height):
            new_img[factor*x][factor*y] = image[x][y]

    return new_img


def create_decimated_image(image):
    factor = int(2)
    i_width, i_height = image.shape
    o_width = int(1/factor * i_width)
    o_height = int(1/factor * i_height)
    new_img = np.zeros((o_width, o_height))

    for x in range(o_width):
        for y in range(o_height):
            new_img[x][y] = image[factor * x][factor * y]

    return new_img


def convolve_image(image, kernel):
    width, height = image.shape

    image_cpy = image.copy()
    for i in range(width):
        cpy_arr = image_cpy[i, :]
        image_cpy[i, :] = np.convolve(cpy_arr, kernel, 'same')

    for j in range(height):
        cpy_arr = image_cpy[:, j]
        image_cpy[:, j] = np.convolve(cpy_arr, kernel, 'same')

    return image_cpy


bilinear_kernel = 0.25*np.array([1, 2, 1])
binomial_kernel = 1/16 * np.array([1, 4, 6, 4, 1])
bicubic_kernel = 1/16 * np.array([-1, 0, 5, 8, 5, 0, -1])
windowed_sinc_kernel = np.array([0.0, -0.0153, 0.0, 0.2684, 0.4939, 0.2684, 0.0, -0.0153, 0.0])

dir = "C:/Users/scday/Documents/coding/ComputerVision/interpolation/"
input_file = dir + "cat.jpg"
greyscale_input = dir + "cat_greyscale.jpg"

im = Image.open(input_file).convert('L')
im.save(greyscale_input)
im_arr = np.asarray(im)

##########################################################################
#############
#############  UPSCALING
#############
##########################################################################

upscaled_arr = create_upscaled_base_image(im_arr)


###
###Using a factor of 4 else the output is too dark
###This is due to averaging via the convolutions over a number of pixels
###of value 0

#bilinear
final_arr = 4*convolve_image(upscaled_arr, bilinear_kernel)
final_im = Image.fromarray(final_arr).convert('L')
final_im.save(dir + "output_bilinear.jpg")

#binomial
final_arr = 4*convolve_image(upscaled_arr, binomial_kernel)
final_im = Image.fromarray(final_arr).convert('L')
final_im.save(dir + "output_binomial.jpg")

#bicubic
final_arr = 4*convolve_image(upscaled_arr, bicubic_kernel)
final_im = Image.fromarray(final_arr).convert('L')
final_im.save(dir + "output_bicubic.jpg")

#windowed sinc
final_arr = 4*convolve_image(upscaled_arr, windowed_sinc_kernel)
final_im = Image.fromarray(final_arr).convert('L')
final_im.save(dir + "output_windowed_sinc.jpg")


########################################################################
####################
####################  DECIMATION
####################
########################################################################

#bilinear
blurred_arr = convolve_image(im_arr, bilinear_kernel)
decimated_arr = create_decimated_image(blurred_arr)
final_im = Image.fromarray(decimated_arr).convert('L')
final_im.save(dir + "output_bilinear_dec.jpg")

#binomial
blurred_arr = convolve_image(im_arr, binomial_kernel)
decimated_arr = create_decimated_image(blurred_arr)
final_im = Image.fromarray(decimated_arr).convert('L')
final_im.save(dir + "output_binomial_dec.jpg")

#bicubic
blurred_arr = convolve_image(im_arr, bicubic_kernel)
decimated_arr = create_decimated_image(blurred_arr)
final_im = Image.fromarray(decimated_arr).convert('L')
final_im.save(dir + "output_bicubic_dec.jpg")

#windowed sinc
blurred_arr = convolve_image(im_arr, windowed_sinc_kernel)
decimated_arr = create_decimated_image(blurred_arr)
final_im = Image.fromarray(decimated_arr).convert('L')
final_im.save(dir + "output_windowed_sinc_dec.jpg")