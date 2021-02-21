from PIL import Image
from os import listdir, mkdir, path
import fnmatch
import numpy as np

std_dev_threshold = 1.5


def classify_background(image, mean_tensor, variation_tensor):
    deviation = np.absolute(image - mean_tensor)
    alphas = np.zeros(image.shape)
    alphas = np.where(deviation > np.sqrt(variation_tensor)*std_dev_threshold, 255, alphas)

    final_alpha = alphas.max(axis=2)
    image[:, :, 3] = final_alpha

    return image


root_dir = "C:/Users/scday/Documents/coding/ComputerVision/Wallflower-test-data/"
test_name = "WavingTrees"
test_dir = root_dir + test_name + "/"
output_dir = root_dir + test_name + "-output/"

if not path.exists(output_dir):
    mkdir(output_dir)

video = []

for file in listdir(test_dir):
    if fnmatch.fnmatch(file, '*.bmp'):
        video.append(file)


#Assuming images are consistently sized

im = Image.open(test_dir + video[0])
width, height = im.size
means = np.zeros((height, width, 4))
variation = np.zeros((height, width, 4))

for img in video:
    im = Image.open(test_dir + img).convert('RGBA')
    img_arr = np.array(im)

    means = means + (img_arr/len(video))

for img in video:
    im = Image.open(test_dir + img).convert('RGBA')
    img_arr = np.array(im)
    variation = variation + (np.square(img_arr - means)/(len(video) - 1))

for img in video:
    im = Image.open(test_dir + img).convert('RGBA')
    img_arr = np.array(im)

    foreground_arr = classify_background(img_arr, means, variation)
    foreground_img = Image.fromarray(foreground_arr)

    background_img = Image.new('RGBA', foreground_img.size, color='blue')
    background_img.paste(foreground_img, (0, 0), foreground_img)
    background_img.save(output_dir + img)

#for img in newVideo:
#    display_image(img)


