import os
import cv2
import random
import numpy as np
import pandas as pd
import pydicom as dicom
from scipy import ndimage
import matplotlib.pyplot as plt

# shift = 25
# height, width = img.shape
# print(width, height)
# top = np.random.randint(0, shift)
# bottom = np.random.randint(0, shift)
# left = np.random.randint(0, shift)
# right = np.random.randint(0, shift)
# print(top, bottom, left, right)
# img = cv2.copyMakeBorder(src=img, top=top, bottom=bottom, left=left, right=right, borderType=cv2.BORDER_ISOLATED)

# rescale_factor_range = (0.75, 1)
# if rescale_factor_range:
#     if rescale_factor_range[0] >= rescale_factor_range[1] or rescale_factor_range[0] > 1 or rescale_factor_range[1] > 1:
#         raise TypeError('inappropriate rescale factor shape')
#     rescale_factor = np.random.random_sample()*(rescale_factor_range[1]-rescale_factor_range[0])+rescale_factor_range[0]
#     height, width = img.shape
#     img = cv2.resize(img, (round(height*rescale_factor), round(width*rescale_factor)), interpolation=cv2.INTER_CUBIC)


# def augment(image, rescale_factor_range=(0.8, 1), crop_factor_range=(0.8, 1), rotation_angle_range=(-20, 20), shift=25,
#             color_inverse=True, flip=True, visualize=False):
#     height, width = image.shape
#     # rescale_factor_range = (-1, 1)
#     if rescale_factor_range:
#         if rescale_factor_range[0] > rescale_factor_range[1] or rescale_factor_range[0] < 0 or rescale_factor_range[1]\
#                 < 0:
#             raise TypeError('inappropriate rescale factor shape')
#         rescale_factor = np.random.random_sample()*(rescale_factor_range[1]-rescale_factor_range[0]) +\
#                          rescale_factor_range[0]
#         new_height = round(height * rescale_factor)
#         new_width = round(height * rescale_factor)
#         if rescale_factor < 1.0:
#             img = np.zeros_like(image)
#             row = (height - new_height) // 2
#             col = (width - new_width) // 2
#             img[row:row+new_height, col:col+new_width] = ndimage.zoom(image, (float(rescale_factor),
#                                                                               float(rescale_factor)),
#                                                                       mode='nearest')[0:new_height, 0:new_width]
#         elif rescale_factor > 1.0:
#             row = (new_height - height) // 2
#             col = (new_width - width) // 2
#             img = ndimage.zoom(image[row:row+new_height, col:col+new_width], (float(rescale_factor),
#                                                                               float(rescale_factor)), mode='nearest')
#             extra_hight = (img.shape[0] - height) // 2
#             extra_width = (img.shape[1] - width) // 2
#             img = img[extra_hight:extra_hight+height, extra_width:extra_width+width]
#
#         else:
#             img = image
#     else:
#         img = image
#
#     # rotation_angle_range = (-20, 20)
#     if rotation_angle_range:
#         if rotation_angle_range[0] >= rotation_angle_range[1]:
#             raise TypeError('inappropriate rotation angle factor shape')
#         angel = np.random.random_sample()*(rotation_angle_range[1]-rotation_angle_range[0])+rotation_angle_range[0]
#         img = ndimage.rotate(img, angel, reshape=False)
#
#     # crop_factor_range = (0.8, 1)
#     if crop_factor_range:
#         if crop_factor_range[0] >= crop_factor_range[1] or crop_factor_range[0] < 0 or crop_factor_range[1] < 0:
#             raise TypeError('inappropriate crop factor shape')
#         crop_factor = np.random.random_sample()*(crop_factor_range[1]-crop_factor_range[0])+crop_factor_range[0]
#         crop_size = (round(crop_factor*height), round(crop_factor*width))
#         height, width = img.shape
#         left = (width - crop_size[0]) // 2
#         right = left + crop_size[0]
#         bottom = (height - crop_size[1]) // 2
#         top = bottom + crop_size[1]
#         if left < 0:
#             left = 0
#         if bottom < 0:
#             bottom = 0
#         img = img[left:right, bottom:top]
#
#     # shift = 25
#     if shift:
#         offset = np.array([[np.random.randint(-shift, shift)], [np.random.randint(-shift, shift)]])
#         img = scipy.ndimage.interpolation.shift(img, (int(offset[0]), int(offset[1])), mode='nearest')
#
#     # color_inverse = True
#     if color_inverse:
#         color_inverse_factor = np.random.randint(-1, 2)
#         while color_inverse_factor == 0:
#             color_inverse_factor = np.random.randint(-1, 2)
#         img = img*color_inverse_factor
#
#     # flip = True
#     if flip:
#         flip_factor = np.random.randint(0, 2)
#         if flip_factor:
#             img = np.fliplr(img)
#         else:
#             img = np.flipud(img)
#
#     print('rescaled by', rescale_factor)
#     print('rotated by', angel)
#     print('shifted by\n', offset)
#     print('color inverse', color_inverse_factor)
#     print('flip', flip_factor)
#     print(img.shape)
#
#     if visualize:
#         plt.imshow(img, cmap='gray')
#         plt.show()
#
#
#     return img


# fig, plot = plt.subplots(1, 2, figsize=(15, 15))
# plot[0].imshow(augment(image, rescale_factor_range=(), crop_factor_range=(0.8, 1.1), rotation_angle_range=(-30, 30),
#                        shift=50, color_inverse=False, flip=True, final_shape=(256, 256)), cmap='gray')
# plot[1].imshow(augment(image, rescale_factor_range=(0.8, 1.1), crop_factor_range=(), rotation_angle_range=(-30, 30),
#                        shift=50, color_inverse=False, flip=True, final_shape=()), cmap='gray')
# plt.show()

def augment(image, rescale_factor_range=(0.8, 1), rotation_angle_range=(-20, 20), shift=25,
            color_inverse=True, flip=True):
    height, width = image.shape
    if rescale_factor_range:
        if rescale_factor_range[0] > rescale_factor_range[1] or rescale_factor_range[0] < 0 or rescale_factor_range[1]\
                < 0:
            raise TypeError('inappropriate rescale factor shape')
        rescale_factor = np.random.random_sample()*(rescale_factor_range[1]-rescale_factor_range[0]) +\
                         rescale_factor_range[0]
        new_height = round(height * rescale_factor)
        new_width = round(height * rescale_factor)
        if rescale_factor < 1.0:
            img = np.zeros_like(image)
            row = (height - new_height) // 2
            col = (width - new_width) // 2
            img[row:row+new_height, col:col+new_width] = ndimage.zoom(image, (float(rescale_factor),
                                                                              float(rescale_factor)),
                                                                      mode='nearest')[0:new_height, 0:new_width]
        elif rescale_factor > 1.0:
            row = (new_height - height) // 2
            col = (new_width - width) // 2
            img = ndimage.zoom(image[row:row+new_height, col:col+new_width], (float(rescale_factor),
                                                                              float(rescale_factor)), mode='nearest')
            extra_hight = (img.shape[0] - height) // 2
            extra_width = (img.shape[1] - width) // 2
            img = img[extra_hight:extra_hight+height, extra_width:extra_width+width]
        else:
            img = image
    else:
        img = image

    if rotation_angle_range:
        if rotation_angle_range[0] >= rotation_angle_range[1]:
            raise TypeError('inappropriate rotation angle factor shape')
        angel = np.random.random_sample()*(rotation_angle_range[1]-rotation_angle_range[0])+rotation_angle_range[0]
        img = ndimage.rotate(img, angel, reshape=False)

    if shift:
        offset = np.array([[np.random.randint(-shift, shift)], [np.random.randint(-shift, shift)]])
        img = ndimage.interpolation.shift(img, (int(offset[0]), int(offset[1])), mode='nearest')

    if color_inverse:
        color_inverse_factor = np.random.randint(-1, 2)
        while color_inverse_factor == 0:
            color_inverse_factor = np.random.randint(-1, 2)
        img = img*color_inverse_factor

    if flip:
        flip_factor = np.random.randint(0, 2)
        if flip_factor:
            img = np.fliplr(img)
        else:
            img = np.flipud(img)

    return img

base_dir = r'E:\Dane\input\sample_images'
pat_dir = r'E:\Dane\input\sample_images\{}'.format(random.choice(os.listdir(base_dir)))
image_dir = os.path.join(pat_dir, random.choice(os.listdir(pat_dir)))
# image_dir = r'E:\Dane\input\sample_images\01f1140c8e951e2a921b61c9a7e782c2\951f21b3a4d868840e36722828d9459b.dcm'
dicom = dicom.read_file(image_dir)
image = dicom.pixel_array.astype(np.float64)
min = image.min()
max = image.max()
image[image < min] = min
image[image > max] = max
image = (image - min) / (max - min)
x = 256
y = 256
image = cv2.resize(image, (x, y), interpolation=cv2.INTER_CUBIC)

i = 0
x = 4
y = 4
fig, plot = plt.subplots(x, y, figsize=(15, 15))
while True:
    scan = augment(image, rescale_factor_range=(0.8, 1.2), rotation_angle_range=(-30, 30), shift=20, color_inverse=True,
                   flip=True)
    plot[i // y, i % y].axis('off')
    plot[i // y, i % y].imshow(scan, cmap='gray')
    i += 1
    if i == x*y:
        break

fname = r'F:\Nowy folder\7\Praca inżynierska\Zdjęcia\Nowy folder\Figure_13.jpg'
plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w', orientation='portrait')
plt.show()

