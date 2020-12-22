import cv2
import os
import math
import glob
import random
import numpy as np
import pandas as pd
import pydicom as dicom
import matplotlib.pyplot as plt
from scipy import ndimage
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

train_valid_fraction = 0.7
train_csv_table = pd.read_csv(r'E:\Test\input\stage1_labels_test.csv')
get_dir = 'Test'
if get_dir == 'Dane':
    train_csv_table = pd.read_csv(r'E:\Dane\input\stage1_labels.csv')

path = r'E:\Test\input\sample_images\00cba091fa4ad62cc3200a657aeb957e\251a10e3af0aa42f1e77d4cf84a0f39d.dcm'


# def load_data(path):
#     dicom1 = dicom.read_file(path)
#     dicom_img = dicom1.pixel_array.astype(np.float64)
#     mn = dicom_img.min()
#     mx = dicom_img.max()
#     # if (mn - mx) != 0:
#     #     dicom_img = (dicom_img - mn)/(mx - mn)
#     # else:
#     #     dicom_img[:, :] = 0
#     dicom_img[dicom_img == -2000] = 0
#     if dicom_img.shape != (x, y):
#         dicom_img = cv2.resize(dicom_img, (x, y), interpolation=cv2.INTER_CUBIC)
#     plt.imshow(dicom_img, cmap='gray')
#     plt.show()
#     return dicom_img

# load_data(path)

data_dir = r'E:\Test\input\sample_images'
patients = os.listdir(data_dir)

""" Wersja 1 """
"""
for patient in patients[:1]:
    path = data_dir + '/' + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices = sorted(slices, key=lambda x: int(x.ImagePositionPatient[2]))
    img = np.zeros((512, 512, len(slices)))
    for idx, slice in enumerate(slices):
        arr = slice.pixel_array
        img[:, :, idx] += arr
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    print(img.shape)
    # fig, plot = plt.subplots(8, 8, figsize=(6, 6))
    # for i in range(img.shape[-1]):
    #     print(i)
    #     plot[idx // 8, idx % 8].axis('off')
    #     plot[idx // 8, idx % 8].imshow()
    # plt.show()
"""

""" Wersja 2 """
# """
desired_depth = 64
desired_width = 128
desired_height = 128
visualise = True


# def process_data(patient, desired_depth=64, desired_width=128, desired_height=128, visualise=False):
for patient in patients[:2]:
    path = data_dir + '/' + patient
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices = sorted(slices, key=lambda x: int(x.ImagePositionPatient[2]))
    pattern = np.zeros((512, 512, len(slices)))
    current_depth = pattern.shape[-1]
    print(current_depth)
    depth = current_depth / desired_depth
    depth = math.floor(depth)
    print(depth)
    start = math.floor((current_depth - (depth * desired_depth))/2)
    print(start)
    img = np.zeros((desired_width, desired_height, desired_depth))
    img_list = []
    for idx, slice in enumerate(slices[start::depth]):
        arr = slice.pixel_array
        arr[arr == -2000] = 0
        if idx <= desired_depth-1:
            if arr.shape != (desired_width, desired_height):
                arr = cv2.resize(arr, (desired_width, desired_width), interpolation=cv2.INTER_CUBIC)
            img[:, :, idx] += arr
            img_list.append(arr)
    # if visualise:
    #     fig, plot = plt.subplots(5, 6, figsize=(15, 15))
    #     for idx, scan in enumerate(img_list):
    #         plot[idx // 6, idx % 6].axis('off')
    #         plot[idx // 6, idx % 6].imshow(scan, cmap='gray')
    #     plt.show()
    # print(img.shape)
#     return img
# # """
#
#
# ids = train_csv_table['id'].values
# split_point = int(round(train_valid_fraction * len(ids)))
# train_data = patients[:split_point]
# valid_data = patients[split_point:]
#
# test = [process_data(patient, desired_depth=20, visualise=True) for patient in patients[5:6]]
#
# x_train = np.array([process_data(patient) for patient in train_data])
# print(x_train.shape)
# x_val = np.array([process_data(patient) for patient in valid_data])
#
# y_train = np.array([train_csv_table.loc[train_csv_table['id'] == patient]['cancer'].values[0] for patient in train_data])
# y_val = np.array([train_csv_table.loc[train_csv_table['id'] == patient]['cancer'].values[0] for patient in valid_data])
#
# print(x_train.shape)
# print(x_val.shape)
# print(y_train)
# print(y_val)
#
# volume = np.arange(0, 1600).reshape(40, 40)
#
#
# def scipy_rotate(volume):
#     angels = [-20, -10, -5, 5, 10, 20]
#     angle = random.choice(angels)
#     plt.imshow(volume)
#     plt.show()
#     volume = ndimage.rotate(volume, angle, reshape=False)
#     # volume[volume < 0] = 0
#     # volume[volume > 1] = 1
#     plt.imshow(volume)
#     plt.show()
#     return volume
#
#
# scipy_rotate(volume)

# augmneted_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)