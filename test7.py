import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pydicom as dicom
from skimage import measure, morphology

base_dir = r'E:\Test2\input\sample_images\0030a160d58723ff36d73f41b170ec21\f3391a9a164c01a9e2b4ae05e9918903.dcm'
scan = dicom.read_file(base_dir)
image = scan.pixel_array
new_image = morphology.dilation(image)

fig, ax = plt.subplots(1, 2, figsize=[10, 10])
ax[0].imshow(image, cmap='gray')
ax[1].imshow(new_image, cmap='gray')
plt.show()

base_dir = r'E:\Test2\input\sample_images'
patient = r'E:\Test2\input\sample_images\0030a160d58723ff36d73f41b170ec21'
path = os.path.join(base_dir, patient)

# def load_scan(path):
slices = [dicom.read_file(os.path.join(path, s)) for s in os.listdir(path)]
slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
try:
    slice_thickness = np.abs(slices[0].ImagePositionPatient[2]-slices[1].ImagePositionPatient[2])
except:
    slice_thickness = np.abs(slices[0].SliceLocation - slices[1].ImagePositioPatient)

for s in slices:
    s.SliceThickness = slice_thickness
print(len(slices))
# return slices


# def get_pixel_hu(slices):
image = np.stack([s.pixel_array for s in slices])
image = image.astype(np.int16)
image[image == -2000] = 0
print(image.shape)
plt.imshow(image[0], cmap='gray')
plt.show()

for slice_number in range(len(slices)):
    intercept = slices[slice_number].RescaleIntercept
    slope = slices[slice_number].RescaleSlope

    if slope != 1:
        image[slice_number] = slope * image[slice_number].astype(np.floate64)
        image[slice_number] = image[slice_number].astype(np.int16)

    image[slice_number] += np.int16(intercept)

arr = np.array(image, dtype=np.int16)

# return np.array(image, dtype=np.int16)
#
# def largest_label_volume(img, bg=-1):
#     vals, counts = np.unique(img, return_counts=True)
#     counts = counts[vals != bg]
#     vals = vals[vals != bg]
#
#     if len(counts) > 0:
#         return vals[np.argmax(counts)]
#     else:
#         return None
#
#
# def segment_lung_mask(image, fill_lung_structures=True):
binary_image = np.array(image > -320, dtype=np.int8) + 1
binary_image = morphology.erosion(morphology.dilation(binary_image))
labels = measure.label(binary_image)
print(labels)
background_labels = [0, 0, 0]
binary_image[background_labels == labels] = 2
fill_lung_structures = True
if fill_lung_structures:
    for i, axial_slice in enumerate(binary_image):
        print(i, axial_slice)
        # axial_slice = axial_slice-1
        # labeling = measure.label(axial_slice)
        # print(labeling)
#             l_max = largest_label_volume(labeling, bg=0)
#             if l_max is not None:
#                 binary_image[i][labeling != l_max] = 1
#
#     binary_image -= 1
#     binary_image = 1 - binary_image
#
#     labels = measure.label(binary_image, background=0)
#     l_max = largest_label_volume(labels, bg=0)
#     if l_max is not None:
#         binary_image[labels != l_max] = 0
#
#     return binary_image
#
#
# input_folder = r'E:\Test2\input\sample_images'
# patient_list = os.listdir(input_folder)[:2]
# path = r'E:\Test2\input\sample_images\0030a160d58723ff36d73f41b170ec21'
#
# ct_list = []
# lung_mask_list = []
#
# for patient in patient_list:
#     ct_pixels = load_scan(path)
#     ct_hu_pixels = get_pixel_hu(ct_pixels)
#
#     lung_mask = segment_lung_mask(ct_hu_pixels, True)
#
#     ct_list.append(ct_hu_pixels)
#     lung_mask_list.append(lung_mask)
#
# for i in range(len(patient_list)):
#     print('patient: {}'.format(patient_list[i]))
#     fig, ax = plt.subplots(1, 2, figsize=[10, 10])
#     ax[0].imshow(ct_list[i], cmap='gray')
#     ax[1].imshow(ct_list[i] * lung_mask_list[i][100], cmap='gray')
#     plt.show()

x = np.eye(3).astype(int)*0.7
labels = measure.label(x)
print(labels)


