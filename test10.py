import os
import cv2
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.filters import roberts
from skimage.segmentation import clear_border
from skimage.measure import regionprops, label
from skimage.morphology import disk, binary_erosion, binary_closing

base_dir = r'E:\Dane\input\sample_images'
fname = r'F:\Nowy folder\7\Praca inżynierska\Zdjęcia\32'
# patient = '1acbe17dc8f9f59d2fd167b2aa6c650f'


def load_and_normalise_dicom(img):
    binary = img < 604
    cleared = clear_border(binary)
    label_image = label(cleared)
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    selem = disk(10)
    binary = binary_closing(binary, selem)
    edges = roberts(binary)
    binary = ndimage.binary_fill_holes(edges)
    get_high_value = binary == 0
    img[get_high_value] = 0
    # plt.imshow(img)
    # plt.show()
    # dicom_img[dicom_img < 604] = 0
    # if img.shape != (x, y):
    #     dicom_img = cv2.resize(img, (x, y), interpolation=cv2.INTER_CUBIC)
    return img


for patient in os.listdir(base_dir):
    path = base_dir + '/' + patient
    if len(os.listdir(path)) == 150:
        print(path)
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices = sorted(slices, key=lambda x: int(x.ImagePositionPatient[2]))
        fig, plot = plt.subplots(10, 15, figsize=(10, 16))
        for idx, slice in enumerate(slices):
            scan = slice.pixel_array
            plot[idx // 15, idx % 15].axis('off')
            plot[idx // 15, idx % 15].imshow(scan, cmap='gray')
        # plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w', orientation='portrait')
        plt.show()
        break


