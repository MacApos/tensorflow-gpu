import pydicom as dicom
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np


base_dir = r'E:\Dane\input\sample_images'


def load_and_normalise_dicom(path):
    dicom1 = dicom.read_file(path)
    dicom_img = dicom1.pixel_array.astype(np.float64)
    plt.subplot(211)
    plt.imshow(dicom_img, cmap='gray')
    plt.axis('off')
    dicom_img[dicom_img == -2000] = 0
    mn = dicom_img.min()
    mx = dicom_img.max()
    if (mx - mn) != 0:
        dicom_img = (dicom_img - mn) / (mx - mn)
        print(mn, mx, (mx - mn), '\n', dicom_img)
    else:
        dicom_img[:, :] = 0
    plt.subplot(212)
    plt.imshow(dicom_img, cmap='gray')
    plt.axis('off')
    plt.show()
    return dicom_img


while True:
    patient = os.path.join(base_dir, np.random.choice(os.listdir(base_dir)))
    path = os.path.join(patient, np.random.choice(os.listdir(patient)))
    scan = load_and_normalise_dicom(path)
    plt.imshow(scan, cmap='gray')
    plt.axis('off')
    plt.show()
