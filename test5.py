import os
import cv2
import math
import numpy as np
import pandas as pd
import pydicom as dicom
import matplotlib.pyplot as plt

df = pd.read_csv(r'E:\Test\new_stage2\new_stage2_labels.csv')
cancer = df.loc[df['cancer'] == 1]
data_dir = r'E:\Dane\input\sample_images'
patient = df.iloc[np.random.randint(0, len(df)-1)].id


def process_data(patient, desired_depth=25, desired_width=128, desired_height=128, visualise=False):
    path = os.path.join(data_dir, patient)
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices = sorted(slices, key=lambda x: int(x.ImagePositionPatient[2]))
    pattern = np.zeros((512, 512, len(slices)))
    print(len(slices))
    current_depth = pattern.shape[-1]
    depth = current_depth / desired_depth
    depth = math.floor(depth)
    start = math.floor((current_depth - (depth * desired_depth))/2)
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
    if visualise:
        x = round(math.sqrt(desired_depth))
        y = round(desired_depth/x)
        print(x, y)

        if x < y:
            cor1 = x
            cor2 = y
        else:
            cor1 = y
            cor2 = x
        fig, plot = plt.subplots(10, 10, figsize=(15, 15))
        for idx, scan in enumerate(img_list):
            plot[idx // cor2, idx % cor2].axis('off')
            plot[idx // cor2, idx % cor2].imshow(scan, cmap='gray')
        plt.show()
    return img


pat_dir = r'E:\Dane\input\sample_images'
for idx, patient in enumerate(os.listdir(pat_dir)):
    path = os.path.join(pat_dir, patient)
    length = len(os.listdir(path))
    if length <= 94:
        print(path, length)
        # test = process_data(patient=path, desired_depth=94, desired_width=128, desired_height=128, visualise=True)
        # break

df = pd.read_csv(r'E:\Test\new_stage2\new_stage2_labels.csv')
patient = df.iloc[np.random.randint(0, len(df)-1)].id
test = process_data(patient=patient, desired_depth=94, desired_width=128, desired_height=128, visualise=True)


# def get_slice_location(dcm):
#     # [0x0020, 0x1041] – Slice Location
#     return float(dcm[0x0020, 0x1041].value)
#
#
# def load_patient(patient_id):
#     # patient_id = np.random.choice(files)
#     files = glob.glob(r'E:\Dane\input\sample_images\{}\*.dcm'.format(patient_id))
#     imgs = {}
#     for file in files:
#         dcm = dicom.read_file(file)
#         img = dcm.pixel_array
#         img[img == -2000] = 0
#         sl = get_slice_location(dcm)
#         imgs[sl] = img
#         sorted_img = [x[1] for x in sorted(imgs.items(), key=lambda x: x[0])]
#     return sorted_img
#
#
# sample_dir = r'E:\Dane\input\sample_images'
# files2 = os.listdir(sample_dir)
# for d in os.listdir(sample_dir):
#     patient_id = os.path.basename(os.path.join(sample_dir, d))
#     x = len(os.listdir(os.path.join(sample_dir, d)))
#     print("Patient '{}' has {} scans".format(d, len(os.listdir(os.path.join(sample_dir, d)))))
#     print(os.path.join(sample_dir, d))
#     pat = load_patient(patient_id)
#     fig, plot = plt.subplots(math.ceil(len(pat)/10), 10,  figsize=(10, 16))
#     for i in range(len(pat)):
#         plot[i // 10, i % 10].axis('off')
#         plot[i // 10, i % 10].imshow(pat[i], cmap='gray')
#     fname = r'C:\Users\Maciej\Desktop\Nowy folder (2)\Figure_6'
#     # plt.show()