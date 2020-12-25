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


def load_and_normalise_dicom(path, x, y):
    dicom1 = dicom.read_file(path)
    dicom_img = dicom1.pixel_array
    dicom_img[dicom_img == -2000] = 0
    binary = dicom_img < 604
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
    dicom_img[get_high_value] = 0
    # dicom_img[dicom_img < 604] = 0
    if dicom_img.shape != (x, y):
        dicom_img = cv2.resize(dicom_img, (x, y), interpolation=cv2.INTER_CUBIC)
    return dicom_img


patient = 'd3d08f2ed5742e96f738e7cdd73f8e8d.dcm'
patient_dir = r'E:\Dane\input\sample_images\2a20e4a4e6411f72374fdffebabfc235'
path = os.path.join(patient_dir, patient)
base_fname = r'C:\Users\Maciej\Desktop\Nowy folder (2)'
index = os.listdir(patient_dir).index(patient)
print(index)


def read_ct_scan(folder):
    slices = [dicom.read_file(os.path.join(folder, patient))]
    # slices = [dicom.read_file(os.path.join(folder, file)) for file in os.listdir(folder)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    slices = np.stack([s.pixel_array for s in slices])
    slices[slices == -2000] = 0
    return slices


ct_scan = read_ct_scan(patient_dir)


def plot_ct_scan2(scan):
    x = 4
    y = 5
    fig, plot = plt.subplots(x, y, figsize=(10, 10))
    for i in range(0, x*y):
        print(i)
        plot[i % x, i // x].axis('off')
        plot[i % x, i // x].imshow(scan[i], cmap=plt.cm.bone)
    plt.show()


def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        print(i, int(i / 20), int((i % 20) / 5))
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone)
    plt.show()


def get_segmneted_lungs(img, plot=False):
    binary1 = img < 604

    cleared = clear_border(binary1)

    label_image = label(cleared)

    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary2 = label_image > 0

    selem = disk(2)
    binary3 = binary_erosion(binary2, selem)

    selem = disk(10)
    binary4 = binary_closing(binary3, selem)

    edges = roberts(binary4)
    binary5 = ndimage.binary_fill_holes(edges)

    get_high_value = binary5 == 0
    img[get_high_value] = 0
    # img[img < 604] = 0

    if plot:
        fig1, plots = plt.subplots(8, 1, figsize=(20, 20))
        plots[0].axis('off')
        plots[0].imshow(binary1, cmap=plt.cm.bone)
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone)
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone)
        plots[3].axis('off')
        plots[3].imshow(binary2, cmap=plt.cm.bone)
        plots[4].axis('off')
        plots[4].imshow(binary3, cmap=plt.cm.bone)
        plots[5].axis('off')
        plots[5].imshow(binary4, cmap=plt.cm.bone)
        plots[6].axis('off')
        plots[6].imshow(binary5, cmap=plt.cm.bone)
        plots[7].axis('off')
        plots[7].imshow(img, cmap=plt.cm.bone)
        fname = os.path.join(base_fname, '0')
        # plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w', orientation='portrait')
        plt.show()
        # img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_CUBIC)
        # plt.imshow(img, cmap=plt.cm.gray)
        # plt.show()

        for idx, val in enumerate((binary1, cleared, label_image, binary2, binary3, binary4, edges, binary5, img)):
            # fname = os.path.join(base_fname, idx+2)
            plt.imshow(val, cmap=plt.cm.gray)
            # plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w', orientation='portrait')
            plt.show()

    return img


fname = os.path.join(base_fname, '1')
img = ct_scan[0]
plt.imshow(img, cmap=plt.cm.gray)
# plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w', orientation='portrait')
get_segmneted_lungs(img, plot=True)

#
# def segment_lung_from_ct_scan(ct_scan):
#     return np.asarray([get_segmneted_lungs(slice) for slice in ct_scan])
#
#
# segmented_ct_scan = segment_lung_from_ct_scan(ct_scan)
# plot_ct_scan(segmented_ct_scan)

# segmented_ct_scan[segmented_ct_scan < 604] = 0
# plot_ct_scan(segmented_ct_scan)