"""
Dane zostały pobrane i rozpakowane do folderu "Dane". Nazwa folderu z danymi treningowymi "stag1" została zmieniona na
"input". Następnie podfoldery w tym folderze, opisane jako id pacjentów, zostały umieszczone w nowym folderze
"sample_data", a plik z etykietami "stage1_labels.csv" w folderze "input".
Ostateczenie w folderze z danymi "Dane", zanjdował się folder z danymi testowymi "stag2" oraz folder z danymi
treningowymi "input", który zawierał plik z etykietami "stage1_labels.csv" oraz podfolder "sample_data" ze skanami pa-
cjentów.
https://www.kaggle.com/zfturbo/keras-vs-cancer
"""

import os
import cv2
import glob
import math
import time
import random
import shutil
import numpy as np
import pandas as pd
import pydicom as dicom
from scipy import ndimage
from skimage.filters import roberts
from skimage.segmentation import clear_border
from skimage.measure import regionprops, label
from skimage.morphology import disk, binary_erosion, binary_closing
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
np.random.seed(2016)

print('Tensorflow: ', tf.__version__)
print('Keras: ', keras.__version__)
print('Numpy: ', np.__version__)

# Konfiguracja modelu:
conf = dict()

# Domyślnie ten parametr powinien być równy 1. Ustawiony na 0 pozwala zdefiniować inny zbiór danych.
conf['use_sample_only'] = 1

# Pozwala zapisać wagi modelu
conf['save_model'] = 1

# Granica podziału danych na zbiór testowy i walidacyjny.
conf['train_valid_fraction'] = 0.7

# Rozmiar wsadu
conf['batch_size'] = 128
print('batch_size = ', conf['batch_size'])

# Liczba epok
conf['nb_epochs'] = 11
print('nb_epochs = ', conf['nb_epochs'])

# Liczba epok bez poprawy wyniku, po których trenowanie się zakończy.
conf['patience'] = 3

# Rozdzielczość skanów
conf['image_shape'] = (160, 160)
print('image_shape = ', conf['image_shape'])

# Liczba warstw
# conf['desired_depth'] = 64
# print('desired_depth = ', conf['desired_depth'])

# Wskaźnik uczenia
conf['learning_rate'] = 1e-3
print('learning_rate = ', conf['learning_rate'])

# Liczba próbek treningowych na epokę
conf['samples_train_per_epoch'] = 20480
print('samples_train_per_epoch = ', conf['samples_train_per_epoch'])

# Liczba próbek walidacyjnych na epokę
conf['samples_valid_per_epoch'] = 14336
print('samples_valid_per_epoch = ', conf['samples_valid_per_epoch'])

# Parametry warstw
conf['level_1_filters'] = 4
conf['level_2_filters'] = 8
conf['dense_layer_size'] = 128
conf['dropout_value'] = 0.5

data_dir = 'Dane'

"""
Do utworzenia zboru testowego można użyć poniższej funkcji.
"""


def create_test_data(train_csv_table, data_size=20, move_back=False):
    new_stage2 = r'E:\{}\new_stage2'.format(data_dir)
    sample_images = r'E:\{}\new_stage2\sample_images'.format(data_dir)
    input_labels = r'E:\{}\input\stage1_labels_train.csv'.format(data_dir)
    new_stage_labels = r'E:\{}\new_stage2\new_stage2_labels.csv'.format(data_dir)
    new_train_csv_table = train_csv_table.head(len(train_csv_table.index)-data_size)
    valid_csv_table = train_csv_table.tail(data_size)

    for folder in (new_stage2, sample_images):
        if not os.path.exists(folder):
            os.mkdir(folder)

    new_train_csv_table.to_csv(input_labels, index=False)
    valid_csv_table.to_csv(new_stage_labels, index=False)

    for patient in valid_csv_table.id:
        if not os.path.exists(os.path.join(sample_images, patient)):
            src = r'E:\{}\input\sample_images\{}'.format(data_dir, patient)
            dst = sample_images
            shutil.move(src, dst)

    if move_back:
        for patient in os.listdir(sample_images):
            src = os.path.join(sample_images, patient)
            dst = r'E:\{}}\input\sample_images'.format(data_dir)
            shutil.move(src, dst)

    print(len(os.listdir(r'E:\{}\input\sample_images'.format(data_dir))))
    print(len(os.listdir(sample_images)))


train_csv_table = pd.read_csv(r'E:\{}\input\stage1_labels.csv'.format(data_dir))
create_test_data(train_csv_table, 100, move_back=False)

train_csv_table = pd.read_csv(r'E:\{}\input\stage1_labels_train.csv'.format(data_dir))


def load_and_normalise_dicom(path, x, y):
    dicom1 = dicom.read_file(path)
    dicom_img = dicom1.pixel_array.astype(np.float64)
    dicom_img[dicom_img == -2000] = 0
    mn = dicom_img.min()
    mx = dicom_img.max()
    if (mn - mx) != 0:
        dicom_img = (dicom_img - mn)/(mx - mn)
    else:
        dicom_img[:, :] = 0
    if dicom_img.shape != (x, y):
        dicom_img = cv2.resize(dicom_img, (x, y), interpolation=cv2.INTER_CUBIC)
    return dicom_img


# def load_and_normalise_dicom(path, x, y):
#     dicom1 = dicom.read_file(path)
#     dicom_img = dicom1.pixel_array
#     dicom_img[dicom_img == -2000] = 0
#     binary = dicom_img < 604
#     cleared = clear_border(binary)
#     label_image = label(cleared)
#     areas = [r.area for r in regionprops(label_image)]
#     areas.sort()
#     if len(areas) > 2:
#         for region in regionprops(label_image):
#             if region.area < areas[-2]:
#                 for coordinates in region.coords:
#                     label_image[coordinates[0], coordinates[1]] = 0
#     binary = label_image > 0
#     selem = disk(2)
#     binary = binary_erosion(binary, selem)
#     selem = disk(10)
#     binary = binary_closing(binary, selem)
#     edges = roberts(binary)
#     binary = ndimage.binary_fill_holes(edges)
#     get_high_value = binary == 0
#     dicom_img[get_high_value] = 0
#     # dicom_img[dicom_img < 604] = 0
#     if dicom_img.shape != (x, y):
#         dicom_img = cv2.resize(dicom_img, (x, y), interpolation=cv2.INTER_CUBIC)
#     return dicom_img


def get_train_single_fold(train_data, fraction):
    ids = train_data['id'].values
    # random.shuffle(ids)
    split_point = int(round(fraction*len(ids)))
    train_list = ids[:split_point]
    valid_list = ids[split_point:]
    return train_list, valid_list


def augment(image, rescale_factor_range=(0.8, 1), rotation_angle_range=(-20, 20), shift=25, color_inverse=True,
            flip=True):
    height, width = image.shape
    if rescale_factor_range:
        if rescale_factor_range[0] > rescale_factor_range[1] or rescale_factor_range[0] < 0 or rescale_factor_range[1] \
                < 0:
            raise TypeError('inappropriate rescale factor shape')
        rescale_factor = np.random.random_sample() * (rescale_factor_range[1] - rescale_factor_range[0]) + \
                         rescale_factor_range[0]
        new_height = round(height * rescale_factor)
        new_width = round(height * rescale_factor)
        if rescale_factor < 1.0:
            img = np.zeros_like(image)
            row = (height - new_height) // 2
            col = (width - new_width) // 2
            img[row:row + new_height, col:col + new_width] = ndimage.zoom(image, (float(rescale_factor),
                                                                                  float(rescale_factor)),
                                                                          mode='nearest')[0:new_height, 0:new_width]
        elif rescale_factor > 1.0:
            row = (new_height - height) // 2
            col = (new_width - width) // 2
            img = ndimage.zoom(image[row:row + new_height, col:col + new_width], (float(rescale_factor),
                                                                                  float(rescale_factor)),
                               mode='nearest')
            extra_hight = (img.shape[0] - height) // 2
            extra_width = (img.shape[1] - width) // 2
            img = img[extra_hight:extra_hight + height, extra_width:extra_width + width]
        else:
            img = image
    else:
        img = image

    if rotation_angle_range:
        if rotation_angle_range[0] >= rotation_angle_range[1]:
            raise TypeError('inappropriate rotation angle factor shape')
        angel = np.random.random_sample() * (rotation_angle_range[1] - rotation_angle_range[0]) + rotation_angle_range[
            0]
        img = ndimage.rotate(img, angel, reshape=False)

    if shift:
        offset = np.array([[np.random.randint(-shift, shift)], [np.random.randint(-shift, shift)]])
        img = ndimage.interpolation.shift(img, (int(offset[0]), int(offset[1])), mode='nearest')

    if color_inverse:
        color_inverse_factor = np.random.randint(-1, 2)
        while color_inverse_factor == 0:
            color_inverse_factor = np.random.randint(-1, 2)
        img = img * color_inverse_factor

    if flip:
        flip_factor = np.random.randint(0, 2)
        if flip_factor:
            img = np.fliplr(img)
        else:
            img = np.flipud(img)

    return img


# """
train_patients, valid_patients = get_train_single_fold(train_csv_table, conf['train_valid_fraction'])
desired_depth = 30
train_files = []
for patient in train_patients:
    slices_path = glob.glob(r'E:\{}\input\sample_images\{}\*dcm'.format(data_dir, patient))
    current_depth = len(slices_path)
    depth = current_depth / desired_depth
    depth = math.floor(depth)
    start = math.floor((current_depth - (depth * desired_depth))/2)
    for idx, slice in enumerate(slices_path[start::depth]):
        if idx <= desired_depth - 1:
            train_files.append(slice)
files = train_files

valid_files = []
for patient in valid_patients:
    slices_path = glob.glob(r'E:\{}\input\sample_images\{}\*dcm'.format(data_dir, patient))
    current_depth = len(slices_path)
    depth = current_depth / desired_depth
    depth = math.floor(depth)
    start = math.floor((current_depth - (depth * desired_depth))/2)
    for idx, slice in enumerate(slices_path[start::depth]):
        if idx <= desired_depth - 1:
            valid_files.append(slice)

print(len(train_files))
print(len(valid_files))

batch_size = conf['batch_size']

do_aug = False
# """


def batch_generator_train(files, train_csv_table, batch_size, do_aug=True):
    number_of_batches = np.ceil(len(files)/batch_size)
    counter = 0
    random.shuffle(files)
    while True:
        batch_files = files[batch_size*counter:batch_size*(counter+1)]
        image_list = []
        mask_list = []
        for f in batch_files:
            image = load_and_normalise_dicom(f, conf["image_shape"][0], conf["image_shape"][0])
            if do_aug:
                image = augment(image, rescale_factor_range=(0.8, 1), rotation_angle_range=(-20, 20), shift=25,
                                color_inverse=True, flip=True)
            # print('Normalising...')
            patient_id = os.path.basename(os.path.dirname(f))
            is_cancer = train_csv_table.loc[train_csv_table['id'] == patient_id]['cancer'].values[0]
            if is_cancer == 0:
                mask = [0]
            else:
                mask = [1]
            image_list.append(image)
            mask_list.append(mask)
        counter += 1
        image_list = np.array(image_list)
        mask_list = np.array(mask_list)
        image_list = np.expand_dims(image_list, axis=3)
        yield image_list, mask_list
        if counter == number_of_batches:
            random.shuffle(files)
            counter = 0


# test = batch_generator_train(train_files, train_csv_table, batch_size, do_aug=True)
# print(test)

#
#
def CNN():
    model = Sequential()
    model.add(layers.Conv2D(filters=conf['level_1_filters'], kernel_size=(3, 3), activation='relu',
                            input_shape=(conf["image_shape"][0], conf["image_shape"][0], 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=conf['level_2_filters'], kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=conf['level_2_filters'], kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=conf['level_2_filters'], kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Conv2D(filters=conf['level_2_filters'], kernel_size=(3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(layers.Conv2D(filters=conf['level_2_filters'], kernel_size=(3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(conf['dense_layer_size'], activation='relu'))
    model.add(layers.Dropout(conf['dropout_value']))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=optimizers.RMSprop(lr=conf['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.SensitivityAtSpecificity(0.5)])

    return model


# desired_depth = conf['desired_depth']


def create_model_and_plots():
    train_patients, valid_patients = get_train_single_fold(train_csv_table, conf['train_valid_fraction'])
    print('Train patients: {}'.format(len(train_patients)))
    print('Valid patients: {}'.format(len(valid_patients)))

    train_files = []
    for patient in train_patients:
        train_files += glob.glob(r'E:\{}\input\sample_images\{}\*.dcm'.format(data_dir, patient))
    print('Number of train files: {}'.format(len(train_files)))

    valid_files = []
    for patient in valid_patients:
        valid_files += glob.glob(r'E:\{}\input\sample_images\{}\*.dcm'.format(data_dir, patient))
    print('Number of valid files: {}'.format(len(valid_files)))

    """
    desired_depth = 10
    train_files = []
    for patient in train_patients:
        slices_path = glob.glob(r'E:\{}\input\sample_images\{}\*.dcm'.format(data_dir, patient))
        current_depth = len(slices_path)
        depth = current_depth / desired_depth
        depth = math.floor(depth)
        start = math.floor((current_depth - (depth * desired_depth))/2)
        for idx, slice in enumerate(slices_path[start::depth]):
            if idx <= desired_depth - 1:
                train_files.append(slice)
    print('Number of train files: {}'.format(len(train_files)))

    valid_files = []
    for patient in valid_patients:
        slices_path = glob.glob(r'E:\{}\input\sample_images\{}\*.dcm'.format(data_dir, patient))
        current_depth = len(slices_path)
        depth = current_depth / desired_depth
        depth = math.floor(depth)
        start = math.floor((current_depth - (depth * desired_depth))/2)
        for idx, slice in enumerate(slices_path[start::depth]):
            if idx <= desired_depth - 1:
                valid_files.append(slice)
    print('Number of valid files: {}'.format(len(valid_files)))
    """

    print('Create and compile model...')
    model = CNN()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=conf['patience'], verbose=0),
        # ModelCheckpoint('best.hd5f', monitor='val_loss', save_best_only=True, verbose=0)
    ]

    print('Fit model')
    # steps_per_epoch = len(train_files)//conf['batch_size']
    # validation_steps = len(train_files)//conf['batch_size']
    steps_per_epoch = conf['samples_train_per_epoch']//conf['batch_size']
    validation_steps = conf['samples_valid_per_epoch']//conf['batch_size']
    print('Sample train: {}, Sample valid: {}'.format(steps_per_epoch, validation_steps))
    # print('Sample train: {}, Sample valid: {}'.format(conf['samples_train_per_epoch'],
    #                                                   conf['samples_valid_per_epoch']))
    start = time.time()
    history = model.fit_generator(generator=batch_generator_train(train_files, train_csv_table,
                                                                  conf['batch_size'], do_aug=True),
                                  steps_per_epoch=steps_per_epoch,
                                  # steps_per_epoch=conf['samples_train_per_epoch'],
                                  epochs=conf['nb_epochs'],
                                  validation_data=batch_generator_train(valid_files, train_csv_table,
                                                                        conf['batch_size'], do_aug=False),
                                  validation_steps=validation_steps,
                                  # validation_steps=conf['samples_valid_per_epoch'],
                                  verbose=1)
    # callbacks = callbacks
    end = time.time()
    print(end-start)
    plt.rcParams.update({'font.size': 20})
    hist = pd.DataFrame(history.history)
    hist.to_csv('hist.csv', index=False)
    epochs = range(1, len(hist['loss']) + 1)
    accuracy, val_accuracy, loss, val_loss = hist['specificity_at_sensitivity'], \
                                             hist['val_specificity_at_sensitivity'], hist['loss'], hist['val_loss']
    plt.subplot(211)
    plt.plot(epochs, accuracy, label='Czułość trenowania', marker='o')
    plt.plot(epochs, val_accuracy, label='Czułość walidacji', marker='o')
    plt.xlabel('Czułość')
    plt.ylabel('Epoki')
    plt.grid(True)
    plt.legend()

    plt.subplot(212)
    plt.plot(epochs, loss, label='Strata trenowania', marker='o')
    plt.plot(epochs, val_loss, label='Strata walidacji', marker='o')
    plt.xlabel('Strata')
    plt.ylabel('Epoki')
    plt.grid(True)
    plt.legend()

    plt.show()

    plt.subplot(211)
    plt.plot(epochs, loss, marker='o')
    plt.xlabel('Strata trenowania')
    plt.ylabel('Epoki')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(epochs, val_loss, marker='o')
    plt.xlabel('Strata walidacji')
    plt.ylabel('Epoki')
    plt.grid(True)

    plt.show()

    plt.subplot(211)
    plt.plot(epochs, accuracy, marker='o')
    plt.xlabel('Czułość trenowania')
    plt.ylabel('Epoki')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(epochs, val_accuracy, marker='o')
    plt.xlabel('Czułość walidacji')
    plt.ylabel('Epoki')
    plt.grid(True)

    plt.show()

    return model


def create_submission_model(model):
    sample_subm = pd.read_csv(r'E:\{}\new_stage2\new_stage2_labels.csv'.format(data_dir))
    ids = sample_subm['id'].values
    for id in ids:
        print('Predict for patient: {}'.format(id))
        files = glob.glob(r'E:\{}\new_stage2\sample_images\{}\*.dcm'.format(data_dir, id))
        image_list = []
        for f in files:
            image = load_and_normalise_dicom(f, conf['image_shape'][0], conf['image_shape'][1])
            image_list.append(image)
        image_list = np.array(image_list)
        image_list = np.expand_dims(image_list, axis=3)
        batch_size = len(image_list)
        prediction = model.predict(image_list, verbose=1, batch_size=batch_size)
        pred_value = prediction[:, 0].mean()
        sample_subm.loc[sample_subm['id'] == id, 'cancer'] = pred_value
    sample_subm.to_csv('subm.csv', index=False)


if __name__ == '__main__':
    model = create_model_and_plots()
    if conf['save_model'] == 1:
        model.save('dsb.h5')
    create_submission_model(model)
