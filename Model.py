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
import random
import shutil
import numpy as np
import pandas as pd
import pydicom as dicom
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
from keras.models import Sequential
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
conf['batch_size'] = 100

# Liczba epok
conf['nb_epochs'] = 5

# Liczba epok bez poprawy wyniku, po których trenowanie się zakończy.
conf['patience'] = 3

# Rozdzielczość skanów
conf['image_shape'] = (64, 64)

# Wskaźnik uczenia
conf['learning_rate'] = 1e-2

# Liczba próbek treningowych na epokę
conf['samples_train_per_epoch'] = 3000

# Liczba próbek walidacyjnych na epokę
conf['samples_valid_per_epoch'] = 500

# Parametry warstw
conf['level_1_filters'] = 4
conf['level_2_filters'] = 8
conf['dense_layer_size'] = 32
conf['dropout_value'] = 0.5


"""
Do utworzenia zboru testowego można użyć poniższej funkcji.
"""


def create_test_data(train_csv_table, data_size=20, move_back=False):
    new_stage2 = r'E:\Dane\new_stage2'
    sample_images = r'E:\Dane\new_stage2\sample_images'
    input_labels = r'E:\Dane\input\stage1_labels_train.csv'
    new_stage_labels = r'E:\Dane\new_stage2\new_stage2_labels.csv'
    new_train_csv_table = train_csv_table.head(len(train_csv_table.index)-data_size)
    print(len(new_train_csv_table.index))
    valid_csv_table = train_csv_table.tail(data_size)
    print(len(valid_csv_table.index))
    new_train_csv_table.to_csv(input_labels, index=False)
    valid_csv_table.to_csv(new_stage_labels, index=False)

    for folder in (new_stage2, sample_images):
        if not os.path.exists(folder):
            os.mkdir(folder)

    for patient in valid_csv_table.id:
        if not os.path.exists(os.path.join(sample_images, patient)):
            src = r'E:\Dane\input\sample_images\{}'.format(patient)
            dst = sample_images
            shutil.move(src, dst)

    if move_back:
        for patient in os.listdir(sample_images):
            src = os.path.join(sample_images, patient)
            dst = r'E:\Dane\input\sample_images'
            shutil.move(src, dst)


train_csv_table = pd.read_csv(r'E:\Dane\input\stage1_labels.csv')
create_test_data(train_csv_table, 20, move_back=False)

get_dir = 'Dane'
train_csv_table = pd.read_csv(r'E:\Dane\input\stage1_labels_train.csv')

if conf['use_sample_only'] != 1:
    # Można zdefiniować inny zbiór danych
    get_dir = 'Test'
    train_csv_table = pd.read_csv(r'E:\Test\input\stage1_labels_test.csv')


def load_and_normalise_dicom(path, x, y):
    dicom1 = dicom.read_file(path)
    dicom_img = dicom1.pixel_array.astype(np.float64)
    mn = dicom_img.min()
    mx = dicom_img.max()
    if (mn - mx) != 0:
        dicom_img = (dicom_img - mn)/(mx - mn)
    else:
        dicom_img[:, :] = 0
    if dicom_img.shape != (x, y):
        dicom_img = cv2.resize(dicom_img, (x, y), interpolation=cv2.INTER_CUBIC)
    return dicom_img


def get_train_single_fold(train_data, fraction):
    ids = train_data['id'].values
    random.shuffle(ids)
    split_point = int(round(fraction*len(ids)))
    train_list = ids[:split_point]
    valid_list = ids[split_point:]
    return train_list, valid_list


def batch_generator_train(files, train_csv_table, batch_size):
    number_of_batches = np.ceil(len(files)/batch_size)
    counter = 0
    random.shuffle(files)
    while True:
        batch_files = files[batch_size*counter:batch_size*(counter+1)]
        image_list = []
        mask_list = []
        for f in batch_files:
            image = load_and_normalise_dicom(f, conf["image_shape"][0], conf["image_shape"][0])
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


def CNN():
    model = Sequential()
    model.add(layers.Conv2D(filters=conf['level_1_filters'], kernel_size=(3, 3), activation='relu',
                            input_shape=(64, 64, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=conf['level_1_filters'], kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=conf['level_2_filters'], kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=conf['level_2_filters'], kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(conf['dense_layer_size'], activation='relu'))
    model.add(layers.Dropout(conf['dropout_value']))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=optimizers.RMSprop(lr=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def create_model_and_plots():
    train_patients, valid_patients = get_train_single_fold(train_csv_table, conf['train_valid_fraction'])
    print('Train patients: {}'.format(len(train_patients)))
    print('Valid patients: {}'.format(len(valid_patients)))

    desired_depth = 10
    train_files = []
    for patient in train_patients:
        slices_path = glob.glob(r'E:\Dane\input\sample_images\{}\*dcm'.format(patient))
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
        slices_path = glob.glob(r'E:\Dane\input\sample_images\{}\*dcm'.format(patient))
        current_depth = len(slices_path)
        depth = current_depth / desired_depth
        depth = math.floor(depth)
        start = math.floor((current_depth - (depth * desired_depth))/2)
        for idx, slice in enumerate(slices_path[start::depth]):
            if idx <= desired_depth - 1:
                valid_files.append(slice)
    print('Number of valid files: {}'.format(len(valid_files)))

    print('Create and compile model...')
    model = CNN()
    callbacks = [
                 EarlyStopping(monitor='val_loss', patience=conf['patience'], verbose=0),
                 # ModelCheckpoint('best.hd5f', monitor='val_loss', save_best_only=True, verbose=0)
                ]

    print('Fit model')
    print('Sample train: {}, Sample valid: {}'.format(conf['samples_train_per_epoch'],
                                                      conf['samples_valid_per_epoch']))
    history = model.fit_generator(generator=batch_generator_train(train_files, train_csv_table,
                                                                  conf['batch_size']),
                                  steps_per_epoch=conf['samples_train_per_epoch'],
                                  epochs=conf['nb_epochs'],
                                  validation_data=batch_generator_train(valid_files, train_csv_table,
                                                                        conf['batch_size']),
                                  validation_steps=conf['samples_valid_per_epoch'],
                                  verbose=1)

# validation_steps = conf['samples_valid_per_epoch']
# steps_per_epoch = conf['samples_train_per_epoch']
# callbacks = callbacks

    hist = pd.DataFrame(history.history)
    epochs = range(1, len(hist['loss']) + 1)
    accuracy, val_accuracy, loss, val_loss = hist['accuracy'], hist['val_accuracy'], hist['loss'], hist['val_loss']
    plt.subplot(211)
    plt.plot(epochs, accuracy, label='Dokładność trenowania', marker='o')
    plt.plot(epochs, val_accuracy, label='Dokładność walidacji', marker='o')
    plt.xlabel('Dokładność')
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

    return model


def create_submission_model(model):
    sample_subm = pd.read_csv(r'E:\Dane\new_stage2\new_stage2_labels.csv')
    ids = sample_subm['id'].values[:5]
    for id in ids:
        print('Predict for patient: {}'.format(id))
        files = glob.glob(r'E:\Dane\new_stage2\sample_images\{}\*.dcm'.format(id))
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
