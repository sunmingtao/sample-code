BATCH_SIZE = 16
GAUSSIAN_NOISE = 0.1

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from keras.optimizers import Adam
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers.core import Lambda, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import GaussianNoise, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Input, concatenate, ZeroPadding2D, Cropping2D
from keras import models

salt_dir = '/Users/msun/Documents/salt'

train_image_dir = os.path.join(salt_dir, 'train/images')
train_masks_dir = os.path.join(salt_dir, 'train/masks')

test_dir = os.path.join(salt_dir, 'test')


train_csv = os.path.join(salt_dir, 'train.csv')
depth_csv = os.path.join(salt_dir, 'depths.csv')

train_pd = pd.read_csv(train_csv)
depth_pd = pd.read_csv(depth_csv)

train_image_names = os.listdir(train_image_dir)

for img_name in train_image_names:
    img_path = os.path.join(train_image_dir, img_name)
    img_binary=imread(img_path)
    break

train_mask_names = os.listdir(train_masks_dir)

for img_name in train_mask_names:
    img_path = os.path.join(train_masks_dir, '0a1742c740.png')
    img_binary=imread(img_path)
    print(img_binary.shape)
    plt.imshow(imread(img_path))
    break

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(101, 101)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def get_raw_train_img_binary(id):
    img_name = id + '.png'
    img_path = os.path.join(train_image_dir, img_name)
    return imread(img_path)

def get_raw_test_img_binary(id):
    img_name = id + '.png'
    img_path = os.path.join(test_dir, img_name)
    return imread(img_path)

def get_mask_img_binary(id):
    img_name = id + '.png'
    img_path = os.path.join(train_masks_dir, img_name)
    return imread(img_path) > 0

def get_mask_img_binary_from_rle(rle):
    if isinstance(rle, str):
        return rle_decode(rle)
    else:
        return rle_decode('')

def get_file_name_without_extension(file_name):
    return os.path.splitext(file_name)[0]

fig, axs = plt.subplots(10, 3, figsize = (10, 10))
for (_, train_data), (ax1, ax2, ax3) in zip(train_pd.iterrows(), axs):
    img_id = train_data[0]
    mask = train_data[1]
    ax1.imshow(get_raw_train_img_binary(img_id))
    ax2.imshow(get_mask_img_binary(img_id))
    ax3.imshow(get_mask_img_binary_from_rle(mask))

img_size_origin = 101
img_size_target = 128

def upsample(img):
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)

def downsample(img):
    return resize(img, (img_size_origin, img_size_origin), mode='constant', preserve_range=True)


def xy_generator(df, batch_size=BATCH_SIZE):
    out_raw_img = []
    out_mask_img = []
    while True:
        shuffled = df.sample(frac=1)
        for index, data in shuffled.iterrows():
            img_id = data[0]
            rle = data[1]
            raw_img = get_raw_train_img_binary(img_id)
            mask_img = np.expand_dims(get_mask_img_binary(img_id), -1)
            out_raw_img += [upsample(raw_img)]
            out_mask_img += [upsample(mask_img)]
            if len(out_raw_img) >= batch_size:
                yield np.array(out_raw_img)/255.0, np.round(np.array(out_mask_img)).astype(int)
                out_raw_img, out_mask_img =[],[]

#gen = xy_generator(train_pd, batch_size=1)
#x, y = next(gen)
#print(x.shape, y.shape)
#print(y)



input_img = Input(shape=(128,128,3), name='RGB_Input')

p0 = GaussianNoise(GAUSSIAN_NOISE)(input_img)
p0 = BatchNormalization()(p0)


c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(p0)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
d4 = Dropout(0.5)(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(d4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)
d5 = Dropout(0.5)(c5)

u6 = UpSampling2D(size=(2, 2))(d5)
up6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)

merge6 = concatenate([d4, up6])
c6 = Conv2D(64, 3, activation = 'relu', padding = 'same')(merge6)
c6 = Conv2D(64, 3, activation = 'relu', padding = 'same')(c6)


u7 = UpSampling2D(size=(2, 2))(c6)
up7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)

merge7 = concatenate([c3, up7])
c7 = Conv2D(32, 3, activation = 'relu', padding = 'same')(merge7)
c7 = Conv2D(32, 3, activation = 'relu', padding = 'same')(c7)

u8 = UpSampling2D(size=(2, 2))(c7)
up8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
merge8 = concatenate([c2, up8])
c8 = Conv2D(16, 3, activation = 'relu', padding = 'same')(merge8)
c8 = Conv2D(16, 3, activation = 'relu', padding = 'same')(c8)

u9 = UpSampling2D(size=(2, 2))(c8)
up9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)

merge9 = concatenate([c1, up9])
c9 = Conv2D(8, 3, activation = 'relu', padding = 'same')(merge9)
c9 = Conv2D(8, 3, activation = 'relu', padding = 'same')(c9)

c9 = Conv2D(2, 3, activation = 'relu', padding = 'same')(c9)
output = Conv2D(1, 1, activation = 'sigmoid')(c9)

model = Model(inputs=[input_img], outputs=output)
model.summary()

def iou(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return iou(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)

def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true) * K.flatten(y_pred))/K.sum(y_true)

def precision(y_true, y_pred, eps=1e-6):
    true_positive_count = K.sum(y_true * K.round(y_pred), axis=[1,2,3])
    false_positive_count = K.sum((1-y_true) * K.round(y_pred), axis=[1,2,3])
    return K.mean((true_positive_count+eps)/(false_positive_count+true_positive_count+eps), axis=0)

def recall(y_true, y_pred, eps=1e-6):
    true_positive_count = K.sum(y_true * K.round(y_pred), axis=[1,2,3])
    false_negative_count = K.sum(y_true * K.round(1-y_pred), axis=[1,2,3])
    return K.mean((true_positive_count+eps)/(false_negative_count+true_positive_count+eps), axis=0)

def my_binary_accuracy(y_true, y_pred):
    true_positive_count = K.sum(y_true * K.round(y_pred), axis=[1, 2, 3])
    true_negative_count = K.sum((1-y_true) * K.round(1-y_pred), axis=[1, 2, 3])
    false_positive_count = K.sum((1-y_true) * K.round(y_pred), axis=[1,2,3])
    false_negative_count = K.sum(y_true * K.round(1-y_pred), axis=[1,2,3])
    return K.mean((true_positive_count + true_negative_count) / (true_positive_count + true_negative_count + false_positive_count + false_negative_count), axis=0)

def f_score(y_true, y_pred, eps=1e-6):
    true_positive_count = K.sum(y_true * K.round(y_pred), axis=[1, 2, 3])
    false_negative_count = K.sum((1 - y_true) * K.round(y_pred), axis=[1, 2, 3])
    false_positive_count = K.sum(y_true * K.round(1 - y_pred), axis=[1, 2, 3])
    return -K.mean((true_positive_count + eps) / (true_positive_count + false_negative_count + false_positive_count + eps), axis=0)

def f_score_diff(y_true, y_pred, eps=1e-6):
    true_positive_count = K.sum(y_true * y_pred, axis=[1, 2, 3])
    false_negative_count = K.sum((1 - y_true) * y_pred, axis=[1, 2, 3])
    false_positive_count = K.sum(y_true * (1 - y_pred), axis=[1, 2, 3])
    return -K.mean(((true_positive_count + eps) * 100) / (true_positive_count + false_negative_count + false_positive_count + eps), axis=0)


train_set, validation_set = train_test_split(train_pd, test_size=0.3, random_state=42)
train_steps = len(train_set) // BATCH_SIZE + 1
validation_steps = len(validation_set) // BATCH_SIZE + 1


model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=[iou, true_positive_rate, f_score, precision, recall, my_binary_accuracy])

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('salt')

checkpoint = ModelCheckpoint(weight_path, monitor='val_f_score', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_f_score', factor=0.2,
                                   patience=10, verbose=1, mode='min',
                                   min_delta=0.0001, cooldown=2, min_lr=1e-7)

early = EarlyStopping(monitor="val_f_score", mode="min", verbose=2, patience=30)

callbacks_list = [checkpoint, early, reduceLROnPlat]

model.fit_generator(xy_generator(train_set), steps_per_epoch=train_steps, epochs=100, callbacks=callbacks_list, validation_data=xy_generator(validation_set), validation_steps=validation_steps, verbose=1)


