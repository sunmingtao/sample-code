import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model

im_width = 128
im_height = 128
im_chan = 1
salt_dir = '/Users/msun/Documents/salt'
path_train = os.path.join(salt_dir, 'train')
path_test = os.path.join(salt_dir, 'test')
train_image_dir = os.path.join(salt_dir, 'train/images')
train_masks_dir = os.path.join(salt_dir, 'train/masks')

ids = ['1f1cc6b3a4', '5b7c160d0d', '6c40978ddf', '7dfdf6eeb8', '7e5a6e5013']
plt.figure(figsize=(20, 10))
for j, img_name in enumerate(ids):
    q = j + 1
    img = load_img(train_image_dir +'/' + img_name + '.png')
    img_mask = load_img(train_masks_dir + '/' + img_name + '.png')

    plt.subplot(1, 2 * (1 + len(ids)), q * 2 - 1)
    plt.imshow(img)
    plt.subplot(1, 2 * (1 + len(ids)), q * 2)
    plt.imshow(img_mask)
plt.show()

train_ids = next(os.walk(path_train+"/images"))[2]
test_ids = next(os.walk(path_test))[2]

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm_notebook(enumerate(train_ids), total=len(train_ids)):
    path = path_train
    img = load_img(path + '/images/' + id_)
    x = img_to_array(img)[:,:,1]
    x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
    X_train[n] = x
    mask = img_to_array(load_img(path + '/masks/' + id_))[:,:,1]
    Y_train[n] = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

print('Done!')

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


inputs = Input((im_height, im_width, im_chan))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()

weight_path = 'model-tgs-salt-1.h5'
earlystopper = EarlyStopping(patience=20, verbose=1)
checkpointer = ModelCheckpoint(weight_path, verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=30,callbacks=[earlystopper, checkpointer])

model.load_weights(weight_path)
model.save('salt_model_100.h5')

from keras import models

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

def upsample(img):
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)

def downsample(img):
    return resize(img, (img_size_origin, img_size_origin), mode='constant', preserve_range=True)

img_size_origin = 101
img_size_target = 128

salt_dir = '/Users/msun/Documents/salt'

train_image_dir = os.path.join(salt_dir, 'train/images')
train_masks_dir = os.path.join(salt_dir, 'train/masks')

test_dir = os.path.join(salt_dir, 'test')
model1 = models.load_model('salt_model_100.h5', compile=False)
final_model = Sequential()
final_model.add(model1)
sub_list = []
test_paths = np.array(os.listdir(test_dir))

for img_name, index in zip(test_paths, range(len(test_paths))):
    img_id = get_file_name_without_extension(img_name)
    raw_img = get_raw_test_img_binary(img_id)
    raw_img = resize(raw_img, (128, 128, 1), mode='constant', preserve_range=True)
    raw_img = np.expand_dims(raw_img, 0) / 255.0
    mask_img = final_model.predict(raw_img)[0]
    mask_img = np.round(downsample(mask_img))
    rle = rle_encode(mask_img)
    sub_list += [[img_id, rle if len(rle) > 0 else None]]
    if index % 500 == 0:
        print('Processed {}'.format(index))


sub = pd.DataFrame(sub_list)
sub.columns = ['id','rle_mask']
sub.to_csv('salt-submission_100.csv', index=False)


