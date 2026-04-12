import os
import numpy as np
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.util.montage import montage2d as montage
from skimage.morphology import label
import gc; gc.enable()
import math

BATCH_SIZE = 16
EDGE_CROP = 16
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
# downsampling inside the network
NET_SCALING = (1, 1)
# downsampling in preprocessing
IMG_SCALING = (4, 4)
# number of validation images to use
VALID_IMG_COUNT = 600
# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 150
AUGMENT_BRIGHTNESS = False

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)

ship_dir = '/Users/msun/data/'
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')



def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
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

def rle_decode(mask_rle, shape=(768, 768)):
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

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    if isinstance(in_mask_list, str):
        all_masks += rle_decode(in_mask_list)
    elif isinstance(in_mask_list, list) and len(in_mask_list) > 0:
        if isinstance(in_mask_list[0], str):
            for mask in in_mask_list:
                all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


masks = pd.read_csv('kaggle-ship-detector/train_ship_segmentations.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
rle_0 = masks.query('ImageId=="000155de5.jpg"')['EncodedPixels']
img_0 = masks_as_image(rle_0.values[0])
ax1.imshow(img_0[:, :, 0]) # reduce dimension from (768,768,1) -> (768,768)
ax1.set_title('Image$_0$')
rle_1 = multi_rle_encode(img_0)
img_1 = masks_as_image(rle_1)
ax2.imshow(img_1[:, :, 0])
ax2.set_title('Image$_1$')
print('Check Decoding->Encoding',
      'RLE_0:', len(rle_0), '->',
      'RLE_1:', len(rle_1))

def get_ship_num(encoded_pixel):
    if isinstance(encoded_pixel[0], str):
        return len(encoded_pixel)
    else:
        return 0

grouped_ship_masks = pd.DataFrame(masks.groupby('ImageId')['EncodedPixels'].apply(lambda a : list(a))).reset_index()
grouped_ship_masks['ship_num'] = grouped_ship_masks['EncodedPixels'].map(get_ship_num)
grouped_ship_masks['file_size_kb'] = grouped_ship_masks['ImageId'].map(lambda c_img_id: os.stat(os.path.join(train_image_dir, c_img_id)).st_size/1024)
grouped_ship_masks = grouped_ship_masks[grouped_ship_masks['file_size_kb'] > 50]
grouped_ship_masks = grouped_ship_masks[grouped_ship_masks['ImageId'] != '6384c3e78.jpg']
#grouped_ship_masks = grouped_ship_masks[grouped_ship_masks['ship_num'] > 0]

#grouped_ship_masks.query('ship_num > 1').sample(10)

#SAMPLES_PER_GROUP = 2000
#grouped_ship_masks = grouped_ship_masks.groupby('ship_num').apply(lambda x: x.sample(SAMPLES_PER_GROUP) if len(x) > SAMPLES_PER_GROUP else x)

from sklearn.model_selection import train_test_split

train_set, validation_set = train_test_split(grouped_ship_masks, test_size=0.3, random_state=42, stratify=grouped_ship_masks['ship_num'])

def xy_generator(df, batch_size=BATCH_SIZE):
    out_raw_img = []
    out_mask_img = []
    while True:
        shuffled = df.sample(frac=1)
        for index, data in shuffled.iterrows():
            img_file = data[0]
            encoded_pixel = data[1]
            img_path = os.path.join(train_image_dir, img_file)
            raw_img = imread(img_path)
            if isinstance(encoded_pixel, list):
                if len(encoded_pixel) > 0 and isinstance(encoded_pixel[0], str):
                    mask_img = masks_as_image(encoded_pixel)
                else:
                    mask_img = masks_as_image([])
            else:
                mask_img = masks_as_image([])
            raw_img = raw_img[::IMG_SCALING[0], ::IMG_SCALING[1]] # Make image a bit smaller
            mask_img = mask_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_raw_img += [raw_img]
            out_mask_img += [mask_img]
            if len(out_raw_img) >= batch_size:
                yield np.array(out_raw_img)/255.0, np.array(out_mask_img)
                out_raw_img, out_mask_img =[],[]


from keras.models import Sequential, Model
from keras.layers.core import Lambda, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import GaussianNoise, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Input, concatenate

input_img = Input(shape=(192,192,3), name='RGB_Input')

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

import keras.backend as K
from keras.losses import binary_crossentropy

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_p_bce(in_gt, in_pred):
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

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
    return -K.mean((5 * true_positive_count + eps) / (5 * true_positive_count + 4 * false_negative_count + false_positive_count + eps), axis=0)

def f_score_diff(y_true, y_pred, eps=1e-6):
    true_positive_count = K.sum(y_true * y_pred, axis=[1, 2, 3])
    false_negative_count = K.sum((1 - y_true) * y_pred, axis=[1, 2, 3])
    false_positive_count = K.sum(y_true * (1 - y_pred), axis=[1, 2, 3])
    return -K.mean(((5 * true_positive_count + eps) * 100) / (5 * true_positive_count + 4 * false_negative_count + false_positive_count + eps), axis=0)

from keras import layers
from keras import losses

x = layers.Input(shape=(None, None, None, 1))
y = layers.Input(shape=(None, None, None, 1))
loss_func = K.Function([x, y], [iou(x, y, 1e-6)])

a1 = np.array([[0,0,0],[0,0,0]])
a1 = np.expand_dims(a1, -1)
a2 = np.array([[0,0,0],[0,0,0]])
a2 = np.expand_dims(a2, -1)
a3 = np.array([[1,1,0],[0,0,0]])
a3 = np.expand_dims(a3, -1)
a_true = np.stack((a1,a2,a3))
b1 = np.array([[0,0,0],[0,0,0]])
b1 = np.expand_dims(b1, -1)
b2 = np.array([[0,0,0],[0,0,0]])
b2 = np.expand_dims(b2, -1)
b3 = np.array([[0,0,0],[0,0,0]])
b3 = np.expand_dims(b3, -1)
b_pred = np.stack((b1,b2,b3))


precision_func = K.Function([x, y], [precision(x, y)])
recall_func = K.Function([x, y], [recall(x, y)])
my_binary_accuracy_func = K.Function([x, y], [my_binary_accuracy(x, y)])
f_score_func = K.Function([x, y], [f_score(x, y)])


print('Precision', precision_func([a_true, b_pred]))
print('Recall', recall_func([a_true, b_pred]))
print('my_binary_accuracy_func', my_binary_accuracy_func([a_true, b_pred]))
print('f score', f_score_func([a_true, b_pred]))
print('Iou', loss_func([a_true, b_pred]))

from keras.preprocessing.image import ImageDataGenerator
dg_args = dict(featurewise_center = False,
                  samplewise_center = False,
                  rotation_range = 15,
                  width_shift_range = 0.1,
                  height_shift_range = 0.1,
                  shear_range = 0.01,
                  zoom_range = [0.9, 1.25],
                  horizontal_flip = True,
                  vertical_flip = True,
                  fill_mode = 'reflect',
                   data_format = 'channels_last')
# brightness can be problematic since it seems to change the labels differently from the images
if AUGMENT_BRIGHTNESS:
    dg_args[' brightness_range'] = [0.5, 1.5]
image_gen = ImageDataGenerator(**dg_args)

if AUGMENT_BRIGHTNESS:
    dg_args.pop('brightness_range')
label_gen = ImageDataGenerator(**dg_args)

def create_aug_gen(in_gen, seed = None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255*in_x,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)

        yield next(g_x)/255.0, next(g_y)

from keras.optimizers import SGD, Adam

model.compile(loss=f_score_diff, optimizer=Adam(1e-4, decay=1e-6), metrics=[dice_p_bce, iou, true_positive_rate, f_score, precision, recall, my_binary_accuracy])

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="{}_weights.best.hdf5".format('simple_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                   patience=10, verbose=1, mode='min',
                                   min_delta=0.0001, cooldown=2, min_lr=1e-7)

early = EarlyStopping(monitor="val_loss", mode="min", verbose=2, patience=30)

callbacks_list = [checkpoint, early, reduceLROnPlat]


model.fit_generator(xy_generator(train_set), steps_per_epoch=train_set.shape[0] // BATCH_SIZE, epochs=100, callbacks=callbacks_list, validation_data=xy_generator(validation_set), validation_steps=validation_set.shape[0] // BATCH_SIZE, verbose=1)


model.load_weights(weight_path)
model.save('simple_model_006.h5')

model.evaluate_generator(xy_generator(grouped_ship_masks), steps=800, verbose=1)

from keras import models, layers

model1 = models.load_model('simple_model_005.h5', compile=False)
final_model = Sequential()
final_model.add(AveragePooling2D(IMG_SCALING, input_shape=(None, None, 3)))
final_model.add(model1)
final_model.add(UpSampling2D(IMG_SCALING))

def get_mask_img(img_name):
    encoded_pixel = grouped_ship_masks.query('ImageId==@img_name')['EncodedPixels']
    if len(encoded_pixel.values) > 0:
        return masks_as_image(encoded_pixel.values[0])
    else:
        return masks_as_image([])


from skimage.morphology import binary_opening, disk

image_names = grouped_ship_masks.query('ship_num > 1').sample(5)['ImageId'].values
fig, axs = plt.subplots(5, 3, figsize = (10, 12))

for (ax1, ax2, ax3), image_name in zip(axs, image_names):
    c_path = os.path.join(train_image_dir, image_name)
    raw_c_img = imread(c_path)
    c_img = np.expand_dims(raw_c_img, 0) / 255.0
    ship_img = final_model.predict(c_img)[0]
    binary_ship_img = ship_img > 0.5
    ground_truth_mask = get_mask_img(image_name)
    ax1.imshow(binary_ship_img[:,:,0])
    ax2.imshow(ground_truth_mask[:,:,0])
    ax3.imshow(raw_c_img)

y_true1 = np.expand_dims(ground_truth_mask, 0)
y_pred1 = np.expand_dims(binary_ship_img, 0)
print('Precision:', precision_func([y_true1, y_pred1]))
print('Recall:', recall_func([y_true1, y_pred1]))
print('F score:', f_score_func([y_true1, y_pred1]))


sum(((1-y_true1) * y_pred1).reshape(-1))

def predict(img, path=test_image_dir):
    c_img = imread(os.path.join(path, c_img_name))
    c_img = np.expand_dims(c_img, 0)/255.0
    cur_seg = final_model.predict(c_img)[0]
    cur_seg = binary_opening(cur_seg>0.5, np.expand_dims(disk(2), -1))
    return cur_seg, c_img

def pred_encode(img):
    cur_seg, _ = predict(img)
    cur_rles = rle_encode(cur_seg)
    return [img, cur_rles if len(cur_rles) > 0 else None]


from tqdm import tqdm_notebook

test_paths = np.array(os.listdir(test_image_dir))
#test_paths = test_paths[0:10]
out_pred_rows = []
for c_img_name, index in zip(tqdm_notebook(test_paths), range(len(test_paths))):
    out_pred_rows += [pred_encode(c_img_name)]
    if index % 200 == 0:
        print('Processed {} test images'.format(index))

sub = pd.DataFrame(out_pred_rows)
sub.columns = ['ImageId', 'EncodedPixels']
sub.to_csv('submission_003.csv', index=False)

test_masks = pd.read_csv('submission_002.csv')
test_masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
test_masks.count()
test_masks.query('ships>0').count()
train_set.shape
train_set.query('ship_num>0').shape


test_paths = os.listdir(test_image_dir)
fig, m_axs = plt.subplots(20, 2, figsize = (10, 40))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for (ax1, ax2), c_img_name in zip(m_axs, test_paths):
    c_path = os.path.join(test_image_dir, c_img_name)
    c_img = imread(c_path)
    first_img = np.expand_dims(c_img, 0)/255.0
    first_seg = final_model.predict(first_img)
    ax1.imshow(first_img[0])
    ax1.set_title('Image')
    ax2.imshow(first_seg[0, :, :, 0], vmin = 0, vmax = 1)
    ax2.set_title('Prediction')
fig.savefig('test_predictions.png')

