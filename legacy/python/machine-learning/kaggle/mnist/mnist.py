import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import matplotlib



from keras.models import Sequential
from keras.layers import Dense , Dropout , Lambda, Flatten
from keras.optimizers import Adam ,RMSprop
from sklearn.model_selection import train_test_split
from keras import  backend as K
from keras.preprocessing.image import ImageDataGenerator

train = pd.read_csv("kaggle-mnist/train.csv")
print(train.shape)
train.head()


test= pd.read_csv("kaggle-mnist/test.csv")
print(test.shape)
test.head()


X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test.values.astype('float32')

X_train.shape

X_train = X_train.reshape(X_train.shape[0], 28, 28)

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x):
    return (x-mean_px)/std_px


from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)
num_classes = y_train.shape[1]
num_classes

seed = 43
np.random.seed(seed)


from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D

model = Sequential()
model.add(Lambda(standardize, input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
print("input shape ", model.input_shape)
print("output shape ", model.output_shape)

from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing import image
gen = image.ImageDataGenerator()

from sklearn.model_selection import train_test_split
X = X_train
y = y_train
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
batches = gen.flow(X_train, y_train, batch_size=64)
val_batches= gen.flow(X_val, y_val, batch_size=64)

history = model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3,
                    validation_data=val_batches, validation_steps=val_batches.n)

history_dict = history.history
history_dict.keys()

predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})

submissions.to_csv("kaggle-mnist/DR.csv", index=False, header=True)