'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from sklearn.model_selection import train_test_split
import pdb
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 2
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

DRAGON_NAME = "data/full_numpy_bitmap_dragon.npy"
TIGER_NAME = "data/full_numpy_bitmap_tiger.npy"
SQUIRREL_NAME = "data/full_numpy_bitmap_squirrel.npy"


def load_all(filenames):
    all_labels = []
    all_features = []
    for i, filename in enumerate(filenames):
        features = load_npy_as_keras_data(filename)
        num_samples = features.shape[0]
        labels = np.ones((num_samples,)) * i
        all_features.append(features)
        all_labels.append(labels)
    all_labels = np.hstack(all_labels).astype(int)
    all_features = np.concatenate(all_features, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, all_labels, test_size=0.2, random_state=42, shuffle=True)
    X_train = np.expand_dims(X_train, 3)
    X_test = np.expand_dims(X_test, 3)
    return (X_train, y_train), (X_test, y_test)


def load_npy_as_keras_data(file, vis=False):
    data = np.load(file)
    num_samples, num_features = data.shape
    width = int(np.sqrt(num_features))
    data = np.reshape(data, (num_samples, width, width))
    if vis:
        for i in range(10):
            plt.imshow(data[i, :, :])
            plt.pause(3)
    return data


# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = load_all(
    [DRAGON_NAME, TIGER_NAME, SQUIRREL_NAME][:2])
#   x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#   x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

pdb.set_trace()
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
preds = model.predict(x_test)
pred_classes = preds.argmax(axis=-1)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
