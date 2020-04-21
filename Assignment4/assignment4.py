from tensorflow.keras import backend

from keras.datasets import cifar10, cifar100

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

import pdb
import cv2
import numpy as np
import matplotlib.pyplot as plt

CIFAR100_LABELS_LIST = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

TIGER_ID = CIFAR100_LABELS_LIST.index('tiger')
BUS_ID = CIFAR100_LABELS_LIST.index('bus')
print(TIGER_ID, BUS_ID)


def retrainCNN_470():
    # load in the data
    (x_train, y_train), (testSet, testLabels) = cifar10.load_data()

    # split the data
    y_train = to_categorical(y_train)
    testLabels = to_categorical(testLabels)

    """ Create the model
    """
    model = Sequential()
    # create a 64 channel convoltional filter with size 3x3
    model.add(Conv2D(32, kernel_size=3, activation='relu',
                     input_shape=(32, 32, 3)))
    # create a 64 channel convoltional filter with size 3x3
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    # create a 3x3 filter
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    # create a max pooling layer with a 3x3 receptive field
    model.add(MaxPooling2D(pool_size=3))
    model.add(Conv2D(128, kernel_size=3, activation='relu'))
    # create a max pooling layer with a 3x3 receptive field
    model.add(MaxPooling2D(pool_size=3))
    model.add(Flatten())  # Flatten to put it in the format for the dense layer
    model.add(Dense(60, activation='relu'))  # dense layer with softmax
    model.add(Dense(30, activation='relu'))  # dense layer with softmax
    model.add(Dense(10, activation='softmax'))  # dense layer with softmax
    # compile the model into one which can be trained
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, validation_data=(
        testSet, testLabels), epochs=5, verbose=1)
    results = model.evaluate(testSet, testLabels, verbose=0)
    print("The test accuracy on CIFAR10 was {}".format(results[1]))

    """ load in the fine tuning data
    """
    (x_train100, y_train100), (testSet100, testLabels100) = cifar100.load_data()

    def convert_to_subset(data, labels, class_labels):
        in_each_class = [(labels == new_class_label)
                         for new_class_label in class_labels]
        in_new_classes = np.squeeze(np.sum(np.asarray(in_each_class), axis=0))
        in_new_classes = in_new_classes.astype(bool)
        updated_data = data[in_new_classes, :]
        updated_labels = labels[in_new_classes]
        for i, class_label in enumerate(class_labels):
            updated_labels[updated_labels == class_label] = i
        updated_labels = to_categorical(updated_labels)
        return updated_data, updated_labels

    tb_x_train, tb_y_train = convert_to_subset(
        x_train100, y_train100, [TIGER_ID, BUS_ID])
    tb_x_test, tb_y_test = convert_to_subset(
        testSet100, testLabels100, [TIGER_ID, BUS_ID])

    """create the new model
    """
    old_weights = model.get_weights()
    results_for_all_runs = []
    trainable = [[False, False, False, False, False, False, False, False, False],
                 [False, False, False, False, False, False, False, False, True],
                 [False, False, False, False, False, False, False, True, True],
                 [False, False, False, False, True, False, False, True, True],
                 [False, False, True, False, True, False, False, True, True],
                 [False, True, True, False, True, False, False, True, True],
                 ]

    for i in range(len(trainable)):
        model.set_weights(old_weights)

        bus_tiger_model = Sequential()
        for j, layer in enumerate(model.layers[:-1]):
            layer.trainable = trainable[i][j]
            bus_tiger_model.add(layer)  # , trainable=is_trainable)

        bus_tiger_model.add(Dense(2, activation='softmax'))
        bus_tiger_model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        bus_tiger_model.fit(tb_x_train, tb_y_train, epochs=10, verbose=0)
        results = bus_tiger_model.evaluate(tb_x_test, tb_y_test, verbose=0)
        results_for_all_runs.append(results[1])
        print("The finetuned test accuracy on bus and tiger is {} with {} retrained layers".format(
            results[1], i))

    plt.plot(np.arange(len(results_for_all_runs)) + 1, results_for_all_runs)
    plt.title(
        "Plot of accuracy versus number of trained layers, including the final one")
    plt.xlabel("Number of trained layers")
    plt.ylabel("Accuracy on test set")
    plt.show()


retrainCNN_470()
