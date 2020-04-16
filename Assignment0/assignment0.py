import pdb

import numpy as np
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


TRAIN_DATA = "../data/train_data.txt"
TRAIN_LABELS = "../data/train_label.txt"
TEST_DATA = "../data/test_data.txt"
TEST_LABELS = "../data/test_label.txt"

SKLEARN_PRED = "scikit_learn_preds.txt"
KERAS_PRED = "keras_preds.txt"


def write_out(first_layer, second_layer, output_file="weights.txt"):
    m = second_layer.shape[1]
    with open(output_file, 'w') as fileh:
        fileh.write("{:d}\n".format(m))
        np.savetxt(fileh, second_layer.transpose())
        np.savetxt(fileh, first_layer.transpose())


train_data = np.loadtxt(TRAIN_DATA)
train_labels = np.loadtxt(TRAIN_LABELS)
test_data = np.loadtxt(TEST_DATA)
with open(TEST_LABELS, 'r') as infile:
    line = infile.readline()
test_labels = np.fromstring(line[1:-1], sep=', ')

classifier = MLPClassifier(hidden_layer_sizes=(20,), activation='relu')
classifier.fit(train_data, train_labels)
training_accuracy = classifier.score(train_data, train_labels)
testing_accuracy = classifier.score(test_data, test_labels)
print("Training accuracy was : {}, testing accuracy was : {}".format(
    training_accuracy, testing_accuracy))

preds = classifier.predict(test_data)
np.savetxt(SKLEARN_PRED, np.expand_dims(preds, 0), fmt="%d")
coefs = classifier.coefs_
write_out(coefs[1], coefs[0])


# drawn heavily from https://towardsdatascience.com/introduction-to-multilayer-neural-networks-with-tensorflows-keras-api-abf4f813959
NUM_HIDDEN = 40
NUM_INPUTS = train_data.shape[1]
NUM_OUTPUTS = 1


def shift(label):
    shifted = ((label + 1) / 2).astype(int)
    return np.expand_dims(shifted, 1)


def convert_to_onehot(label):
    shifted = shift(label)
    return keras.utils.to_categorical(shifted), shifted


train_labels_onehot, train_labels_shifted = convert_to_onehot(train_labels)
test_labels_onehot, test_labels_shifted = convert_to_onehot(test_labels)

# process the data so it conforms to the one-hot requirement
model = Sequential()
model.add(Dense(NUM_HIDDEN, activation='relu', input_dim=NUM_INPUTS))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data, train_labels_shifted,
                    epochs=10, batch_size=32, validation_split=0.1)

# calculate training accuracy
train_labels_pred = model.predict_classes(train_data, verbose=0)
correct_preds = np.sum(train_labels_shifted == train_labels_pred, axis=0)
train_acc = correct_preds / train_labels_shifted.shape[0]
print("Training accuracy: {}".format(train_acc))

# calculate testing accuracy
test_labels_pred = model.predict_classes(test_data, verbose=0)
correct_preds = np.sum(test_labels_shifted == test_labels_pred, axis=0)
test_acc = correct_preds / test_labels_shifted.shape[0]
print("Test accuracy {}".format(test_acc))
np.savetxt(KERAS_PRED, 2 * test_labels_pred.transpose() - 1, fmt="%d")

weights = model.get_weights()

write_out(weights[2], weights[0], "Keras.csv")
