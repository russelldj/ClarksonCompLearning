import pdb

import numpy as np
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow import keras

TRAIN_DATA = "../data/train_data.txt"
TRAIN_LABELS = "../data/train_label.txt"
TEST_DATA = "../data/test_data.txt"
TEST_LABELS = "../data/test_label.txt"

SKLEARN_PRED = "scikit_learn_preds.txt"

def write_out(coefs, output_file="weights.txt"):
    first_layer, second_layer = coefs
    m = second_layer.shape[0]
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
testing_accuracy  = classifier.score(test_data, test_labels)
print("Training accuracy was : {}, testing accuracy was : {}".format(training_accuracy, testing_accuracy))

preds = classifier.predict(test_data)
np.savetxt(SKLEARN_PRED, np.expand_dims(preds, 0), fmt="%d")
coefs = classifier.coefs_
write_out(coefs)


# the tensorflow section
# drawn heavily from https://towardsdatascience.com/introduction-to-multilayer-neural-networks-with-tensorflows-keras-api-abf4f813959
NUM_HIDDEN = 20
NUM_INPUTS = train_data.shape[1]
NUM_OUTPUTS = 2

def shift(label):
    return ((label + 1) / 2).astype(int)

def convert_to_onehot(label):
    shifted = shift(label)
    return keras.utils.to_categorical(shifted), shifted

train_labels_onehot, train_labels_shifted = convert_to_onehot(train_labels)
test_labels_onehot, test_labels_shifted = convert_to_onehot(test_labels)

# process the data so it conforms to the one-hot requirement
model = keras.models.Sequential()

# add input layer
model.add(keras.layers.Dense(
    units=NUM_HIDDEN,
    input_dim=NUM_INPUTS,
    bias_initializer='zeros',
    activation='tanh')
)
model.add(
    keras.layers.Dense(
        units=NUM_OUTPUTS,
        input_dim=NUM_HIDDEN,
        bias_initializer='zeros',
        activation='softmax')
    )

# define SGD optimizer
sgd_optimizer = keras.optimizers.SGD(
    lr=0.001, decay=1e-7, momentum=0.9
)
# compile model
model.compile(
    optimizer=sgd_optimizer,
    loss='categorical_crossentropy'
)
# train model
history = model.fit(
    train_data, train_labels_onehot,
    batch_size=64, epochs=50,
    verbose=1, validation_split=0.1
)

# calculate training accuracy
train_labels_pred = model.predict_classes(train_data, verbose=0)
correct_preds = np.sum(train_labels_shifted == train_labels_pred, axis=0)
train_acc = correct_preds / train_labels_shifted.shape[0]

print("Training accuracy: {}".format(train_acc))
pdb.set_trace()

# calculate testing accuracy
y_test_pred = model.predict_classes(X_test_centered, verbose=0)
correct_preds = np.sum(y_test == y_test_pred, axis=0)
test_acc = correct_preds / y_test.shape[0]

print(f'Test accuracy: {(test_acc * 100):.2f}')


#THRESHOLDS = np.linspace(0.1, 0.9, 9)
#print(THRESHOLDS)
#num_samples = len(train)
#for threshold in THRESHOLDS:
#    indices = np.random.choice(num_samples, int(np.floor(threshold * num_samples)), replace=False)
#    train_sample = train[indices]
#    train_labels = labels[indices]
#exit()
#model = tf.keras.models.Sequential([
#  tf.keras.layers.Dense(128, activation='relu'),
#  tf.keras.layers.Dense(2)
#])
#
#print(model)
#pdb.set_trace()
#model()
