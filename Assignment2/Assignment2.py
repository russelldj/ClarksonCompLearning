import numpy as np
from numpy.random import normal
import pandas as pd
import pdb
import matplotlib.pyplot as plt


class neuron():
    def __init__(self, size):
        self.weights = normal(0, 1, size)

    def forward(self, x):
        pass

    def weight(self, x):
        if not np.equal(x.shape, self.weights.shape):
            raise Exception("The shape of the weights did not match the input")
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)


def relu(x):
    pass


def sigmoid(x):
    out = 1 / (1 + np.exp(-x))
    return out


def sigmoid_prime(x):
    sigmoid_x = sigmoid(x)
    out = sigmoid_x * (1 - sigmoid_x)
    return out


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    out = 1 - tanh(x) ** 2


class layer():
    def __init__(self, in_dim, out_dim, activation="tanh"):
        self.weights = normal(0, 1, (in_dim, out_dim))
        self.activation = activation
        if activation == "sigmoid":
            self.nonlinearity = sigmoid
            self.nonlinearity_prime = sigmoid_prime
        if activation == "tanh":
            self.nonlinearity = tanh
            self.nonlinearity_prime = sigmoid_prime
        if activation == "relu":
            self.nonlinearity = None

    def __str__(self):
        return "Layer with {} activation".format(self.activation)

    def forward(self, x):
        if self.weights.shape[1] != len(x):
            raise Exception("The size of the input was {} when it should have been {}".format(
                self.weights.shape, len(x)))

        y = np.matmul(self.weights, x)
        z = self.nonlinearity(y)
        return y, z

    def backward(self):
        pass


class NN():
    def __init__(self, sizes):
        self.layers = self.create_layers(sizes)
        self.sizes = sizes
        self.zs = None

    def create_layers(self, sizes):
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(layer(sizes[i+1], sizes[i]))
        # since it's classifcation with [-1, 1]
        layers.append(layer(sizes[-1], sizes[-2], 'tanh'))
        return layers

    def forward(self, x):
        xs = [x]
        ys = []
        for layer in self.layers:
            y, x = layer.forward(x)
            xs.append(x)
            ys.append(y)
        return xs[-1], xs, ys

    def compute_backprop(self):
        deltas = []
        for i, size in enumerate(self.sizes[-2::-1]):
            deltas.append([])  # add the new layer
            num_uses_as_input = self.sizes[-(i + 1)]
            layer = self.layers[-(i + 1)]
            print(layer, num_uses_as_input, i, size)
            for j in range(size):  # iterate over the current neurons
                deltas[-1].append(0)
                for k in range(num_uses_as_input):  # this is the summation
                    deltas[-1][-1] += 1
        print(deltas)
        print(self.zs)

    def compute_sigmas(self):
        sigmas = []
        for size in self.sizes:
            print(size)

    def compute_matrix_backprop(self, label, xs, ys, alpha=1):
        """
        taken heavily from https://sudeepraja.github.io/Neural

        label : ArrayLike
            The thing we are supposed to be predicting
        xs : list[ArrayLike]
            the output at each stage, with the first element being the input
            and each subsequent one being the result of a weight multiplication
            and a nonlinearity. Thereby the last one is the output
        ys : list[ArrayLike]
            the intermediate step of each of the first n-1 xs being multiplied
            by the associated weight matrix. Each of the last n-1 elements in
            xs is the same size as the coresponding element in ys
        alpha : number
            The learning rate
        """

        sigmas = [None] * len(ys)  # fill each element with None
        nonlinear_prime = self.layers[-1].nonlinearity_prime(ys[-1])
        error = (xs[-1] - label)
        sigma = error * nonlinear_prime
        sigmas[-1] = sigma
        for i in range(len(ys) - 2, -1, -1):
            recuresive_part = np.matmul(np.transpose(self.layers[i+1].weights),
                                        sigmas[i+1])
            # the xs[i-1] is removed because that list is longer
            mult_part = np.matmul(self.layers[i].weights, xs[i])
            nonlinear_part = self.layers[i].nonlinearity_prime(mult_part)
            sigma = np.multiply(recuresive_part, nonlinear_part)  # elementwise
            sigmas[i] = sigma

        # do the gradient calculations and updates
        for i in range(len(ys)):
            sigma = np.asarray(sigmas[i])
            sigma = np.expand_dims(sigma, axis=0)
            sigma_transpose = np.transpose(sigma)
            x = np.asarray(xs[i])
            x = np.expand_dims(x, axis=0)
            # compute the gradient matrix
            grad_matrix = np.matmul(sigma_transpose, x)
            if grad_matrix.shape != self.layers[i].weights.shape:
                raise AssertionError(
                    "gradients are not the same size as the weights")
            # perform the update
            self.layers[i].weights -= alpha * grad_matrix

    def train(self, features, labels, num_iterations=100, alpha=0.001, vis=False):
        all_losses = []
        for iteration in range(num_iterations):
            loss = 0
            for (feature, label) in zip(features, labels):
                pred, xs, ys = self.forward(feature)
                if len(pred) != 1:
                    raise Exception(
                        "The length of the prediction was expected to be one")
                loss += 1/2.0*(pred[0] - label)**2
                self.compute_matrix_backprop(label, xs, ys, alpha)
            average_loss = loss / num_iterations
            all_losses.append(average_loss)
            print("iteration : {}, average loss : {}".format(
                iteration, average_loss))

            if vis:
                plt.scatter(iteration, average_loss, c='b')
                plt.pause(0.1)
        plt.plot(all_losses)

    def score(self, features, labels):
        """
        assumes that labels are [-1, 1]
        """
        preds = self.pred(features)
        preds = np.squeeze(np.asarray(preds))
        preds = (preds > 0).astype(int) * 2 - 1
        equal = preds == labels
        accuracy = np.sum(equal) / equal.shape[0]
        print("The accuracy was {}".format(accuracy))

        pdb.set_trace()

    def pred(self, features):
        preds = []
        for feature in features:
            pred, _, _ = self.forward(feature)
            preds.append(pred)
        return preds

    def write_out(self, filename="output.txt"):
        with open(filename, "w") as outfile:
            num_hidden = self.layers[0].weights.shape[0]
            outfile.write("{}\n".format(num_hidden))
            print("Writing weights to {}".format(filename))
            print("The shape of the output layer is {} and the input weights are {}".format(
                self.layers[1].weights.shape, self.layers[0].weights.shape))
            np.savetxt(
                outfile,
                self.layers[1].weights,
                fmt="%.5f",
                delimiter=" ")
            np.savetxt(
                outfile,
                self.layers[0].weights,
                fmt="%.5f",
                delimiter=" ")


def read_data(values_file, labels_file):
    features = pd.read_csv(values_file, sep=' ',
                           header=None)  # read in the data
    features = features.dropna(axis=1).values  # drop the nan column
    labels = pd.read_csv(labels_file, sep=' ', header=None)
    labels = np.squeeze(labels.values)
    return features, labels


(features, labels) = read_data(u"/home/david/dev/ClarksonCompLearning/data/train_data.txt",
                               u"/home/david/dev/ClarksonCompLearning/data/train_label.txt")
net = NN([1000, 200, 1])
net.train(features, labels, 2000, 0.001)
net.score(features, labels)
net.write_out()
