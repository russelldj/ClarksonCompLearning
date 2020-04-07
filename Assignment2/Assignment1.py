import numpy as np
from numpy.random import normal
import pdb

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
            self.nonliniarity = sigmoid
            self.nonliniarity_prime = sigmoid_prime
        if activation == "tanh":
            self.nonliniarity = tanh
            self.nonlinearity_prime = sigmoid_prime
        if activation == "relu":
            self.nonliniarity = None

    def __str__(self):
        return "Layer with {} activation".format(self.activation)

    def forward(self, x):
        if self.weights.shape[1] != len(x):
            raise Exception("The size of the input was {} when it should have been {}".format(self.weights.shape, len(x)))

        y = np.matmul(self.weights, x)
        z = self.nonliniarity(y)
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
        layers.append(layer(sizes[-1], sizes[-2], 'sigmoid'))
        return layers

    def forward(self, x):
        ys = []
        for layer in self.layers:
            y, x = layer.forward(x)
            ys.append(y)
        return x, ys

    def compute_backprop(self):
        deltas = []
        for i, size in enumerate(self.sizes[-2::-1]):
            deltas.append([]) # add the new layer
            num_uses_as_input = self.sizes[-(i + 1)]
            layer = self.layers[-(i + 1)]
            print(layer, num_uses_as_input, i, size)
            for j in range(size): # iterate over the current neurons
                deltas[-1].append(0)
                for k in range(num_uses_as_input): # this is the summation
                    deltas[-1][-1] += 1
        print(deltas)
        print(self.zs)

    def compute_sigmas(self):
        sigmas = []
        for size in self.sizes:
            print(size)

    def train(self, features, labels):
        for (feature, y) in zip(features, labels):
            y_pred, self.zs = self.forward(feature)
            if len(y_pred) != 1:
                raise Exception("The length of the prediction was expected to be one")
            loss = 1/2.0*(y_pred[0] - y)**2
            print("y_pred : {}, y : {}, loss : {}".format(y_pred, y, loss))
        self.compute_backprop()

net = NN([3,5,8,1])
net.train([[1, 1, 1],[2,2,2]], [0.5,-0.5])
