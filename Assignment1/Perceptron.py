import pandas as pd
import numpy as np


class Perceptron:
    def __init__(self, data_file="train-a1-449.txt"):
        data = pd.read_csv(data_file, sep=' ', header=None)  # read in the data
        data = data.dropna(axis=1)  # drop the nan column
        labels = data[1024]  # Get the last row as labels
        # now remove the labels from the main dataset
        data.drop(labels=1024, axis=1, inplace=True)

        self.data = data.values
        # massage the labels into the correct format
        self.labels = labels.values == 'Y'
        self.labels = self.labels.astype(int) * 2 - 1

    def train(self):
        # step 1: Initialize the
        self.weights = np.zeros((self.data.shape[1],))
        while True:
            ret = self.find_wrong()  # See if there are any incorrect predictions
            if ret is None:  # If not, break the loop
                break

            sample, label = ret
            self.weights += sample * label  # Perform the update

    def find_wrong(self):
        """
        return the weight and label if it's wrong else false
        """
        for label, sample in zip(
                self.labels, self.data):  # iterate over all the samples
            # multiply the weights with the data
            val = np.dot(sample, self.weights)
            # get the sign by seeing if it's aboe 0 and then rescaleing
            sign = int(val > 0) * 2 - 1
            if sign != label:  # this means the prediction was incorrect
                return sample, label
        return None  # no error was found

    def validate(self):
        predicted = self.predict(self.data)
        # see if the predictions are the same as the labels
        matching = predicted == self.labels
        if np.all(matching):
            print("All the predictions matched the labels")
        else:
            print("There were inconsistent predictions")

    def predict(self, data):
        # calculate the product of each data point with the weights
        val = np.dot(data, self.weights)
        signs = (val > 0).astype(int) * 2 - 1  # take the signs and rescale
        return signs

    def print_weight(self, file="weights.txt"):
        print("Writing the data to: {}".format(file))
        np.savetxt(
            file,
            np.expand_dims(
                self.weights,
                axis=0),
            fmt="%10.5f",
            delimiter=" ")


data_file = input("Please enter the file name for the data file: ")
perceptron = Perceptron(data_file)
perceptron.train()
perceptron.validate()
perceptron.print_weight()
