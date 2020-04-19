import matplotlib.pyplot as plt
from sklearn import datasets
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause


# Load the digits dataset
digits = datasets.load_digits()

# Display the first digit
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# https://en.wikipedia.org/wiki/AdaBoost


class AdaBoost():
    def __init__(self):
        data = datasets.load_digits())
        data=data.reshape(len(data), -1)  # flatten
        self.data /= 255  # noramalize


    def run(learners, T = 1000):
        """
        learners : ArrayLike[function : data -> label]
        """
        pass
