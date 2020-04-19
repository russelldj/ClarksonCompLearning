import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
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

#Reshape, normalize
n_samples = len(digits.images)
data = digits.images.reshape(n_samples, -1)
data = data /255

X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size = 0.2, random_state = 42)

"""
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred)) """

class DecisionTreeWrapper(object):
    """What this does."""

    def __init__(self, test_size = 0.2):
        self.test_size = test_size
        self.clf = tree.DecisionTreeClassifier()

    def train(self, X, y, weights = None):
        """
        Trains a decision tree on MNIST digit dataset
        X: 3D array, training set images
        y: vector, corresponding labels
        weights: vector, corresponding weights
        """
        self.clf.fit(X, y, sample_weight = weights)

    def predict(self, X):
        """
        Generates predictions from a fit decision tree.
        X: 3D array, test set images
        returns vector of predictions
        """
        return self.clf.predict(X)

my_decision_tree = DecisionTreeWrapper()

my_decision_tree.train(X = X_train, y = y_train)
y_pred = my_decision_tree.predict(X_test)

print(y_pred[:20])
print(y_test[:20])

print(my_decision_tree)
