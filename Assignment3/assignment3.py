import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import tree
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import pdb
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause


# Load the digits dataset
digits = datasets.load_digits()

# Display the first digit
#plt.figure(1, figsize=(3, 3))
##plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()


# Reshape, normalize
n_samples = len(digits.images)
data = digits.images.reshape(n_samples, -1)
data = data / 255

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.2, random_state=42)

"""
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred)) """


class DecisionTreeWrapper(object):
    """What this does."""

    def __init__(self, test_size=0.2):
        self.test_size = test_size
        self.clf = tree.DecisionTreeClassifier()

    def fit(self, X, y, weights=None):
        """
        Trains a decision tree on MNIST digit dataset
        X: 3D array, training set images
        y: vector, corresponding labels
        weights: vector, corresponding weights
        """
        self.clf.fit(X, y, sample_weight=weights)

    def predict(self, X):
        """
        Generates predictions from a fit decision tree.
        X: 3D array, test set images
        returns vector of predictions
        """
        return self.clf.predict(X)


#my_decision_tree = DecisionTreeWrapper()
#
#my_decision_tree.train(X=X_train, y=y_train)
#y_pred = my_decision_tree.predict(X_test)
# print(y_pred[:20])
# print(y_test[:20])
#
# print(my_decision_tree)

# https://en.wikipedia.org/wiki/AdaBoost


class AdaBoost():
    """
    So, annoyingly, the algorithm we've been taught only works for the two
    class problem. This paper [1] https://web.stanford.edu/~hastie/Papers/samme.pdf
    seems to present an effective mutliclass algorithm
    """

    def __init__(self):
        all_data = datasets.load_digits()  # all_data is actually a dict
        features = all_data['data']
        labels = all_data['target']

        num_samples = len(features)
        features = features.reshape(num_samples, -1)  # flatten
        normalized_features = features / 255

        self.features = normalized_features  # noramalize
        self.labels = labels
        self.weights = np.ones((num_samples,)) / \
            num_samples  # n weights of 1 / n
        self.is_boolean = False
        self.alphas = None  # the weights on our classifiers
        self.hypotheses = None  # trained classifiers
        self.SMALL_NUMBER = 0.00000000000001  # to avoid division by zero

    def get_features(self):
        return self.features

    def set_data(self, features, labels):
        self.features = features
        self.labels = labels
        num_points = len(labels)
        self.weights = np.ones((num_points,)) / num_points

    def set_data_boolean(self):
        """
        transform the data so that half of the classes are now labeled -1
        and the other half are 1
        """
        unique_labels = np.unique(self.labels)
        half_num_labels = int(len(unique_labels) / 2)
        first_half_labels = unique_labels[:half_num_labels]
        # compute which samples have the first two labels
        samples_with_first_half_labels = np.isin(self.labels,
                                                 first_half_labels)
        self.labels[samples_with_first_half_labels] = 1
        self.labels[np.logical_not(samples_with_first_half_labels)] = -1
        self.is_boolean = True

    def boost(self, learner, M=1000):
        """
        learner : function : data -> label

        M : int
            the number of epochs to run for. This notation is taken from [1]
        """
        self.alphas = np.zeros((M,))
        unweighted_errors = np.zeros((M,))
        self.hypotheses = []
        for m in range(M):  # loop from 0..(M-1)
            # step a) from from paper, training the classifiers
            current_learner = clone(learner)  # avoid issues with deep copies
            current_learner.fit(self.features, self.labels, self.weights)
            epsilon, incorrect_indicator, unweighted_error = self.compute_error(
                current_learner)
            # small number is added so we don't get division by zero
            alpha = 1 / 2 * np.log((1 - epsilon) /
                                   (epsilon + self.SMALL_NUMBER))
            self.alphas[m] = alpha
            unweighted_errors[m] = unweighted_error
            self.hypotheses.append(current_learner)
            self.update_weights(incorrect_indicator, alpha)

        print("Boosting complete")
        plt.plot(unweighted_errors)
        plt.xlabel("iteration")
        plt.ylabel("The erorr rate")
        plt.show()

        self.predict_final_model(self.features)

    def predict_final_model(self, features, evaluate_accuracy=True):
        all_weighted_preds = []
        # iterate over the two lists at the same time
        for hypothesis, alpha in zip(self.hypotheses, self.alphas):
            preds = hypothesis.predict(features)
            weighted_preds = preds * alpha
            all_weighted_preds.append(weighted_preds)
        # convert to numpy array for easier math
        all_weighted_preds = np.asarray(all_weighted_preds)
        pred_sample_sum_preds = np.sum(all_weighted_preds, axis=0)
        print(pred_sample_sum_preds.shape)

        def sign(x):
            """
            acts elementwise on x
            if x > 0 -> 1
            else -> -1
            """
            is_positive = (x > 0).astype(int)
            labels = (is_positive * 2) - 1
            return labels

        pred_labels = sign(pred_sample_sum_preds)

        if evaluate_accuracy:
            correct = np.equal(pred_labels, self.labels).astype(int)
            num_correct = np.sum(correct)
            num_samples = len(self.labels)
            accuracy = num_correct / num_samples
            print("The accuracy was : {} on {} samples".format(accuracy,
                                                               num_samples))

        return pred_labels

    def compute_error(self, learner):
        """
        compute the error rate w.r.t. the current weights
        """
        preds = learner.predict(self.features)
        # comptute the locations of the errors
        incorrect_indicator = np.not_equal(preds, self.labels).astype(int)
        # elementwise weighting
        weighted_incorrect = incorrect_indicator * self.weights
        sum_errors = np.sum(weighted_incorrect)
        total_weights = np.sum(self.weights)
        epsilon = sum_errors / total_weights
        unweighted_error = np.sum(incorrect_indicator) / len(self.labels)
        return epsilon, incorrect_indicator, unweighted_error

    def update_weights(self, incorrect_indicator, alpha):
        """
        # if the samples was correctly predicted, we need to dewieght it by a
        factor of e^(-alpha)
        if incorrect, upweight by e^alpha
        """
        exponents = 2 * incorrect_indicator - 1  # map from (0, 1) -> (-1, 1)
        exponentials = np.exp(exponents * alpha)
        self.weights = self.weights * exponentials
        self.weights /= np.sum(self.weights)  # normalize


my_decision_tree = tree.DecisionTreeClassifier(
    max_depth=2)  # Try to make it worse, it was too good
#my_decision_tree = DecisionTreeWrapper()
my_adaboost = AdaBoost()
my_adaboost.set_data_boolean()
my_adaboost.boost(my_decision_tree, 200)
features = my_adaboost.get_features()
my_adaboost.predict_final_model(features)
