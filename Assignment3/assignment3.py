import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import tree
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import pdb


class DecisionTreeWrapper(object):
    """What this does."""

    def __init__(self, test_size=0.2):
        self.test_size = test_size
        self.clf = tree.DecisionTreeClassifier()

    def fit(self, X, y, weights=None):
        """
        Trains a decision tree on MNIST digit dataset.

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
    So, as far as I can tell, the algorithm we were taught in class only works
    for two classes. This paper
    https://web.stanford.edu/~hastie/Papers/samme.pdf
    seems to present an effective mutliclass algorithm
    """

    def __init__(self, train_features=None, train_labels=None, num_classes=10):

        self.train_features = train_features
        if train_features is None or train_labels is None:
            digits = datasets.load_digits()
            features = digits["data"]
            labels = digits["target"]
            num_samples = len(features)
            features = features.reshape(num_samples, -1)  # flatten
            normalized_features = features / 255
            self.features = normalized_features
            self.labels = labels
        else:
            self.features = train_features  # noramalize
            self.labels = train_labels

        self.num_classes = num_classes
        num_samples = len(self.labels)
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

    def boost(self, learner, filename="ErrorRateOverTime.png", M=1000):
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
            alpha = np.log((1 - epsilon) /
                           (epsilon + self.SMALL_NUMBER)) + np.log(self.num_classes - 1)
            self.alphas[m] = alpha
            unweighted_errors[m] = unweighted_error
            self.hypotheses.append(current_learner)
            self.update_weights(incorrect_indicator, alpha)

        print("Boosting complete")
        plt.clf()
        plt.plot(unweighted_errors)
        plt.xlabel("Iteration")
        plt.ylabel("The erorr rate")
        plt.savefig(filename)
        print("figure written to {}".format(filename))
        plt.pause(2)

    def predict_final_model(self, features, labels=None):
        """
        If labels are not none, the accuracy will be computed
        """
        num_samples = len(features)
        per_class_predictions = np.zeros((num_samples, self.num_classes))
        samples_indices = np.arange(0, num_samples, 1)

        # iterate over the two lists at the same time
        for hypothesis, alpha in zip(self.hypotheses, self.alphas):
            preds = hypothesis.predict(features)
            # Assumes that class labels are distinct contigious integers
            # starting at 0
            per_class_predictions[samples_indices, preds] += alpha
        pred_labels = np.argmax(per_class_predictions, axis=1)

        if labels is not None:
            correct = np.equal(pred_labels, labels).astype(int)
            num_correct = np.sum(correct)
            num_samples = len(labels)
            accuracy = num_correct / num_samples
            print("Our implementation's accuracy was : {} on {} samples".format(accuracy,
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


# Load the digits dataset
NUM_ESTIMATORS = 500
for dataset_name in ["digits", "iris"]:
    print("Doing experiments for the {} dataset\n".format(dataset_name))
    if dataset_name == "iris":
        mydataset = datasets.load_iris()
    elif dataset_name == "digits":
        mydataset = datasets.load_digits()

    # normalize to [0, 1]
    mydataset.data /= np.max(mydataset.data)
    X_train, X_test, y_train, y_test = train_test_split(
        mydataset.data, mydataset.target, test_size=0.2, random_state=42)

    #  experiments, decision tree
    print("Decision tree tests for {}".format(dataset_name))
    decision_tree = tree.DecisionTreeClassifier(
        max_depth=2)  # Try to make it worse, it was too good
    my_adaboost = AdaBoost(X_train, y_train)
    my_adaboost.boost(decision_tree, M=NUM_ESTIMATORS,
                      filename="{}DecisionTreeErrorRate.png".format(dataset_name))
    my_adaboost.predict_final_model(X_test, y_test)

    abc = AdaBoostClassifier(
        n_estimators=NUM_ESTIMATORS, base_estimator=decision_tree, learning_rate=1)
    model = abc.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy for scikit learn: {}".format(
        metrics.accuracy_score(y_test, y_pred)))

    # svm
    print("\nSVC tests for {}".format(dataset_name))
    svc = SVC(probability=True, kernel='linear')

    my_adaboost = AdaBoost(X_train, y_train)
    my_adaboost.boost(svc, M=NUM_ESTIMATORS,
                      filename="{}SVCErrorRate.png".format(dataset_name))
    my_adaboost.predict_final_model(X_test, y_test)

    abc = AdaBoostClassifier(
        n_estimators=NUM_ESTIMATORS, base_estimator=svc, learning_rate=1)
    model = abc.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy for scikit learn: {}".format(
        metrics.accuracy_score(y_test, y_pred)))
