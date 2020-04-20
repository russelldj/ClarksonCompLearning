# -*- coding: utf-8 -*-

from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC


svc = SVC(probability = True, kernel = 'linear')

# Load the digits dataset
digits = datasets.load_digits()
n = len(digits.images)
X = digits.images.reshape(n, -1)
X = X / 255
#X = digits.data 
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test


abc = AdaBoostClassifier(n_estimators = 50, base_estimator = svc, learning_rate = 1)
model = abc.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

