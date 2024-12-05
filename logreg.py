

import numpy as np
from collections import defaultdict, Counter

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, max_iter=1000, learning_rate=0.01, tol=1e-4, fit_intercept=True):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.coefficients = None

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.float64)

        if self.fit_intercept:
            self.X_train = np.hstack((np.ones((self.X_train.shape[0], 1)), self.X_train))

        self.coefficients = np.zeros(self.X_train.shape[1])

        for _ in range(self.max_iter):
            linear_model = np.dot(self.X_train, self.coefficients)
            print (linear_model)
            predictions = sigmoid(linear_model)
            
            difference = predictions - self.y_train
            gradient = np.dot(self.X_train.T, difference) / self.X_train.shape[0]
            
            new_coefficients = self.coefficients - self.learning_rate * gradient
            if np.linalg.norm(new_coefficients - self.coefficients, ord=1) < self.tol:
                break
            
            self.coefficients = new_coefficients


    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=np.float64)

        if self.fit_intercept:
            X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

        linear_model = np.dot(X_test, self.coefficients)

        probabilities = sigmoid(linear_model)

        predictions = (probabilities >= 0.1).astype(int)  

        return predictions

