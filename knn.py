import numpy as np
from collections import defaultdict, Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3, weights="uniform"):
        self.k = k
        self.weights = weights

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train)

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        return np.array([self._predict(x) for x in X_test])

    def _predict(self, x):
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]

            if self.weights == "distance":
                return self._weighted_predict(k_nearest_labels, k_indices, distances)
            else:
                return self._unweighted_predict(k_nearest_labels)

    def _unweighted_predict(self, k_nearest_labels):
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _weighted_predict(self, k_nearest_labels, k_indices, distances):
        label_weights = {}
        
        for i, label in zip(k_indices, k_nearest_labels):
            distance = distances[i]
            weight = 1 / (distance + 1e-5) 
            if label in label_weights:
                label_weights[label] += weight
            else:
                label_weights[label] = weight
        return max(label_weights, key=label_weights.get)
