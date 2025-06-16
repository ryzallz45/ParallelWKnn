import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from multiprocessing import Pool, cpu_count
import time
import matplotlib.pyplot as plt

# === Dataset Generator ===
def generate_dataset(n_samples, n_features, n_classes=3):
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=int(n_features * 0.8),
                               n_redundant=int(n_features * 0.2),
                               n_classes=n_classes, random_state=42)
    return X, y

# === Vanilla KNN ===
def knn_vanilla(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            dist = np.linalg.norm(test_point - train_point)
            distances.append((dist, y_train[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = [label for _, label in distances[:k]]
        prediction = max(set(neighbors), key=neighbors.count)
        predictions.append(prediction)
    return predictions
