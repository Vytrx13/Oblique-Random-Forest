import time
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter


class Node:
    def __init__(self, feature_index=None, th_star=None, left=None, right=None, label=None):
        self.feature_index = feature_index 
        self.th_star = th_star            
        self.left = left
        self.right = right
        self.label = label


class OrthogonalDecisionTreeClassifier:
    def __init__(self, max_depth=None, max_features=None):
        self.root = None
        self.max_depth = max_depth
        self.max_features = max_features

    def build_tree(self, X, Y, depth=0):
        if len(np.unique(Y)) == 1 or (
            self.max_depth is not None and depth >= self.max_depth
        ):
            return Node(label=self.calculate_leaf_label(Y))

        feature_index, th_star = self.get_best_split(X, Y)

        if feature_index is None:
            return Node(label=self.calculate_leaf_label(Y))

        left_indices = X[:, feature_index] <= th_star
        right_indices = X[:, feature_index] > th_star

        Xi_left = X[left_indices]
        Yi_left = Y[left_indices]
        Xi_right = X[right_indices]
        Yi_right = Y[right_indices]

        if len(Xi_left) == 0 or len(Xi_right) == 0:
            return Node(label=self.calculate_leaf_label(Y))

        left_subtree = self.build_tree(Xi_left, Yi_left, depth + 1)
        right_subtree = self.build_tree(Xi_right, Yi_right, depth + 1)

        return Node(
            feature_index=feature_index,
            th_star=th_star,
            left=left_subtree,
            right=right_subtree,
        )

    def get_best_split(self, X, Y):
        feature_star, th_star = None, None
        max_info_gain = -float("inf")
        m = X.shape[1]

        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(m)))
            features_to_try = np.random.choice(m, size=k, replace=False)
        elif self.max_features == "log2":
            k = max(1, int(np.log2(m)))
            features_to_try = np.random.choice(m, size=k, replace=False)
        elif isinstance(self.max_features, int):
            k = min(self.max_features, m)
            features_to_try = np.random.choice(m, size=k, replace=False)
        else:
            features_to_try = np.arange(m)

        for j in features_to_try:
            thresholds = np.unique(X[:, j])

            for th in thresholds:
                left_indices = X[:, j] <= th
                right_indices = X[:, j] > th

                Xi_left = X[left_indices]
                Yi_left = Y[left_indices]
                Xi_right = X[right_indices]
                Yi_right = Y[right_indices]

                if len(Xi_left) == 0 or len(Xi_right) == 0:
                    continue

                gain = self.information_gain(Y, Yi_left, Yi_right)

                if gain > max_info_gain:
                    max_info_gain = gain
                    feature_star = j
                    th_star = th

        return feature_star, th_star

    def entropy_calc(self, y):
        values, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return entropy(p, base=2)

    def information_gain(self, parent, l_child, r_child):
        total_parent = len(parent)
        weight_l = len(l_child) / total_parent
        weight_r = len(r_child) / total_parent

        gain = self.entropy_calc(parent) - (
            weight_l * self.entropy_calc(l_child)
            + weight_r * self.entropy_calc(r_child)
        )
        return gain

    def calculate_leaf_label(self, Y):
        values, counts = np.unique(Y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X, Y):
        self.root = self.build_tree(X, Y)

    def predict(self, X):
        predictions = []
        for x in X:
            y_hat = self.make_prediction(x, self.root)
            predictions.append(y_hat)
        return np.array(predictions)

    def make_prediction(self, x, tree):
        if tree.label is not None:
            return tree.label

        if x[tree.feature_index] <= tree.th_star:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)


class OrthogonalRandomForest:
    def __init__(self, n_estimators=10, max_depth=None, max_features="sqrt"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)

            tree = OrthogonalDecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self.max_features,
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        predictions = predictions.T

        final_predictions = []
        for sample_predictions in predictions:
            vote = Counter(sample_predictions).most_common(1)[0][0]
            final_predictions.append(vote)

        return np.array(final_predictions)


def main():
    np.random.seed(67)
    start_time = time.time()
    X, y = make_classification(
        n_samples=1000, n_features=34, n_informative=20, n_classes=3, random_state=67
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = OrthogonalRandomForest(
        n_estimators=50,
        max_depth=10,
        max_features="sqrt",
    )

    model.fit(X_train, y_train)

    y_hat = model.predict(X_test)

    acc = accuracy_score(y_test, y_hat)
    print(f"Accuracy: {acc}")

    end_time = time.time()
    print(f"\nTempo de execucao: {end_time - start_time:.2f} segundos")

if __name__ == "__main__":
    main()