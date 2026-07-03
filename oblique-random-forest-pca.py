import numpy as np
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.decomposition import PCA
import pandas as pd


class Node:
    def __init__(self, w_star=None, th_star=None, left=None, right=None, label=None):
        self.w_star = w_star
        self.th_star = th_star
        self.left = left
        self.right = right
        self.label = label


class ObliqueDecisionTreeClassifier:
    def __init__(self, max_depth=None, n_projections=15, max_features="sqrt"):
        self.root = None
        self.max_depth = max_depth
        self.n_projections = n_projections
        self.max_features = max_features

    def build_tree(self, X, Y, depth=0):
        if len(np.unique(Y)) == 1 or (
            self.max_depth is not None and depth >= self.max_depth
        ):
            return Node(label=self.calculate_leaf_label(Y))

        w_star, th_star = self.get_best_split(X, Y)

        if w_star is None:
            return Node(label=self.calculate_leaf_label(Y))

        z = np.dot(X, w_star)
        left_indices = z <= th_star
        right_indices = z > th_star

        Xi_left = X[left_indices]
        Yi_left = Y[left_indices]

        Xi_right = X[right_indices]
        Yi_right = Y[right_indices]

        if len(Xi_left) == 0 or len(Xi_right) == 0:
            return Node(label=self.calculate_leaf_label(Y))

        left_subtree = self.build_tree(Xi_left, Yi_left, depth + 1)
        right_subtree = self.build_tree(Xi_right, Yi_right, depth + 1)

        return Node(
            w_star=w_star, th_star=th_star, left=left_subtree, right=right_subtree
        )

    def get_best_split(self, X, Y):
        w_star, th_star = None, None
        max_info_gain = -float("inf")

        m = X.shape[1]

        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(m)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(m)))
        elif isinstance(self.max_features, int):
            k = min(self.max_features, m)
        else:
            k = m

        for _ in range(self.n_projections):
            features = np.random.choice(m, size=k, replace=False)
            X_subset = X[:, features]

            W = np.zeros(m)

            # NOVO: Verifica se há variação nos dados antes de rodar o PCA.
            # Se a soma das variâncias das colunas sorteadas for próxima de zero,
            # os dados são constantes e o PCA falhará.
            total_variance = np.var(X_subset, axis=0).sum()

            if X_subset.shape[0] > 1 and total_variance > 1e-6:
                pca = PCA(n_components=1)
                try:
                    pca.fit(X_subset)
                    W_subset = pca.components_[0]
                except Exception:
                    W_subset = np.random.randn(k)
            else:
                # Fallback ativado: amostra insuficiente ou dados constantes
                W_subset = np.random.randn(k)

            norm = np.linalg.norm(W_subset)
            if norm > 0:
                W_subset /= norm

            W[features] = W_subset

            feature_values = X @ W

            thresholds = np.percentile(feature_values, np.arange(5, 96, 5))

            for th in thresholds:
                left_indices = feature_values <= th
                right_indices = feature_values > th

                Xi_left = X[left_indices]
                Yi_left = Y[left_indices]
                Xi_right = X[right_indices]
                Yi_right = Y[right_indices]

                if len(Xi_left) == 0 or len(Xi_right) == 0:
                    continue

                gain = self.information_gain(Y, Yi_left, Yi_right)

                if gain > max_info_gain:
                    max_info_gain = gain
                    w_star = W
                    th_star = th

        return w_star, th_star

    def entropy_calc(self, y):
        values, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return entropy(p, base=2)

    # retorna a reducao de entropia obtida por um corte (quero maximizar esse valor)
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

        feature_val = np.dot(x, tree.w_star)
        if feature_val <= tree.th_star:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)


class ObliqueRandomForest:
    def __init__(
        self,
        n_estimators=10,
        max_depth=None,
        n_projections=15,
        max_features="sqrt",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_projections = n_projections
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

            # Repassando os parâmetros para cada árvore construída
            tree = ObliqueDecisionTreeClassifier(
                max_depth=self.max_depth,
                n_projections=self.n_projections,
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
    data = np.load("data.npz")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]

    X_train_local, X_val, y_train_local, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    scaler_local = StandardScaler()
    X_train_local = scaler_local.fit_transform(X_train_local)
    X_val = scaler_local.transform(X_val)

    print("Treinando modelo na base local (80%)...")
    classifier_local = ObliqueRandomForest(
        n_estimators=10,
        max_depth=10,
        n_projections=15,
        max_features="sqrt",
    )
    classifier_local.fit(X_train_local, y_train_local)

    preds_val = classifier_local.predict(X_val)
    acc_local = accuracy_score(y_val, preds_val)
    print(f"Acurácia de Validação Local: {acc_local:.4f}\n")

    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    X_test_scaled = scaler_final.transform(X_test)

    print("Treinando modelo na base completa (100%) para submissão...")
    clf_final = ObliqueRandomForest(
        n_estimators=40,
        max_depth=10,
        n_projections=50,
        max_features="sqrt",
    )
    clf_final.fit(X_train_scaled, y_train)

    final_predictions = clf_final.predict(X_test_scaled)

    num_samples = X_test.shape[0]
    submission_df = pd.DataFrame(
        {"ID": np.arange(1, num_samples + 1), "Prediction": final_predictions}
    )

    submission_df.to_csv("submission_pca.csv", index=False)
    print("Arquivo 'submission_pca.csv' gerado com sucesso.")


main()
