import time
import numpy as np
import pandas as pd
import warnings
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier # usada para comparaçao com a obliqua

def main():
    np.random.seed(67)
    start_time = time.time()

    compare_with_traditional()
    # X, y = make_classification(n_samples=1000, n_features=34, n_informative=20, n_classes=3, random_state=67)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # model = ObliqueRandomForest(
    #     n_estimators=50,
    #     max_depth=10,
    #     n_projections=50,
    #     max_features="sqrt",
    # )

    # model.fit(X_train, y_train)
    
    # y_hat = model.predict(X_test)

    # acc = accuracy_score(y_test, y_hat)
    # print(f"Accuracy: {acc}")

    # end_time = time.time()
    # print(f"\nTempo de execucao: {end_time - start_time:.2f} segundos")


    # data = np.load("data.npz")
    # X_train = data["X_train"]
    # y_train = data["y_train"]
    # X_test = data["X_test"]

    # clf_final = ObliqueRandomForest(
    #     n_estimators=50,
    #     max_depth=10,
    #     n_projections=50,
    #     max_features="sqrt",
    # )
    # clf_final.fit(X_train_scaled, y_train)
    # final_predictions = clf_final.predict(X_test_scaled)

    # num_samples = X_test.shape[0]
    # submission_df = pd.DataFrame(
    #     {"ID": np.arange(1, num_samples + 1), "Prediction": final_predictions}
    # )
    # submission_df.to_csv("submission_pls.csv", index=False)
    # print("Arquivo 'submission_pls.csv' gerado com sucesso.")

def compare_with_traditional():
    for n_samples in [200, 500, 1000]:
        print(f"--- Testando para {n_samples} amostras ---")

        X, y = make_classification(
            n_samples=n_samples,
            n_features=34,
            n_informative=20,
            n_classes=3,
            random_state=67,
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=67
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        start_time_orf = time.time()
        orf = ObliqueRandomForest(
            n_estimators=50,
            max_depth=10,
            n_projections=50,
            max_features="sqrt",
        )
        orf.fit(X_train, y_train)
        preds_orf = orf.predict(X_test)
        acc_orf = accuracy_score(y_test, preds_orf)
        end_time_orf = time.time()

        start_time_rf = time.time()
        rf = RandomForestClassifier(
            n_estimators=50, max_depth=10, max_features="sqrt", random_state=67
        )
        rf.fit(X_train, y_train)
        preds_rf = rf.predict(X_test)
        acc_rf = accuracy_score(y_test, preds_rf)
        end_time_rf = time.time()

        print(
            f"Oblique Random Forest - Acuracia: {acc_orf:.4f} | Tempo: {end_time_orf - start_time_orf:.4f}s"
        )
        print(
            f"Traditional Random Forest - Acuracia: {acc_rf:.4f} | Tempo: {end_time_rf - start_time_rf:.4f}s\n"
        )


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

    # Gera n_projections direcoes via PLS (em subconjuntos de features) e
    # retorna a combinacao de vetor e limiar com maior ganho de informacao.
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

        # PLS precisa de uma saida numerica. Se multiclasse, usamos one-hot.
        classes = np.unique(Y)
        if len(classes) > 2:
            Y_target = np.eye(len(classes))[np.searchsorted(classes, Y)]
        else:
            Y_target = (Y == classes[-1]).astype(float).reshape(-1, 1)

        for _ in range(self.n_projections):
            features = np.random.choice(m, size=k, replace=False)
            X_subset = X[:, features]

            W = np.zeros(m)

            # Verifica se ha amostras/variacao suficiente antes de rodar o PLS.
            # Com poucas amostras (nos profundos da arvore) ou colunas quase
            # constantes, o ajuste do PLS fica degenerado (score com variancia
            # zero), o que gera divisao por zero (0/0) e pesos NaN.
            total_variance = np.var(X_subset, axis=0).sum()
            min_samples_for_pls = k + 2  # regra pratica para evitar ajuste degenerado

            if X_subset.shape[0] >= min_samples_for_pls and total_variance > 1e-6:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        pls = PLSRegression(n_components=1)
                        pls.fit(X_subset, Y_target)
                        W_subset = pls.x_weights_[:, 0]
                    if not np.all(np.isfinite(W_subset)):
                        W_subset = np.random.randn(k)
                except Exception:
                    W_subset = np.random.randn(k)
            else:
                # Fallback: amostra insuficiente ou dados constantes
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

    # Retorna a reducao de entropia obtida por um corte (queremos maximizar esse valor).
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

main()
