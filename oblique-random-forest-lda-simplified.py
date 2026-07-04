import time
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import entropy


class Node:
    def __init__(
        self, w_star=None, th_star=None, left=None, right=None, label=None, proba=None
    ):
        self.w_star = w_star
        self.th_star = th_star
        self.left = left
        self.right = right
        self.label = label
        self.proba = proba


class ObliqueDecisionTreeClassifier:
    def __init__(
        self, max_depth=None, n_projections=15, max_features="sqrt", min_samples_leaf=2
    ):
        self.root = None
        self.max_depth = max_depth
        self.n_projections = n_projections
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.lda_success_count = 0
        self.fallback_inconsistent_data_count = 0
        self.fallback_lda_exception_count = 0

    def fit(self, X, Y, classes):
        self.classes_ = np.asarray(classes)
        self.root = self.build_tree(X, Y, depth=0)

    def build_tree(self, X, Y, depth=0):
        # pureza max ou maxima profundidade => para e cria folha
        if len(np.unique(Y)) == 1 or (
            self.max_depth is not None and depth >= self.max_depth
        ):
            return self._make_leaf(Y)

        w_star, th_star = self.get_best_split(X, Y)

        if w_star is None:
            return self._make_leaf(Y)

        z = np.dot(X, w_star)
        left_mask = z <= th_star
        right_mask = ~left_mask

        # impede divisões que criam folhas muito pequenas
        if (
            left_mask.sum() < self.min_samples_leaf
            or right_mask.sum() < self.min_samples_leaf
        ):
            return self._make_leaf(Y)

        left_subtree = self.build_tree(X[left_mask], Y[left_mask], depth + 1)
        right_subtree = self.build_tree(X[right_mask], Y[right_mask], depth + 1)

        return Node(
            w_star=w_star, th_star=th_star, left=left_subtree, right=right_subtree
        )

    # determinar a classe prevista e a distr de prob das classes no nó
    def _make_leaf(self, Y):
        inverse = np.searchsorted(self.classes_, Y)
        counts = np.bincount(inverse, minlength=len(self.classes_))
        proba = counts / counts.sum()
        label = self.classes_[np.argmax(proba)]
        return Node(label=label, proba=proba)

    def get_best_split(self, X, Y):
        w_star, th_star = None, None
        max_info_gain = -float("inf")
        m = X.shape[1]

        parent_entropy = self.entropy_calc(Y)
        total_parent = len(Y)

        if isinstance(self.max_features, float):
            k = max(1, int(m * self.max_features))
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(m)))
        else:
            k = m

        for _ in range(self.n_projections):
            # sorteia algumas features
            features = np.random.choice(m, size=k, replace=False)
            X_subset = X[:, features]
            W = np.zeros(m)

            classes_in_node, class_counts = np.unique(Y, return_counts=True)

            # so usa o LDA se houver dados consistentes, senão usa projeção aleatória
            if len(classes_in_node) > 1:
                lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
                try:
                    import warnings

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        lda.fit(X_subset, Y)
                    W_subset = lda.scalings_[:, 0]
                    if np.isnan(W_subset).any() or np.isinf(W_subset).any():
                        W_subset = np.random.randn(k)
                        self.fallback_lda_exception_count += 1
                    else:
                        self.lda_success_count += 1
                except Exception:
                    W_subset = np.random.randn(k)
                    self.fallback_lda_exception_count += 1
            else:
                W_subset = np.random.randn(k)
                self.fallback_inconsistent_data_count += 1

            norm = np.linalg.norm(W_subset)
            if norm > 0:
                W_subset /= norm

            W[features] = W_subset
            feature_values = X @ W

            # testo 2%, 4%, ... 98%
            thresholds = np.percentile(feature_values, np.arange(2, 99, 2))

            for th in thresholds:
                left_mask = feature_values <= th
                right_mask = ~left_mask

                if (
                    left_mask.sum() < self.min_samples_leaf
                    or right_mask.sum() < self.min_samples_leaf
                ):
                    continue

                gain = self.information_gain(
                    parent_entropy, total_parent, Y[left_mask], Y[right_mask]
                )

                if gain > max_info_gain:
                    max_info_gain = gain
                    w_star = W
                    th_star = th

        return w_star, th_star

    def entropy_calc(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return entropy(p, base=2)

    def information_gain(self, parent_entropy, total_parent, l_child, r_child):
        weight_l = len(l_child) / total_parent
        weight_r = len(r_child) / total_parent

        gain = parent_entropy - (
            weight_l * self.entropy_calc(l_child)
            + weight_r * self.entropy_calc(r_child)
        )
        return gain

    def predict_proba(self, X):
        probabilities = []
        for sample in X:
            sample_probability = self._proba_single(sample, self.root)
            probabilities.append(sample_probability)
        return np.array(probabilities)

    # define o caaminho de uma amostra ao longo da arvore
    def _proba_single(self, sample, node):
        is_leaf_node = node.label is not None
        if is_leaf_node:
            return node.proba

        projection = np.dot(sample, node.w_star)
        goes_to_left = projection <= node.th_star

        if goes_to_left:
            return self._proba_single(sample, node.left)
        else:
            return self._proba_single(sample, node.right)


class ObliqueRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=10,
        max_depth=None,
        n_projections=15,
        max_features="sqrt",
        min_samples_leaf=2,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_projections = n_projections
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.total_lda_success = 0
        self.total_fallback_inconsistent = 0
        self.total_fallback_exception = 0

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            # bootstrap => novo conj de dados p cada arvore
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]

            tree = ObliqueDecisionTreeClassifier(
                max_depth=self.max_depth,
                n_projections=self.n_projections,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_sample, y_sample, self.classes_)
            self.trees.append(tree)

            self.total_lda_success += tree.lda_success_count
            self.total_fallback_inconsistent += tree.fallback_inconsistent_data_count
            self.total_fallback_exception += tree.fallback_lda_exception_count

        return self

    def predict(self, X):
        # soft voting
        proba_sum = np.zeros((X.shape[0], len(self.classes_)))
        for tree in self.trees:
            proba_sum += tree.predict_proba(X)

        proba_avg = proba_sum / len(self.trees)
        return self.classes_[np.argmax(proba_avg, axis=1)]


def otimizar_hiperparametros(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    param_distributions = {
        "n_estimators": [10, 50, 100],
        "max_depth": [5, 12, 20, None],
        "n_projections": [10, 30, 50],
        "max_features": [0.3, 0.5, "sqrt"],
        "min_samples_leaf": [1, 2, 5],
    }

    rf = ObliqueRandomForest()

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        random_state=42,
    )

    random_search.fit(X_train_scaled, y_train)

    print(random_search.best_params_)
    print(random_search.best_score_)

    return random_search.best_estimator_


def run_with_local_train_test_split(X_train, y_train):
    X_train_local, X_val, y_train_local, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    scaler_local = StandardScaler()
    X_train_local = scaler_local.fit_transform(X_train_local)
    X_val = scaler_local.transform(X_val)

    print("Treinando modelo na base local (80%)...")
    classifier_local = ObliqueRandomForest(
        n_estimators=5,
        max_depth=12,
        n_projections=20,
        max_features=0.5,
        min_samples_leaf=2,
    )
    classifier_local.fit(X_train_local, y_train_local)

    preds_val = classifier_local.predict(X_val)
    acc_local = accuracy_score(y_val, preds_val)

    print(f"Acuracia de Validacao Local: {acc_local:.4f}\n")
    print("Matriz de confusao (validacao):")
    print(confusion_matrix(y_val, preds_val))

    print("\nEstatisticas de Projecao (Treinamento Local):")
    print(f"Sucesso no LDA: {classifier_local.total_lda_success}")
    print(
        f"Fallback para aleatorio (Dados Inconsistentes): {classifier_local.total_fallback_inconsistent}"
    )
    print(
        f"Fallback para aleatorio (Excecao/NaN no LDA): {classifier_local.total_fallback_exception}"
    )


def run_prediction_submission(X_train, y_train, X_test, classifier):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTreinando modelo otimizado na base completa (100%) para submissao...")
    classifier.fit(X_train_scaled, y_train)

    final_predictions = classifier.predict(X_test_scaled)

    num_samples = X_test.shape[0]
    submission_df = pd.DataFrame(
        {"ID": np.arange(1, num_samples + 1), "Prediction": final_predictions}
    )

    submission_df.to_csv("submission_lda.csv", index=False)
    print("Arquivo 'submission_lda.csv' gerado com sucesso.")


def main():
    np.random.seed(67)
    start_time = time.time()
    data = np.load("data.npz")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]

    # run_with_local_train_test_split(X_train, y_train)

    # run_prediction_submission(X_train, y_train, X_test)

    print("Iniciando busca pelos melhores hiperparametros...")
    classificador_otimo = otimizar_hiperparametros(X_train, y_train)

    run_prediction_submission(X_train, y_train, X_test, classificador_otimo)

    end_time = time.time()
    print(f"\nTempo de execucao: {end_time - start_time:.2f} segundos")


if __name__ == "__main__":
    main()
