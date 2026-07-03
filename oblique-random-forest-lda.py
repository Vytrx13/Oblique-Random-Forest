import time
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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
        self,
        max_depth=None,
        n_projections=15,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=None,
        lda_shrinkage="auto",
        random_state=None,
    ):
        self.root = None
        self.max_depth = max_depth
        self.n_projections = n_projections
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.lda_shrinkage = lda_shrinkage
        self.random_state = random_state

    def fit(self, X, Y, classes=None):
        X = np.asarray(X)
        Y = np.asarray(Y)
        self.classes_ = np.unique(Y) if classes is None else np.asarray(classes)
        self.rng_ = np.random.default_rng(self.random_state)

        if self.class_weight == "balanced":
            inverse = np.searchsorted(self.classes_, Y)
            counts = np.bincount(inverse, minlength=len(self.classes_))
            n_samples = len(Y)
            n_classes = len(self.classes_)
            class_w = n_samples / (n_classes * np.maximum(counts, 1))
            sample_weight = class_w[inverse]
        else:
            sample_weight = np.ones(len(Y))

        self.root = self.build_tree(X, Y, sample_weight, depth=0)
        return self

    def build_tree(self, X, Y, sample_weight, depth=0):
        if (
            len(np.unique(Y)) == 1
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(Y) < self.min_samples_split
        ):
            return self._make_leaf(Y, sample_weight)

        w_star, th_star = self.get_best_split(X, Y, sample_weight)

        if w_star is None:
            return self._make_leaf(Y, sample_weight)

        z = X @ w_star
        left_mask = z <= th_star
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return self._make_leaf(Y, sample_weight)

        left_subtree = self.build_tree(
            X[left_mask], Y[left_mask], sample_weight[left_mask], depth + 1
        )
        right_subtree = self.build_tree(
            X[right_mask], Y[right_mask], sample_weight[right_mask], depth + 1
        )

        return Node(
            w_star=w_star, th_star=th_star, left=left_subtree, right=right_subtree
        )

    def _make_leaf(self, Y, sample_weight):
        inverse = np.searchsorted(self.classes_, Y)
        weighted_counts = np.bincount(
            inverse, weights=sample_weight, minlength=len(self.classes_)
        )
        total = weighted_counts.sum()
        proba = (
            weighted_counts / total
            if total > 0
            else np.ones(len(self.classes_)) / len(self.classes_)
        )
        label = self.classes_[np.argmax(proba)]
        return Node(label=label, proba=proba)

    def _n_features_to_use(self, m):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(m)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(m)))
        elif isinstance(self.max_features, int):
            return min(self.max_features, m)
        return m

    def get_best_split(self, X, Y, sample_weight):
        w_star, th_star = None, None
        max_info_gain = -float("inf")

        m = X.shape[1]
        k = self._n_features_to_use(m)
        parent_entropy = self.entropy_calc(
            Y, sample_weight
        ) 

        for _ in range(self.n_projections):
            features = self.rng_.choice(m, size=k, replace=False)
            X_subset = X[:, features]

            W = np.zeros(m)

            classes_in_node, class_counts = np.unique(Y, return_counts=True)
            variances = np.var(X_subset, axis=0)

            if (
                len(classes_in_node) > 1
                and np.all(variances > 1e-6)
                and np.min(class_counts) >= 2
            ):
                lda = LinearDiscriminantAnalysis(
                    solver="eigen", shrinkage=self.lda_shrinkage
                )
                try:
                    lda.fit(X_subset, Y)
                    n_valid = len(lda.explained_variance_ratio_)
                    comp_idx = int(self.rng_.integers(0, n_valid)) if n_valid > 1 else 0
                    W_subset = lda.scalings_[:, comp_idx]

                    if np.isnan(W_subset).any() or np.isinf(W_subset).any():
                        W_subset = self.rng_.standard_normal(k)
                except Exception:
                    W_subset = self.rng_.standard_normal(k)
            else:
                W_subset = self.rng_.standard_normal(k)

            norm = np.linalg.norm(W_subset)
            if norm > 0:
                W_subset /= norm

            W[features] = W_subset

            feature_values = X @ W
            thresholds = np.percentile(feature_values, np.arange(5, 96, 5))

            for th in thresholds:
                left_mask = feature_values <= th
                right_mask = ~left_mask

                n_left = left_mask.sum()
                n_right = right_mask.sum()

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gain = self.information_gain(
                    sample_weight, Y, left_mask, right_mask, parent_entropy
                )

                if gain > max_info_gain:
                    max_info_gain = gain
                    w_star = W
                    th_star = th

        return w_star, th_star

    def entropy_calc(self, y, sample_weight):
        _, inverse = np.unique(y, return_inverse=True)
        weighted_counts = np.bincount(inverse, weights=sample_weight)
        total = weighted_counts.sum()
        if total <= 0:
            return 0.0
        p = weighted_counts / total
        return entropy(p, base=2)

    def information_gain(self, sample_weight, Y, left_mask, right_mask, parent_entropy):
        w_left = sample_weight[left_mask]
        w_right = sample_weight[right_mask]
        total_w = sample_weight.sum()

        weight_l = w_left.sum() / total_w
        weight_r = w_right.sum() / total_w

        ent_l = self.entropy_calc(Y[left_mask], w_left)
        ent_r = self.entropy_calc(Y[right_mask], w_right)

        return parent_entropy - (weight_l * ent_l + weight_r * ent_r)

    def predict_proba(self, X):
        X = np.asarray(X)
        return np.array([self._proba_single(x, self.root) for x in X])

    def _proba_single(self, x, node):
        if node.label is not None:
            return node.proba
        z = np.dot(x, node.w_star)
        if z <= node.th_star:
            return self._proba_single(x, node.left)
        return self._proba_single(x, node.right)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class ObliqueRandomForest:
    def __init__(
        self,
        n_estimators=10,
        max_depth=None,
        n_projections=15,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=None,
        lda_shrinkage="auto",
        random_state=None,
        oob_score=False,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_projections = n_projections
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.lda_shrinkage = lda_shrinkage
        self.random_state = random_state
        self.oob_score = oob_score
        self.trees = []
        self.oob_score_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)

        self.trees = []
        oob_proba_sum = np.zeros((n_samples, len(self.classes_)))
        oob_count = np.zeros(n_samples)

        for _ in range(self.n_estimators):
            tree_seed = int(rng.integers(0, 2**32 - 1))
            boot_rng = np.random.default_rng(tree_seed)
            idx = boot_rng.integers(0, n_samples, size=n_samples)

            tree = ObliqueDecisionTreeClassifier(
                max_depth=self.max_depth,
                n_projections=self.n_projections,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                class_weight=self.class_weight,
                lda_shrinkage=self.lda_shrinkage,
                random_state=tree_seed,
            )
            tree.fit(X[idx], y[idx], classes=self.classes_)
            self.trees.append(tree)

            if self.oob_score:
                oob_mask = np.ones(n_samples, dtype=bool)
                oob_mask[idx] = False
                if oob_mask.any():
                    proba_oob = tree.predict_proba(X[oob_mask])
                    oob_proba_sum[oob_mask] += proba_oob
                    oob_count[oob_mask] += 1

        if self.oob_score:
            has_oob = oob_count > 0
            if has_oob.any():
                oob_pred = self.classes_[np.argmax(oob_proba_sum[has_oob], axis=1)]
                self.oob_score_ = float(accuracy_score(y[has_oob], oob_pred))
            else:
                self.oob_score_ = None
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        proba_sum = np.zeros((X.shape[0], len(self.classes_)))
        for tree in self.trees:
            proba_sum += tree.predict_proba(X)
        return proba_sum / len(self.trees)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def main():
    start_time = time.time()

    data = np.load("data.npz")

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]

    X_train_local, X_val, y_train_local, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
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
        min_samples_leaf=1,
        # class_weight="balanced",
        lda_shrinkage="auto",
        random_state=42,
        oob_score=True,
    )
    classifier_local.fit(X_train_local, y_train_local)

    preds_val = classifier_local.predict(X_val)
    acc_local = accuracy_score(y_val, preds_val)

    print(f"Acuracia de Validacao Local:  {acc_local:.4f}")
    if classifier_local.oob_score_ is not None:
        print(f"Acuracia OOB (no treino):     {classifier_local.oob_score_:.4f}")
        
    print("\nMatriz de confusao (validacao):")
    print(confusion_matrix(y_val, preds_val))

    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    X_test_scaled = scaler_final.transform(X_test)

    print("Treinando modelo na base completa (100%) para submissao...")
    clf_final = ObliqueRandomForest(
        n_estimators=50,
        max_depth=10,
        n_projections=50,
        max_features="sqrt",
        min_samples_leaf=1,
        # class_weight="balanced",
        lda_shrinkage="auto",
        random_state=42,
    )
    clf_final.fit(X_train_scaled, y_train)
    
    final_predictions = clf_final.predict(X_test_scaled)
    
    num_samples = X_test.shape[0]
    submission_df = pd.DataFrame(
        {"ID": np.arange(1, num_samples + 1), "Prediction": final_predictions}
    )
    
    submission_df.to_csv("submission_lda.csv", index=False)
    print("Arquivo 'submission_lda.csv' gerado com sucesso.")

    end_time = time.time()
    print(f"\nTempo de execucao: {end_time - start_time:.2f} segundos")


if __name__ == "__main__":
    main()