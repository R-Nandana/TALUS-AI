"""
TALUS Machine Learning Engine — Custom Robust Vectorized Classifiers & Explainer
=================================================================================
Provides scikit-learn compatible classifiers (Logistic Regression, SVM, Random Forest, 
Gradient Boosting), preprocessing pipelines, evaluation metrics, and local explanation engines.
Built using NumPy, Pandas, and Joblib for high performance and total cross-platform safety.
"""

import numpy as np
import pandas as pd
import math
from typing import Dict, List, Tuple, Any, Optional


class PreprocessingPipeline:
    """
    StandardScaler and Preprocessor for tabular mine sector features.
    Fits feature statistics exclusively on training data to prevent leakage.
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.means = {}
        self.stds = {}
        self.imputers = {}
        self.fitted = False

    def fit(self, X_df: pd.DataFrame):
        X = X_df[self.feature_names].copy()
        for col in self.feature_names:
            # Impute missing values with median
            median_val = X[col].median() if not X[col].dropna().empty else 0.0
            self.imputers[col] = median_val
            clean_series = X[col].fillna(median_val)
            
            mean_val = float(clean_series.mean())
            std_val = float(clean_series.std())
            if std_val < 1e-6:
                std_val = 1.0
            self.means[col] = mean_val
            self.stds[col] = std_val
        self.fitted = True
        return self

    def transform(self, X_df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before calling transform!")
        X = X_df.copy()
        X_scaled = np.zeros((len(X), len(self.feature_names)))
        for idx, col in enumerate(self.feature_names):
            if col in X.columns:
                series = X[col].fillna(self.imputers[col]).values
            else:
                series = np.full(len(X), self.imputers[col])
            X_scaled[:, idx] = (series - self.means[col]) / self.stds[col]
        return X_scaled

    def fit_transform(self, X_df: pd.DataFrame) -> np.ndarray:
        return self.fit(X_df).transform(X_df)


# ==========================================
# METRICS FUNCTIONS
# ==========================================

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, num_classes: int = 4) -> Dict[str, float]:
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "roc_auc": 0.0}

    acc = float(np.mean(y_true == y_pred))

    precisions = []
    recalls = []
    f1s = []

    for c in range(num_classes):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    # ROC AUC One-vs-Rest calculation
    auc_scores = []
    for c in range(num_classes):
        y_binary = (y_true == c).astype(int)
        p_score = y_probs[:, c] if y_probs.shape[1] > c else np.zeros(n)
        
        # Sort by p_score
        sort_idx = np.argsort(-p_score)
        y_binary_sorted = y_binary[sort_idx]
        
        n_pos = np.sum(y_binary)
        n_neg = n - n_pos
        
        if n_pos > 0 and n_neg > 0:
            tp_cum = np.cumsum(y_binary_sorted)
            fp_cum = np.cumsum(1 - y_binary_sorted)
            tpr = tp_cum / n_pos
            fpr = fp_cum / n_neg
            
            # Trapezoidal integration for AUC
            if hasattr(np, "trapezoid"):
                auc = np.trapezoid(tpr, fpr)
            elif hasattr(np, "trapz"):
                auc = np.trapz(tpr, fpr)
            else:
                auc = np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0)
            auc_scores.append(abs(float(auc)))
        else:
            auc_scores.append(0.5)

    macro_auc = float(np.mean(auc_scores))

    return {
        "accuracy": round(acc, 4),
        "precision": round(macro_prec, 4),
        "recall": round(macro_rec, 4),
        "f1_score": round(macro_f1, 4),
        "roc_auc": round(macro_auc, 4),
    }


# ==========================================
# ML MODELS IMPLEMENTATION
# ==========================================

class MultiClassLogisticRegression:
    def __init__(self, lr: float = 0.05, n_epochs: int = 300, reg: float = 0.01):
        self.lr = lr
        self.n_epochs = n_epochs
        self.reg = reg
        self.W = None
        self.b = None
        self.num_classes = 4

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int = 4):
        self.num_classes = num_classes
        n_samples, n_features = X.shape
        self.W = np.random.randn(n_features, num_classes) * 0.01
        self.b = np.zeros(num_classes)

        # One-hot encode targets
        Y_onehot = np.zeros((n_samples, num_classes))
        for i, val in enumerate(y):
            Y_onehot[i, int(val)] = 1.0

        for epoch in range(self.n_epochs):
            logits = np.dot(X, self.W) + self.b
            probs = self._softmax(logits)

            dw = (1 / n_samples) * np.dot(X.T, (probs - Y_onehot)) + self.reg * self.W
            db = (1 / n_samples) * np.sum(probs - Y_onehot, axis=0)

            self.W -= self.lr * dw
            self.b -= self.lr * db
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.dot(X, self.W) + self.b
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def feature_importances(self) -> np.ndarray:
        # Mean absolute weight per feature across classes
        return np.mean(np.abs(self.W), axis=1)


class SupportVectorMachineOvR:
    def __init__(self, C: float = 1.0, n_epochs: int = 250, lr: float = 0.01):
        self.C = C
        self.n_epochs = n_epochs
        self.lr = lr
        self.weights = []
        self.biases = []
        self.num_classes = 4

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int = 4):
        self.num_classes = num_classes
        n_samples, n_features = X.shape
        self.weights = []
        self.biases = []

        for c in range(num_classes):
            w = np.zeros(n_features)
            b = 0.0
            y_binary = np.where(y == c, 1.0, -1.0)

            for epoch in range(1, self.n_epochs + 1):
                eta = self.lr / (1.0 + 0.01 * epoch)
                for i in range(n_samples):
                    margin = y_binary[i] * (np.dot(X[i], w) + b)
                    if margin < 1.0:
                        w = w - eta * (w / self.C - y_binary[i] * X[i])
                        b = b + eta * y_binary[i]
                    else:
                        w = w - eta * (w / self.C)

            self.weights.append(w)
            self.biases.append(b)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        scores = np.zeros((n_samples, self.num_classes))
        for c in range(self.num_classes):
            scores[:, c] = np.dot(X, self.weights[c]) + self.biases[c]
        # Softmax calibration for probabilities
        exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def feature_importances(self) -> np.ndarray:
        W_arr = np.array(self.weights)
        return np.mean(np.abs(W_arr), axis=0)


class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 6, min_samples_split: int = 4):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.num_classes = 4

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y.astype(int), minlength=self.num_classes)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> DecisionNode:
        n_samples, n_features = X.shape
        counts = np.bincount(y.astype(int), minlength=self.num_classes)
        node_probs = counts / max(1, n_samples)

        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return DecisionNode(value=node_probs)

        # Feature sub-sampling for trees
        feat_sub = np.random.choice(n_features, max(1, int(math.sqrt(n_features))), replace=False)
        
        best_gini = self._gini(y)
        best_feat, best_thresh = None, None
        best_left_idx, best_right_idx = None, None

        for f_idx in feat_sub:
            thresholds = np.percentile(X[:, f_idx], np.linspace(10, 90, 5))
            for thresh in thresholds:
                left_mask = X[:, f_idx] <= thresh
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                weighted_gini = (np.sum(left_mask) / n_samples) * left_gini + (np.sum(right_mask) / n_samples) * right_gini

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feat = f_idx
                    best_thresh = thresh
                    best_left_idx = left_mask
                    best_right_idx = right_mask

        if best_feat is None:
            return DecisionNode(value=node_probs)

        left_child = self._build_tree(X[best_left_idx], y[best_left_idx], depth + 1)
        right_child = self._build_tree(X[best_right_idx], y[best_right_idx], depth + 1)
        return DecisionNode(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int = 4):
        self.num_classes = num_classes
        self.root = self._build_tree(X, y, 0)
        return self

    def _predict_row(self, node: DecisionNode, x: np.ndarray) -> np.ndarray:
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(self.root, row) for row in X])


class RandomForestClassifierModel:
    def __init__(self, n_estimators: int = 15, max_depth: int = 6, min_samples_split: int = 4):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.num_classes = 4
        self.feat_importances = None

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int = 4):
        self.num_classes = num_classes
        n_samples, n_features = X.shape
        self.trees = []
        feat_counts = np.zeros(n_features)

        for _ in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[boot_idx], y[boot_idx]

            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_boot, y_boot, num_classes=num_classes)
            self.trees.append(tree)

            # Count feature usage for importance
            def accumulate_importance(node):
                if node and node.feature is not None:
                    feat_counts[node.feature] += 1.0
                    accumulate_importance(node.left)
                    accumulate_importance(node.right)
            accumulate_importance(tree.root)

        total = np.sum(feat_counts)
        self.feat_importances = feat_counts / total if total > 0 else np.ones(n_features) / n_features
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        tree_probs = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(tree_probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def feature_importances(self) -> np.ndarray:
        return self.feat_importances


class GradientBoostingClassifierModel:
    def __init__(self, n_estimators: int = 15, lr: float = 0.1, max_depth: int = 4):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.trees = []
        self.num_classes = 4
        self.base_probs = None
        self.feat_importances = None

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: int = 4):
        self.num_classes = num_classes
        n_samples, n_features = X.shape

        counts = np.bincount(y.astype(int), minlength=num_classes)
        self.base_probs = counts / max(1, n_samples)

        # Build boosted decision tree ensemble per class
        self.trees = []
        feat_counts = np.zeros(n_features)

        # Train weak trees targeting residuals
        for class_idx in range(num_classes):
            y_binary = (y == class_idx).astype(int)
            class_trees = []
            
            for stage in range(self.n_estimators):
                # Calculate current predictions
                if stage == 0:
                    current_p = np.full(n_samples, self.base_probs[class_idx])
                else:
                    raw_logits = np.zeros(n_samples)
                    for t in class_trees:
                        raw_logits += self.lr * (t.predict_proba(X)[:, 1] - 0.5)
                    current_p = 1.0 / (1.0 + np.exp(-raw_logits))
                
                residuals = y_binary - current_p
                # Fit decision tree on residuals
                pseudo_y = np.where(residuals > 0, 1, 0)
                tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=4)
                tree.fit(X, pseudo_y, num_classes=2)
                class_trees.append(tree)

                def acc_imp(node):
                    if node and node.feature is not None:
                        feat_counts[node.feature] += 1.0
                        acc_imp(node.left)
                        acc_imp(node.right)
                acc_imp(tree.root)

            self.trees.append(class_trees)

        total = np.sum(feat_counts)
        self.feat_importances = feat_counts / total if total > 0 else np.ones(n_features) / n_features
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        scores = np.zeros((n_samples, self.num_classes))

        for class_idx in range(self.num_classes):
            raw_scores = np.full(n_samples, np.log(max(1e-4, self.base_probs[class_idx])))
            for tree in self.trees[class_idx]:
                raw_scores += self.lr * (tree.predict_proba(X)[:, 1] - 0.5)
            scores[:, class_idx] = raw_scores

        exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def feature_importances(self) -> np.ndarray:
        return self.feat_importances


# ==========================================
# EXPLAINABLE AI ENGINE (SHAP / CONTRIBUTION)
# ==========================================

class ModelExplainer:
    """
    Computes local feature contribution scores for any prediction instance X_single.
    Calculates impact of each feature relative to baseline average features.
    """
    def __init__(self, model, preprocessor: PreprocessingPipeline, feature_names: List[str], background_X: np.ndarray):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.background_X = background_X
        self.baseline_probs = np.mean(self.model.predict_proba(background_X), axis=0)

    def explain_instance(self, X_single_scaled: np.ndarray, pred_class: int) -> List[Dict[str, Any]]:
        n_features = len(self.feature_names)
        contributions = []
        
        # Get baseline prediction for target class
        base_val = float(self.baseline_probs[pred_class])
        
        # Get actual model feature importances
        if hasattr(self.model, 'feature_importances'):
            imp = self.model.feature_importances()
        elif hasattr(self.model, 'feature_importances_'):
            imp = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            coef = np.asarray(self.model.coef_)
            imp = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        else:
            imp = np.ones(n_features) / n_features
            
        # Perturbation marginal impact analysis
        bg_mean = np.mean(self.background_X, axis=0)
        
        for idx, feat in enumerate(self.feature_names):
            x_perturbed = X_single_scaled.copy()
            x_perturbed[0, idx] = bg_mean[idx]  # Neutralize feature to baseline mean
            
            perturbed_prob = float(self.model.predict_proba(x_perturbed)[0, pred_class])
            current_prob = float(self.model.predict_proba(X_single_scaled)[0, pred_class])
            
            # Marginal impact on target class probability
            delta = current_prob - perturbed_prob
            
            # Feature value in original unscaled space
            unscaled_val = X_single_scaled[0, idx] * self.preprocessor.stds[feat] + self.preprocessor.means[feat]
            
            contributions.append({
                "feature": feat,
                "value": round(float(unscaled_val), 3),
                "impact_score": round(float(delta), 4),
                "importance_weight": round(float(imp[idx]), 4),
                "direction": "increases_risk" if delta > 0 else "decreases_risk"
            })
            
        # Sort by absolute impact score descending
        contributions.sort(key=lambda x: abs(x["impact_score"]), reverse=True)
        return contributions
