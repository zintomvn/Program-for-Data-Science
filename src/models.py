import numpy as np
from typing import Optional, Tuple

# helper function
def _add_intercept(X: np.ndarray) -> np.ndarray:
    """Thêm cột bias (intercept) vào ma trận X."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    ones = np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.hstack([ones, X])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_clip = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clip))


# Metric
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    prec = precision_score(y_true, y_pred, pos_label)
    rec = recall_score(y_true, y_pred, pos_label)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }


# Logistic regression
class LogisticRegressionNumpy:
    """
    Logistic Regression (nhị phân) dùng NumPy thuần.
    - Hàm mất mát: Binary cross-entropy + L2 (tùy chọn)
    - Tối ưu: Gradient Descent đơn giản
    """

    def __init__(
        self,
        lr: float = 0.1,
        n_iters: int = 1000,
        l2: float = 0.0,
        fit_intercept: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.lr = float(lr)
        self.n_iters = int(n_iters)
        self.l2 = float(l2)
        self.fit_intercept = bool(fit_intercept)
        self.random_state = random_state
        self.coef_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionNumpy":
        if X.ndim != 2:
            raise ValueError("X phải có shape (n_samples, n_features)")
        y = y.astype(float).reshape(-1)
        if set(np.unique(y)) - {0.0, 1.0}:
            raise ValueError("y chỉ được chứa 0/1 cho bài toán nhị phân")

        Xw = _add_intercept(X) if self.fit_intercept else X
        n_samples, n_features = Xw.shape

        rng = np.random.default_rng(self.random_state)
        self.coef_ = rng.normal(0.0, 0.01, size=(n_features,))

        for _ in range(self.n_iters):
            logits = Xw @ self.coef_
            probs = _sigmoid(logits)
            error = probs - y
            grad = (Xw.T @ error) / n_samples
            if self.l2 > 0:
                if self.fit_intercept:
                    reg = np.r_[0.0, self.coef_[1:]]
                else:
                    reg = self.coef_
                grad = grad + self.l2 * reg / n_samples
            self.coef_ -= self.lr * grad

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model chưa được fit")
        Xw = _add_intercept(X) if self.fit_intercept else X
        logits = Xw @ self.coef_
        p1 = _sigmoid(logits)
        return np.vstack([1.0 - p1, p1]).T

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


# Naive bayes
class GaussianNaiveBayesNumpy:
    """
    Gaussian Naive Bayes cho bài toán nhị phân.
    Giả định các feature tuân theo phân phối chuẩn và độc lập có điều kiện.
    """

    def __init__(self) -> None:
        self.classes_: Optional[np.ndarray] = None
        self.priors_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None  # shape (n_classes, n_features)
        self.vars_: Optional[np.ndarray] = None   # shape (n_classes, n_features)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayesNumpy":
        if X.ndim != 2:
            raise ValueError("X phải là mảng 2 chiều")
        y = y.astype(int).reshape(-1)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.means_[idx, :] = X_c.mean(axis=0)
            self.vars_[idx, :] = X_c.var(axis=0) + 1e-9  # thêm epsilon để tránh chia 0
            self.priors_[idx] = X_c.shape[0] / X.shape[0]
        
        return self

    def _calculate_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            # Log prior
            log_prior = np.log(self.priors_[idx])
            
            # Log likelihood: log p(x|c) = sum log N(x_i; mu_i, sigma_i^2)
            # log N(x; mu, sigma^2) = -0.5 * log(2*pi*sigma^2) - (x-mu)^2/(2*sigma^2)
            var = self.vars_[idx, :]
            mean = self.means_[idx, :]
            
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var))
            log_likelihood += -0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            
            log_probs[:, idx] = log_prior + log_likelihood
        
        return log_probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model chưa được fit")
        
        log_probs = self._calculate_log_likelihood(X)
        # Normalize bằng log-sum-exp trick
        log_sum = np.logaddexp.reduce(log_probs, axis=1, keepdims=True)
        probs = np.exp(log_probs - log_sum)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_probs = self._calculate_log_likelihood(X)
        return self.classes_[np.argmax(log_probs, axis=1)]


# K-Nearest Neighbors
class KNeighborsClassifierNumpy:
    """
    Euclidean để tìm k láng giềng gần nhất.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = 'uniform', 
    ) -> None:
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNeighborsClassifierNumpy":
        """Lưu trữ dữ liệu huấn luyện (lazy learning)."""
        if X.ndim != 2:
            raise ValueError("X phải là mảng 2 chiều")
        self.X_train_ = X.copy()
        self.y_train_ = y.astype(int).reshape(-1)
        self.classes_ = np.unique(self.y_train_)
        return self

    def _euclidean_distance(self, x1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum((X2 - x1) ** 2, axis=1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None:
            raise RuntimeError("Model chưa được fit")
        
        predictions = np.zeros(X.shape[0], dtype=int)
        
        for i, x in enumerate(X):
            # Tính khoảng cách đến tất cả điểm trong tập train
            distances = self._euclidean_distance(x, self.X_train_)
            
            # Tìm k láng giềng gần nhất
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = self.y_train_[k_indices]
            
            if self.weights == 'uniform':
                # Vote đơn giản
                predictions[i] = np.bincount(k_nearest_labels).argmax()
            else:  # distance-weighted
                k_distances = distances[k_indices]
                # Tránh chia cho 0
                weights = 1.0 / (k_distances + 1e-9)
                weighted_votes = np.zeros(len(self.classes_))
                for label, weight in zip(k_nearest_labels, weights):
                    weighted_votes[label] += weight
                predictions[i] = self.classes_[np.argmax(weighted_votes)]
        
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None:
            raise RuntimeError("Model chưa được fit")
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        
        for i, x in enumerate(X):
            distances = self._euclidean_distance(x, self.X_train_)
            k_indices = np.argsort(distances)[:self.n_neighbors]
            k_nearest_labels = self.y_train_[k_indices]
            
            if self.weights == 'uniform':
                for label in k_nearest_labels:
                    proba[i, label] += 1.0 / self.n_neighbors
            else:
                k_distances = distances[k_indices]
                weights = 1.0 / (k_distances + 1e-9)
                weights /= weights.sum()
                for label, weight in zip(k_nearest_labels, weights):
                    proba[i, label] += weight
        
        return proba


__all__ = [
    "LogisticRegressionNumpy",
    "GaussianNaiveBayesNumpy",
    "KNeighborsClassifierNumpy",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "classification_report",
]
