import numpy as np
from collections import Counter

def knn_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int = 3) -> np.ndarray:
    """
    对X_test用KNN分类。
    """
    preds = []
    for x in X_test:
        dists = np.linalg.norm(X_train - x, axis=1)
        k_idxs = np.argsort(dists)[:k]
        k_labels = y_train[k_idxs]
        pred = Counter(k_labels).most_common(1)[0][0]
        preds.append(pred)
        
    return np.array(preds)
