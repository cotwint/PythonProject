import numpy as np

def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    计算两个向量之间的欧氏距离。
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))
