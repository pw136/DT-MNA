
import numpy as np

def distribution_tv(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("distributions must have equal shape")
    return 0.5 * float(np.abs(left - right).sum())

def kernel_row_tv(kernel_left, kernel_right):
    left = np.asarray(kernel_left, dtype=float)
    right = np.asarray(kernel_right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("kernels must have equal shape")
    return 0.5 * np.abs(left - right).sum(axis=1)

def max_kernel_tv(kernel_left, kernel_right, rows=None):
    distances = kernel_row_tv(kernel_left, kernel_right)
    if rows is not None:
        rows = tuple(rows)
        distances = distances[list(rows)] if rows else np.asarray([0.0])
    return float(distances.max()) if len(distances) else 0.0

def mean_kernel_tv(kernel_left, kernel_right, row_weights=None):
    distances = kernel_row_tv(kernel_left, kernel_right)
    if row_weights is None:
        return float(distances.mean())
    weights = np.asarray(row_weights, dtype=float)
    weights = weights / weights.sum()
    return float(np.dot(weights, distances))

