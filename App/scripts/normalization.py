
"""Leakage-safe normalization utilities for CSI amplitude windows."""

import numpy as np


def old_log1p_per_window_zscore(X, eps=1e-6):
    """Apply log1p and z-score normalization independently per window.

    Parameters
    ----------
    X : array-like
        CSI amplitude data with shape (n_samples, 150, 256).
        Values must be finite and non-negative.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Normalized float32 data with the same shape as X.
    """
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3 or X.shape[1:] != (150, 256):
        raise ValueError(
            f"Expected shape (n_samples, 150, 256), got {X.shape}"
        )

    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")

    if (X < 0).any():
        raise ValueError("log1p normalization requires non-negative values.")

    if eps <= 0:
        raise ValueError("eps must be positive.")

    Z = np.log1p(X)

    means = Z.mean(axis=(1, 2), keepdims=True)
    stds = Z.std(axis=(1, 2), keepdims=True)

    X_normalized = (Z - means) / np.maximum(stds, eps)
    X_normalized = X_normalized.astype(np.float32, copy=False)

    if not np.isfinite(X_normalized).all():
        raise ValueError("Normalization produced non-finite values.")

    return X_normalized


def log1p_per_window_zscore(X, eps=1e-6):
    """
    Apply log1p and per-window z-score normalization.

    Parameters
    ----------
    X : array-like, shape (n_samples, time_steps, 256)
        Non-negative CSI amplitude windows.
    eps : float
        Lower bound for the per-window standard deviation.

    Returns
    -------
    np.ndarray
        Normalized float32 array with the same shape as X.
    """
    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 3 or X.shape[1] < 2 or X.shape[2] != 256:
        raise ValueError(
            "Expected shape (n_samples, time_steps >= 2, 256); "
            f"got {X.shape}"
        )

    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample.")

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")

    if np.any(X < 0):
        raise ValueError("X must contain non-negative amplitude values.")

    if eps <= 0:
        raise ValueError("eps must be positive.")

    Z = np.log1p(X)
    means = Z.mean(axis=(1, 2), keepdims=True)
    stds = Z.std(axis=(1, 2), keepdims=True)

    X_normalized = (Z - means) / np.maximum(stds, eps)
    X_normalized = X_normalized.astype(np.float32)

    if not np.isfinite(X_normalized).all():
        raise ValueError("Normalized output contains non-finite values.")

    return X_normalized
