
"""Utilities for room-fingerprint classification."""

import numpy as np


def build_room_classification_dataset(X_room_a, X_room_b):
    """Combine room tensors and create binary room labels.

    Room A is encoded as 0; Room B is encoded as 1.

    Returns
    -------
    X : np.ndarray
        Combined CSI windows, shape (n_samples, 150, 256).
    y : np.ndarray
        Room labels, shape (n_samples,).
    """
    X_room_a = np.asarray(X_room_a, dtype=np.float32)
    X_room_b = np.asarray(X_room_b, dtype=np.float32)

    for name, X in (("Room A", X_room_a), ("Room B", X_room_b)):
        if X.ndim != 3 or X.shape[1:] != (150, 256):
            raise ValueError(
                f"{name} must have shape (n_samples, 150, 256); "
                f"received {X.shape}"
            )
        if X.shape[0] == 0:
            raise ValueError(f"{name} has no samples.")
        if not np.isfinite(X).all():
            raise ValueError(f"{name} contains non-finite values.")

    X = np.concatenate([X_room_a, X_room_b], axis=0)

    y = np.concatenate([
        np.zeros(len(X_room_a), dtype=np.int64),
        np.ones(len(X_room_b), dtype=np.int64),
    ])

    return X, y



def extract_room_features(X):
    """
    Extract compact numerical features from CSI windows.

    Parameters
    ----------
    X : array-like, shape (n_windows, 150, 256)
        CSI amplitude windows.

    Returns
    -------
    np.ndarray, shape (n_windows, 524)
        Numerical features for room classification.
    """
    X = np.asarray(X, dtype=np.float32)

    # if X.ndim != 3 or X.shape[1:] != (150, 256):
    #     raise ValueError(
    #         f"Expected shape (n_windows, 150, 256), got {X.shape}"
    #     )
    if X.ndim != 3 or X.shape[1] < 2 or X.shape[2] != 256:
        raise ValueError(
            f"Expected shape (n_windows, time_steps >= 2, 256), "
            f"got {X.shape}"
        )

    if X.shape[0] == 0:
        raise ValueError("Input must contain at least one window.")

    if not np.isfinite(X).all():
        raise ValueError("Input contains NaN or infinite values.")

    # Seven global statistics per window.
    global_features = [
        X.mean(axis=(1, 2)),
        X.std(axis=(1, 2)),
        np.median(X, axis=(1, 2)),
        X.min(axis=(1, 2)),
        X.max(axis=(1, 2)),
        np.percentile(X, 25, axis=(1, 2)),
        np.percentile(X, 75, axis=(1, 2)),
    ]

    # Two statistics for each of the 256 subcarriers.
    subcarrier_mean = X.mean(axis=1)
    subcarrier_std = X.std(axis=1)

    # Temporal variation within each window.
    temporal_diff = np.diff(X, axis=1)
    temporal_features = [
        np.mean(np.abs(temporal_diff), axis=(1, 2)),
        np.std(temporal_diff, axis=(1, 2)),
    ]

    features = np.column_stack(
        global_features
        + [subcarrier_mean, subcarrier_std]
        + temporal_features
    ).astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError("Extracted features contain non-finite values.")

    return features
