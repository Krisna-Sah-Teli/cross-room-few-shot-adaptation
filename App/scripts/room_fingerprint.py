"""Normalization and features for CSI windows (all 256 subcarriers).

About 70% of Room A's observations come from 20 MHz frames that fill only the
lowest 20 MHz subchannel (indices 134..186), while Room B almost always records
the full 80 MHz channel. `narrow_band_fraction` measures that recording-
configuration difference; it is used only as a diagnostic, never as a model feature.
"""

import numpy as np

from scripts.csi_quality import classify_band_coverage


def normalize(X, method, eps=1e-6):
    """Normalize amplitude windows using only each window's own values (no fitted statistics).

    window_zscore: log1p, then z-score over the whole window.
    row_gain:      log1p, then subtract each observation's mean over its subcarriers.
                   This removes the per-packet gain/autoscaling factor, so only the
                   shape of the frequency response is left.
    """
    Z = np.log1p(np.asarray(X, dtype=np.float32))
    if method == "window_zscore":
        mean = Z.mean(axis=(1, 2), keepdims=True)
        std = Z.std(axis=(1, 2), keepdims=True)
        return (Z - mean) / np.maximum(std, eps)
    if method == "row_gain":
        return Z - Z.mean(axis=2, keepdims=True)
    raise ValueError(f"Unknown normalization {method!r}")


def window_features(Z):
    """Per-subcarrier mean and std, global statistics and temporal variation.

    Works for any window length >= 1, so the same function serves the official
    150-observation windows and the shorter windows of the window-length study.
    """
    Z = np.asarray(Z, dtype=np.float32)
    n, length, _ = Z.shape
    flat = Z.reshape(n, -1)
    global_stats = [
        flat.mean(1), flat.std(1), np.median(flat, 1), flat.min(1), flat.max(1),
        np.percentile(flat, 25, axis=1), np.percentile(flat, 75, axis=1),
    ]
    if length > 1:
        diff = np.diff(Z, axis=1).reshape(n, -1)
        temporal = [np.abs(diff).mean(1), diff.std(1)]
    else:
        temporal = [np.zeros(n, np.float32), np.zeros(n, np.float32)]
    features = np.column_stack(global_stats + [Z.mean(1), Z.std(1)] + temporal).astype(np.float32)
    assert np.isfinite(features).all()
    return features


def narrow_band_fraction(X, quality_config):
    """Fraction of observations in each window that are not full 80 MHz frames.

    This single number is the recording-configuration shortcut; it is used only
    as a diagnostic baseline, never as a feature of the reported models.
    """
    n, length, width = X.shape
    classes = classify_band_coverage(np.asarray(X).reshape(-1, width), quality_config).reshape(n, length)
    return (classes != "full").mean(axis=1, keepdims=True).astype(np.float32)


def center_crop(X, length):
    """Deterministic crop of `length` consecutive observations from the middle of each window."""
    start = (X.shape[1] - length) // 2
    return X[:, start:start + length]
