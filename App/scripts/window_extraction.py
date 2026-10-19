
"""Window extraction for fixed-format CSI amplitude sequences."""

import numpy as np


def extract_windows(
    amplitudes: np.ndarray,
    window_size: int = 150,
    stride: int = 150,
) -> np.ndarray:
    """Split a CSI amplitude sequence into fixed-length windows.

    Args:
        amplitudes: Array shaped (observations, 256).
        window_size: Number of consecutive observations per window.
        stride: Step between successive window starts.

    Returns:
        Float32 array shaped (n_windows, window_size, 256).
        Incomplete trailing observations are discarded.
    """
    amplitudes = np.asarray(amplitudes, dtype=np.float32)

    if amplitudes.ndim != 2:
        raise ValueError(
            f"Expected a 2D array (observations, subcarriers), "
            f"got shape {amplitudes.shape}."
        )

    if amplitudes.shape[1] != 256:
        raise ValueError(
            f"Expected 256 subcarriers, got {amplitudes.shape[1]}."
        )

    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive.")

    if not np.isfinite(amplitudes).all():
        raise ValueError("Amplitudes contain NaN or infinite values.")

    if amplitudes.shape[0] < window_size:
        return np.empty((0, window_size, 256), dtype=np.float32)

    starts = range(0, amplitudes.shape[0] - window_size + 1, stride)

    windows = np.stack(
        [amplitudes[start:start + window_size] for start in starts],
        axis=0,
    )

    return np.ascontiguousarray(windows, dtype=np.float32)
