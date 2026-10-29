
"""Window extraction rule for fixed-format CSI amplitude sequences."""


def centered_tiling_starts(n_observations: int, window_size: int = 150) -> list[int]:
    """Start rows of non-overlapping windows centred in a capture.

    floor(n / window_size) windows are placed back to back, and the unused
    remainder is split evenly before the first and after the last window
    (the extra row of an odd remainder goes to the end). A capture shorter
    than `window_size` yields no window.
    """
    n_windows = n_observations // window_size
    offset = (n_observations - n_windows * window_size) // 2
    return [offset + i * window_size for i in range(n_windows)]
