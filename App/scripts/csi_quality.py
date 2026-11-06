"""Observation-level CSI quality control shared by both rooms.

Both rooms store the 256 subcarriers in raw FFT order: index 0 is DC, indices
1..127 are positive frequencies and 128..255 are negative frequencies. The
80 MHz channel consists of four 20 MHz subchannels. A frame transmitted in
only one 20 MHz subchannel leaves the other three near zero, even though the
receiver still reports all 256 values.

An observation is classified by which subchannels carry energy:
  full       all four subchannels (80 MHz frame)
  low_20mhz  only the lowest subchannel, indices 134..186 (Room A's narrow frames)
  partial    any other subset
  empty      no subchannel carries CSI (zero-information)
"""

import numpy as np

BAND_CLASSES = ("full", "low_20mhz", "partial", "empty")


def subchannel_means(amplitude, quality_config):
    """Return (n, 4) mean amplitude of each 20 MHz subchannel core per observation."""
    amplitude = np.asarray(amplitude, dtype=np.float64)
    ranges = quality_config["subchannels"].values()
    return np.stack([amplitude[..., start:stop].mean(axis=-1) for start, stop in ranges], axis=-1)


def classify_band_coverage(amplitude, quality_config):
    """Return an array of band-class names, one per row of `amplitude` (n, 256)."""
    means = subchannel_means(amplitude, quality_config)
    strongest = means.max(axis=-1)
    occupied = means > quality_config["occupied_fraction"] * strongest[..., None]

    classes = np.full(strongest.shape, "partial", dtype=object)
    classes[occupied.all(axis=-1)] = "full"
    classes[(occupied == [True, False, False, False]).all(axis=-1)] = "low_20mhz"
    classes[strongest < quality_config["empty_amplitude"]] = "empty"
    return classes


def band_counts(classes):
    """Count observations per band class (all classes present as keys)."""
    classes = np.asarray(classes)
    return {f"n_{name}": int((classes == name).sum()) for name in BAND_CLASSES}


def consecutive_duplicate_mask(amplitude):
    """True for rows identical to the row immediately before them."""
    amplitude = np.asarray(amplitude)
    mask = np.zeros(len(amplitude), dtype=bool)
    if len(amplitude) > 1:
        mask[1:] = np.all(amplitude[1:] == amplitude[:-1], axis=1)
    return mask
