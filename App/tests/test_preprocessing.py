"""Preprocessing checks. Run with `python App/tests/test_preprocessing.py` (or pytest)."""

import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import load_config
from scripts.csi_quality import classify_band_coverage
from scripts.room_a_pcap import (
    decode_bcm4366c0_batch, decode_bcm4366c0_words, list_pcap_members,
    read_room_a_capture_qc,
)
from scripts.window_extraction import centered_tiling_starts

CONFIG = load_config()


def test_batch_decoder_matches_reference_on_random_words():
    rng = np.random.default_rng(0)
    words = rng.integers(0, 2**32, size=(20, 256), dtype=np.uint64).astype(np.uint32)
    words[3] = 0                                   # an all-zero observation
    words[4, :200] = 0                             # a sparse observation
    real, imag = decode_bcm4366c0_batch(words)
    for i in range(len(words)):
        ref_real, ref_imag = decode_bcm4366c0_words(words[i])
        assert np.array_equal(real[i], ref_real) and np.array_equal(imag[i], ref_imag), i


def test_batch_decoder_matches_reference_on_real_capture():
    zip_path = CONFIG["paths"]["room_a_zip"]
    member = next(m for m in list_pcap_members(zip_path) if m.endswith("/walk/walk1.pcap"))
    with zipfile.ZipFile(zip_path) as archive:
        capture = read_room_a_capture_qc(archive.read(member), drop_consecutive_duplicates=False)
    from scripts.room_a_pcap import read_room_a_capture
    _, reference = read_room_a_capture(str(zip_path), member, keep_zero_observations=True)
    assert np.array_equal(capture.amplitude, reference)


def test_truncated_capture_is_salvaged():
    zip_path = CONFIG["paths"]["room_a_zip"]
    member = next(m for m in list_pcap_members(zip_path) if m.endswith("/sit/sit109.pcap"))
    with zipfile.ZipFile(zip_path) as archive:
        capture = read_room_a_capture_qc(archive.read(member))
    assert capture.counts["truncated"] and capture.counts["packets"] == 178


def test_band_classification():
    q = CONFIG["quality"]
    full = np.full((1, 256), 500.0)
    low_20 = np.full((1, 256), 3.0); low_20[0, 134:187] = 900.0
    upper_40 = full.copy(); upper_40[0, 128:] = 3.0
    empty = np.zeros((1, 256))
    classes = classify_band_coverage(np.vstack([full, low_20, upper_40, empty]), q)
    assert list(classes) == ["full", "low_20mhz", "partial", "empty"]


def test_model_input_keeps_all_256_subcarriers():
    from scripts.ssl_encoder import prepare_input
    windows = np.load(CONFIG["paths"]["data_dir"] / "windows_X.npy", mmap_mode="r")
    inputs = prepare_input(windows[:8])
    assert inputs.shape == (8, 150, 256) and inputs.dtype == np.float32
    assert np.isfinite(inputs).all()
    assert np.allclose(inputs.mean(axis=2), 0.0, atol=1e-4)    # each observation's gain is removed


def test_centered_tiling():
    assert centered_tiling_starts(149) == []
    assert centered_tiling_starts(150) == [0]
    assert centered_tiling_starts(216) == [33]
    assert centered_tiling_starts(500) == [25, 175, 325]


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
