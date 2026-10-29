"""End-to-end checks of the deployed system. Run with `python App/tests/test_episode.py`."""

import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import PROJECT_ROOT, load_config
from scripts.episode import (build_bundle, load_bundle, load_episode_inputs, predict_windows, save_bundle,
                             windows_from_capture)

CONFIG = load_config()
WINDOWS = pd.read_csv(CONFIG["paths"]["data_dir"] / "windows_manifest.csv")
CAPTURES = pd.read_csv(CONFIG["paths"]["data_dir"] / "capture_manifest.csv")
BUNDLE_PATH = CONFIG["paths"]["final_model_dir"] / "bundle.pkl"
if not BUNDLE_PATH.is_file():
    save_bundle(build_bundle(CONFIG), BUNDLE_PATH)
BUNDLE = load_bundle(BUNDLE_PATH)
VAL = WINDOWS[WINDOWS.split == "validation"]


def load_windows(rows):
    return np.stack([np.load(PROJECT_ROOT / path) for path in rows.window_file])


def test_reproduces_task3_validation_predictions():
    """The deployed system is the evaluated system: same predictions as Task 3 (K=20, draw 0)."""
    k = CONFIG["final_model"]["k"]
    for source, target in (("A", "B"), ("B", "A")):
        rows = VAL[VAL.room == target]
        ours = predict_windows(BUNDLE, load_windows(rows))
        reported = pd.read_csv(CONFIG["paths"]["results_dir"] / "task3" / f"predictions_{source}_to_{target}_K{k}.csv")
        reported = reported.set_index("window_id").loc[rows.window_id]
        assert (ours.room.to_numpy() == reported.room.to_numpy()).all()
        agreement = np.mean(ours.label.to_numpy() == reported.label.to_numpy())
        assert agreement >= 0.99, f"{source}->{target}: only {agreement:.3f} of labels agree"


def test_raw_captures_give_the_dataset_windows():
    """A raw .pcap or .csv goes through exactly the preprocessing of the dataset build."""
    for room, zip_key, suffix in (("A", "room_a_zip", ".pcap"), ("B", "room_b_zip", ".csv")):
        capture = CAPTURES[(CAPTURES.room == room) & (CAPTURES.split == "validation")
                           & (CAPTURES.observations >= 150)].iloc[0]
        expected = load_windows(WINDOWS[WINDOWS.capture_id == capture.capture_id])
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(CONFIG["paths"][zip_key]) as archive:
            path = Path(tmp) / f"capture{suffix}"
            path.write_bytes(archive.read(capture.source_member))
            produced = windows_from_capture(path, CONFIG)
        assert len(produced) == len(expected)
        assert all(np.array_equal(window, ref) for (_, window), ref in zip(produced, expected))


def test_output_contract_and_label_spaces():
    rows = pd.concat([VAL[VAL.room == r].head(6) for r in ("A", "B")])
    with tempfile.TemporaryDirectory() as tmp:
        for i, path in enumerate(rows.window_file, start=1):
            np.save(Path(tmp) / f"window_{i:04d}.npy", np.load(PROJECT_ROOT / path))
        ids, X, _ = load_episode_inputs(tmp, CONFIG)
    assert ids == [f"window_{i:04d}" for i in range(1, 13)] and X.shape == (12, 150, 256)
    predictions = predict_windows(BUNDLE, X)
    assert list(predictions.room) == list(rows.room)
    for room, label in zip(predictions.room, predictions.label):
        assert label in BUNDLE["label_sets"][room]
    assert predictions.confidence.between(0, 1).all()


def test_episode_support_replaces_default_support():
    rows = WINDOWS[(WINDOWS.split == "development") & (WINDOWS.room == "A") & (WINDOWS.label == "SITDOWN")].head(2)
    support = [(np.load(PROJECT_ROOT / p), "SITDOWN", "A") for p in rows.window_file]
    X = load_windows(VAL[VAL.room == "A"].head(5))
    predictions = predict_windows(BUNDLE, X, support)
    assert set(predictions.label) <= set(BUNDLE["label_sets"]["A"])
    try:
        predict_windows(BUNDLE, X, [(support[0][0], "LYING", "A")])   # LYING is not a Room A activity
    except ValueError:
        pass
    else:
        raise AssertionError("an invalid support label must be rejected")


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
