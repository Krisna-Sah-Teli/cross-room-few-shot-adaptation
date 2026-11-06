"""Leakage checks: hiding every label the rules forbid changes no prediction.

Run with `python App/tests/test_leakage.py` (about two minutes: it refits every Task 3 episode and rebuilds
the deployed system).

Task 3 rules: for a target room, only the labels of the K support captures per target-exclusive
activity may be used, besides the development labels of the source room; validation labels never.
If the code read any other label, replacing it by "HIDDEN" (or scrambling it) would change the output.
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import load_config
from scripts.episode import build_bundle, predict_windows
from scripts.few_shot_adaptation import select_support
from scripts.room_structure import structure_features
from scripts.task3_cross_room import DIRECTIONS, Experiment, build_representations

CONFIG = load_config()
TASK3, FINAL = CONFIG["task3"], CONFIG["final_model"]
WINDOWS = pd.read_csv(CONFIG["paths"]["data_dir"] / "windows_manifest.csv")
X = np.load(CONFIG["paths"]["data_dir"] / "windows_X.npy", mmap_mode="r")
DEV = (WINDOWS.split == "development").to_numpy()


@lru_cache(maxsize=None)
def inputs():
    """Features and structure-prior features of every window (computed from the windows alone)."""
    reps = build_representations(X, CONFIG, CONFIG["paths"]["results_dir"] / "task2")
    return reps, structure_features(X, CONFIG["quality"])


def hide(table, mask):
    hidden = table.copy()
    hidden.loc[mask, "label"] = "HIDDEN"
    return hidden


def scramble_validation(table, seed=0):
    scrambled = table.copy()
    scrambled.loc[~DEV, "label"] = np.random.default_rng(seed).permutation(scrambled.label.to_numpy()[~DEV])
    return scrambled


def supports(target, draw):
    return select_support(WINDOWS, target, TASK3["exclusive"][target], CONFIG["few_shot"]["k_values"],
                          CONFIG["few_shot"]["support_seed"], draw)


def adapt(table, source, target, support):
    """Validation predictions of the reported Task 3 system, fitted with every label read from `table`."""
    reps, structure = inputs()
    return Experiment(table, CONFIG).reported_system(X, reps, structure, FINAL, source, target, support)[0]


def test_task3_reads_only_the_support_labels_of_the_target_room():
    """Both directions, every K and support draw: hiding every target development label except those
    of the K support captures, or scrambling the validation labels, changes no prediction."""
    for source, target in DIRECTIONS:
        target_dev = DEV & (WINDOWS.room == target).to_numpy()
        for draw in range(TASK3["support_draws"]):
            for k, support in supports(target, draw).items():
                hidden = target_dev & ~WINDOWS.capture_id.isin(support).to_numpy()
                assert hidden.sum() >= 0.9 * target_dev.sum()
                reference = adapt(WINDOWS, source, target, support)
                assert (adapt(hide(WINDOWS, hidden), source, target, support) == reference).all(), \
                    f"{source}->{target} K={k} draw {draw}: a hidden target label was used"
                assert (adapt(scramble_validation(WINDOWS), source, target, support) == reference).all(), \
                    f"{source}->{target} K={k} draw {draw}: a validation label was used"


def test_the_check_detects_label_use():
    """Negative control: hiding the support labels as well does change the predictions."""
    support = supports("B", 0)[5]
    reference = adapt(WINDOWS, "A", "B", support)
    target_dev = DEV & (WINDOWS.room == "B").to_numpy()
    assert (adapt(hide(WINDOWS, target_dev), "A", "B", support) != reference).any()


def test_deployed_system_reads_no_forbidden_label():
    """run_episode's model for a room is unchanged when that room's development SIT/STAND/WALK labels
    are hidden and all validation labels are scrambled."""
    reference = build_bundle(CONFIG)
    shared = WINDOWS.label.isin(CONFIG["labels"]["shared"]).to_numpy()
    for room in ("A", "B"):
        in_room = (WINDOWS.room == room).to_numpy()
        bundle = build_bundle(CONFIG, scramble_validation(hide(WINDOWS, DEV & in_room & shared)))
        val = np.asarray(X[np.flatnonzero(in_room & ~DEV)])
        expected, ours = predict_windows(reference, val), predict_windows(bundle, val)
        routed = expected.room.to_numpy() == room
        assert routed.any() and (ours.room == expected.room).all()
        assert (ours.label.to_numpy()[routed] == expected.label.to_numpy()[routed]).all(), \
            f"Room {room}: the deployed model used a hidden label"


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
