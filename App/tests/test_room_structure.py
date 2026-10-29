"""Checks of the room-structure prior. Run with `python App/tests/test_room_structure.py`."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import PROJECT_ROOT, load_config
from scripts.room_structure import StructurePrior, applicable, posture_examples

CONFIG = load_config()
QUALITY = CONFIG["quality"]
WINDOWS = pd.read_csv(CONFIG["paths"]["data_dir"] / "windows_manifest.csv")


def dev_windows(room, label, n):
    rows = WINDOWS[(WINDOWS.split == "development") & (WINDOWS.room == room) & (WINDOWS.label == label)].head(n)
    return [np.load(PROJECT_ROOT / p) for p in rows.window_file]


def test_applies_only_where_transitions_are_k_shot_activities():
    assert applicable(CONFIG["task3"]["exclusive"]["A"])
    assert not applicable(CONFIG["task3"]["exclusive"]["B"])


def test_transition_windows_yield_posture_examples_in_time_order():
    windows = dev_windows("A", "SITDOWN", 1) + dev_windows("A", "STANDUP", 1) + dev_windows("A", "FALL", 1)
    shapes, labels, sources = posture_examples(windows, ["SITDOWN", "STANDUP", "FALL"], QUALITY)
    assert list(labels) == ["SITDOWN", "STAND", "SIT", "STANDUP", "SIT", "STAND", "FALL"]
    assert list(sources) == ["SITDOWN"] * 3 + ["STANDUP"] * 3 + ["FALL"]
    assert shapes.shape == (7, 256)                      # all 256 subcarriers


def test_prior_needs_sit_and_stand_examples_and_never_target_shared_labels():
    shapes = np.random.default_rng(0).normal(size=(6, 256))
    try:
        StructurePrior.fit(shapes, np.ones(6), shapes, np.array(["FALL"] * 3 + ["SITDOWN"] * 3))
    except ValueError:
        pass
    else:
        raise AssertionError("a prior without transition-derived SIT/STAND examples must be rejected")


def test_revision_rules():
    rng = np.random.default_rng(1)
    shapes = rng.normal(size=(40, 256))
    labels = np.array(["SIT", "STAND", "FALL", "SITDOWN"] * 10)
    prior = StructurePrior.fit(shapes, np.r_[np.full(20, 0.2), np.full(20, 5.0)], shapes, labels)
    predicted = np.array(["WALK", "SIT", "FALL", "SITDOWN", "STAND"])
    motion = np.array([5.0, 0.1, 0.1, 0.1, 5.0])               # low score = strong fluctuation
    revised, conf = prior.revise(rng.normal(size=(5, 256)), motion, predicted, np.full(5, 0.7), ["SIT", "STAND", "WALK"])
    assert revised[0] in ("SIT", "STAND")                       # calm window predicted WALK -> posture
    assert revised[1] == "WALK" and revised[3] == "WALK"        # fluctuating windows -> WALK
    assert revised[2] == "FALL"                                 # FALL is never overridden
    assert revised[4] in ("SIT", "STAND")
    assert ((conf >= 0) & (conf <= 1)).all()


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
