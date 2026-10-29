"""Leakage checks for few-shot support selection and class balancing. Run with `python App/tests/test_few_shot.py`."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import load_config
from scripts.few_shot_adaptation import balancing_weights, select_support

CONFIG = load_config()
WINDOWS = pd.read_csv(CONFIG["paths"]["data_dir"] / "windows_manifest.csv")
K_VALUES = CONFIG["few_shot"]["k_values"]


def test_support_sets_are_development_k_shot_nested_and_exclusive_only():
    captures = WINDOWS.drop_duplicates("capture_id").set_index("capture_id")
    for room, exclusive in CONFIG["task3"]["exclusive"].items():
        for draw in range(CONFIG["task3"]["support_draws"]):
            sets = select_support(WINDOWS, room, exclusive, K_VALUES, CONFIG["few_shot"]["support_seed"], draw)
            for k in K_VALUES:
                chosen = captures.loc[sets[k]]
                assert (chosen.split == "development").all()
                assert (chosen.room == room).all()
                assert set(chosen.label) == set(exclusive)
                assert (chosen.label.value_counts() == k).all() and chosen.index.is_unique
            for small, large in zip(K_VALUES, K_VALUES[1:]):
                assert set(sets[small]) <= set(sets[large])


def test_support_selection_is_deterministic():
    a = select_support(WINDOWS, "B", ["EMPTY", "LYING"], K_VALUES, 42, 0)
    b = select_support(WINDOWS, "B", ["EMPTY", "LYING"], K_VALUES, 42, 0)
    assert all(np.array_equal(a[k], b[k]) for k in K_VALUES)


def test_balancing_equalizes_mean_class_probabilities():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(500, 4))
    logits[:, 0] += 2.0                                   # class 0 is strongly over-predicted
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    weights = balancing_weights(p)
    q = p * weights
    q /= q.sum(axis=1, keepdims=True)
    assert np.allclose(q.mean(axis=0), 0.25, atol=1e-4)  # every class now has the same mean probability
    assert weights[0] < weights[1:].min()                # the over-predicted class is down-weighted
    assert np.isclose(weights.sum(), 1.0)


def test_balancing_leaves_balanced_predictions_unchanged():
    p = np.full((100, 3), 1 / 3)
    assert np.allclose(balancing_weights(p), 1 / 3)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS", name)
