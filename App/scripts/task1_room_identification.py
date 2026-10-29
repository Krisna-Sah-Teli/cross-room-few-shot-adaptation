"""Task 1: recorded-environment (Room A vs Room B) identification.

Every variant uses all 256 subcarriers of the 150 x 256 window:
  shortcut          one feature: fraction of narrow (non-80 MHz) observations in the window.
                    Diagnostic only - shows how much a recording-setting difference explains.
  full_window_z     log1p + per-window z-score (alternative normalization).
  full_row_gain     log1p minus each observation's own mean, i.e. per-packet gain removed (headline).

All models are fitted on development windows only. Development accuracy comes
from 5-fold cross-validation grouped by capture; validation windows are scored
once with the model refitted on all development windows.

A final control keeps only full 80 MHz observations from every window, so the
two rooms are compared on the same frame type (still all 256 subcarriers).

Usage:
    python App/scripts/task1_room_identification.py [--config App/configs/experiment.yaml]
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import load_config
from scripts.csi_quality import classify_band_coverage
from scripts.room_fingerprint import center_crop, narrow_band_fraction, normalize, window_features

VARIANTS = {
    "shortcut": dict(norm=None, title="Shortcut: narrow-frame fraction only"),
    "full_window_z": dict(norm="window_zscore", title="All 256 subcarriers, per-window z-score"),
    "full_row_gain": dict(norm="row_gain", title="All 256 subcarriers, per-observation gain removed"),
}
HEADLINE = "full_row_gain"
WINDOW_LENGTHS = [1, 2, 5, 10, 25, 50, 100, 150]
MATCHED_COUNTS = [1, 5, 30]
ROOM_NAMES = np.array(["A", "B"])


def featurize(X, variant, config, chunk=400):
    spec = VARIANTS[variant]
    parts = []
    for start in range(0, len(X), chunk):
        block = np.asarray(X[start:start + chunk], dtype=np.float32)
        if spec["norm"] is None:
            parts.append(narrow_band_fraction(block, config["quality"]))
        else:
            parts.append(window_features(normalize(block, spec["norm"])))
    return np.vstack(parts)


def make_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000),
    )


def grouped_cv_accuracy(F, y, groups, seed):
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []
    for train, test in folds.split(F, y, groups):
        model = make_model().fit(F[train], y[train])
        scores.append(accuracy_score(y[test], model.predict(F[test])))
    return float(np.mean(scores)), float(np.std(scores))


def evaluate(F_dev, y_dev, groups, F_val, y_val, seed):
    cv_mean, cv_std = grouped_cv_accuracy(F_dev, y_dev, groups, seed)
    model = make_model().fit(F_dev, y_dev)
    proba = model.predict_proba(F_val)
    pred = proba.argmax(1)
    return model, proba, {
        "dev_cv_accuracy": cv_mean, "dev_cv_accuracy_std": cv_std,
        "validation_accuracy": accuracy_score(y_val, pred),
        "validation_balanced_accuracy": balanced_accuracy_score(y_val, pred),
        "validation_errors": int((pred != y_val).sum()),
    }


def plot_window_length(curve, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink_muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    styles = {
        "shortcut": dict(color="#8a8984", linestyle="--", marker="s", label="Shortcut (narrow-frame fraction)"),
        "full_window_z": dict(color="#eb6834", linestyle="-", marker="o", label="Per-window z-score"),
        "full_row_gain": dict(color="#2a78d6", linestyle="-", marker="o", label="Per-packet gain removed (headline)"),
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=surface)
    ax.set_facecolor(surface)
    for variant, style in styles.items():
        rows = curve[curve.variant == variant].sort_values("window_length")
        ax.plot(rows.window_length, rows.validation_accuracy, linewidth=2, markersize=7,
                markeredgecolor=surface, markeredgewidth=1.5, **style)
    ax.set_xscale("log")
    ax.set_xticks(WINDOW_LENGTHS, [str(v) for v in WINDOW_LENGTHS])
    ax.minorticks_off()
    ax.set_xlim(0.8, 190)
    ax.set_ylim(0.80, 1.005)
    ax.set_xlabel("Window length (consecutive CSI observations, log scale)", color=ink_muted)
    ax.set_ylabel("Validation room accuracy", color=ink_muted)
    ax.set_title("Room identification vs window length (fixed validation windows, 256 subcarriers)",
                 color=ink, loc="left", fontsize=11)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=ink_muted)
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    data_dir = config["paths"]["data_dir"]
    out_dir = config["paths"]["results_dir"] / "task1"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = config["few_shot"]["support_seed"]

    X = np.load(data_dir / "windows_X.npy", mmap_mode="r")
    windows = pd.read_csv(data_dir / "windows_manifest.csv")
    y = (windows.room == "B").to_numpy(np.int64)          # Room A = 0, Room B = 1
    dev = np.flatnonzero(windows.split == "development")
    val = np.flatnonzero(windows.split == "validation")
    groups = windows.capture_id.to_numpy()[dev]
    print(f"development windows: {len(dev)}   validation windows: {len(val)}")

    # 1. Headline comparison at the official 150-observation length.
    results, metric_rows = [], []
    for variant in VARIANTS:
        F = featurize(X, variant, config)
        model, proba, scores = evaluate(F[dev], y[dev], groups, F[val], y[val], seed)
        pred = proba.argmax(1)
        cm = confusion_matrix(y[val], pred, labels=[0, 1])
        results.append({"variant": variant, "description": VARIANTS[variant]["title"], **scores,
                        "val_A_as_B": int(cm[0, 1]), "val_B_as_A": int(cm[1, 0])})
        for metric in ("validation_accuracy", "validation_balanced_accuracy", "dev_cv_accuracy"):
            metric_rows.append({"experiment": f"task1_{variant}", "target_room": "A+B", "K": "",
                                "metric": metric, "value": round(scores[metric], 4)})
        print(f"{variant:16s} dev-CV {scores['dev_cv_accuracy']:.4f}  "
              f"validation {scores['validation_accuracy']:.4f}  (errors {scores['validation_errors']})")
        if variant == HEADLINE:
            pd.DataFrame({
                "window_id": windows.window_id.to_numpy()[val],
                "room": ROOM_NAMES[pred],
                "label": "",
                "confidence": proba.max(1).round(4),
            }).to_csv(out_dir / "predictions_validation.csv", index=False)
            with open(out_dir / "room_router.pkl", "wb") as f:
                pickle.dump({"model": model, "norm": VARIANTS[variant]["norm"],
                             "window_length": 150, "subcarriers": 256}, f)

    # 2. Required figure: accuracy against window length (centre crops of the fixed windows).
    curve = []
    for variant in ("shortcut", "full_window_z", "full_row_gain"):
        for length in WINDOW_LENGTHS:
            F = featurize(center_crop(X, length), variant, config)
            _, _, scores = evaluate(F[dev], y[dev], groups, F[val], y[val], seed)
            curve.append({"variant": variant, "window_length": length, **scores})
            metric_rows.append({"experiment": f"task1_length_{variant}_L{length}", "target_room": "A+B", "K": "",
                                "metric": "validation_accuracy", "value": round(scores["validation_accuracy"], 4)})
            print(f"  length {length:3d} {variant:16s} validation {scores['validation_accuracy']:.4f}")
    curve = pd.DataFrame(curve)

    # 3. Packet-type-matched control: use only the first M full 80 MHz observations of
    #    each window, so both rooms are compared on identical frame types.
    full_masks = [classify_band_coverage(np.asarray(X[i]), config["quality"]) == "full" for i in range(len(X))]
    for m in MATCHED_COUNTS:
        keep = np.array([i for i, mask in enumerate(full_masks) if mask.sum() >= m])
        S = np.stack([np.asarray(X[i])[full_masks[i]][:m] for i in keep])
        k_dev = np.isin(keep, dev)
        k_val = np.isin(keep, val)
        F = window_features(normalize(S, "row_gain"))
        _, _, scores = evaluate(F[k_dev], y[keep][k_dev], windows.capture_id.to_numpy()[keep][k_dev],
                                F[k_val], y[keep][k_val], seed)
        name = f"matched80_M{m}"
        results.append({"variant": name, "description": f"First {m} full 80 MHz observations, 256 subcarriers, gain removed",
                        **scores, "windows_kept_A": int((y[keep] == 0).sum()), "windows_kept_B": int((y[keep] == 1).sum())})
        metric_rows.append({"experiment": f"task1_{name}", "target_room": "A+B", "K": "",
                            "metric": "validation_accuracy", "value": round(scores["validation_accuracy"], 4)})
        print(f"{name:20s} validation {scores['validation_accuracy']:.4f}")

    pd.DataFrame(results).to_csv(out_dir / "variants.csv", index=False)
    curve.to_csv(out_dir / "window_length.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)
    plot_window_length(curve, out_dir / "fig_room_accuracy_vs_window_length.png")
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
