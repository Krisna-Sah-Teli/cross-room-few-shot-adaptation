"""Task 2: learn a CSI representation without labels, then test it with a linear probe.

Input: the full 150 x 256 windows (all 256 subcarriers).
  1. Train the contrastive encoder on development windows of BOTH rooms
     (no activity or room labels are read).
  2. Freeze it and embed every window.
  3. Fit a linear classifier (logistic regression) on labelled development
     embeddings; its regularization C is chosen by capture-grouped
     cross-validation on development data only.
  4. Score the fixed validation windows (macro-F1, balanced accuracy).
Baselines with the same probe: handcrafted window features, and the same
encoder architecture with random weights. The predictions file comes from the
SSL encoder, the method this task evaluates.

Usage:
    python App/scripts/task2_self_supervised.py [--config ...] [--epochs N]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import load_config
from scripts.room_fingerprint import window_features
from scripts.ssl_encoder import embed, prepare_input, random_encoder, train_ssl


def make_probe(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=3000))


def choose_c(F, y, groups, grid, seed):
    """Pick C by 5-fold capture-grouped CV macro-F1 on development data."""
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(F, y, groups))
    scores = {}
    for c in grid:
        fold_scores = []
        for train, test in folds:
            probe = make_probe(c).fit(F[train], y[train])
            fold_scores.append(f1_score(y[test], probe.predict(F[test]), average="macro"))
        scores[c] = float(np.mean(fold_scores))
    best = max(scores, key=scores.get)
    return best, scores[best]


def linear_probe(name, F, labels, rooms, groups, dev, val, grid, seed):
    best_c, cv_f1 = choose_c(F[dev], labels[dev], groups[dev], grid, seed)
    probe = make_probe(best_c).fit(F[dev], labels[dev])
    proba = probe.predict_proba(F[val])
    pred = probe.classes_[proba.argmax(1)]
    truth = labels[val]
    row = {
        "representation": name, "dim": F.shape[1], "probe_C": best_c, "dev_cv_macro_f1": round(cv_f1, 4),
        "val_macro_f1": round(f1_score(truth, pred, average="macro"), 4),
        "val_balanced_accuracy": round(balanced_accuracy_score(truth, pred), 4),
    }
    for room in ("A", "B"):
        in_room = rooms[val] == room
        room_labels = sorted(set(truth[in_room]))
        row[f"val_macro_f1_room_{room}"] = round(
            f1_score(truth[in_room], pred[in_room], labels=room_labels, average="macro"), 4)
    report = classification_report(truth, pred, zero_division=0)
    return row, pred, proba.max(1), report


def plot_embeddings(points_by_name, rooms, labels, out_dir):
    """t-SNE maps: one figure coloured by room, one faceted by activity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink_muted, surface, rest = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d3"
    room_colors = {"A": "#2a78d6", "B": "#eb6834"}

    names = list(points_by_name)
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.8), facecolor=surface)
    for ax, name in zip(np.atleast_1d(axes), names):
        P = points_by_name[name]
        for room, color in room_colors.items():
            sel = rooms == room
            ax.scatter(P[sel, 0], P[sel, 1], s=10, color=color, edgecolors=surface, linewidths=0.4,
                       label=f"Room {room}")
        ax.set_title(name, color=ink, fontsize=10, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(rest)
        ax.set_facecolor(surface)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("t-SNE of validation embeddings, coloured by room (qualitative)", color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_tsne_by_room.png", dpi=200, facecolor=surface)
    plt.close(fig)

    P = points_by_name[names[0]]
    activities = sorted(set(labels))
    cols = 4
    rows_n = int(np.ceil(len(activities) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(3.4 * cols, 3.2 * rows_n), facecolor=surface)
    for ax, activity in zip(axes.ravel(), activities):
        sel = labels == activity
        ax.scatter(P[~sel, 0], P[~sel, 1], s=5, color=rest, linewidths=0)
        for room, marker in (("A", "o"), ("B", "^")):
            pick = sel & (rooms == room)
            if pick.any():
                ax.scatter(P[pick, 0], P[pick, 1], s=14, marker=marker, color="#2a78d6",
                           edgecolors=surface, linewidths=0.4)
        rooms_here = "/".join(sorted(set(rooms[sel])))
        ax.set_title(f"{activity} (Room {rooms_here})", color=ink, fontsize=9, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(rest)
        ax.set_facecolor(surface)
    for ax in axes.ravel()[len(activities):]:
        ax.axis("off")
    fig.suptitle(f"t-SNE by activity - {names[0]} (circle = Room A, triangle = Room B)",
                 color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_tsne_by_activity.png", dpi=200, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override ssl.epochs (e.g. for a smoke test)")
    args = parser.parse_args()
    config = load_config(args.config)
    ssl = config["ssl"]
    epochs = args.epochs or ssl["epochs"]
    seed = ssl["seed"]
    torch.set_num_threads(ssl["threads"])

    data_dir = config["paths"]["data_dir"]
    out_dir = config["paths"]["results_dir"] / "task2"
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(data_dir / "windows_X.npy", mmap_mode="r")
    windows = pd.read_csv(data_dir / "windows_manifest.csv")
    labels = windows.label.to_numpy()
    rooms = windows.room.to_numpy()
    groups = windows.capture_id.to_numpy()
    dev = np.flatnonzero(windows.split == "development")
    val = np.flatnonzero(windows.split == "validation")

    inputs = prepare_input(X)
    started = time.time()
    encoder, history = train_ssl(inputs[dev], epochs=epochs, batch_size=ssl["batch_size"], lr=ssl["lr"],
                                 temperature=ssl["temperature"], seed=seed)
    print(f"SSL training took {time.time() - started:.0f}s", flush=True)
    torch.save({"state_dict": encoder.state_dict(), "in_channels": inputs.shape[2], "epochs": epochs},
               out_dir / "encoder.pt")
    pd.DataFrame({"epoch": np.arange(1, len(history) + 1), "loss": history}).to_csv(
        out_dir / "ssl_loss.csv", index=False)

    candidates = {
        "SSL encoder": embed(encoder, inputs),
        "Random-weight encoder": embed(random_encoder(inputs.shape[2], seed), inputs),
        "Handcrafted features": window_features(inputs),
    }
    results, reports, embeddings = [], {}, {}
    for name, F in candidates.items():
        row, pred, confidence, report = linear_probe(name, F, labels, rooms, groups, dev, val,
                                                     ssl["probe_C_grid"], seed)
        results.append(row)
        reports[name] = report
        print(f"  {name:22s} val macro-F1 {row['val_macro_f1']:.4f}  bal-acc {row['val_balanced_accuracy']:.4f}  "
              f"(A {row['val_macro_f1_room_A']:.3f} / B {row['val_macro_f1_room_B']:.3f})", flush=True)
        if name == "SSL encoder":
            ssl_result = (row, pred, confidence)
        if name != "Handcrafted features":
            embeddings[name] = F[val]

    results = pd.DataFrame(results)
    results.to_csv(out_dir / "linear_probe_results.csv", index=False)
    (out_dir / "classification_reports.txt").write_text(
        "\n\n".join(f"== {name} ==\n{report}" for name, report in reports.items()), encoding="utf-8")

    metric_rows = []
    for row in results.itertuples(index=False):
        for metric in ("val_macro_f1", "val_balanced_accuracy", "val_macro_f1_room_A", "val_macro_f1_room_B"):
            metric_rows.append({"experiment": f"task2_{row.representation}", "target_room": "A+B", "K": "",
                                "metric": metric, "value": getattr(row, metric)})
    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)

    # Predictions of the SSL encoder's linear probe; the room column comes from the Task 1 router.
    ssl_row, ssl_pred, ssl_conf = ssl_result
    router = config["paths"]["results_dir"] / "task1" / "predictions_validation.csv"
    predictions = pd.DataFrame({"window_id": windows.window_id.to_numpy()[val], "label": ssl_pred,
                                "confidence": ssl_conf.round(4)})
    if router.is_file():
        predictions = predictions.merge(pd.read_csv(router)[["window_id", "room"]], on="window_id", how="left")
    predictions[["window_id", "room", "label", "confidence"]].to_csv(out_dir / "predictions_validation.csv", index=False)
    print(f"\nSSL encoder: val macro-F1 {ssl_row['val_macro_f1']:.4f}")

    from sklearn.manifold import TSNE
    points = {name: TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(F)
              for name, F in embeddings.items()}
    plot_embeddings(points, rooms[val], labels[val], out_dir)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
