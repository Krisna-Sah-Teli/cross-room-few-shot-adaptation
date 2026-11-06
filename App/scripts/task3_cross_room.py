"""Task 3: cross-room few-shot adaptation (A -> B and B -> A, K = 1, 5, 10, 20).

Grid (all evaluated on the fixed target validation windows, oracle routing):
  representation  ssl (Task 2 encoder), handcrafted features, or random - the same encoder
                  architecture with random weights, a baseline that measures what SSL pretraining
                  contributes and is never selected (representation_effect.csv); all on 256 subcarriers
  alignment       pooled  - one standardization fitted on development windows of both rooms
                  per_room - each room standardized with its own development statistics
  head            logreg (balanced logistic regression) or prototype (nearest class mean)
  balancing       none, or uniform - class probabilities reweighted so that every activity is
                  predicted equally often on the room's unlabelled development windows
  structure_prior none, or transitions - for a target room whose K-shot activities include
                  SITDOWN and STANDUP: label-free walking detector + SIT/STAND examples from
                  the start and end of the transition support windows (room_structure.py)
Every (direction, K) is repeated for several support draws; the mean and std are reported.

Refinements, added one at a time to the best grid configuration (refinement_effect.csv). They
are meant for a target room whose shared activities come from the structure prior (Room A),
where the classifier mainly has to recognise the K-shot activities; they are also computed for
target Room B, where they do not help, and the reported system uses them only with the prior:
  sway       postural sway also votes on SIT vs STAND (room_structure.py)
  ensemble   average of the balanced probabilities of the SSL and handcrafted models
  smoothing  each window's probabilities combined with the mean probabilities of its k-means
             cluster of unlabelled target development windows

Routing study for the reported system (the best grid configuration with its refinements where
the prior applies; each model balanced, if enabled, on the unlabelled development windows of
the room(s) it serves):
  oracle     the true room selects the room-specific normalization and label space
  predicted  the Task 1 room router selects them (a window routed to the source room
             is classified by a source-room model with the source label space)
  none       one model for both rooms: pooled normalization and the union label space

Usage:
    python App/scripts/task3_cross_room.py [--config ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import load_config
from scripts.few_shot_adaptation import (ClusterSmoother, apply_stats, balanced_proba, balancing_weights, fit_stats,
                                         make_head, predict, score, select_support)
from scripts.room_fingerprint import window_features
from scripts.room_structure import StructurePrior, applicable, posture_examples, structure_features
from scripts.ssl_encoder import CSIEncoder, embed, prepare_input, random_encoder

DIRECTIONS = [("A", "B"), ("B", "A")]
METRICS = ["primary_macro_f1", "exclusive_macro_f1", "shared_macro_f1", "balanced_accuracy", "accuracy"]
ENSEMBLE = ("ssl", "handcrafted")
BASELINES = ("random",)                 # compared in the grid, never selected as the reported configuration
# Row order of the per-activity K = 5 table, one column block per target room.
TABLE_ROWS = {"A": ["SIT", "STAND", "WALK", "SITDOWN", "STANDUP", "FALL"], "B": ["EMPTY", "LYING", "SIT", "STAND", "WALK"]}


def build_representations(X, config, task2_dir):
    """SSL embeddings (frozen Task 2 encoder), handcrafted features and the embeddings of the same encoder
    architecture with random weights (the Task 2 baseline) of every window."""
    inputs = prepare_input(X)
    checkpoint = torch.load(task2_dir / "encoder.pt", weights_only=False)
    encoder = CSIEncoder(checkpoint["in_channels"])
    encoder.load_state_dict(checkpoint["state_dict"])
    reps = {"ssl": embed(encoder, inputs), "handcrafted": window_features(inputs),
            "random": embed(random_encoder(checkpoint["in_channels"], config["ssl"]["seed"]), inputs)}
    print("  representations ready (256 subcarriers)", flush=True)
    return reps


class Experiment:
    """Holds the windows table and room masks; runs one adaptation episode."""

    def __init__(self, windows, config):
        self.w = windows
        self.labels = windows.label.to_numpy()
        self.capture = windows.capture_id.to_numpy()
        self.cfg = config["task3"]
        self.quality = config["quality"]
        self.shared_labels = config["labels"]["shared"]
        self.exclusive = {"A": self.cfg["exclusive"]["A"], "B": self.cfg["exclusive"]["B"]}
        dev = (windows.split == "development").to_numpy()
        self.dev = {r: dev & (windows.room == r).to_numpy() for r in "AB"}
        self.val = {r: (~dev) & (windows.room == r).to_numpy() for r in "AB"}
        self.all_dev = dev

    def stats(self, F, alignment, room):
        mask = self.dev[room] if alignment == "per_room" else self.all_dev
        return fit_stats(F[mask])

    def target_model(self, F, alignment, head, source, target, support_captures):
        """Classifier over the target label space: shared from source, exclusive from support."""
        src = self.dev[source] & np.isin(self.labels, self.shared_labels)
        sup = self.dev[target] & np.isin(self.capture, support_captures)
        s_stats, t_stats = self.stats(F, alignment, source), self.stats(F, alignment, target)
        train = np.vstack([apply_stats(F[src], s_stats), apply_stats(F[sup], t_stats)])
        y = np.concatenate([self.labels[src], self.labels[sup]])
        return make_head(head, self.cfg["logreg_C"]).fit(train, y), t_stats

    def source_model(self, F, alignment, head, source):
        """Classifier for windows routed to the source room: all source labels."""
        stats = self.stats(F, alignment, source)
        src = self.dev[source]
        return make_head(head, self.cfg["logreg_C"]).fit(apply_stats(F[src], stats), self.labels[src]), stats

    def unrouted_model(self, F, head, source, target, support_captures):
        """No routing: pooled normalization, union of source labels and target-exclusive labels."""
        stats = fit_stats(F[self.all_dev])
        src = self.dev[source]
        sup = self.dev[target] & np.isin(self.capture, support_captures)
        train = apply_stats(F[src | sup], stats)
        return make_head(head, self.cfg["logreg_C"]).fit(train, self.labels[src | sup]), stats

    def balance(self, model, F, stats, reference_mask, balancing):
        """Class weights from the unlabelled windows in `reference_mask`; None when balancing is off."""
        if balancing == "none":
            return None
        return balancing_weights(model.predict_proba(apply_stats(F[reference_mask], stats)))

    def evaluate(self, pred, target):
        return score(self.labels[self.val[target]], pred, self.shared_labels, self.exclusive[target])

    def prior(self, X, structure, target, support_captures):
        """Room-structure prior from unlabelled target development windows and the support windows."""
        shapes, motion, _ = structure
        sup = np.flatnonzero(self.dev[target] & np.isin(self.capture, support_captures))
        ex_shapes, ex_labels, _ = posture_examples(X[sup], self.labels[sup], self.quality,
                                                   self.cfg["structure_segment"])
        return StructurePrior.fit(shapes[self.dev[target]], motion[self.dev[target]], ex_shapes, ex_labels,
                                  C=self.cfg["logreg_C"])

    def smoother(self, F, room):
        """(cluster smoother, standardization) of the room's unlabelled development windows."""
        stats = fit_stats(F[self.dev[room]])
        n_clusters = self.cfg["smoothing_clusters_per_class"] * (len(self.shared_labels) + len(self.exclusive[room]))
        smoother = ClusterSmoother(n_clusters, self.cfg["smoothing_components"])
        return smoother.fit(apply_stats(F[self.dev[room]], stats)), stats

    def system(self, reps, structure, settings, source, target, support_captures, refinements=(), prior=None,
               smoother=None):
        """Validation (labels, confidences) of one episode: the adapted classifier, revised by `prior` if given.

        refinements: "ensemble" averages the balanced probabilities of the SSL and handcrafted models,
        "smoothing" applies `smoother` (from `self.smoother` on SSL features), "sway" adds the sway vote.
        """
        dev, val = self.dev[target], self.val[target]
        P_dev, P_val = [], []
        for name in (ENSEMBLE if "ensemble" in refinements else [settings["representation"]]):
            F = reps[name]
            model, stats = self.target_model(F, settings["alignment"], settings["head"], source, target, support_captures)
            weights = self.balance(model, F, stats, dev, settings["balancing"])
            P_dev.append(balanced_proba(model, apply_stats(F[dev], stats), weights))
            P_val.append(balanced_proba(model, apply_stats(F[val], stats), weights))
        classes, P_dev, P_val = model.classes_, np.mean(P_dev, axis=0), np.mean(P_val, axis=0)
        if "smoothing" in refinements:
            clusters, stats = smoother
            P_val = clusters.smooth(P_val, clusters.assign(apply_stats(reps["ssl"][val], stats)),
                                    clusters.cluster_means(P_dev))
        pred, conf = classes[P_val.argmax(axis=1)], P_val.max(axis=1)
        if prior is None:
            return pred, conf
        shapes, motion, sway = structure
        if "sway" in refinements:
            prior = prior.with_sway(sway[dev], motion[dev], classes[P_dev.argmax(axis=1)], self.shared_labels)
        return prior.revise(shapes[val], motion[val], pred, conf, self.shared_labels, sway[val])

    def reported_system(self, X, reps, structure, settings, source, target, support_captures):
        """The configuration in `settings` (e.g. final_model), fitted from scratch: structure prior and
        refinements only where the prior applies."""
        if not (settings["structure_prior"] == "transitions" and applicable(self.exclusive[target])):
            return self.system(reps, structure, settings, source, target, support_captures)
        refinements = settings.get("refinements", ())
        smoother = self.smoother(reps["ssl"], target) if "smoothing" in refinements else None
        return self.system(reps, structure, settings, source, target, support_captures, refinements,
                           self.prior(X, structure, target, support_captures), smoother)


INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style_k_axis(ax, source, target):
    ax.set_title(f"Target Room {target} (source Room {source})", color=INK, fontsize=10, loc="left")
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 20], ["1", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlabel("K labelled target captures per exclusive activity", color=MUTED)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def _k_curves(frame, series, ylabel, title, out_path):
    """Two panels (one per target room) of metric-vs-K curves; legend below the panels."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), facecolor=SURFACE, sharey=True)
    for ax, (source, target) in zip(axes, DIRECTIONS):
        for select, metric, color, style, label in series:
            rows = frame[(frame.target_room == target) & select(frame)].sort_values("K")
            ax.errorbar(rows.K, rows[f"{metric}_mean"], yerr=rows[f"{metric}_std"], color=color, linestyle=style,
                        linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, capsize=3, label=label)
        _style_k_axis(ax, source, target)
    axes[0].set_ylabel(ylabel, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(series), frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_label_efficiency(summary, out_path, title):
    every = lambda f: f.K > 0
    _k_curves(summary, [
        (every, "primary_macro_f1", "#2a78d6", "-", "All target activities (primary)"),
        (every, "exclusive_macro_f1", "#eb6834", "-", "Target-labelled (K-shot) activities"),
        (every, "shared_macro_f1", "#1baf7a", "-", "Transferred shared activities"),
    ], "Validation macro-F1 (mean ± std over draws)", title, out_path)


def plot_routing(routing, out_path, title):
    mode = lambda name: (lambda f: f.routing == name)
    _k_curves(routing, [
        (mode("none"), "primary_macro_f1", "#eb6834", "--", "No routing (one model, pooled normalization)"),
        (mode("predicted"), "primary_macro_f1", "#2a78d6", "-", "Predicted-room routing (Task 1 router)"),
        (mode("oracle"), "primary_macro_f1", "#1baf7a", ":", "Oracle-room routing"),
    ], "Validation primary macro-F1", title, out_path)


def plot_confusions(confusions, out_path, title):
    """Confusion matrix (counts, shaded by row share) above a per-class precision/recall/F1 table."""
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [3, 1.6]})
    for col, (target, (cm, labels, per_class)) in enumerate(confusions.items()):
        ax = axes[0, col]
        shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(shown, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if shown[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8.5, color=MUTED)
        ax.set_yticks(range(len(labels)), labels, fontsize=8.5, color=MUTED)
        ax.set_xlabel("Predicted", color=MUTED)
        ax.set_ylabel("True", color=MUTED)
        ax.set_title(f"Target Room {target}, K = 5", color=INK, fontsize=10, loc="left")

        table_ax = axes[1, col]
        table_ax.axis("off")
        table = table_ax.table(cellText=[[f"{p:.2f}", f"{r:.2f}", f"{f:.2f}"] for p, r, f in per_class],
                               rowLabels=labels, colLabels=["precision", "recall", "F1"],
                               loc="upper center", cellLoc="center", colWidths=[0.22, 0.22, 0.22])
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.25)
        for cell in table.get_celld().values():
            cell.set_edgecolor(GRID)
    fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def write_per_class_table(confusions, exclusive, out_dir):
    """Per-activity precision, recall and F1 at K = 5 with both target rooms side by side, as Markdown and as a
    LaTeX (booktabs) table; * marks the target-exclusive (K-shot) activities."""
    blocks = []
    for target in ("A", "B"):
        _, labels, per_class = confusions[target]
        scores = dict(zip(labels, per_class))
        blocks.append([(name, name in exclusive[target], scores[name]) for name in TABLE_ROWS[target]])
    md = ["| Target Room A | P | R | F1 | Target Room B | P | R | F1 |", "|---|---:|---:|---:|---|---:|---:|---:|"]
    tex = []
    for i in range(max(len(block) for block in blocks)):
        md_cells, tex_cells = [], []
        for block in blocks:
            if i < len(block):
                name, starred, (p, r, f) = block[i]
                values = [f"{p:.3f}", f"{r:.3f}", f"{f:.3f}"]
                md_cells += [name + ("\\*" if starred else "")] + values
                tex_cells += [name + ("$^{*}$" if starred else "")] + values
            else:
                md_cells += [""] * 4
                tex_cells += [""] * 4
        md.append("| " + " | ".join(md_cells) + " |")
        tex.append("    " + " & ".join(tex_cells) + r" \\")
    (out_dir / "table_per_class_K5.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "table_per_class_K5.tex").write_text("\n".join([
        r"% Generated by App/scripts/task3_cross_room.py (needs \usepackage{booktabs}).",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Per-activity validation precision (P), recall (R) and F1 at $K = 5$ (support draw 0). "
        r"$^{*}$ marks the target-exclusive activities, learned from the $K$ labelled target captures; "
        r"SIT, STAND and WALK are never labelled in the target room.}",
        r"  \label{tab:per_class_k5}",
        r"  \begin{tabular}{lccc@{\qquad}lccc}",
        r"    \toprule",
        r"    \multicolumn{4}{c}{\textbf{Target Room A}} & \multicolumn{4}{c}{\textbf{Target Room B}} \\",
        r"    \cmidrule(lr){1-4}\cmidrule(lr){5-8}",
        r"    Activity & P & R & F1 & Activity & P & R & F1 \\",
        r"    \midrule",
        *tex,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]) + "\n", encoding="utf-8")


def make_figures(out_dir, windows, config):
    """Draw every Task 3 figure, and the per-activity K = 5 table, from the saved result files."""
    ranking = pd.read_csv(out_dir / "grid_ranking.csv")
    ranking = ranking[~ranking.representation.isin(BASELINES)].iloc[0]
    label = f"{ranking.representation}, {ranking.alignment} alignment, {ranking['head']} head"
    if "balancing" in ranking:
        label += f", {ranking.balancing} balancing"
    if "structure_prior" in ranking and ranking.structure_prior != "none":
        label += f", {ranking.structure_prior} prior"
        if config["task3"].get("refinements"):
            label += f"\n+ {', '.join(config['task3']['refinements'])} where the prior applies (target Room A)"
    routing_summary = pd.read_csv(out_dir / "routing_summary.csv")
    plot_label_efficiency(routing_summary[routing_summary.routing == "predicted"],
                          out_dir / "fig_label_efficiency.png", f"Label efficiency (predicted-room routing) - {label}")
    plot_routing(routing_summary, out_dir / "fig_routing_contribution.png",
                 f"Routing contribution - {label}\n"
                 "(predicted-room and oracle routing coincide: the Task 1 router is 100% correct)")

    truth_by_id = windows.set_index("window_id").label
    shared = config["labels"]["shared"]
    confusions, per_class_rows = {}, []
    for source, target in DIRECTIONS:
        predictions = pd.read_csv(out_dir / f"predictions_{source}_to_{target}_K5.csv")
        truth = truth_by_id.loc[predictions.window_id].to_numpy()
        labels = sorted(shared) + sorted(config["task3"]["exclusive"][target])
        cm = confusion_matrix(truth, predictions.label, labels=labels)
        p, r, f, s = precision_recall_fscore_support(truth, predictions.label, labels=labels, zero_division=0)
        confusions[target] = (cm, labels, list(zip(p, r, f)))
        per_class_rows += [{"target_room": target, "K": 5, "label": lab, "precision": round(pi, 4),
                            "recall": round(ri, 4), "f1": round(fi, 4), "support": int(si)}
                           for lab, pi, ri, fi, si in zip(labels, p, r, f, s)]
    pd.DataFrame(per_class_rows).to_csv(out_dir / "per_class_K5.csv", index=False)
    plot_confusions(confusions, out_dir / "fig_confusion_K5.png",
                    f"Confusion matrices at K = 5 (support draw 0) - {label}")
    write_per_class_table(confusions, config["task3"]["exclusive"], out_dir)


def summarize(frame, keys):
    agg = frame.groupby(keys)[METRICS].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--replot", action="store_true",
                        help="only redraw the figures and the K = 5 table from saved results")
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config["task3"]
    k_values = config["few_shot"]["k_values"]
    seed = config["few_shot"]["support_seed"]
    torch.set_num_threads(config["ssl"]["threads"])

    data_dir = config["paths"]["data_dir"]
    results_dir = config["paths"]["results_dir"]
    out_dir = results_dir / "task3"
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.read_csv(data_dir / "windows_manifest.csv")
    if args.replot:
        make_figures(out_dir, windows, config)
        return
    X = np.load(data_dir / "windows_X.npy", mmap_mode="r")
    print("Computing representations...", flush=True)
    reps = build_representations(X, config, results_dir / "task2")
    exp = Experiment(windows, config)

    # Support sets (capture level), saved for provenance.
    supports, support_rows = {}, []
    for source, target in DIRECTIONS:
        for draw in range(cfg["support_draws"]):
            sets = select_support(windows, target, exp.exclusive[target], k_values, seed, draw)
            supports[(target, draw)] = sets
            for k, captures in sets.items():
                for capture in captures:
                    label = windows.loc[windows.capture_id == capture, "label"].iloc[0]
                    n = int((windows.capture_id == capture).sum())
                    support_rows.append({"source_room": source, "target_room": target, "draw": draw, "K": k,
                                         "label": label, "capture_id": capture, "windows": n})
    support_table = pd.DataFrame(support_rows)
    assert (windows.set_index("capture_id").loc[support_table.capture_id, "split"] == "development").all()
    support_table.to_csv(out_dir / "support_sets.csv", index=False)

    # Target-room structure priors: fitted only from unlabelled target development windows and the
    # support windows (never from target SIT/STAND/WALK labels); independent of the representation.
    use_prior = {room: applicable(exp.exclusive[room]) for room in "AB"}
    priors, structure = {}, None
    if "transitions" in cfg["structure_prior"] and any(use_prior.values()):
        structure = structure_features(X, config["quality"])
        for source, target in DIRECTIONS:
            if not use_prior[target]:
                continue
            for draw in range(cfg["support_draws"]):
                for k in k_values:
                    priors[(target, draw, k)] = exp.prior(X, structure, target, supports[(target, draw)][k])
        print(f"  structure prior fitted for target room(s) {[r for r in 'AB' if use_prior[r]]}", flush=True)

    def revise(pred, conf, target, draw, k, prior):
        if prior == "none" or not use_prior[target]:
            return pred, conf
        val = exp.val[target]
        shapes, motion, _ = structure
        return priors[(target, draw, k)].revise(shapes[val], motion[val], pred, conf, exp.shared_labels)

    # 1. Grid with oracle routing.
    rows = []
    for rep_name, F in reps.items():
        for alignment in cfg["alignments"]:
            for head in cfg["heads"]:
                for source, target in DIRECTIONS:
                    for k in k_values:
                        for draw in range(cfg["support_draws"]):
                            model, t_stats = exp.target_model(F, alignment, head, source, target,
                                                              supports[(target, draw)][k])
                            F_val = apply_stats(F[exp.val[target]], t_stats)
                            for balancing in cfg["balancing"]:
                                weights = exp.balance(model, F, t_stats, exp.dev[target], balancing)
                                pred, conf = predict(model, F_val, weights)
                                for prior in cfg["structure_prior"]:
                                    final, _ = revise(pred, conf, target, draw, k, prior)
                                    rows.append({"representation": rep_name, "alignment": alignment, "head": head,
                                                 "balancing": balancing, "structure_prior": prior,
                                                 "source_room": source, "target_room": target,
                                                 "K": k, "draw": draw, **exp.evaluate(final, target)})
                config_rows = pd.DataFrame(rows)
                last = config_rows[(config_rows.representation == rep_name) & (config_rows.alignment == alignment)
                                   & (config_rows["head"] == head)]
                means = last.groupby(["balancing", "structure_prior"]).primary_macro_f1.mean()
                print(f"  {rep_name:12s} {alignment:8s} {head:9s} mean primary macro-F1 "
                      + "  ".join(f"{b}/{p}: {value:.4f}" for (b, p), value in means.items()), flush=True)
    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "grid_all_runs.csv", index=False)
    config_keys = ["representation", "alignment", "head", "balancing", "structure_prior"]
    grid_summary = summarize(grid, config_keys + ["target_room", "K"])
    grid_summary.to_csv(out_dir / "grid_summary.csv", index=False)
    overall = grid.groupby(config_keys)["primary_macro_f1"].mean().sort_values(ascending=False)
    overall.rename("mean_primary_macro_f1").reset_index().to_csv(out_dir / "grid_ranking.csv", index=False)
    candidates = overall[~overall.index.get_level_values("representation").isin(BASELINES)]
    best_rep, best_align, best_head, best_balancing, best_prior = candidates.index[0]
    print(f"\nBest configuration on validation: {best_rep} / {best_align} / {best_head} / {best_balancing} balancing"
          f" / {best_prior} structure prior (mean primary macro-F1 {candidates.iloc[0]:.4f})", flush=True)

    # What SSL pretraining contributes: every representation with the best alignment, head, balancing and prior.
    same_settings = grid[(grid.alignment == best_align) & (grid["head"] == best_head)
                         & (grid.balancing == best_balancing) & (grid.structure_prior == best_prior)]
    effect = same_settings.groupby(["target_room", "K", "representation"])[
        ["primary_macro_f1", "exclusive_macro_f1", "shared_macro_f1"]].mean().unstack("representation").round(4)
    effect.to_csv(out_dir / "representation_effect.csv")
    print("\nRepresentations with the best settings (mean primary macro-F1 over K and draws):")
    print(same_settings.groupby(["target_room", "representation"]).primary_macro_f1.mean().unstack("target_room")
          .round(4).to_string())

    # Effect of balancing and of the structure prior for the best representation, alignment and head.
    same_model = grid[(grid.representation == best_rep) & (grid.alignment == best_align) & (grid["head"] == best_head)]
    for name, key, fixed in (("balancing", "balancing", ("structure_prior", best_prior)),
                             ("structure_prior", "structure_prior", ("balancing", best_balancing))):
        effect = same_model[same_model[fixed[0]] == fixed[1]].groupby(["target_room", "K", key])[
            ["primary_macro_f1", "exclusive_macro_f1", "shared_macro_f1"]].mean().unstack(key).round(4)
        effect.to_csv(out_dir / f"{name}_effect.csv")
        print(f"\nEffect of {name} (mean over draws, {fixed[0]} = {fixed[1]}):")
        print(effect.to_string())

    # 2. Refinements added one at a time to the best configuration (sway needs the structure prior).
    best = {"representation": best_rep, "alignment": best_align, "head": best_head, "balancing": best_balancing,
            "structure_prior": best_prior, "refinements": list(cfg["refinements"])}
    smoothers = {room: exp.smoother(reps["ssl"], room) for room in "AB"}
    steps = [cfg["refinements"][:i] for i in range(len(cfg["refinements"]) + 1)]
    rows = []
    for source, target in DIRECTIONS:
        for draw in range(cfg["support_draws"]):
            for k in k_values:
                prior = priors.get((target, draw, k)) if best_prior == "transitions" else None
                for step in steps:
                    pred, _ = exp.system(reps, structure, best, source, target, supports[(target, draw)][k], step,
                                         prior, smoothers[target])
                    rows.append({"target_room": target, "K": k, "draw": draw,
                                 "refinements": " + ".join(step) or "none", **exp.evaluate(pred, target)})
    refinement = pd.DataFrame(rows)
    effect = refinement.groupby(["target_room", "K", "refinements"])[
        ["primary_macro_f1", "exclusive_macro_f1", "shared_macro_f1"]].mean().unstack("refinements").round(4)
    effect.to_csv(out_dir / "refinement_effect.csv")
    print("\nRefinements added one at a time (mean primary macro-F1 over K and draws; "
          "the reported system uses them only where the structure prior applies):")
    print(refinement.groupby(["target_room", "refinements"]).primary_macro_f1.mean().unstack("target_room")
          .round(4).to_string())

    # 3. Routing study, predictions and confusion matrices for the reported system.
    F = reps[best_rep]
    router = pd.read_csv(results_dir / "task1" / "predictions_validation.csv").set_index("window_id")["room"]
    routing_rows = []
    for source, target in DIRECTIONS:
        val_ids = windows.window_id.to_numpy()[exp.val[target]]
        routed_room = router.reindex(val_ids).to_numpy()
        source_model, s_stats = exp.source_model(F, best_align, best_head, source)
        source_weights = exp.balance(source_model, F, s_stats, exp.dev[source], best_balancing)
        source_pred, source_conf = predict(source_model, apply_stats(F[exp.val[target]], s_stats), source_weights)
        with_prior = best_prior == "transitions" and use_prior[target]
        refinements = list(cfg["refinements"]) if with_prior else []
        for k in k_values:
            for draw in range(cfg["support_draws"]):
                support = supports[(target, draw)][k]
                oracle_pred, oracle_conf = exp.system(reps, structure, best, source, target, support, refinements,
                                                      priors[(target, draw, k)] if with_prior else None,
                                                      smoothers[target])
                to_target = routed_room == target
                routed_pred = np.where(to_target, oracle_pred, source_pred)
                routed_conf = np.where(to_target, oracle_conf, source_conf)
                unrouted, u_stats = exp.unrouted_model(F, best_head, source, target, support)
                u_weights = exp.balance(unrouted, F, u_stats, exp.all_dev, best_balancing)
                none_pred, _ = predict(unrouted, apply_stats(F[exp.val[target]], u_stats), u_weights)
                for mode, pred in (("oracle", oracle_pred), ("predicted", routed_pred), ("none", none_pred)):
                    routing_rows.append({"source_room": source, "target_room": target, "K": k, "draw": draw,
                                         "routing": mode, **exp.evaluate(pred, target)})
                if draw == 0:
                    pd.DataFrame({"window_id": val_ids, "room": routed_room, "label": routed_pred,
                                  "confidence": routed_conf.round(4)}).to_csv(
                        out_dir / f"predictions_{source}_to_{target}_K{k}.csv", index=False)
    routing = pd.DataFrame(routing_rows)
    routing.to_csv(out_dir / "routing_all_runs.csv", index=False)
    routing_summary = summarize(routing, ["source_room", "target_room", "routing", "K"])
    routing_summary.to_csv(out_dir / "routing_summary.csv", index=False)

    # 4. Metrics in the challenge format (mean over support draws; std as its own metric).
    best_summary = routing_summary[routing_summary.routing == "predicted"]
    metric_rows = []
    for row in best_summary.itertuples(index=False):
        for metric in METRICS:
            for stat in ("mean", "std"):
                metric_rows.append({"experiment": f"task3_{row.source_room}_to_{row.target_room}",
                                    "target_room": row.target_room, "K": row.K,
                                    "metric": metric if stat == "mean" else f"{metric}_std",
                                    "value": round(getattr(row, f"{metric}_{stat}"), 4)})
    for row in grid_summary.itertuples(index=False):
        metric_rows.append({"experiment": f"task3_grid_{row.representation}_{row.alignment}_{row.head}_{row.balancing}"
                                          f"_{row.structure_prior}",
                            "target_room": row.target_room, "K": row.K, "metric": "primary_macro_f1",
                            "value": round(row.primary_macro_f1_mean, 4)})
    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)

    make_figures(out_dir, windows, config)
    print("\nPredicted-routing results of the best configuration (mean over draws):")
    print(best_summary[["source_room", "target_room", "K", "primary_macro_f1_mean", "exclusive_macro_f1_mean",
                        "shared_macro_f1_mean", "balanced_accuracy_mean"]].round(3).to_string(index=False))
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
