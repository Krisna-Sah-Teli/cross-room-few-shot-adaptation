"""End-to-end inference for a challenge episode (used by run_episode.py).

An episode is a folder of inputs to label:
  - window files: window_*.npy, each a (150, 256) float32 CSI-amplitude window, and/or
  - raw captures: *.pcap (Nexmon BCM4366c0, Room A format) or *.csv (256 amplitude
    columns, Room B format). Raw captures go through the same preprocessing as the
    dataset build: decoding, quality control and centred 150-observation windows.
Optional files in the folder:
  - support.csv: labelled target-room windows (columns window_file,label[,room]) for
    few-shot adaptation. For every label it contains, it replaces the default support
    (development captures) of that room.
  - labels.csv: ground truth (columns window_id,label[,room]) used only to print scores.

Each window is routed to a room by the Task 1 router and classified by that room's Task 3
model: shared activities learned from the other room's development windows, target-exclusive
activities from the support set, with per-room alignment and class balancing computed on
unlabelled development windows. For a room whose K-shot activities include SITDOWN and
STANDUP (Room A), the shared activities are then revised by the room-structure prior
(room_structure.py). All inputs use the full 256 subcarriers.
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from scripts.config import preprocessing_fingerprint
from scripts.csi_quality import classify_band_coverage, consecutive_duplicate_mask
from scripts.few_shot_adaptation import (apply_stats, balancing_weights, fit_stats, make_head, predict,
                                         select_support)
from scripts.room_a_pcap import read_room_a_capture_qc
from scripts.room_fingerprint import normalize, window_features
from scripts.room_structure import StructurePrior, applicable, posture_examples, structure_features
from scripts.ssl_encoder import CSIEncoder, embed, prepare_input
from scripts.window_extraction import centered_tiling_starts

ROOMS = ("A", "B")
RESERVED_FILES = {"support.csv", "labels.csv", "predictions.csv", "predictions_details.csv"}


# ------------------------------------------------------------------ inputs

def validate_window(array, name):
    array = np.asarray(array, dtype=np.float32)
    if array.shape != (150, 256):
        raise ValueError(f"{name}: expected shape (150, 256), got {array.shape}")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name}: amplitudes must be finite and non-negative")
    return array


def windows_from_capture(path, config):
    """Raw capture -> list of (window_id, (150, 256) window), with the dataset-build preprocessing."""
    path = Path(path)
    quality = config["quality"]
    if path.suffix.lower() == ".pcap":
        capture = read_room_a_capture_qc(
            path.read_bytes(),
            expected_subcarriers=config["room_a"]["expected_subcarriers"],
            udp_port=config["room_a"]["udp_port"],
            salvage_truncated=config["room_a"]["salvage_truncated"],
            drop_consecutive_duplicates=quality["drop_consecutive_duplicates"],
        )
        # The dataset build stores decoded amplitudes with 4 decimals in the intermediate CSV.
        amplitude = np.round(capture.amplitude.astype(np.float64), 4)
    else:
        values = pd.read_csv(path).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if values.shape[1] != 256:
            raise ValueError(f"{path.name}: expected 256 amplitude columns, got {values.shape[1]}")
        values = values[np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)]
        if quality["drop_consecutive_duplicates"]:
            values = values[~consecutive_duplicate_mask(values)]
        amplitude = values
    amplitude = amplitude[classify_band_coverage(amplitude, quality) != "empty"].astype(np.float32)
    length = config["window"]["length"]
    return [(f"{path.stem}_w{i}", amplitude[start:start + length])
            for i, start in enumerate(centered_tiling_starts(len(amplitude), length))]


def load_episode_inputs(folder, config):
    """Collect every window of an episode folder: window files first, then raw captures."""
    folder = Path(folder)
    ids, windows, sources = [], [], []
    for path in sorted(folder.glob("*.npy")):
        ids.append(path.stem)
        windows.append(validate_window(np.load(path), path.name))
        sources.append(path.name)
    raw = sorted(p for p in folder.iterdir()
                 if p.suffix.lower() in (".pcap", ".csv") and p.name.lower() not in RESERVED_FILES)
    for path in raw:
        produced = windows_from_capture(path, config)
        if not produced:
            print(f"  note: {path.name} has fewer than {config['window']['length']} valid observations; no window")
        for window_id, window in produced:
            ids.append(window_id)
            windows.append(validate_window(window, window_id))
            sources.append(path.name)
    if not windows:
        raise FileNotFoundError(f"No window_*.npy files or raw captures (.pcap/.csv) found in {folder}")
    if len(set(ids)) != len(ids):
        raise ValueError("Window ids must be unique within an episode")
    return ids, np.stack(windows), sources


# ------------------------------------------------------------------ the trained system

def compute_features(bundle, X):
    inputs = prepare_input(X)
    if bundle["representation"] == "handcrafted":
        return window_features(inputs)
    encoder = CSIEncoder(bundle["encoder"]["in_channels"])
    encoder.load_state_dict(bundle["encoder"]["state_dict"])
    return embed(encoder, inputs)


def route(bundle, X):
    """Task 1 router: (room per window, router confidence)."""
    router = bundle["router"]
    features = np.vstack([window_features(normalize(X[i:i + 400], router["norm"])) for i in range(0, len(X), 400)])
    proba = router["model"].predict_proba(features)
    return np.array(ROOMS)[proba.argmax(axis=1)], proba.max(axis=1)


def fit_room_model(bundle, room, support_features, support_labels):
    """Classifier for windows routed to `room`: shared activities from the other room plus the support set."""
    settings = bundle["settings"]
    other = bundle["rooms"]["B" if room == "A" else "A"]
    train = np.vstack([other["shared_features"], support_features])
    labels = np.concatenate([other["shared_labels"], support_labels])
    model = make_head(settings["head"], settings["logreg_C"]).fit(train, labels)
    weights = None
    if settings["balancing"] == "uniform":
        weights = balancing_weights(model.predict_proba(bundle["rooms"][room]["dev_features"]))
    return model, weights


def build_bundle(config):
    """Train the deployed system from the development data and the Task 1 / Task 2 outputs."""
    started = time.time()
    settings = dict(config["final_model"], logreg_C=config["task3"]["logreg_C"],
                    structure_segment=config["task3"]["structure_segment"])
    quality = config["quality"]
    paths = config["paths"]
    windows = pd.read_csv(paths["data_dir"] / "windows_manifest.csv")
    X = np.load(paths["data_dir"] / "windows_X.npy", mmap_mode="r")
    with open(paths["results_dir"] / "task1" / "room_router.pkl", "rb") as f:
        router = pickle.load(f)
    checkpoint = torch.load(paths["results_dir"] / "task2" / "encoder.pt", weights_only=False)
    bundle = {
        "settings": settings,
        "representation": settings["representation"],
        "router": router,
        "encoder": {"state_dict": checkpoint["state_dict"], "in_channels": checkpoint["in_channels"]},
        "label_sets": {room: list(config["labels"]["shared"]) + list(config["task3"]["exclusive"][room])
                       for room in ROOMS},
        "preprocessing_id": preprocessing_fingerprint(config),
        "quality": quality,
        "shared_labels": list(config["labels"]["shared"]),
        "rooms": {},
    }
    dev = (windows.split == "development").to_numpy()
    dev_index = np.flatnonzero(dev)
    dev_features = compute_features(bundle, np.asarray(X[dev_index]))
    features = np.zeros((len(windows), dev_features.shape[1]), dtype=np.float32)
    features[dev_index] = dev_features
    labels = windows.label.to_numpy()
    capture = windows.capture_id.to_numpy()

    pooled = fit_stats(features[dev])
    for room in ROOMS:
        in_room = dev & (windows.room == room).to_numpy()
        stats = fit_stats(features[in_room]) if settings["alignment"] == "per_room" else pooled
        shared = in_room & np.isin(labels, config["labels"]["shared"])
        support_ids = select_support(windows, room, config["task3"]["exclusive"][room], [settings["k"]],
                                     config["few_shot"]["support_seed"], settings["support_draw"])[settings["k"]]
        support = in_room & np.isin(capture, support_ids)
        structure = None
        if settings["structure_prior"] == "transitions" and applicable(config["task3"]["exclusive"][room]):
            dev_shapes, dev_motion = structure_features(X[np.flatnonzero(in_room)], quality)
            sup_index = np.flatnonzero(support)
            examples = posture_examples(X[sup_index], labels[sup_index], quality, settings["structure_segment"])
            structure = {"prior": StructurePrior.fit(dev_shapes, dev_motion, examples[0], examples[1],
                                                     C=settings["logreg_C"]),
                         "examples": examples}
        bundle["rooms"][room] = {
            "stats": stats,
            "dev_features": apply_stats(features[in_room], stats).astype(np.float32),
            "shared_features": apply_stats(features[shared], stats).astype(np.float32),
            "shared_labels": labels[shared],
            "support_features": apply_stats(features[support], stats).astype(np.float32),
            "support_labels": labels[support],
            "support_captures": list(support_ids),
            "structure": structure,
        }
    for room in ROOMS:
        r = bundle["rooms"][room]
        r["model"], r["weights"] = fit_room_model(bundle, room, r["support_features"], r["support_labels"])
    bundle["build_seconds"] = round(time.time() - started, 1)
    return bundle


def save_bundle(bundle, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------ prediction

def predict_windows(bundle, X, support=None):
    """Predict room and activity for windows X (n, 150, 256).

    `support` (optional) is a list of (window, label, room or None) from the episode.
    Returns a DataFrame with room, room_confidence, label, confidence.
    """
    rooms, room_confidence = route(bundle, X)
    features = compute_features(bundle, X)
    models = {room: (bundle["rooms"][room]["model"], bundle["rooms"][room]["weights"]) for room in ROOMS}
    priors = {room: (bundle["rooms"][room].get("structure") or {}).get("prior") for room in ROOMS}

    if support:
        S = np.stack([validate_window(w, "support window") for w, _, _ in support])
        s_labels = np.array([label for _, label, _ in support])
        given = [room for _, _, room in support]
        s_rooms = np.where([g in ROOMS for g in given], given, route(bundle, S)[0])
        s_features = compute_features(bundle, S)
        for room in ROOMS:
            mine = s_rooms == room
            if not mine.any():
                continue
            unknown = set(s_labels[mine]) - set(bundle["label_sets"][room])
            if unknown:
                raise ValueError(f"Support labels {sorted(unknown)} are not activities of Room {room}")
            r = bundle["rooms"][room]
            keep = ~np.isin(r["support_labels"], s_labels[mine])      # default support for the other labels
            new_features = np.vstack([r["support_features"][keep], apply_stats(s_features[mine], r["stats"])])
            new_labels = np.concatenate([r["support_labels"][keep], s_labels[mine]])
            models[room] = fit_room_model(bundle, room, new_features, new_labels)
            if r.get("structure"):
                shapes, ex_labels, sources = r["structure"]["examples"]
                new = posture_examples(S[mine], s_labels[mine], bundle["quality"],
                                       bundle["settings"]["structure_segment"])
                kept = ~np.isin(sources, s_labels[mine])                    # default examples of other labels
                all_labels = np.concatenate([ex_labels[kept], new[1]])
                priors[room] = (r["structure"]["prior"].refit(np.vstack([shapes[kept], new[0]]), all_labels,
                                                               bundle["settings"]["logreg_C"])
                                if {"SIT", "STAND"} <= set(all_labels) else None)

    label = np.empty(len(X), dtype=object)
    confidence = np.zeros(len(X))
    for room in ROOMS:
        mine = rooms == room
        if mine.any():
            model, weights = models[room]
            label[mine], confidence[mine] = predict(model, apply_stats(features[mine], bundle["rooms"][room]["stats"]),
                                                    weights)
            if priors[room] is not None:
                shapes, motion = structure_features(X[mine], bundle["quality"])
                label[mine], confidence[mine] = priors[room].revise(shapes, motion, label[mine], confidence[mine],
                                                                    bundle["shared_labels"])
    return pd.DataFrame({"room": rooms, "room_confidence": room_confidence.round(4),
                         "label": label, "confidence": confidence.round(4)})


def read_support(folder):
    """support.csv -> list of (window, label, room or None)."""
    folder = Path(folder)
    table = pd.read_csv(folder / "support.csv")
    missing = {"window_file", "label"} - set(table.columns)
    if missing:
        raise ValueError(f"support.csv is missing columns {sorted(missing)}")
    return [(np.load(folder / row.window_file), str(row.label).upper(),
             str(getattr(row, "room", "")).upper() or None) for row in table.itertuples()]


def score_predictions(predictions, labels_file):
    """Print accuracy and macro-F1 against an episode's labels.csv."""
    truth = pd.read_csv(labels_file)
    merged = predictions.merge(truth, on="window_id", suffixes=("", "_true"))
    if merged.empty:
        print("labels.csv has no window_id in common with the predictions")
        return {}
    scores = {
        "windows scored": len(merged),
        "activity accuracy": accuracy_score(merged.label_true, merged.label),
        "activity balanced accuracy": balanced_accuracy_score(merged.label_true, merged.label),
        "activity macro-F1": f1_score(merged.label_true, merged.label, average="macro", zero_division=0),
    }
    if "room_true" in merged:
        scores["room accuracy"] = accuracy_score(merged.room_true.astype(str), merged.room)
    for name, value in scores.items():
        print(f"  {name:28s} {value if isinstance(value, int) else round(value, 4)}")
    return scores
