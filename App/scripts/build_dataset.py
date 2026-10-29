"""Rebuild the common CSI dataset from the raw challenge archives.

Stages (all outputs are regenerated on every run):
  1. Room A: decode every PCAP with the project's Python decoder.
  2. Room B: create the fixed capture-level validation split (seed 42) and read
     every amplitude CSV.
  3. Quality control for both rooms: reject zero-information and repeated
     observations, classify band coverage, record every rejection.
  4. Export one intermediate CSV per capture (256 amplitude columns).
  5. Cut 150 x 256 windows from the intermediate CSVs only, and write each window
     as its own window_XXXXX.npy file (the model-input contract).
  6. Write manifests, provenance, and a QC report.

Usage:
    python App/scripts/build_dataset.py [--config App/configs/experiment.yaml]
    python App/scripts/build_dataset.py --export-windows-only     # window files from an existing build
"""

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import PROJECT_ROOT, load_config, preprocessing_fingerprint
from scripts.csi_quality import BAND_CLASSES, band_counts, classify_band_coverage, consecutive_duplicate_mask
from scripts.room_a_pcap import list_pcap_members, read_room_a_capture_qc
from scripts.room_b_split import create_room_b_split
from scripts.window_extraction import centered_tiling_starts

COMMON_COLUMNS = [f"subcarrier_{i:03d}" for i in range(1, 257)]
ROOM_A_ACTIVITIES = {"fall", "sit", "sitdown", "stand", "standup", "walk"}


def room_a_identity(member):
    """Return (activity, split) from a Room A archive path; labels only, never features."""
    parts = [p.lower() for p in PurePosixPath(member).parts]
    after = parts[parts.index("csidata") + 1:]
    split = "validation" if "validation" in after else "development"
    activity = next(p for p in after if p in ROOM_A_ACTIVITIES)
    return activity, split


def reject_empty(amplitude, quality):
    """Drop zero-information observations; return kept rows, band classes and drop count."""
    classes = classify_band_coverage(amplitude, quality)
    keep = classes != "empty"
    return amplitude[keep], classes[keep], int((~keep).sum())


def write_intermediate(amplitude, path, float_format=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(amplitude, columns=COMMON_COLUMNS).to_csv(path, index=False, float_format=float_format)


def process_room_a(config, intermediate_dir):
    quality = config["quality"]
    zip_path = config["paths"]["room_a_zip"]
    captures, observations = [], []
    with zipfile.ZipFile(zip_path) as archive:
        for member in list_pcap_members(zip_path):
            activity, split = room_a_identity(member)
            capture_id = f"A__{split}__{activity}__{PurePosixPath(member).stem}"
            capture = read_room_a_capture_qc(
                archive.read(member),
                expected_subcarriers=config["room_a"]["expected_subcarriers"],
                udp_port=config["room_a"]["udp_port"],
                salvage_truncated=config["room_a"]["salvage_truncated"],
                drop_consecutive_duplicates=quality["drop_consecutive_duplicates"],
            )
            classes = classify_band_coverage(capture.amplitude, quality)
            keep = classes != "empty"
            amplitude = capture.amplitude[keep]
            csv_path = intermediate_dir / "room_A" / split / f"{capture_id}.csv"
            if len(amplitude):
                write_intermediate(amplitude, csv_path, float_format="%.4f")

            counts = capture.counts
            captures.append({
                "capture_id": capture_id, "room": "A", "split": split, "activity": activity,
                "label": config["labels"]["room_a"][activity], "source_member": member,
                "intermediate_csv": csv_path.relative_to(PROJECT_ROOT).as_posix() if len(amplitude) else "",
                "raw_records": counts["packets"], "truncated": counts["truncated"],
                "rejected_not_csi": counts["not_udp"] + counts["wrong_port"] + counts["bad_magic"],
                "rejected_bad_length": counts["bad_length"], "rejected_zero": counts["zero_csi"] + int((~keep).sum()),
                "rejected_duplicate": counts["duplicate"], "observations": len(amplitude),
                **band_counts(classes[keep]),
            })
            for row, (ts, seq, pidx, band) in enumerate(zip(
                capture.timestamp_seconds[keep], capture.sequence[keep],
                capture.packet_index[keep], classes[keep],
            )):
                observations.append((capture_id, row, round(float(ts), 6), int(seq), int(pidx), band))
    obs_frame = pd.DataFrame(observations, columns=[
        "capture_id", "row", "timestamp_s", "sequence", "packet_index", "band_class"])
    return pd.DataFrame(captures), obs_frame


def process_room_b(config, intermediate_dir, data_dir):
    quality = config["quality"]
    zip_path = config["paths"]["room_b_zip"]
    split_rows = create_room_b_split(
        zip_path, data_dir / "room_b_capture_split.csv",
        validation_per_class=config["room_b"]["validation_per_class"],
        seed=config["room_b"]["split_seed"],
    )
    captures = []
    with zipfile.ZipFile(zip_path) as archive:
        for row in split_rows:
            member, activity, split = row["capture_path"], row["activity"], row["split"]
            capture_id = f"B__{split}__{activity}__{PurePosixPath(member).stem}"
            frame = pd.read_csv(archive.open(member))
            record = {
                "capture_id": capture_id, "room": "B", "split": split, "activity": activity,
                "label": config["labels"]["room_b"][activity], "source_member": member,
                "intermediate_csv": "", "raw_records": len(frame), "truncated": False,
                "rejected_not_csi": 0, "rejected_bad_length": 0, "rejected_zero": 0,
                "rejected_duplicate": 0, "observations": 0,
                **{f"n_{name}": 0 for name in BAND_CLASSES},
            }
            if frame.shape[1] != 256:
                record["rejected_bad_length"] = len(frame)
                captures.append(record)
                continue
            values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
            malformed = ~np.isfinite(values).all(axis=1) | (values < 0).any(axis=1)
            values = values[~malformed]
            duplicate = consecutive_duplicate_mask(values) if quality["drop_consecutive_duplicates"] else np.zeros(len(values), bool)
            values = values[~duplicate]
            values, classes, n_empty = reject_empty(values, quality)

            csv_path = intermediate_dir / "room_B" / split / f"{capture_id}.csv"
            if len(values):
                as_int = np.array_equal(values, np.round(values))
                write_intermediate(values.astype(np.int64) if as_int else values, csv_path,
                                   float_format=None if as_int else "%.4f")
            record.update({
                "intermediate_csv": csv_path.relative_to(PROJECT_ROOT).as_posix() if len(values) else "",
                "rejected_bad_length": int(malformed.sum()), "rejected_zero": n_empty,
                "rejected_duplicate": int(duplicate.sum()), "observations": len(values),
                **band_counts(classes),
            })
            captures.append(record)
    return pd.DataFrame(captures)


def build_windows(captures, config, fingerprint):
    """Cut windows from the intermediate CSVs (the common form) and record provenance."""
    length = config["window"]["length"]
    quality = config["quality"]
    tensors, rows = [], []
    for capture in captures.itertuples(index=False):
        if not capture.intermediate_csv:
            continue
        amplitude = pd.read_csv(PROJECT_ROOT / capture.intermediate_csv).to_numpy(dtype=np.float32)
        assert amplitude.shape[1] == 256 and np.isfinite(amplitude).all()
        for window_index, start in enumerate(centered_tiling_starts(len(amplitude), length)):
            window = amplitude[start:start + length]
            tensors.append(window)
            rows.append({
                "window_id": f"w{len(rows):05d}", "tensor_index": len(rows),
                "capture_id": capture.capture_id, "room": capture.room, "split": capture.split,
                "activity": capture.activity, "label": capture.label,
                "window_index": window_index, "start_row": start, "end_row": start + length,
                "intermediate_csv": capture.intermediate_csv, "source_member": capture.source_member,
                "preprocessing_id": fingerprint,
                **band_counts(classify_band_coverage(window, quality)),
            })
    X = np.stack(tensors).astype(np.float32)
    return X, pd.DataFrame(rows)


def export_window_files(X, windows, windows_dir):
    """Write every window as its own window_XXXXX.npy (150 x 256 float32, raw amplitude), one folder per split.

    The number in the file name is the window's tensor index, so window_id w00123 is window_00123.npy.
    Window files left over from a previous build are removed first. Returns the relative paths.
    """
    for split in ("development", "validation"):
        for stale in (windows_dir / split).glob("window_*.npy"):
            stale.unlink()
    paths = []
    for row in windows.itertuples(index=False):
        path = windows_dir / row.split / f"window_{row.tensor_index:05d}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.ascontiguousarray(X[row.tensor_index], dtype=np.float32))
        paths.append(path.relative_to(PROJECT_ROOT).as_posix())
    return paths


def qc_report(captures, windows, length):
    lines = ["# Data quality report", ""]
    for room in ("A", "B"):
        room_caps = captures[captures.room == room]
        lines += [f"## Room {room}", ""]
        totals = room_caps[["raw_records", "rejected_not_csi", "rejected_bad_length", "rejected_zero",
                            "rejected_duplicate", "observations"]].sum()
        lines += ["| quantity | count |", "|---|---:|"]
        lines += [f"| {name} | {int(value)} |" for name, value in totals.items()]
        bands = room_caps[[f"n_{name}" for name in BAND_CLASSES]].sum()
        kept = max(int(bands.sum()), 1)
        lines += ["", "Band coverage of kept observations:", ""]
        lines += [f"- {name[2:]}: {int(v)} ({100 * v / kept:.1f}%)" for name, v in bands.items()]
        truncated = room_caps[room_caps.truncated]
        if len(truncated):
            lines += ["", "Truncated captures (complete packets before the truncation were kept):", ""]
            lines += [f"- `{r.source_member}`: {r.observations} observations kept" for r in truncated.itertuples()]
        short = room_caps[room_caps.observations < length]
        lines += ["", f"Captures with fewer than {length} valid observations (no window): {len(short)}", ""]
        lines += [f"- `{r.capture_id}` ({r.split}): {r.observations}" for r in short.itertuples()]
        table = (windows[windows.room == room].groupby(["label", "split"]).size()
                 .unstack(fill_value=0))
        lines += ["", "Windows per label and split:", "", markdown_table(table), ""]
    return "\n".join(lines)


def markdown_table(frame):
    header = [frame.index.name or ""] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for index, row in frame.iterrows():
        lines.append("| " + " | ".join([str(index)] + [str(v) for v in row]) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--export-windows-only", action="store_true",
                        help="only write the window_XXXXX.npy files from an existing build")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    if args.export_windows_only:
        windows = pd.read_csv(paths["data_dir"] / "windows_manifest.csv")
        X = np.load(paths["data_dir"] / "windows_X.npy", mmap_mode="r")
        windows["window_file"] = export_window_files(X, windows, paths["windows_dir"])
        windows.to_csv(paths["data_dir"] / "windows_manifest.csv", index=False)
        print(f"Wrote {len(windows)} window files to {paths['windows_dir']}")
        return
    for key in ("room_a_zip", "room_b_zip"):
        if not paths[key].is_file():
            sys.exit(f"Missing source archive: {paths[key]}")
    data_dir, intermediate_dir = paths["data_dir"], paths["intermediate_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = preprocessing_fingerprint(config)

    started = time.time()
    room_a, room_a_obs = process_room_a(config, intermediate_dir)
    print(f"Room A: {len(room_a)} captures decoded ({time.time() - started:.0f}s)")
    room_b = process_room_b(config, intermediate_dir, data_dir)
    print(f"Room B: {len(room_b)} captures read ({time.time() - started:.0f}s)")

    captures = pd.concat([room_a, room_b], ignore_index=True)
    X, windows = build_windows(captures, config, fingerprint)
    windows["window_file"] = export_window_files(X, windows, paths["windows_dir"])
    print(f"Windows: {X.shape}, one file each in {paths['windows_dir']} ({time.time() - started:.0f}s)")

    captures.to_csv(data_dir / "capture_manifest.csv", index=False)
    room_a_obs.to_csv(data_dir / "room_a_observations.csv", index=False)
    windows.to_csv(data_dir / "windows_manifest.csv", index=False)
    np.save(data_dir / "windows_X.npy", X)
    with open(data_dir / "preprocessing_config.json", "w", encoding="utf-8") as f:
        json.dump({"preprocessing_id": fingerprint,
                   **{k: config[k] for k in ("room_a", "room_b", "quality", "window", "labels")}}, f, indent=2)
    (data_dir / "qc_report.md").write_text(qc_report(captures, windows, config["window"]["length"]), encoding="utf-8")
    print(f"Done. Outputs in {data_dir} (preprocessing_id={fingerprint})")


if __name__ == "__main__":
    main()
