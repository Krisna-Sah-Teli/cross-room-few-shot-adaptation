"""Create practice episodes from VALIDATION data, to rehearse the live run_episode.py demonstration.

  Episodes/demo_windows/  window_0001.npy ... (shuffled validation windows) + labels.csv
  Episodes/demo_raw/      capture_01.pcap / capture_02.csv ... (validation captures) + labels.csv

File names are neutral on purpose: the system never uses file names, folders or capture
length. labels.csv is only used to print scores after prediction.

Usage:
    python App/scripts/make_demo_episode.py [--per-class 2] [--raw-per-room 3]
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.config import PROJECT_ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--per-class", type=int, default=2, help="validation windows per room and activity")
    parser.add_argument("--raw-per-room", type=int, default=3, help="raw validation captures per room")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    windows = pd.read_csv(config["paths"]["data_dir"] / "windows_manifest.csv")
    captures = pd.read_csv(config["paths"]["data_dir"] / "capture_manifest.csv")
    val = windows[windows.split == "validation"]

    # 1. Window-file episode.
    out = PROJECT_ROOT / "Episodes" / "demo_windows"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    picked = pd.concat([g.sample(min(args.per_class, len(g)), random_state=args.seed)
                        for _, g in val.groupby(["room", "label"])])
    picked = picked.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    labels = []
    for i, row in enumerate(picked.itertuples(), start=1):
        window_id = f"window_{i:04d}"
        shutil.copy(PROJECT_ROOT / row.window_file, out / f"{window_id}.npy")
        labels.append({"window_id": window_id, "room": row.room, "label": row.label})
    pd.DataFrame(labels).to_csv(out / "labels.csv", index=False)
    print(f"{out}: {len(labels)} windows")

    # 2. Raw-capture episode (validation captures that produce at least one window).
    out = PROJECT_ROOT / "Episodes" / "demo_raw"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    usable = captures[(captures.split == "validation") & (captures.observations >= config["window"]["length"])]
    labels, n = [], 0
    for room, zip_key, suffix in (("A", "room_a_zip", ".pcap"), ("B", "room_b_zip", ".csv")):
        chosen = usable[usable.room == room].sample(args.raw_per_room, random_state=args.seed)
        with zipfile.ZipFile(config["paths"][zip_key]) as archive:
            for row in chosen.itertuples():
                n += 1
                stem = f"capture_{n:02d}"
                (out / f"{stem}{suffix}").write_bytes(archive.read(row.source_member))
                count = int((windows.capture_id == row.capture_id).sum())
                labels += [{"window_id": f"{stem}_w{i}", "room": room, "label": row.label} for i in range(count)]
    pd.DataFrame(labels).to_csv(out / "labels.csv", index=False)
    print(f"{out}: {n} raw captures, {len(labels)} windows")


if __name__ == "__main__":
    main()
