"""Build the deployed system used by run_episode.py.

Trains, from development data only, the per-room classifiers of the Task 3 configuration in
`final_model` (experiment.yaml), with the Task 1 router and the Task 2 encoder, and saves
everything to App/results/final_model/bundle.pkl.

Usage:
    python App/scripts/build_final_model.py [--config App/configs/experiment.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from scripts.config import load_config
from scripts.episode import build_bundle, save_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(config["ssl"]["threads"])

    bundle = build_bundle(config)
    out_dir = config["paths"]["final_model_dir"]
    save_bundle(bundle, out_dir / "bundle.pkl")
    summary = {
        "settings": bundle["settings"],
        "preprocessing_id": bundle["preprocessing_id"],
        "label_sets": bundle["label_sets"],
        "support_captures": {room: r["support_captures"] for room, r in bundle["rooms"].items()},
        "structure_prior_rooms": [room for room, r in bundle["rooms"].items() if r.get("structure")],
        "build_seconds": bundle["build_seconds"],
    }
    (out_dir / "bundle_info.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {out_dir / 'bundle.pkl'} ({(out_dir / 'bundle.pkl').stat().st_size / 1e6:.1f} MB, "
          f"built in {bundle['build_seconds']}s)")
    for room, r in bundle["rooms"].items():
        print(f"  Room {room}: {len(r['support_captures'])} support captures, classes {list(r['model'].classes_)}")


if __name__ == "__main__":
    main()
