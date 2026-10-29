"""Run the trained cross-room CSI system on a challenge episode and write predictions.

Usage:
    python run_episode.py EPISODE_DIR [--output FILE] [--config FILE] [--rebuild-model]

EPISODE_DIR contains window_XXXX.npy files (150 x 256 float32 CSI amplitude) and/or raw
captures (*.pcap in the Room A Nexmon format, *.csv with 256 amplitude columns in the
Room B format). Optional files:
    support.csv   few-shot support for the target room (columns window_file,label[,room])
    labels.csv    ground truth (columns window_id,label[,room]); scores are printed
Writes predictions.csv (window_id, room, label, confidence) into EPISODE_DIR, and
predictions_details.csv with the router confidence and the input file of each window.

The trained system is App/results/final_model/bundle.pkl. It is built automatically from
the development data the first time (or with --rebuild-model).
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "App"))

import torch

from scripts.config import load_config
from scripts.episode import (build_bundle, load_bundle, load_episode_inputs, predict_windows, read_support,
                             save_bundle, score_predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("episode", help="folder with window_XXXX.npy files and/or raw .pcap/.csv captures")
    parser.add_argument("--output", default=None, help="predictions file (default: EPISODE_DIR/predictions.csv)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--rebuild-model", action="store_true", help="rebuild the final model from the development data")
    args = parser.parse_args()

    started = time.time()
    config = load_config(args.config)
    torch.set_num_threads(config["ssl"]["threads"])
    episode = Path(args.episode)
    if not episode.is_dir():
        sys.exit(f"Episode folder not found: {episode}")

    bundle_path = config["paths"]["final_model_dir"] / "bundle.pkl"
    if args.rebuild_model or not bundle_path.is_file():
        print("Building the final model from the development data ...")
        save_bundle(build_bundle(config), bundle_path)
    bundle = load_bundle(bundle_path)

    ids, X, sources = load_episode_inputs(episode, config)
    support = read_support(episode) if (episode / "support.csv").is_file() else None
    print(f"Episode {episode}: {len(ids)} windows from {len(set(sources))} input file(s)"
          + (f", {len(support)} support windows" if support else ""))

    predictions = predict_windows(bundle, X, support)
    predictions.insert(0, "window_id", ids)
    output = Path(args.output) if args.output else episode / "predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions[["window_id", "room", "label", "confidence"]].to_csv(output, index=False)
    predictions.assign(input_file=sources).to_csv(output.with_name(output.stem + "_details.csv"), index=False)

    print("Predicted rooms:     ", predictions.room.value_counts().sort_index().to_dict())
    print("Predicted activities:", predictions.label.value_counts().sort_index().to_dict())
    if (episode / "labels.csv").is_file():
        print("Scores against labels.csv:")
        score_predictions(predictions, episode / "labels.csv")
    print(f"Wrote {output} ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
