"""Load the pipeline configuration and resolve its paths."""

import hashlib
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "App" / "configs" / "experiment.yaml"


def load_config(path=None):
    """Return the configuration dict with every entry of `paths` as an absolute Path."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    resolved = {}
    for key, value in config["paths"].items():
        value = Path(value)
        resolved[key] = value if value.is_absolute() else PROJECT_ROOT / value
    config["paths"] = resolved
    config["config_path"] = str(path)
    return config


def preprocessing_fingerprint(config):
    """Short hash of every setting that changes the derived windows (provenance tag)."""
    relevant = {key: config[key] for key in ("room_a", "room_b", "quality", "window")}
    blob = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:10]
