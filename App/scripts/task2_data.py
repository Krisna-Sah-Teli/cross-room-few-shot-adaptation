
"""Reusable data loading and integrity checks for Task 2.

Loads the existing Room A and Room B window tensors and metadata.
This module does not create or modify dataset splits.
"""

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_METADATA_COLUMNS = {
    "window_id",
    "capture_path",
    "activity",
    "split",
}

ALLOWED_SPLITS = {"development", "validation"}
EXPECTED_WINDOW_SHAPE = (150, 256)


def load_room_artifacts(results_dir, room):
    """Load one room's saved tensor and metadata.

    Parameters
    ----------
    results_dir : str or pathlib.Path
        Directory containing the existing Task 2 result artifacts.
    room : str
        Either "a" or "b".

    Returns
    -------
    X : numpy.ndarray
        Window tensor with shape (n_windows, 150, 256).
    metadata : pandas.DataFrame
        Metadata aligned row-for-row with X.
    """
    results_dir = Path(results_dir)
    room = str(room).lower()

    if room not in {"a", "b"}:
        raise ValueError("room must be either 'a' or 'b'.")

    tensor_path = results_dir / f"X_room_{room}.npy"
    metadata_path = results_dir / f"room_{room}_window_metadata.csv"

    for path in (tensor_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    X = np.load(tensor_path, allow_pickle=False)
    metadata = pd.read_csv(metadata_path)

    audit_room_artifacts(X, metadata, room=room)

    return X, metadata


def audit_room_artifacts(X, metadata, room="unknown"):
    """Validate tensor/metadata alignment and split integrity.

    Raises
    ------
    ValueError
        If a required integrity check fails.
    """
    missing_columns = REQUIRED_METADATA_COLUMNS - set(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"Room {room}: missing metadata columns "
            f"{sorted(missing_columns)}."
        )

    if X.ndim != 3 or X.shape[1:] != EXPECTED_WINDOW_SHAPE:
        raise ValueError(
            f"Room {room}: expected shape (n, 150, 256), got {X.shape}."
        )

    if len(X) != len(metadata):
        raise ValueError(
            f"Room {room}: {len(X)} tensor windows but "
            f"{len(metadata)} metadata rows."
        )

    if not np.isfinite(X).all():
        raise ValueError(f"Room {room}: tensor contains non-finite values.")

    for column in ("window_id", "capture_path", "activity", "split"):
        if metadata[column].isna().any():
            raise ValueError(
                f"Room {room}: missing values in {column!r}."
            )

    if metadata["window_id"].duplicated().any():
        raise ValueError(f"Room {room}: duplicate window IDs detected.")

    observed_splits = set(metadata["split"].unique())
    unexpected_splits = observed_splits - ALLOWED_SPLITS

    if unexpected_splits:
        raise ValueError(
            f"Room {room}: unexpected split values "
            f"{sorted(unexpected_splits)}."
        )

    # A source capture must not contribute windows to both splits.
    splits_per_capture = metadata.groupby("capture_path")["split"].nunique()

    if (splits_per_capture > 1).any():
        raise ValueError(
            f"Room {room}: source captures occur in both splits."
        )

    return True


def summarize_room_artifacts(X, metadata, room="unknown"):
    """Return a compact, reusable summary of a room's artifacts."""
    split_counts = (
        metadata["split"]
        .value_counts()
        .reindex(["development", "validation"], fill_value=0)
    )

    return {
        "room": str(room).upper(),
        "n_windows": int(len(X)),
        "window_shape": tuple(X.shape[1:]),
        "dtype": str(X.dtype),
        "n_captures": int(metadata["capture_path"].nunique()),
        "n_activities": int(metadata["activity"].nunique()),
        "development_windows": int(split_counts["development"]),
        "validation_windows": int(split_counts["validation"]),
        "finite_values": bool(np.isfinite(X).all()),
        "unique_window_ids": int(metadata["window_id"].nunique()),
    }


def prepare_unlabeled_development_data(
    X_room_a,
    metadata_room_a,
    X_room_b,
    metadata_room_b,
):
    """Prepare normalized development windows for self-supervised training.

    Activity labels, room labels, and validation windows are not returned.
    Existing dataset arrays and metadata are not modified.
    """
    import numpy as np

    from scripts.normalization import log1p_per_window_zscore

    datasets = {
        "A": (X_room_a, metadata_room_a),
        "B": (X_room_b, metadata_room_b),
    }

    normalized_development_windows = []
    audit_rows = []

    for room, (X, metadata) in datasets.items():
        if len(X) != len(metadata):
            raise ValueError(
                f"Room {room}: array and metadata row counts differ."
            )

        if "split" not in metadata.columns:
            raise ValueError(
                f"Room {room}: metadata must contain a 'split' column."
            )

        split_values = set(metadata["split"].dropna().unique())
        if not split_values.issubset({"development", "validation"}):
            raise ValueError(
                f"Room {room}: unexpected split values {split_values}."
            )

        development_mask = (
            metadata["split"].to_numpy() == "development"
        )
        selected_indices = np.flatnonzero(development_mask)

        if len(selected_indices) == 0:
            raise ValueError(
                f"Room {room}: no development windows were found."
            )

        # Select only development rows before normalization.
        X_development = X[selected_indices]
        X_normalized = log1p_per_window_zscore(X_development)

        if X_normalized.shape != X_development.shape:
            raise ValueError(
                f"Room {room}: normalization changed the array shape."
            )

        if X_normalized.dtype != np.float32:
            raise ValueError(
                f"Room {room}: expected float32 normalized data, "
                f"got {X_normalized.dtype}."
            )

        if not np.isfinite(X_normalized).all():
            raise ValueError(
                f"Room {room}: normalized data contain non-finite values."
            )

        normalized_development_windows.append(X_normalized)

        audit_rows.append({
            "room": room,
            "development_windows": len(selected_indices),
            "validation_windows_excluded": int(
                (metadata["split"].to_numpy() == "validation").sum()
            ),
            "normalized_shape": tuple(X_normalized.shape),
            "dtype": str(X_normalized.dtype),
            "all_finite": bool(np.isfinite(X_normalized).all()),
        })

    # Concatenate windows only. No labels or metadata enter the training data.
    X_unlabeled = np.concatenate(
        normalized_development_windows, axis=0
    ).astype(np.float32, copy=False)

    expected_count = sum(
        row["development_windows"] for row in audit_rows
    )
    if X_unlabeled.shape != (expected_count, 150, 256):
        raise ValueError(
            "Unexpected combined development-data shape: "
            f"{X_unlabeled.shape}."
        )

    return X_unlabeled, audit_rows
