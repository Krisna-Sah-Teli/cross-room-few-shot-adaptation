
"""Build and validate model-ready datasets for cross-room adaptation."""

import numpy as np
import pandas as pd


def build_training_dataset(
    adaptation_direction,
    K,
    combined_manifest,
    room_tensor_by_name,
):
    """Build X and y for one few-shot adaptation experiment.

    Shared-source windows are extracted from the source room.
    Target-exclusive support windows are extracted from the target room.

    Returns
    -------
    X : np.ndarray
        CSI windows with shape (n_samples, 150, 256).
    y : np.ndarray
        Standardized activity labels as strings.
    experiment : pd.DataFrame
        The manifest rows used to construct this dataset.
    """
    required_columns = {
        "adaptation_direction",
        "K",
        "source_room",
        "target_room",
        "training_role",
        "target_label",
        "tensor_index",
    }
    missing = required_columns - set(combined_manifest.columns)
    if missing:
        raise ValueError(
            f"Combined manifest is missing columns: {sorted(missing)}"
        )

    experiment = combined_manifest.loc[
        (combined_manifest["adaptation_direction"] == adaptation_direction)
        & (combined_manifest["K"] == K)
    ].copy()

    if experiment.empty:
        raise ValueError(
            f"No rows found for direction={adaptation_direction!r}, K={K}"
        )

    allowed_roles = {"shared_source", "target_exclusive_support"}
    unexpected_roles = set(experiment["training_role"].unique()) - allowed_roles
    if unexpected_roles:
        raise ValueError(f"Unexpected training roles: {unexpected_roles}")

    if experiment["target_label"].isna().any():
        raise ValueError("Missing target labels in the experiment manifest.")

    X_parts = []
    y_parts = []

    for row_index, row in experiment.iterrows():
        if row["training_role"] == "shared_source":
            room_name = row["source_room"]
        else:
            room_name = row["target_room"]

        if room_name not in room_tensor_by_name:
            raise KeyError(f"No CSI tensor configured for {room_name!r}")

        tensor = room_tensor_by_name[room_name]
        tensor_index = int(row["tensor_index"])

        if not 0 <= tensor_index < len(tensor):
            raise IndexError(
                f"Manifest row {row_index}: tensor index {tensor_index} "
                f"is out of bounds for {room_name}"
            )

        X_parts.append(tensor[tensor_index])
        y_parts.append(str(row["target_label"]))

    X = np.stack(X_parts).astype(np.float32, copy=False)
    y = np.asarray(y_parts, dtype=str)

    validate_training_dataset(X, y, experiment)

    return X, y, experiment


def validate_training_dataset(X, y, experiment):
    """Validate shapes, numeric values, labels, and sample counts."""
    if X.ndim != 3 or X.shape[1:] != (150, 256):
        raise ValueError(f"Unexpected X shape: {X.shape}")

    if X.shape[0] != len(y) or len(y) != len(experiment):
        raise ValueError(
            "Sample count mismatch between X, y, and experiment manifest."
        )

    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")

    if pd.isna(y).any():
        raise ValueError("y contains missing labels.")

    if len(y) == 0:
        raise ValueError("The training dataset is empty.")

    return True
