
"""Self-supervised training utilities for the Task 2 CSI autoencoder."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.task2_encoder import CSIAutoencoder


def train_autoencoder(
    X_unlabeled: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str | None = None,
) -> tuple[CSIAutoencoder, list[dict[str, float]]]:
    """Train an autoencoder using only unlabeled development CSI windows.

    Returns:
        The trained autoencoder and a list of per-epoch training metrics.

    No validation data or labels are accepted by this function.
    """
    if not isinstance(X_unlabeled, np.ndarray):
        raise TypeError("X_unlabeled must be a NumPy array.")

    if X_unlabeled.ndim != 3 or X_unlabeled.shape[1:] != (150, 256):
        raise ValueError(
            "Expected X_unlabeled shape (n, 150, 256); "
            f"got {X_unlabeled.shape}."
        )

    if len(X_unlabeled) == 0:
        raise ValueError("X_unlabeled must contain at least one window.")

    if not np.isfinite(X_unlabeled).all():
        raise ValueError("X_unlabeled contains non-finite values.")

    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError(
            "epochs and batch_size must be positive, "
            "and learning_rate must be greater than zero."
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    training_device = torch.device(device)

    if training_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    # Copy to a CPU tensor; each mini-batch is transferred to the device.
    X_tensor = torch.from_numpy(
        np.ascontiguousarray(X_unlabeled, dtype=np.float32)
    )

    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(training_device.type == "cuda"),
        drop_last=False,
    )

    model = CSIAutoencoder(embedding_dim=128).to(training_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    criterion = nn.MSELoss()

    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_squared_error = 0.0
        total_values = 0

        for (batch,) in loader:
            batch = batch.to(
                training_device,
                non_blocking=(training_device.type == "cuda"),
            )

            optimizer.zero_grad(set_to_none=True)

            _, reconstruction = model(batch)
            loss = criterion(reconstruction, batch)

            loss.backward()
            optimizer.step()

            # Aggregate by number of scalar input values, not by batch.
            n_values = batch.numel()
            total_squared_error += loss.detach().item() * n_values
            total_values += n_values

        epoch_mse = total_squared_error / total_values
        history.append({"epoch": float(epoch), "train_mse": epoch_mse})

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"training reconstruction MSE: {epoch_mse:.6f}"
        )

    return model, history



def extract_embeddings(
    encoder: torch.nn.Module,
    X: np.ndarray,
    batch_size: int = 128,
    device: str | None = None,
) -> np.ndarray:
    """Extract frozen embeddings without accessing labels or metadata."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    if X.ndim != 3 or X.shape[1:] != (150, 256):
        raise ValueError(
            f"Expected shape (n, 150, 256); got {X.shape}."
        )

    if len(X) == 0:
        raise ValueError("X must contain at least one window.")

    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values.")

    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    encoder = encoder.to(target_device)
    encoder.eval()

    for parameter in encoder.parameters():
        if parameter.requires_grad:
            raise ValueError("Encoder must be frozen before extraction.")

    tensor_X = torch.from_numpy(
        np.ascontiguousarray(X, dtype=np.float32)
    )
    loader = DataLoader(
        TensorDataset(tensor_X),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(target_device.type == "cuda"),
    )

    embedding_batches = []

    with torch.inference_mode():
        for (batch,) in loader:
            batch = batch.to(
                target_device,
                non_blocking=(target_device.type == "cuda"),
            )
            embeddings = encoder(batch)

            if embeddings.ndim != 2:
                raise ValueError(
                    "Encoder must return a 2-D embedding matrix."
                )

            embedding_batches.append(embeddings.cpu().numpy())

    result = np.concatenate(embedding_batches, axis=0).astype(
        np.float32, copy=False
    )

    if result.shape[0] != len(X):
        raise RuntimeError("Embedding count does not match input count.")

    if not np.isfinite(result).all():
        raise ValueError("Extracted embeddings contain non-finite values.")

    return result
