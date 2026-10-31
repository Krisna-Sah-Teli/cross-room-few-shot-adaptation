
"""Self-supervised convolutional autoencoder for CSI windows."""

from __future__ import annotations

import torch
from torch import nn


class CSIEncoder(nn.Module):
    """Map a CSI window of shape (batch, time, subcarriers) to an embedding."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()

        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive.")

        self.embedding_dim = embedding_dim

        # Conv1d expects (batch, channels, time).
        self.features = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accept (batch, time, 256); return (batch, embedding_dim)."""
        if x.ndim != 3:
            raise ValueError(
                f"Expected a 3-D input (batch, time, 256); got {tuple(x.shape)}."
            )

        if x.shape[-1] != 256:
            raise ValueError(
                f"Expected 256 subcarriers; got {x.shape[-1]}."
            )

        x = x.transpose(1, 2).contiguous()
        return self.features(x)


class CSIAutoencoder(nn.Module):
    """Convolutional autoencoder for unlabeled CSI-window reconstruction."""

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()

        self.encoder = CSIEncoder(embedding_dim=embedding_dim)

        # Decode the embedding into the original 150 x 256 window.
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128 * 19),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 19)),

            nn.ConvTranspose1d(
                128, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose1d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 256, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the embedding and a reconstruction matching input shape."""
        if x.ndim != 3 or x.shape[1:] != (150, 256):
            raise ValueError(
                "Expected input shape (batch, 150, 256); "
                f"got {tuple(x.shape)}."
            )

        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)

        # The decoder produces 152 time steps; crop to the 150-step input.
        reconstruction = reconstruction[..., : x.shape[1]]
        reconstruction = reconstruction.transpose(1, 2).contiguous()

        return embedding, reconstruction
