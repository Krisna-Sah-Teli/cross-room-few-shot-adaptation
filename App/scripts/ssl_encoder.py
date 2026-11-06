"""Self-supervised CSI encoder (contrastive, SimCLR-style).

Input:  (n, T, 256) windows after `prepare_input` (log amplitude with each
        observation's gain removed), all 256 subcarriers.
Output: (n, 256) embedding = mean and std over time of the last conv layer.

Training uses no activity or room labels. Two random views of each window are
created by augmentation, and the NT-Xent loss pulls views of the same window
together while pushing other windows in the batch away.

Augmentations, chosen for CSI:
  - random subset of observations (order kept): robustness to missing packets and
    to Room B's weak temporal ordering
  - additive Gaussian noise: measurement noise
  - random smooth spectral tilt: a slowly varying gain across subcarriers, as a
    different hardware filter or static channel would produce
  - masking a contiguous block of subcarriers
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scripts.room_fingerprint import normalize


def prepare_input(X, chunk=500):
    """Raw amplitude windows (n, 150, 256) -> float32 (n, 150, 256) model input."""
    parts = []
    for start in range(0, len(X), chunk):
        block = np.asarray(X[start:start + chunk], dtype=np.float32)
        parts.append(normalize(block, "row_gain").astype(np.float32))
    return np.concatenate(parts)


class CSIEncoder(nn.Module):
    """Temporal 1D CNN with subcarriers as input channels and statistics pooling."""

    def __init__(self, in_channels, width=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=5, padding=2, stride=2), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width, kernel_size=3, padding=1, stride=2), nn.BatchNorm1d(width), nn.ReLU(),
        )
        self.out_dim = 2 * width

    def forward(self, x):                       # x: (batch, T, C)
        h = self.net(x.transpose(1, 2))         # (batch, width, T')
        return torch.cat([h.mean(dim=2), h.std(dim=2)], dim=1)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Linear(in_dim, out_dim))

    def forward(self, z):
        return self.net(z)


def augment(x, generator, keep=120, noise=0.1, tilt=0.3, mask_fraction=0.1):
    """Return one random view of a batch x: (B, T, C)."""
    batch, length, channels = x.shape
    order = torch.rand(batch, length, generator=generator).argsort(dim=1)[:, :keep].sort(dim=1).values
    view = torch.gather(x, 1, order.unsqueeze(-1).expand(-1, -1, channels))

    view = view + noise * torch.randn(view.shape, generator=generator)

    # Smooth tilt: random quadratic curve across subcarriers, added in the log domain.
    grid = torch.linspace(-1, 1, channels)
    coeffs = tilt * torch.randn(batch, 3, generator=generator)
    curve = coeffs[:, :1] + coeffs[:, 1:2] * grid + coeffs[:, 2:3] * (grid ** 2 - 1 / 3)
    view = view + curve.unsqueeze(1)

    width = max(1, int(mask_fraction * channels))
    starts = torch.randint(0, channels - width + 1, (batch,), generator=generator)
    mask = (torch.arange(channels).unsqueeze(0) >= starts.unsqueeze(1)) & \
           (torch.arange(channels).unsqueeze(0) < (starts + width).unsqueeze(1))
    return view.masked_fill(mask.unsqueeze(1), 0.0)


def nt_xent(z1, z2, temperature):
    """Normalized-temperature cross-entropy loss over 2B views."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    similarity = z @ z.T / temperature
    similarity.fill_diagonal_(float("-inf"))
    n = len(z1)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(similarity, targets)


def train_ssl(X, *, epochs, batch_size, lr, temperature, seed, log_every=10):
    """Train an encoder on unlabeled windows X (n, T, C). Returns (encoder, loss history)."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    data = torch.from_numpy(X)
    encoder = CSIEncoder(X.shape[2])
    head = ProjectionHead(encoder.out_dim)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(epochs):
        encoder.train(); head.train()
        permutation = torch.randperm(len(data), generator=generator)
        losses = []
        for start in range(0, len(data) - batch_size + 1, batch_size):   # drop the last partial batch
            batch = data[permutation[start:start + batch_size]]
            z1 = head(encoder(augment(batch, generator)))
            z2 = head(encoder(augment(batch, generator)))
            loss = nt_xent(z1, z2, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        history.append(float(np.mean(losses)))
        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(f"    epoch {epoch + 1:3d}/{epochs}  contrastive loss {history[-1]:.4f}", flush=True)
    return encoder, history


def random_encoder(in_channels, seed):
    """The same architecture with untrained random weights (baseline)."""
    torch.manual_seed(seed)
    return CSIEncoder(in_channels)


@torch.no_grad()
def embed(encoder, X, batch_size=512):
    """Frozen-encoder embeddings for windows X (n, T, C)."""
    encoder.eval()
    data = torch.from_numpy(np.ascontiguousarray(X))
    return np.concatenate([encoder(data[i:i + batch_size]).numpy() for i in range(0, len(data), batch_size)])
