"""Datasets for offline hand-gesture temporal segmentation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from finger_ml.features import NUM_NODES
from finger_ml.labels import BACKGROUND_LABEL, NUM_CLASSES

IGNORE_INDEX = -100


@dataclass(frozen=True)
class Session:
    path: Path
    features: np.ndarray
    labels: np.ndarray
    train_mask: np.ndarray
    fps: float


def subject_from_stem(stem: str) -> str:
    return stem.rsplit("_", 1)[-1].lower() if "_" in stem else ""


def split_feature_files(
    features_dir: str | Path,
    *,
    val_ratio: float = 0.2,
    subjects: list[str] | None = None,
) -> tuple[list[Path], list[Path]]:
    features_dir = Path(features_dir)
    files = sorted(features_dir.glob("*.npz"))
    if subjects:
        allowed = {s.lower() for s in subjects}
        files = [p for p in files if subject_from_stem(p.stem) in allowed]
    if not files:
        raise FileNotFoundError(f"No feature .npz files found in {features_dir}")
    if len(files) == 1:
        return files, files

    frame_counts = []
    for path in files:
        with np.load(path) as data:
            frame_counts.append(int(data["labels"].shape[0]))
    target = sum(frame_counts) * val_ratio
    val: list[Path] = []
    total = 0
    for path, count in zip(reversed(files), reversed(frame_counts)):
        val.append(path)
        total += count
        if total >= target:
            break
    val_set = set(val)
    train = [p for p in files if p not in val_set]
    if not train:
        train = val[:1]
        val = val[1:] or val[:1]
    return train, sorted(val)


def load_session(path: Path) -> Session:
    with np.load(path) as data:
        features = (data["features"] if "features" in data else data["landmarks"]).astype(np.float32)
        labels = data["labels"].astype(np.int64)
        train_mask = (
            data["train_mask"].astype(bool)
            if "train_mask" in data
            else np.ones_like(labels, dtype=bool)
        )
        fps = float(data["fps"]) if "fps" in data else 30.0
    if features.ndim != 3 or features.shape[1] != NUM_NODES:
        raise ValueError(f"{path}: expected features [T, 21, C], got {features.shape}")
    return Session(path, features, labels, train_mask, fps)


def boundary_targets(labels: np.ndarray, radius: int = 2) -> np.ndarray:
    """Return [2, T] start/end targets from frame labels."""
    n = len(labels)
    out = np.zeros((2, n), dtype=np.float32)
    prev = np.full(n, BACKGROUND_LABEL, dtype=np.int64)
    prev[1:] = labels[:-1]
    starts = np.where((labels != BACKGROUND_LABEL) & (prev == BACKGROUND_LABEL))[0]
    ends = np.where((labels == BACKGROUND_LABEL) & (prev != BACKGROUND_LABEL))[0] - 1
    if n and labels[-1] != BACKGROUND_LABEL:
        ends = np.append(ends, n - 1)
    for channel, indices in enumerate((starts, ends)):
        for idx in indices:
            lo = max(0, int(idx) - radius)
            hi = min(n, int(idx) + radius + 1)
            out[channel, lo:hi] = 1.0
    return out


class GestureSegmentationDataset(Dataset):
    """Fixed-length chunks for dense frame labeling.

    Each item is ``x [C,T,21]``, ``y [T]``, ``boundary [2,T]`` and ``mask [T]``.
    Padding and ignored annotation-transition frames use ``IGNORE_INDEX``.
    """

    def __init__(
        self,
        files: list[Path],
        *,
        chunk_len: int = 256,
        hop: int | None = None,
        augment: bool = False,
        boundary_radius: int = 2,
    ) -> None:
        self.sessions = [load_session(p) for p in files]
        self.chunk_len = int(chunk_len)
        self.hop = int(hop if hop is not None else chunk_len)
        self.augment = augment
        self.boundary_radius = int(boundary_radius)
        self.input_channels = max(int(s.features.shape[2]) for s in self.sessions)
        self.index: list[tuple[int, int]] = []
        for si, session in enumerate(self.sessions):
            n = len(session.labels)
            if n <= self.chunk_len:
                self.index.append((si, 0))
                continue
            for start in range(0, n, self.hop):
                self.index.append((si, start))
                if start + self.chunk_len >= n:
                    break

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        si, start = self.index[idx]
        session = self.sessions[si]
        n = len(session.labels)
        if self.augment and n > self.chunk_len:
            jitter = random.randint(-self.hop // 2, self.hop // 2)
            start = min(max(0, start + jitter), max(0, n - self.chunk_len))

        end = min(n, start + self.chunk_len)
        length = end - start
        c = self.input_channels
        x = np.zeros((self.chunk_len, NUM_NODES, c), dtype=np.float32)
        x[:length, :, : session.features.shape[2]] = session.features[start:end]
        y = np.full(self.chunk_len, IGNORE_INDEX, dtype=np.int64)
        mask = np.zeros(self.chunk_len, dtype=bool)
        y[:length] = session.labels[start:end]
        mask[:length] = session.train_mask[start:end]
        y[~mask] = IGNORE_INDEX

        b_all = boundary_targets(session.labels, self.boundary_radius)
        b = np.zeros((2, self.chunk_len), dtype=np.float32)
        b[:, :length] = b_all[:, start:end]

        if self.augment:
            # Coordinate and velocity channels are normalized palm-local values.
            noise_channels = min(6, c)
            x[:, :, :noise_channels] += np.random.normal(
                0.0, 0.008, x[:, :, :noise_channels].shape
            ).astype(np.float32)
            if random.random() < 0.15:
                drop = np.random.rand(self.chunk_len) < 0.04
                x[drop, :, :] = 0.0

        return (
            torch.from_numpy(x).permute(2, 0, 1),
            torch.from_numpy(y),
            torch.from_numpy(b),
            torch.from_numpy(mask),
        )

    def class_counts(self) -> np.ndarray:
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for session in self.sessions:
            valid = session.train_mask
            counts += np.bincount(session.labels[valid], minlength=NUM_CLASSES)
        return counts

    def __repr__(self) -> str:
        names = ", ".join(s.path.stem for s in self.sessions)
        return (
            f"GestureSegmentationDataset(n_chunks={len(self)}, "
            f"chunk_len={self.chunk_len}, files=[{names}])"
        )
