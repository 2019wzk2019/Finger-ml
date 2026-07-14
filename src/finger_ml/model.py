"""ST-GCN + MS-TCN style model for offline gesture segmentation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from finger_ml.features import NUM_NODES
from finger_ml.labels import BACKGROUND_LABEL, GESTURE_ORDER, NUM_CLASSES

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
    (4, 8), (4, 12), (4, 16), (4, 20),
]


def build_adjacency(num_nodes: int = NUM_NODES, edges: list[tuple[int, int]] | None = None) -> torch.Tensor:
    edges = EDGES if edges is None else edges
    a = np.eye(num_nodes, dtype=np.float32)
    for i, j in edges:
        a[i, j] = 1.0
        a[j, i] = 1.0
    d = a.sum(axis=1)
    d_inv = np.where(d > 0, d ** -0.5, 0.0)
    return torch.from_numpy(np.diag(d_inv) @ a @ np.diag(d_inv))


class AdaptiveGraphConv(nn.Module):
    """Graph convolution with a fixed anatomical graph plus a learned residual graph."""

    def __init__(self, c_in: int, c_out: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("a_fixed", adjacency)
        self.a_residual = nn.Parameter(torch.zeros_like(adjacency))
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.a_fixed + torch.tanh(self.a_residual) * 0.25
        x = self.proj(x)
        return torch.einsum("bctv,vw->bctw", x, a)


class STGCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, adjacency: torch.Tensor, dropout: float = 0.0) -> None:
        super().__init__()
        self.gcn = AdaptiveGraphConv(c_in, c_out, adjacency)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(c_out),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Sequential(nn.Conv2d(c_in, c_out, 1), nn.BatchNorm2d(c_out))
            if c_in != c_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.tcn(self.gcn(x)) + self.residual(x), inplace=True)


class STGCNEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        a = build_adjacency()
        self.blocks = nn.Sequential(
            STGCNBlock(input_channels, 48, a, dropout=0.0),
            STGCNBlock(48, 96, a, dropout=dropout * 0.5),
            STGCNBlock(96, hidden_dim, a, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,T,V] -> [B,D,T]
        return self.blocks(x).mean(dim=-1)


class DilatedResidualLayer(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TemporalStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, channels, 1)
        self.layers = nn.ModuleList(
            DilatedResidualLayer(channels, 2 ** i, dropout)
            for i in range(num_layers)
        )
        self.classifier = nn.Conv1d(channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.in_proj(x)
        for layer in self.layers:
            feat = layer(feat)
        return self.classifier(feat), feat


class GestureSegmenter(nn.Module):
    """Dense frame-label model.

    Returns:
        logits: [B, num_classes, T]
        boundary_logits: [B, 2, T] for start/end boundary probabilities
        stage_logits: list of refinement stage logits
    """

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 128,
        temporal_channels: int = 128,
        temporal_layers: int = 6,
        temporal_stages: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.encoder = STGCNEncoder(input_channels, hidden_dim, dropout)
        self.stage0 = TemporalStage(hidden_dim, temporal_channels, temporal_layers, num_classes, dropout)
        self.refiners = nn.ModuleList(
            TemporalStage(num_classes, temporal_channels, temporal_layers, num_classes, dropout)
            for _ in range(max(0, temporal_stages - 1))
        )
        self.boundary_head = nn.Sequential(
            nn.Conv1d(temporal_channels, temporal_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(temporal_channels // 2, 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        encoded = self.encoder(x)
        logits, feat = self.stage0(encoded)
        stages = [logits]
        for refiner in self.refiners:
            logits, _ = refiner(F.softmax(logits, dim=1))
            stages.append(logits)
        boundary = self.boundary_head(feat)
        return logits, boundary, stages


def probabilities_to_events(
    probs: np.ndarray,
    fps: float,
    *,
    boundary_probs: np.ndarray | None = None,
    conf_threshold: float = 0.55,
    min_event_ms: int = 120,
    max_gap_ms: int = 120,
    smooth: int = 7,
) -> list[dict]:
    """Convert frame probabilities ``[T,C]`` to gesture event dictionaries."""
    if probs.size == 0:
        return []
    labels = probs.argmax(axis=1).astype(np.int64)
    confs = probs[np.arange(len(labels)), labels]
    labels = np.where(confs >= conf_threshold, labels, BACKGROUND_LABEL)
    if smooth > 1:
        labels = _median_like_smooth(labels, probs, smooth)
    min_frames = max(1, int(round(min_event_ms * fps / 1000.0)))
    max_gap = max(0, int(round(max_gap_ms * fps / 1000.0)))

    raw: list[dict] = []
    i = 0
    while i < len(labels):
        label = int(labels[i])
        if label == BACKGROUND_LABEL:
            i += 1
            continue
        j = i + 1
        while j < len(labels) and int(labels[j]) == label:
            j += 1
        if j - i >= min_frames:
            start, end = _refine_boundaries(i, j - 1, boundary_probs)
            mean_conf = float(probs[i:j, label].mean())
            raw.append(_event(label, start, end, fps, mean_conf))
        i = j
    return _merge_events(raw, max_gap, fps)


def _median_like_smooth(labels: np.ndarray, probs: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    out = labels.copy()
    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)
        votes: dict[int, float] = {}
        for k in range(lo, hi):
            label = int(labels[k])
            if label == BACKGROUND_LABEL:
                continue
            votes[label] = votes.get(label, 0.0) + float(probs[k, label])
        out[i] = max(votes, key=votes.get) if votes else BACKGROUND_LABEL
    return out


def _refine_boundaries(start: int, end: int, boundary_probs: np.ndarray | None) -> tuple[int, int]:
    if boundary_probs is None or boundary_probs.size == 0:
        return start, end
    n = boundary_probs.shape[1]
    pad = max(3, min(12, (end - start + 1) // 2))
    s0, s1 = max(0, start - pad), min(n, start + pad + 1)
    e0, e1 = max(0, end - pad), min(n, end + pad + 1)
    if s1 > s0:
        start = s0 + int(np.argmax(boundary_probs[0, s0:s1]))
    if e1 > e0:
        end = e0 + int(np.argmax(boundary_probs[1, e0:e1]))
    if end < start:
        end = start
    return start, end


def _event(label: int, start: int, end: int, fps: float, confidence: float) -> dict:
    name = GESTURE_ORDER[label] if 0 <= label < len(GESTURE_ORDER) else str(label)
    return {
        "gesture": name,
        "label": int(label),
        "start_frame": int(start),
        "end_frame": int(end),
        "start_ms": int(round(start * 1000 / fps)),
        "end_ms": int(round(end * 1000 / fps)),
        "duration_ms": int(round((end - start + 1) * 1000 / fps)),
        "mean_conf": round(float(confidence), 4),
    }


def _merge_events(events: list[dict], max_gap: int, fps: float) -> list[dict]:
    if not events:
        return []
    merged = [events[0].copy()]
    for ev in events[1:]:
        prev = merged[-1]
        gap = ev["start_frame"] - prev["end_frame"] - 1
        if ev["label"] == prev["label"] and gap <= max_gap:
            prev_len = prev["end_frame"] - prev["start_frame"] + 1
            cur_len = ev["end_frame"] - ev["start_frame"] + 1
            prev["end_frame"] = ev["end_frame"]
            prev["end_ms"] = ev["end_ms"]
            prev["duration_ms"] = int(round((prev["end_frame"] - prev["start_frame"] + 1) * 1000 / fps))
            prev["mean_conf"] = round(
                (prev["mean_conf"] * prev_len + ev["mean_conf"] * cur_len) / (prev_len + cur_len),
                4,
            )
        else:
            merged.append(ev.copy())
    return merged
