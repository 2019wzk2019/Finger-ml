"""Hand skeleton normalization and feature engineering."""

from __future__ import annotations

from typing import Optional

import numpy as np

NUM_NODES = 21
KEEP_INDICES = list(range(NUM_NODES))

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_TIP = 12
PINKY_MCP = 17

NODE_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)

FEATURE_NAMES: tuple[str, ...] = (
    "x", "y", "z",
    "dx", "dy", "dz",
    "thumb_index_dist",
    "thumb_middle_dist",
    "thumb_index_dx",
    "thumb_index_dy",
    "thumb_index_dz",
    "valid",
)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Convert MediaPipe 21-point landmarks to a palm-local coordinate system."""
    landmarks = landmarks.astype(np.float32, copy=False)
    centered = landmarks - landmarks[WRIST]

    x_axis = centered[INDEX_MCP].copy()
    pinky_vec = centered[PINKY_MCP].copy()
    scale = float(np.linalg.norm(x_axis))
    if scale > 1e-6:
        x_axis = x_axis / scale
    else:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        scale = 1.0

    z_axis = np.cross(x_axis, pinky_vec)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm > 1e-6:
        z_axis = z_axis / z_norm
    else:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm > 1e-6:
        y_axis = y_axis / y_norm
    else:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    basis = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
    return ((centered @ basis) / scale).astype(np.float32)


def build_motion_features(
    landmarks: np.ndarray,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build coordinate, velocity, contact/direction and validity features."""
    landmarks = landmarks.astype(np.float32, copy=False)
    n, v, _ = landmarks.shape
    delta = np.zeros_like(landmarks, dtype=np.float32)
    if n > 1:
        delta[1:] = landmarks[1:] - landmarks[:-1]

    thumb_tip = landmarks[:, THUMB_TIP]
    index_tip = landmarks[:, INDEX_TIP]
    middle_tip = landmarks[:, MIDDLE_TIP]
    thumb_index_vec = thumb_tip - index_tip
    thumb_middle_vec = thumb_tip - middle_tip
    thumb_index_dist = np.linalg.norm(thumb_index_vec, axis=1, keepdims=True)
    thumb_middle_dist = np.linalg.norm(thumb_middle_vec, axis=1, keepdims=True)

    global_feats = np.concatenate(
        [thumb_index_dist, thumb_middle_dist, thumb_index_vec],
        axis=1,
    ).astype(np.float32)
    global_feats = np.repeat(global_feats[:, None, :], v, axis=1)

    if valid is None:
        valid_channel = np.ones((n, v, 1), dtype=np.float32)
    else:
        valid_channel = np.repeat(valid.astype(np.float32)[:, None, None], v, axis=1)

    return np.concatenate([landmarks, delta, global_feats, valid_channel], axis=2).astype(np.float32)
