"""MediaPipe hand tracking helpers shared by collection, preprocessing and detection."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    mp_python = None
    mp_vision = None
    MEDIAPIPE_AVAILABLE = False


DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12

HIGHLIGHT_COLORS = {
    THUMB_TIP: (255, 210, 50),
    INDEX_TIP: (50, 230, 80),
    MIDDLE_TIP: (30, 140, 255),
}


def resolve_model(model_cache_dir: str) -> str:
    cache_dir = Path(model_cache_dir)
    model_path = cache_dir / "hand_landmarker.task"
    if model_path.exists():
        return str(model_path)
    print(f"[info] Downloading hand_landmarker.task -> {model_path} ...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DEFAULT_MODEL_URL, model_path)
    print("[info] Download complete.")
    return str(model_path)


def make_landmarker(
    model_path: str,
    *,
    running_mode: str = "IMAGE",
    delegate: str = "CPU",
):
    if not MEDIAPIPE_AVAILABLE:
        return None
    delegate_name = delegate.upper()
    base_kwargs = {"model_asset_path": model_path}
    if delegate_name == "GPU":
        base_kwargs["delegate"] = mp_python.BaseOptions.Delegate.GPU
    elif delegate_name == "CPU":
        base_kwargs["delegate"] = mp_python.BaseOptions.Delegate.CPU
    base_opts = mp_python.BaseOptions(**base_kwargs)
    mode = getattr(mp_vision.RunningMode, running_mode.upper())
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mode,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def detect(frame_bgr: np.ndarray, landmarker, hand_side: Optional[str] = None) -> Optional[list]:
    """Detect a hand in IMAGE mode."""
    if landmarker is None:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    return _select_hand(result, hand_side)


def detect_video(
    frame_bgr: np.ndarray,
    landmarker,
    timestamp_ms: int,
    hand_side: Optional[str] = None,
) -> Optional[list]:
    """Detect/track a hand in VIDEO mode."""
    if landmarker is None:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_img, timestamp_ms)
    return _select_hand(result, hand_side)


def _select_hand(result, hand_side: Optional[str]) -> Optional[list]:
    if not result.hand_landmarks:
        return None
    if hand_side is None:
        return result.hand_landmarks[0]
    for i, handedness in enumerate(result.handedness):
        if handedness[0].category_name == hand_side:
            return result.hand_landmarks[i]
    return None
