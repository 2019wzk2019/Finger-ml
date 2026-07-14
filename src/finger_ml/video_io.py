"""Shared video writing utilities."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class VideoWriter(Protocol):
    def write(self, frame: np.ndarray) -> None:
        ...

    def release(self) -> None:
        ...

    def isOpened(self) -> bool:
        ...


def make_writer(path: Path, fps: float, width: int, height: int) -> VideoWriter:
    """Create a broadly playable MP4 writer.

    Prefer ffmpeg/libx264 because OpenCV's MP4 writer often emits MPEG-4 Part 2
    (`mp4v`), which VS Code/browser players may not decode.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        return FFMPEGWriter(path, fps, width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {path}")
    return writer


class FFMPEGWriter:
    """Streaming H.264 writer for BGR OpenCV frames."""

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        self.path = path
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", f"{fps:.4f}",
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "24",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-loglevel", "error",
            str(path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError(f"Invalid frame shape: {frame.shape}")
        if self.proc.stdin is not None:
            self.proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with code {code} while writing {self.path}")

    def isOpened(self) -> bool:
        return self.proc.poll() is None
