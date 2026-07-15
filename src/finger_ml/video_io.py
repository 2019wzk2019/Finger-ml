"""共享视频写出工具。

优先使用 ffmpeg/libx264 写 MP4，因为 OpenCV 默认的 mp4v 编码器
输出的 MPEG-4 Part 2 格式在浏览器和 VS Code 中可能无法播放。
系统没有 ffmpeg 时自动回退到 OpenCV 的 VideoWriter。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class VideoWriter(Protocol):
    """视频写出器协议，定义 write/release/isOpened 三个方法。"""
    def write(self, frame: np.ndarray) -> None:
        ...
    def release(self) -> None:
        ...
    def isOpened(self) -> bool:
        ...


def make_writer(path: Path, fps: float, width: int, height: int) -> VideoWriter:
    """创建视频写出器，优先 ffmpeg/libx264，回退 OpenCV mp4v。

    ffmpeg 输出 H.264 编码的 MP4，兼容性最好（浏览器/播放器通用）。
    OpenCV 的 mp4v 输出 MPEG-4 Part 2，部分播放器不支持。

    Args:
        path: 输出文件路径
        fps: 帧率
        width: 画面宽度
        height: 画面高度

    Returns:
        FFMPEGWriter 或 cv2.VideoWriter 实例
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # 优先尝试 ffmpeg
    if shutil.which("ffmpeg"):
        return FFMPEGWriter(path, fps, width, height)
    # 回退到 OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {path}")
    return writer


class FFMPEGWriter:
    """流式 H.264 视频写出器，接收 BGR 格式帧通过管道传给 ffmpeg。

    编码参数：
        - libx264 编码器
        - ultrafast 预设（速度优先，采集时避免积压）
        - CRF 24（质量与体积平衡）
        - yuv420p 像素格式（最大兼容性）
        - faststart 标记（支持流式播放）
    """

    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        self.path = path
        cmd = [
            "ffmpeg",
            "-y",                        # 覆盖已有文件
            "-f", "rawvideo",            # 输入格式：原始视频
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",   # 画面尺寸
            "-pix_fmt", "bgr24",         # 输入像素格式（OpenCV BGR）
            "-r", f"{fps:.4f}",          # 帧率
            "-i", "-",                   # 从 stdin 读取
            "-c:v", "libx264",           # H.264 编码
            "-preset", "ultrafast",      # 编码速度优先
            "-crf", "24",                # 质量因子（越小质量越高）
            "-pix_fmt", "yuv420p",       # 输出像素格式（兼容性好）
            "-movflags", "+faststart",   # 元数据前置，支持流式播放
            "-loglevel", "error",        # 只输出错误信息
            str(path),
        ]
        # 启动 ffmpeg 子进程，通过 stdin 管道写入帧数据
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        """将一帧 BGR 图像写入 ffmpeg 管道。"""
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            raise ValueError(f"Invalid frame shape: {frame.shape}")
        if self.proc.stdin is not None:
            self.proc.stdin.write(frame.tobytes())

    def release(self) -> None:
        """关闭管道并等待 ffmpeg 编码完成。"""
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        code = self.proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with code {code} while writing {self.path}")

    def isOpened(self) -> bool:
        """检查 ffmpeg 进程是否仍在运行。"""
        return self.proc.poll() is None
