"""MediaPipe 手部检测/跟踪封装 — 采集、预处理、检测共用。

提供两种检测模式：
    IMAGE 模式（detect()）：逐帧独立检测，用于采集器实时叠加
    VIDEO 模式（detect_video()）：帧间跟踪，用于预处理和检测的视频流处理

首次使用时自动下载 hand_landmarker.task 模型到 .models/ 目录。
GPU delegate 创建失败时会静默回退到 CPU。
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# 尝试导入 mediapipe，不可用时设置标志位（采集器需要检测这个标志）
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


# MediaPipe Hand Landmarker 模型下载地址（float16 版本，体积更小）
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# 手部骨架连接关系，用于绘制骨架线条
# 格式：(节点A索引, 节点B索引)，共 23 条边 + 4 条指尖辅助边
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # 腕→拇指
    (0, 5), (5, 6), (6, 7), (7, 8),       # 腕→食指
    (0, 9), (9, 10), (10, 11), (11, 12),  # 腕→中指
    (0, 13), (13, 14), (14, 15), (15, 16),# 腕→无名指
    (0, 17), (17, 18), (18, 19), (19, 20),# 腕→小指
    (5, 9), (9, 13), (13, 17),            # 掌骨横连
]

# 关键指尖节点索引常量
THUMB_TIP = 4    # 拇指尖
INDEX_TIP = 8    # 食指尖
MIDDLE_TIP = 12  # 中指尖

# 关键指尖的高亮颜色（BGR 格式），用于 HUD 绘制
HIGHLIGHT_COLORS = {
    THUMB_TIP: (255, 210, 50),   # 金黄色
    INDEX_TIP: (50, 230, 80),    # 绿色
    MIDDLE_TIP: (30, 140, 255),  # 蓝色
}


def resolve_model(model_cache_dir: str) -> str:
    """解析 MediaPipe 模型路径，不存在时自动下载。

    模型文件为 hand_landmarker.task，保存到 model_cache_dir 目录。
    首次运行时从 Google Storage 下载（约 10MB），之后直接使用缓存。

    Args:
        model_cache_dir: 模型缓存目录路径

    Returns:
        模型文件的绝对路径字符串
    """
    cache_dir = Path(model_cache_dir)
    model_path = cache_dir / "hand_landmarker.task"
    if model_path.exists():
        return str(model_path)
    # 缓存不存在，自动下载
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
    """创建 MediaPipe HandLandmarker 实例。

    注意事项：
        - VIDEO 模式的 landmarker 要求 timestamps 单调递增，不能跨视频复用
        - GPU delegate 可能因平台/MediaPipe 版本不支持而失败，代码中会回退 CPU
        - num_hands=1：只检测一只手，减少误检

    Args:
        model_path: 模型文件路径
        running_mode: "IMAGE"（逐帧）或 "VIDEO"（帧间跟踪）
        delegate: "CPU" 或 "GPU"

    Returns:
        HandLandmarker 实例，或 None（mediapipe 不可用时）
    """
    if not MEDIAPIPE_AVAILABLE:
        return None

    # 配置 delegate（CPU/GPU）
    delegate_name = delegate.upper()
    base_kwargs = {"model_asset_path": model_path}
    if delegate_name == "GPU":
        base_kwargs["delegate"] = mp_python.BaseOptions.Delegate.GPU
    elif delegate_name == "CPU":
        base_kwargs["delegate"] = mp_python.BaseOptions.Delegate.CPU

    # 构建 BaseOptions 和 HandLandmarkerOptions
    base_opts = mp_python.BaseOptions(**base_kwargs)
    mode = getattr(mp_vision.RunningMode, running_mode.upper())
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mode,
        num_hands=1,                           # 只检测一只手
        min_hand_detection_confidence=0.5,     # 手部检测最低置信度
        min_hand_presence_confidence=0.5,      # 手部存在最低置信度
        min_tracking_confidence=0.5,           # 跟踪最低置信度
    )
    return mp_vision.HandLandmarker.create_from_options(opts)


def detect(frame_bgr: np.ndarray, landmarker, hand_side: Optional[str] = None) -> Optional[list]:
    """IMAGE 模式检测 — 逐帧独立检测，用于采集器实时叠加。

    Args:
        frame_bgr: BGR 格式的图像帧
        landmarker: IMAGE 模式的 HandLandmarker 实例
        hand_side: 手性过滤 ("Left"/"Right")，None 表示取第一只手

    Returns:
        21 个 NormalizedLandmark 的列表，或 None（未检测到手）
    """
    if landmarker is None:
        return None
    # BGR → RGB → MediaPipe Image
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
    """VIDEO 模式检测 — 帧间跟踪，用于预处理和检测的视频流。

    关键约束：
        - timestamps 必须单调递增
        - 一个 landmarker 实例只能处理一个视频（不能跨视频复用）
        - 速度比 IMAGE 模式快，因为帧间可复用 tracking 状态

    Args:
        frame_bgr: BGR 格式的图像帧
        landmarker: VIDEO 模式的 HandLandmarker 实例
        timestamp_ms: 当前帧的时间戳（毫秒），必须单调递增
        hand_side: 手性过滤 ("Left"/"Right")，None 表示取第一只手

    Returns:
        21 个 NormalizedLandmark 的列表，或 None（未检测到手）
    """
    if landmarker is None:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_img, timestamp_ms)
    return _select_hand(result, hand_side)


def _select_hand(result, hand_side: Optional[str]) -> Optional[list]:
    """从检测结果中选择指定手性的手。

    当检测到多只手时，根据 hand_side 参数选择：
        - None: 返回第一只手
        - "Left"/"Right": 返回指定手性的手，找不到则返回 None

    Args:
        result: MediaPipe HandLandmarker 的检测结果
        hand_side: 目标手性

    Returns:
        21 个 NormalizedLandmark 的列表，或 None
    """
    if not result.hand_landmarks:
        return None
    # 不指定手性，取第一只
    if hand_side is None:
        return result.hand_landmarks[0]
    # 遍历检测到的手，匹配手性
    for i, handedness in enumerate(result.handedness):
        if handedness[0].category_name == hand_side:
            return result.hand_landmarks[i]
    # 没有匹配的手性
    return None
