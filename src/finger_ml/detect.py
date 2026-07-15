"""Offline video gesture event detection.

The command extracts MediaPipe hand landmarks in VIDEO mode, runs the dense
segmentation model over the whole sequence, and writes gesture events with
start/end times.
"""

# 推理管线总体流程:
#   1. extract_video_features  — 用 MediaPipe VIDEO 模式逐帧提取手部关键点，
#                                 归一化后构建运动特征 (shape: T×21×C)
#   2. predict_probabilities   — 将特征分块送入 GestureSegmenter 模型，
#                                 重叠区取平均，得到逐帧类别概率和边界概率
#   3. probabilities_to_events — 根据概率后处理（平滑/阈值/合并），
#                                 输出结构化手势事件列表
#   4. 输出 JSON + 可选叠加可视化视频
#
# 关键设计决策:
#   - MediaPipe 使用 VIDEO 模式而非 IMAGE 模式，以获得帧间追踪，
#     提升关键点时序连贯性；但 VIDEO 模式的 landmarker 单次使用，
#     时间戳必须单调递增，因此每段视频需新建 landmarker 实例
#   - 分块推理 (chunked prediction) 支持任意长度视频，
#     通过重叠区平均 (overlap averaging) 消除边界效应
#   - 边界头 (boundary head) 额外预测手势起止帧概率，
#     辅助事件边界的精确定位

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from finger_ml.features import build_motion_features, normalize_landmarks, KEEP_INDICES
from finger_ml.hand_tracking import (
    HAND_CONNECTIONS,
    HIGHLIGHT_COLORS,
    detect_video,
    make_landmarker,
    resolve_model,
)
from finger_ml.labels import GESTURE_ZH, LABEL_NAMES, NUM_CLASSES
from finger_ml.model import GestureSegmenter, probabilities_to_events
from finger_ml.video_io import make_writer


# 每个类别的可视化颜色 (BGR 格式)，与 LABEL_NAMES 顺序对应
# 6 种手势 + 1 个背景类，共 7 种颜色
COLORS = [
    (50, 230, 80),     # 绿色 — 类别 0
    (30, 140, 255),    # 橙色 — 类别 1
    (255, 210, 50),    # 黄色 — 类别 2
    (200, 80, 255),    # 紫色 — 类别 3
    (50, 200, 255),    # 浅橙 — 类别 4
    (255, 80, 80),     # 红色 — 类别 5
    (100, 100, 100),   # 灰色 — 类别 6 (背景)
]


def select_device() -> torch.device:
    """自动选择最优计算设备。

    优先级: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU。
    该函数在推理开始时调用一次，决定模型和张量的运行设备。

    Returns:
        torch.device: 选中的设备对象
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_video_features(
    video_path: Path,
    *,
    model_cache_dir: str,
    hand_side: str | None,
    delegate: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int]:
    """从视频中提取手部关键点特征，用于后续模型推理。

    逐帧读取视频，使用 MediaPipe HandLandmarker (VIDEO 模式) 检测手部关键点，
    归一化后构建运动特征。当某帧检测不到手时，复用上一帧的关键点并标记为无效。

    算法流程:
        1. 初始化 MediaPipe landmarker (VIDEO 模式，支持帧间追踪)
        2. 逐帧读取视频，计算时间戳 (毫秒)，调用 detect_video 获取关键点
        3. 对检测到的关键点: 选取 KEEP_INDICES 子集 → 归一化 → 存储世界坐标和图像坐标
        4. 对未检测到手的帧: 复用上一帧关键点，标记 valid=False
        5. 将所有帧的关键点堆叠为 numpy 数组，调用 build_motion_features 构建运动特征

    Args:
        video_path: 视频文件路径
        model_cache_dir: MediaPipe 模型文件的缓存目录
        hand_side: 指定检测左手/右手，None 表示自动选择第一个检测到的手
        delegate: MediaPipe 推理委托类型 ("CPU" 或 "GPU")

    Returns:
        tuple 包含 6 个元素:
        - features: 运动特征数组，shape (T, 21, C)，C 由 build_motion_features 决定
        - image_landmarks: 图像坐标系下的关键点，shape (T, 21, 2)，用于可视化骨架绘制
        - valid: 每帧 MediaPipe 是否成功检测到手，shape (T,)
        - fps: 视频帧率
        - width: 视频宽度 (像素)
        - height: 视频高度 (像素)
    """
    # 下载或定位 MediaPipe hand_landmarker.task 模型文件
    model_path = resolve_model(model_cache_dir)
    # 尝试创建 landmarker 实例；GPU 委托不可用时自动回退 CPU
    try:
        lmkr = make_landmarker(model_path, running_mode="VIDEO", delegate=delegate)
    except Exception as exc:
        if delegate == "GPU":
            print(f"[warn] MediaPipe GPU delegate 不可用，回退 CPU：{exc}")
            lmkr = make_landmarker(model_path, running_mode="VIDEO", delegate="CPU")
        else:
            raise

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    # 读取视频元信息；若元信息缺失则使用合理默认值
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 逐帧累积关键点数据
    landmarks: list[np.ndarray] = []        # 归一化后的世界坐标系关键点 (21, 3)
    image_landmarks: list[np.ndarray] = []   # 图像坐标系关键点 (21, 2)，用于可视化
    valid: list[bool] = []                   # MediaPipe 是否成功检测到手
    last = np.zeros((21, 3), dtype=np.float32)      # 上一帧归一化关键点 (用于缺失帧填充)
    last_img = np.zeros((21, 2), dtype=np.float32)   # 上一帧图像坐标关键点
    t0 = time.time()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # VIDEO 模式要求时间戳单调递增，按帧索引和帧率计算毫秒时间戳
        timestamp_ms = int(round(frame_idx * 1000 / fps))
        lms = detect_video(frame, lmkr, timestamp_ms, hand_side=hand_side)
        if lms is None:
            # 未检测到手: 复用上一帧关键点，标记为无效
            # 设计决策: 填充而非跳帧，保持时序连续性，模型可利用运动特征推断
            landmarks.append(last.copy())
            image_landmarks.append(last_img.copy())
            valid.append(False)
        else:
            # 检测到手: 提取原始关键点 → 选取子集 → 归一化
            raw = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            last_img = raw[KEEP_INDICES, :2].copy()       # 保存图像坐标供可视化用
            last = normalize_landmarks(raw[KEEP_INDICES])  # 归一化为相对手腕的局部坐标
            landmarks.append(last.copy())
            image_landmarks.append(last_img.copy())
            valid.append(True)
        frame_idx += 1
        # 每 200 帧输出一次进度信息
        if frame_idx % 200 == 0:
            pct = frame_idx / max(total, 1) * 100
            speed = frame_idx / max(time.time() - t0, 1e-6)
            print(f"\r  MediaPipe {frame_idx}/{total} ({pct:.0f}%) {speed:.1f} fps", end="", flush=True)
    cap.release()
    # 关闭 landmarker，释放 MediaPipe 资源
    if lmkr is not None:
        lmkr.close()
    print(f"\r  MediaPipe done: {frame_idx} frames.                     ")

    # 空视频的特殊处理: 返回零维数组
    if not landmarks:
        return (
            np.zeros((0, 21, 12), dtype=np.float32),
            np.zeros((0, 21, 2), dtype=np.float32),
            np.zeros(0, dtype=bool),
            fps,
            width,
            height,
        )
    # 将列表堆叠为 numpy 数组，并构建运动特征 (含速度/加速度等时序信息)
    arr = np.stack(landmarks, axis=0).astype(np.float32)
    img_arr = np.stack(image_landmarks, axis=0).astype(np.float32)
    valid_arr = np.array(valid, dtype=bool)
    return build_motion_features(arr, valid_arr), img_arr, valid_arr, fps, width, height


@torch.no_grad()
def predict_probabilities(
    model: GestureSegmenter,
    features: np.ndarray,
    device: torch.device,
    *,
    chunk_len: int,
    overlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """分块推理: 将长视频特征分段送入模型，重叠区取平均，输出逐帧概率。

    算法流程 (分块重叠推理):
        1. 计算步长 stride = chunk_len - overlap
        2. 按步长滑动窗口，每段送入模型前向推理
        3. 对重叠区域的预测结果进行累加，最后除以该帧被覆盖的次数
        4. 输出归一化后的类别概率和边界概率

    设计决策 — 为什么需要分块重叠推理:
        - 模型输入长度固定 (chunk_len)，但视频长度任意
        - 直接分段会导致段边界处预测质量下降 (缺乏上下文)
        - 重叠区取平均可以平滑边界效应，提升整体预测一致性
        - overlap=128, chunk_len=512 时步长为 384，每帧至少被覆盖 1 次，
          重叠区帧被覆盖 2+ 次，取平均后更鲁棒

    通道数对齐:
        若特征通道数与模型输入通道数不匹配，自动裁剪或零填充。
        这保证了旧模型 (少通道) 与新特征 (多通道) 之间的兼容性。

    Args:
        model: 训练好的 GestureSegmenter 模型实例
        features: 运动特征数组，shape (T, 21, C)
        device: 推理设备 (cuda / mps / cpu)
        chunk_len: 每个推理块的帧数长度
        overlap: 相邻块之间的重叠帧数

    Returns:
        tuple 包含 2 个元素:
        - probs: 逐帧类别概率，shape (T, NUM_CLASSES)，每行概率之和为 1
        - boundary_probs: 逐帧边界概率，shape (2, T)，
          boundary_probs[0] 为手势起始概率，boundary_probs[1] 为手势结束概率
    """
    n = len(features)
    # 空序列直接返回零数组
    if n == 0:
        return np.zeros((0, NUM_CLASSES), dtype=np.float32), np.zeros((2, 0), dtype=np.float32)
    # 通道数对齐: 裁剪或零填充至模型期望的输入通道数
    c_expected = model.input_channels
    if features.shape[2] < c_expected:
        # 通道不足: 在末尾零填充
        pad = np.zeros((*features.shape[:2], c_expected - features.shape[2]), dtype=np.float32)
        features = np.concatenate([features, pad], axis=2)
    elif features.shape[2] > c_expected:
        # 通道过多: 裁剪到模型期望的通道数
        features = features[:, :, :c_expected]

    # 分块推理核心逻辑
    stride = max(1, chunk_len - overlap)   # 滑动步长，确保至少为 1
    logits_sum = np.zeros((n, NUM_CLASSES), dtype=np.float32)   # 类别概率累加器
    boundary_sum = np.zeros((2, n), dtype=np.float32)           # 边界概率累加器
    counts = np.zeros(n, dtype=np.float32)                       # 每帧被覆盖次数计数器
    for start in range(0, n, stride):
        end = min(n, start + chunk_len)
        length = end - start
        # 构造定长 chunk，不足部分零填充
        chunk = np.zeros((chunk_len, 21, c_expected), dtype=np.float32)
        chunk[:length] = features[start:end]
        # 维度变换: (T, 21, C) → (1, C, T, 21) 以匹配模型输入格式
        x = torch.from_numpy(chunk).permute(2, 0, 1).unsqueeze(0).to(device)
        # 模型前向推理，返回类别 logits、边界 logits 和辅助输出
        logits, boundary_logits, _ = model(x)
        # 类别 logits → softmax 概率 (仅取有效长度部分，忽略零填充帧)
        probs = torch.softmax(logits[:, :, :length], dim=1).squeeze(0).T.cpu().numpy()
        # 边界 logits → sigmoid 概率 (起始/结束各一个概率值)
        b_probs = torch.sigmoid(boundary_logits[:, :, :length]).squeeze(0).cpu().numpy()
        # 累加到对应帧位置 (重叠区会被多次累加)
        logits_sum[start:end] += probs
        boundary_sum[:, start:end] += b_probs
        counts[start:end] += 1.0
        # 已到达序列末尾，无需继续
        if end >= n:
            break
    # 取平均: 累加值 / 被覆盖次数，确保 counts 至少为 1 防止除零
    counts = np.maximum(counts, 1.0)
    return logits_sum / counts[:, None], boundary_sum / counts[None, :]


def load_model(checkpoint: Path, device: torch.device) -> GestureSegmenter:
    """从检查点文件加载训练好的 GestureSegmenter 模型。

    模型超参数从检查点的 args 字段恢复，确保与训练时一致。
    若检查点缺少某参数，则使用默认值。

    Args:
        checkpoint: 模型检查点文件路径 (.pt)
        device: 模型加载到的目标设备

    Returns:
        GestureSegmenter: 已加载权重并设为 eval 模式的模型实例
    """
    ckpt = torch.load(checkpoint, map_location=device)
    # 从检查点恢复训练时的模型超参数
    args = ckpt.get("args", {})
    model = GestureSegmenter(
        input_channels=int(args.get("input_channels", 12)),
        hidden_dim=int(args.get("hidden_dim", 128)),
        temporal_channels=int(args.get("temporal_channels", 128)),
        temporal_layers=int(args.get("temporal_layers", 6)),
        temporal_stages=int(args.get("temporal_stages", 2)),
        dropout=float(args.get("dropout", 0.25)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _draw_panel(frame: np.ndarray, alpha: float = 0.72) -> None:
    """在视频帧顶部和底部绘制半透明信息面板背景。

    顶部面板用于显示预测结果文字，底部面板用于显示概率条和边界条。
    使用 addWeighted 实现半透明效果，避免遮挡视频主体内容。

    Args:
        frame: 视频帧 (原地修改)，BGR 格式
        alpha: 面板不透明度，1.0 为完全不透明，0.0 为完全透明
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    # 顶部面板: 高度 118 像素，深灰底色
    cv2.rectangle(overlay, (0, 0), (w, 118), (18, 18, 18), -1)
    # 底部面板: 高度 178 像素，深灰底色
    cv2.rectangle(overlay, (0, h - 178), (w, h), (18, 18, 18), -1)
    # 将半透明面板混合到原始帧上
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _draw_skeleton(frame: np.ndarray, pts_norm: np.ndarray, valid: bool) -> None:
    """在视频帧上绘制手部骨架和关键点。

    将归一化坐标 (0~1) 转换为像素坐标后绘制 HAND_CONNECTIONS 连线和关键点圆圈。
    有效帧 (MediaPipe 检测成功) 使用彩色关键点，无效帧 (填充帧) 使用蓝色标记。

    设计决策 — 区分有效/无效帧的视觉反馈:
        - 有效帧: 骨架连线为白色，关键点按 HIGHLIGHT_COLORS 着色 (指尖等高亮)
        - 无效帧: 骨架和关键点全部使用蓝色，提示用户该帧为推断填充

    Args:
        frame: 视频帧 (原地修改)，BGR 格式
        pts_norm: 归一化图像坐标关键点，shape (21, 2)，值域 [0, 1]
        valid: 该帧 MediaPipe 是否成功检测到手
    """
    if pts_norm.size == 0:
        return
    h, w = frame.shape[:2]
    # 将归一化坐标转换为像素坐标
    pts = np.round(pts_norm * np.array([w, h], dtype=np.float32)).astype(np.int32)
    # 根据有效性选择骨架连线颜色
    line_color = (230, 230, 230) if valid else (80, 80, 220)
    # 绘制骨架连线 (手指骨骼连接关系)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), line_color, 2, cv2.LINE_AA)
    # 绘制关键点圆圈
    for i, pt in enumerate(pts):
        color = HIGHLIGHT_COLORS.get(i, (180, 180, 180))
        if not valid:
            color = (80, 80, 220)  # 无效帧统一蓝色
        # 高亮关键点 (指尖等) 使用较大半径
        radius = 7 if i in HIGHLIGHT_COLORS else 4
        cv2.circle(frame, tuple(pt), radius, color, -1, cv2.LINE_AA)
        # 高亮关键点额外绘制白色外圈，增强视觉辨识度
        if i in HIGHLIGHT_COLORS:
            cv2.circle(frame, tuple(pt), radius + 2, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_prob_bars(frame: np.ndarray, probs: np.ndarray, x: int, y: int, width: int) -> None:
    """在视频帧上绘制各类别概率条形图。

    每个类别一行，包含: 暗色背景条 + 彩色概率填充条 + 类别名和百分比文字。
    条形长度与概率值成正比，直观展示模型对各类别的置信度。

    Args:
        frame: 视频帧 (原地修改)，BGR 格式
        probs: 当前帧的类别概率数组，shape (NUM_CLASSES,)
        x: 条形图左上角 x 坐标
        y: 条形图左上角 y 坐标
        width: 条形图最大宽度 (像素)
    """
    for i, (name, prob) in enumerate(zip(LABEL_NAMES, probs)):
        yy = y + i * 22  # 每行间隔 22 像素
        color = COLORS[i]
        # 暗色背景条 (满宽度)
        cv2.rectangle(frame, (x, yy), (x + width, yy + 14), (48, 48, 48), -1)
        # 彩色概率填充条 (宽度与概率成正比)
        cv2.rectangle(frame, (x, yy), (x + int(width * float(prob)), yy + 14), color, -1)
        # 类别名 + 百分比文字
        cv2.putText(
            frame,
            f"{name:<18} {float(prob) * 100:5.1f}%",
            (x + width + 10, yy + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )


def _draw_boundary_strip(
    frame: np.ndarray,
    idx: int,
    boundary_probs: np.ndarray,
    events: list[dict],
    fps: float,
) -> None:
    """在视频帧底部绘制边界概率时间条。

    可视化当前帧附近 ±window 帧范围内的边界概率分布:
        - 每个竖线代表一帧的边界概率高度 (取起始和结束概率的较大值)
        - 青色竖线: 起始边界概率 >= 结束边界概率 (手势开始倾向)
        - 橙色竖线: 结束边界概率 > 起始边界概率 (手势结束倾向)
        - 中心白色竖线: 当前帧位置
        - 上下圆点: 标记检测到的事件起止帧

    设计决策 — 边界条的作用:
        帮助用户直观判断模型是否准确捕获了手势的起止边界，
        便于调试后处理参数 (conf_threshold, min_event_ms 等)。

    Args:
        frame: 视频帧 (原地修改)，BGR 格式
        idx: 当前帧索引
        boundary_probs: 逐帧边界概率，shape (2, T)
        events: 已检测到的手势事件列表
        fps: 视频帧率
    """
    h, w = frame.shape[:2]
    left, right = 20, w - 20  # 时间条左右边界
    y = h - 40               # 时间条垂直位置
    # 绘制暗色背景条
    cv2.rectangle(frame, (left, y), (right, y + 16), (45, 45, 45), -1)
    # 时间窗口: 当前帧前后各 window 帧
    window = 150
    lo = max(0, idx - window)
    hi = min(boundary_probs.shape[1], idx + window + 1)
    if hi > lo:
        # 取起始和结束概率的最大值作为竖线高度
        values = np.maximum(boundary_probs[0, lo:hi], boundary_probs[1, lo:hi])
        for k, value in enumerate(values):
            x = left + int((right - left) * k / max(1, len(values) - 1))
            # 颜色区分: 青色=起始倾向，橙色=结束倾向
            color = (60, 210, 255) if boundary_probs[0, lo + k] >= boundary_probs[1, lo + k] else (255, 130, 80)
            cv2.line(frame, (x, y + 16), (x, y + 16 - int(16 * float(value))), color, 1)
    # 当前帧位置标记 (白色竖线)
    center_x = left + (right - left) // 2
    cv2.line(frame, (center_x, y - 4), (center_x, y + 20), (255, 255, 255), 2, cv2.LINE_AA)
    # 标记检测到的事件起止帧位置 (彩色圆点)
    for ev in events:
        # 事件起始帧: 时间条上方圆点
        if lo <= ev["start_frame"] <= hi:
            x = left + int((right - left) * (ev["start_frame"] - lo) / max(1, hi - lo))
            cv2.circle(frame, (x, y - 5), 4, COLORS[ev["label"]], -1, cv2.LINE_AA)
        # 事件结束帧: 时间条下方圆点
        if lo <= ev["end_frame"] <= hi:
            x = left + int((right - left) * (ev["end_frame"] - lo) / max(1, hi - lo))
            cv2.circle(frame, (x, y + 24), 4, COLORS[ev["label"]], -1, cv2.LINE_AA)
    # 图例文字
    cv2.putText(
        frame,
        f"boundary strip +/- {window / max(fps, 1e-6):.1f}s   cyan=start orange=end",
        (left, y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def _topk_text(row: np.ndarray, k: int = 3) -> str:
    """将概率数组格式化为 top-k 类别文字，用于叠加视频显示。

    按概率降序排列，取前 k 个类别，格式为 "类名=概率%  类名=概率%  ..."。

    Args:
        row: 单帧类别概率数组，shape (NUM_CLASSES,)
        k: 取前 k 个类别

    Returns:
        str: 格式化的 top-k 文字串
    """
    ids = np.argsort(row)[::-1][:k]
    return "  ".join(f"{LABEL_NAMES[i]}={row[i] * 100:.1f}%" for i in ids)


def write_overlay_video(
    video_path: Path,
    out_path: Path,
    events: list[dict],
    probs: np.ndarray,
    boundary_probs: np.ndarray,
    image_landmarks: np.ndarray,
    valid: np.ndarray,
    fps: float,
) -> None:
    """生成叠加了预测结果的可视化视频。

    逐帧读取原始视频，在每帧上绘制:
        1. 半透明信息面板 (顶部/底部)
        2. 手部骨架和关键点 (有效帧彩色，无效帧蓝色)
        3. 预测类别、置信度、事件信息 (顶部文字)
        4. 类别概率条形图 (左下角)
        5. 边界概率时间条 (右下角)

    设计决策 — 可视化的目的:
        用于人工审查模型推理质量，特别是:
        - 手部关键点是否稳定 (观察骨架)
        - 分类是否准确 (观察顶部预测文字和概率条)
        - 事件边界是否合理 (观察边界条和事件标记)

    Args:
        video_path: 原始视频文件路径
        out_path: 输出叠加视频文件路径
        events: 已检测到的手势事件列表
        probs: 逐帧类别概率，shape (T, NUM_CLASSES)
        boundary_probs: 逐帧边界概率，shape (2, T)
        image_landmarks: 图像坐标关键点，shape (T, 21, 2)
        valid: 每帧 MediaPipe 是否有效，shape (T,)
        fps: 视频帧率
    """
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = make_writer(out_path, fps, width, height)

    # 构建帧到事件的映射: 每帧可能属于一个事件或不在任何事件中
    # 同一帧若被多个事件覆盖，取最后一个 (通常不会重叠)
    event_by_frame: list[dict | None] = [None] * len(probs)
    for ev in events:
        for i in range(max(0, ev["start_frame"]), min(len(probs), ev["end_frame"] + 1)):
            event_by_frame[i] = ev

    # 逐帧读取原始视频并叠加可视化
    idx = 0
    while idx < len(probs):
        ok, frame = cap.read()
        if not ok:
            break
        # 当前帧的预测类别和置信度
        label = int(probs[idx].argmax())
        conf = float(probs[idx, label])
        color = COLORS[label]
        active = event_by_frame[idx]
        valid_now = bool(valid[idx]) if idx < len(valid) else False

        # 绘制半透明面板背景
        _draw_panel(frame)
        # 绘制手部骨架
        if idx < len(image_landmarks):
            _draw_skeleton(frame, image_landmarks[idx], valid_now)

        # --- 顶部信息文字 ---
        # 第一行: 帧号、时间戳、MediaPipe 状态
        title = LABEL_NAMES[label] if active is None else active["gesture"]
        zh = GESTURE_ZH.get(title, title)
        status = "MP_OK" if valid_now else "MP_MISS"
        cv2.putText(frame, f"#{idx:05d}  t={idx / max(fps, 1e-6):7.2f}s  {status}",
                    (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (235, 235, 235), 2, cv2.LINE_AA)
        # 第二行: 预测类别 (英文名/中文名) + 置信度
        cv2.putText(frame, f"PRED {title} / {zh}  {conf * 100:.1f}%",
                    (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.86, color, 2, cv2.LINE_AA)
        # 第三行: 事件信息 (若有事件则显示起止时间和持续时间，否则显示 none)
        if active is not None:
            cv2.putText(
                frame,
                f"EVENT {active['start_ms']}ms - {active['end_ms']}ms  duration={active['duration_ms']}ms  mean={active['mean_conf'] * 100:.1f}%",
                (18, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(frame, "EVENT none", (18, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (170, 170, 170), 1, cv2.LINE_AA)

        # --- 右上角辅助信息 ---
        # top-3 类别概率
        start_p = float(boundary_probs[0, idx]) if boundary_probs.size else 0.0
        end_p = float(boundary_probs[1, idx]) if boundary_probs.size else 0.0
        cv2.putText(frame, f"top3: {_topk_text(probs[idx])}",
                    (width - 760, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (235, 235, 235), 1, cv2.LINE_AA)
        # 边界概率数值
        cv2.putText(frame, f"boundary_start={start_p:.3f}  boundary_end={end_p:.3f}",
                    (width - 760, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (60, 210, 255), 1, cv2.LINE_AA)
        # 模型信息
        cv2.putText(frame, f"model: class_probs + boundary_head + postprocess",
                    (width - 760, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (210, 210, 210), 1, cv2.LINE_AA)

        # --- 底部可视化 ---
        # 左下角: 类别概率条形图
        _draw_prob_bars(frame, probs[idx], 22, height - 164, 190)
        # 右下角: 边界概率时间条
        _draw_boundary_strip(frame, idx, boundary_probs, events, fps)
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()


def detect_events_in_video(args: argparse.Namespace) -> None:
    """完整的离线视频手势事件检测管线入口函数。

    管线步骤:
        1. 选择推理设备 (GPU/CPU)
        2. 提取视频特征 (MediaPipe 关键点 + 运动特征构建)
        3. 加载模型
        4. 分块推理获取逐帧概率
        5. 后处理: 概率 → 结构化事件列表
        6. 输出 JSON 文件 (包含事件列表和逐帧详细概率)
        7. 可选: 输出叠加可视化视频

    Args:
        args: 命令行参数命名空间，包含以下字段:
            - video: 输入视频路径
            - checkpoint: 模型检查点路径
            - out_json: 输出 JSON 路径 (可选，默认与视频同名)
            - out_video: 输出叠加视频路径 (可选)
            - hand_side: 手部侧别 ("Left"/"Right"/"Any")
            - model_cache_dir: MediaPipe 模型缓存目录
            - delegate: MediaPipe 委托类型
            - chunk_len: 推理块长度
            - overlap: 重叠帧数
            - conf_threshold: 事件置信度阈值
            - min_event_ms: 最短事件持续时间 (毫秒)
            - max_gap_ms: 事件间最大间隔 (毫秒)
            - smooth: 概率平滑窗口大小
    """
    video = Path(args.video)
    checkpoint = Path(args.checkpoint)
    out_json = Path(args.out_json) if args.out_json else video.with_suffix(".events.json")
    if not video.exists():
        raise FileNotFoundError(video)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    # 步骤 1: 选择推理设备
    device = select_device()
    print(f"[info] torch device={device}")

    # 步骤 2: 提取视频特征
    features, image_landmarks, valid, fps, width, height = extract_video_features(
        video,
        model_cache_dir=args.model_cache_dir,
        hand_side=None if args.hand_side == "Any" else args.hand_side,
        delegate=args.delegate,
    )
    print(f"[info] video={video.name} {width}x{height} @ {fps:.3f}fps frames={len(features)} valid={valid.mean() if len(valid) else 0:.1%}")

    # 步骤 3: 加载模型
    model = load_model(checkpoint, device)

    # 步骤 4: 分块推理，获取逐帧类别概率和边界概率
    probs, boundary_probs = predict_probabilities(
        model,
        features,
        device,
        chunk_len=args.chunk_len,
        overlap=args.overlap,
    )

    # 步骤 5: 后处理 — 概率 → 事件列表
    # probabilities_to_events 内部执行: 概率平滑 → 阈值过滤 → 连续段检测 → 间隔合并
    events = probabilities_to_events(
        probs,
        fps,
        boundary_probs=boundary_probs,
        conf_threshold=args.conf_threshold,
        min_event_ms=args.min_event_ms,
        max_gap_ms=args.max_gap_ms,
        smooth=args.smooth,
    )

    # 打印检测结果摘要
    print(f"[result] events={len(events)}")
    for ev in events:
        zh = GESTURE_ZH.get(ev["gesture"], ev["gesture"])
        print(f"  {ev['start_ms']:>6}ms - {ev['end_ms']:>6}ms  {ev['gesture']:<20} {zh:<8} conf={ev['mean_conf']:.3f}")

    # 步骤 6: 构建并输出 JSON 文件
    # JSON 包含两层信息: (1) 结构化事件列表 (2) 逐帧详细概率 (供精细分析)
    frame_rows = [
        {
            "frame": i,
            "time_ms": int(round(i * 1000 / fps)),
            "label": int(row.argmax()),
            "label_name": LABEL_NAMES[int(row.argmax())],
            "confidence": round(float(row.max()), 4),
            # 边界概率: 起始和结束
            "boundary_start": round(float(boundary_probs[0, i]), 4) if boundary_probs.size else 0.0,
            "boundary_end": round(float(boundary_probs[1, i]), 4) if boundary_probs.size else 0.0,
            # top-3 类别概率
            "top3": [
                {
                    "label": int(j),
                    "label_name": LABEL_NAMES[int(j)],
                    "prob": round(float(row[int(j)]), 4),
                }
                for j in np.argsort(row)[::-1][:3]
            ],
            # MediaPipe 是否检测到手
            "mediapipe_valid": bool(valid[i]) if i < len(valid) else False,
        }
        for i, row in enumerate(probs)
    ]
    payload = {
        "task": "offline_gesture_event_detection",
        "video": str(video),
        "checkpoint": str(checkpoint),
        "fps": fps,
        "total_frames": len(probs),
        # MediaPipe 配置和有效性统计
        "mediapipe": {
            "running_mode": "VIDEO",
            "delegate": args.delegate,
            "valid_rate": float(valid.mean()) if len(valid) else 0.0,
        },
        # 后处理参数记录，便于复现和调试
        "postprocess": {
            "conf_threshold": args.conf_threshold,
            "min_event_ms": args.min_event_ms,
            "max_gap_ms": args.max_gap_ms,
            "smooth": args.smooth,
        },
        "events": events,
        "frames": frame_rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out_json}")

    # 步骤 7: 可选 — 生成叠加可视化视频
    if args.out_video:
        write_overlay_video(
            video,
            Path(args.out_video),
            events,
            probs,
            boundary_probs,
            image_landmarks,
            valid,
            fps,
        )
        print(f"[save] {args.out_video}")


def main() -> None:
    """CLI 入口: 解析命令行参数并执行手势事件检测。

    命令行参数说明:
        --video         输入视频路径 (必需)
        --checkpoint    模型检查点路径 (默认 checkpoints/best.pt)
        --out-json      输出 JSON 路径 (默认与视频同名 .events.json)
        --out-video     输出叠加视频路径 (可选)
        --hand-side     检测哪只手: Left/Right/Any (默认 Right)
        --model-cache-dir  MediaPipe 模型缓存目录 (默认 .models)
        --delegate      MediaPipe 推理委托: CPU/GPU (默认 CPU)
        --chunk-len     推理块帧数长度 (默认 512)
        --overlap       相邻块重叠帧数 (默认 128)
        --conf-threshold  事件置信度阈值 (默认 0.55)
        --min-event-ms  最短事件持续时间 (默认 120ms)
        --max-gap-ms    事件间可合并最大间隔 (默认 120ms)
        --smooth        概率平滑窗口大小 (默认 7 帧)
    """
    ap = argparse.ArgumentParser(description="Detect gesture events and start/end times in a video")
    ap.add_argument("--video", required=True)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-video", default=None)
    ap.add_argument("--hand-side", default="Right", choices=["Left", "Right", "Any"])
    ap.add_argument("--model-cache-dir", default=".models")
    ap.add_argument("--delegate", default="CPU", choices=["CPU", "GPU"], help="MediaPipe delegate")
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--conf-threshold", type=float, default=0.55)
    ap.add_argument("--min-event-ms", type=int, default=120)
    ap.add_argument("--max-gap-ms", type=int, default=120)
    ap.add_argument("--smooth", type=int, default=7)
    detect_events_in_video(ap.parse_args())


if __name__ == "__main__":
    main()
