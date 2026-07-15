"""
finger_ml.preprocess — 从采集视频中提取手部骨架特征并生成帧级标签。

输出：data/features/<session>_<subject>.npz
  landmarks : float32 [N_frames, 21, 3]  — 手掌局部坐标系下的骨架坐标
  features  : float32 [N_frames, 21, C]  — 扩展特征，含坐标/速度/距离/方向/有效性
  labels    : int64   [N_frames]          — 0-5 手势, 6=背景
  valid     : bool    [N_frames]          — MediaPipe 是否检测到手

用法：
    uv run finger-preprocess --data-dir data/
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from finger_ml.features import (
    FEATURE_NAMES,
    KEEP_INDICES,
    NODE_NAMES,
    NUM_NODES,
    build_motion_features,
    normalize_landmarks,
)
from finger_ml.hand_tracking import detect_video, make_landmarker, resolve_model
from finger_ml.labels import BACKGROUND_LABEL, LABEL_NAMES
from finger_ml.video_io import make_writer


@contextlib.contextmanager
def _suppress_native_stderr(enabled: bool):
    """屏蔽 MediaPipe/TFLite 原生 C++ 代码的 stderr 噪声日志。

    MediaPipe 的 INFO/WARNING 日志来自原生 C++ 层，Python 的 warnings 过滤器
    无法控制它们。通过重定向文件描述符 fd=2 来隐藏这些后端消息，同时保持
    stdout 的进度输出可见。

    参数:
        enabled: 是否启用屏蔽。为 False 时直接 yield，不做任何重定向。
    """
    if not enabled:
        yield
        return
    # 保存原始 stderr 的文件描述符副本
    sys.stderr.flush()
    saved = os.dup(2)
    # 打开 /dev/null 用于丢弃输出
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        # 将 fd=2 指向 /dev/null，屏蔽所有 stderr 输出
        os.dup2(devnull, 2)
        yield
    finally:
        # 恢复原始 stderr
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)

# ── 常量 ───────────────────────────────────────────────────────────────────

# 手部骨架的 21 个关键点之间的连接关系，用于调试视频中绘制骨架线条。
# 每个元组 (i, j) 表示第 i 个关键点与第 j 个关键点之间有骨骼连接。
# 连接结构: 手腕(0) → 拇指(1-4), 食指(5-8), 中指(9-12), 无名指(13-16), 小指(17-20)
# 加上掌骨间的横向连接: 5-9, 9-13, 13-17
_DEBUG_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

# 手势边界裁剪：在标注的起止帧各向内缩减 2 帧。
# 原因：手势起止边界处往往是过渡帧，标签模糊（如刚按下/刚松开），
# 这些帧的标签不可靠，裁掉可提升训练数据质量。
BOUNDARY_MARGIN  = 2

# 动作开始前的预备/过渡帧数：在动作起始帧(raw_s)前 pre_ignore_frames 帧范围内，
# 将 train_mask 设为 False，避免把"预备动作"当作背景训练。
# 默认 4 帧，即动作标签起始前 4 帧不计入训练。
PRE_IGNORE_FRAMES = 4

# 动作结束后的回弹段屏蔽秒数：手势结束后手指恢复原位的过程可能看起来像
# 反向手势，默认屏蔽 1 秒。运行时根据视频 fps 换算为帧数。
POST_IGNORE_SECONDS = 1.0


def _longest_false_run(valid: np.ndarray) -> int:
    """计算 bool 数组中连续 False 值的最长长度。

    用于检测手部追踪的最长连续丢失段。如果丢失段过长，
    说明该区间内骨架数据全部是填充的，质量堪忧。

    参数:
        valid: 帧级 bool 数组，True 表示该帧检测到手，False 表示丢失。

    返回:
        连续 False 的最大帧数。
    """
    longest = 0
    current = 0
    for ok in valid:
        if ok:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _consecutive_step_norms(
    landmarks: np.ndarray,
    valid: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """计算相邻有效帧之间每个节点的位移量（L2 范数）。

    仅当相邻两帧都是有效帧（valid 均为 True）时才计算位移，
    避免因检测丢失导致帧间距离被填充数据污染。

    参数:
        landmarks: [N_frames, 21, 3] 手掌局部坐标系下的骨架坐标。
        valid: [N_frames] bool 数组，True 表示该帧检测到手。
        mask: [N_frames] 可选 bool 数组，进一步限定哪些帧参与计算。
              例如只计算背景帧的抖动时传入 labels==BACKGROUND_LABEL。

    返回:
        [M, NUM_NODES] float32 数组，M 为满足条件的相邻帧对数。
        每行是两个相邻有效帧之间各节点的 L2 位移。
    """
    if len(landmarks) < 2:
        return np.empty((0, NUM_NODES), dtype=np.float32)
    # 仅计算相邻两帧都是有效帧的位移，跳过检测丢失帧
    pair_valid = valid[1:] & valid[:-1]
    # 如果提供了额外 mask，进一步过滤
    if mask is not None:
        pair_valid &= mask[1:] & mask[:-1]
    if not np.any(pair_valid):
        return np.empty((0, NUM_NODES), dtype=np.float32)
    # 逐帧差分后取 L2 范数
    diff = landmarks[1:] - landmarks[:-1]
    return np.linalg.norm(diff[pair_valid], axis=2).astype(np.float32)


def _p95(values: np.ndarray) -> float:
    """计算数组的第 95 百分位数。

    使用 P95 而非最大值是因为骨架数据可能存在个别极端异常帧，
    P95 对离群值更鲁棒，更能反映典型抖动水平。

    参数:
        values: 一维数值数组。

    返回:
        第 95 百分位数值，空数组返回 0.0。
    """
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 95))


def _score_high_is_good(value: float, bad_at: float) -> float:
    """线性评分函数：值越低得分越高，达到 bad_at 时得分为 0。

    评分公式: score = clip(1.0 - value / bad_at, 0, 1)
    即 value=0 时得 1.0（满分），value>=bad_at 时得 0.0（零分）。

    参数:
        value: 待评分的原始指标值。
        bad_at: 认为该指标完全不可接受的阈值。

    返回:
        [0.0, 1.0] 范围内的得分。
    """
    if bad_at <= 0:
        return 1.0
    return float(np.clip(1.0 - value / bad_at, 0.0, 1.0))


def compute_quality_metrics(
    landmarks: np.ndarray,
    valid: np.ndarray,
    labels: np.ndarray,
    fps: float,
) -> dict:
    """计算 session 级别的骨架数据质量指标。

    质量评分由四部分加权组成:
      - 检测率 (detect_score, 权重 0.45): MediaPipe 成功检测到手部帧的比例
      - 抖动 (jitter_score, 权重 0.25): 背景帧的骨架稳定性
      - 丢失间隔 (gap_score, 权重 0.15): 最长连续检测丢失时长
      - 类别覆盖 (coverage_score, 权重 0.15): 出现的手势类别数占 6 类的比例

    抖动以手掌局部归一化坐标下每帧位移为单位。背景帧抖动是最有用的
    稳定性信号，因为排除了手势动作期间的有意运动。

    参数:
        landmarks: [N_frames, 21, 3] 手掌局部坐标系下的骨架坐标。
        valid: [N_frames] bool，True 表示该帧 MediaPipe 检测到手。
        labels: [N_frames] int64，帧级标签 (0-5 手势, 6=背景)。
        fps: 视频帧率，用于将帧数换算为秒数。

    返回:
        包含所有质量指标的字典，含总分和各分项。
    """
    n_frames = int(len(valid))
    # 检测率：成功检测帧占总帧数的比例
    valid_rate = float(valid.mean()) if n_frames else 0.0
    # 最长连续检测丢失段（帧数和秒数）
    longest_gap_frames = _longest_false_run(valid)
    longest_gap_sec = float(longest_gap_frames / fps) if fps > 0 else 0.0

    # 提取背景帧 mask，用于单独计算背景帧抖动（排除手势运动的影响）
    bg_mask = labels == BACKGROUND_LABEL
    # 全帧和背景帧的逐帧位移
    all_steps = _consecutive_step_norms(landmarks, valid)
    bg_steps = _consecutive_step_norms(landmarks, valid, bg_mask)
    # 如果没有背景帧数据，退而使用全帧位移
    jitter_steps = bg_steps if bg_steps.size else all_steps

    # 逐节点计算 P95 抖动，找出最不稳定的节点
    per_node_p95 = (
        np.percentile(jitter_steps, 95, axis=0).astype(np.float32)
        if jitter_steps.size
        else np.zeros(NUM_NODES, dtype=np.float32)
    )
    worst_node_idx = int(per_node_p95.argmax()) if len(per_node_p95) else 0

    # 统计出现的手势类别数和类别覆盖率
    counts = np.bincount(labels, minlength=BACKGROUND_LABEL + 1)
    present_gesture_classes = int((counts[:BACKGROUND_LABEL] > 0).sum())
    class_coverage = present_gesture_classes / BACKGROUND_LABEL

    # ── 四项评分 ─────────────────────────────────────────────────────
    detect_score = valid_rate
    # 最长丢失间隔评分：0.5 秒以上认为完全不可接受
    gap_score = _score_high_is_good(longest_gap_sec, bad_at=0.5)
    # 抖动评分：手掌局部归一化坐标系下，
    #   0.02 单位/帧以下非常稳定，0.12 以上肉眼可见抖动
    #   线性映射: [0.02, 0.12] → [1.0, 0.0]
    jitter_p95 = _p95(jitter_steps)
    jitter_score = 1.0 - float(np.clip((jitter_p95 - 0.02) / 0.10, 0.0, 1.0))
    coverage_score = class_coverage
    # 加权总分，映射到 0-100 分
    quality_score = (
        0.45 * detect_score
        + 0.25 * jitter_score
        + 0.15 * gap_score
        + 0.15 * coverage_score
    ) * 100.0

    return {
        "quality_score": round(float(quality_score), 1),
        "valid_rate": round(valid_rate, 6),
        "detect_fail_frames": int(n_frames - int(valid.sum())),
        "longest_detect_fail_run_frames": int(longest_gap_frames),
        "longest_detect_fail_run_sec": round(longest_gap_sec, 3),
        "jitter_p50": round(float(np.median(jitter_steps)) if jitter_steps.size else 0.0, 6),
        "jitter_p95": round(jitter_p95, 6),
        "background_jitter_p95": round(_p95(bg_steps), 6),
        "motion_step_p95": round(_p95(all_steps), 6),
        "worst_jitter_node": NODE_NAMES[worst_node_idx],
        "worst_jitter_node_index": worst_node_idx,
        "worst_jitter_node_p95": round(float(per_node_p95[worst_node_idx]), 6),
        "per_node_jitter_p95": {
            name: round(float(value), 6)
            for name, value in zip(NODE_NAMES, per_node_p95)
        },
        "present_gesture_classes": present_gesture_classes,
        "class_coverage": round(float(class_coverage), 3),
        "score_parts": {
            "detect": round(float(detect_score * 100), 1),
            "jitter": round(float(jitter_score * 100), 1),
            "gap": round(float(gap_score * 100), 1),
            "coverage": round(float(coverage_score * 100), 1),
        },
    }


# ── Debug 视频渲染 ──────────────────────────────────────────────────────────

def _render_progress_bar(done: int, total: int, width: int = 28) -> str:
    """渲染 ASCII 进度条字符串。

    参数:
        done: 已完成的帧数。
        total: 总帧数。
        width: 进度条字符宽度。

    返回:
        如 "[=====>...........]" 格式的进度条字符串。
    """
    if total <= 0:
        return "[?]"
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(width * ratio)
    # 未完成时最后一个填充字符用 ">" 表示进行中
    if done < total and filled < width:
        bar = "=" * max(0, filled - 1) + ">" + "." * (width - filled)
    else:
        bar = "=" * width
    return f"[{bar}]"


def _print_frame_progress(
    done: int,
    total: int,
    started_at: float,
    *,
    final: bool = False,
) -> None:
    """在终端打印单行帧处理进度（覆盖式更新）。

    使用 \\r 回车符实现原地更新，不换行（final=True 时换行）。

    参数:
        done: 已处理帧数。
        total: 总帧数。
        started_at: 开始时间（time.monotonic 返回值），用于计算实时 fps。
        final: 是否为最终输出，为 True 时打印换行符。
    """
    elapsed = max(time.monotonic() - started_at, 1e-6)
    fps = done / elapsed
    percent = (done / total * 100.0) if total > 0 else 0.0
    text = (
        f"\r      {_render_progress_bar(done, total)} "
        f"{done:>6}/{total:<6} frames "
        f"{percent:6.2f}% "
        f"{fps:6.1f} fps"
    )
    sys.stdout.write(text)
    if final:
        sys.stdout.write("\n")
    sys.stdout.flush()

def _draw_skeleton(
    frame: np.ndarray,
    pts: np.ndarray,      # [21, 2]，图像归一化坐标 [0,1] 或像素坐标
    color: tuple,
    is_normalized: bool,
    fw: int,
    fh: int,
) -> None:
    """在 frame 上绘制 21 节点骨架。

    参数:
        frame: OpenCV BGR 图像，直接在其上绘制（原地修改）。
        pts: [21, 2] 关键点坐标，可以是 [0,1] 归一化坐标或像素坐标。
        color: BGR 颜色元组，如 (60, 60, 255) 为红色。
        is_normalized: pts 是否为 [0,1] 归一化坐标。为 True 时乘以宽高转换为像素。
        fw: 图像宽度（像素）。
        fh: 图像高度（像素）。
    """
    if is_normalized:
        # 归一化坐标 [0,1] 转像素坐标
        px = (pts[:, 0] * fw).astype(int)
        py = (pts[:, 1] * fh).astype(int)
    else:
        px, py = pts[:, 0].astype(int), pts[:, 1].astype(int)

    # 绘制骨骼连线
    for i, j in _DEBUG_EDGES:
        cv2.line(frame, (px[i], py[i]), (px[j], py[j]), color, 1, cv2.LINE_AA)
    # 绘制关键点圆点，指尖（4,8,12,16,20）半径更大以便区分
    for k in range(len(px)):
        r = 5 if k in (4, 8, 12, 16, 20) else 3  # 指尖更大
        cv2.circle(frame, (px[k], py[k]), r, color, -1, cv2.LINE_AA)


def _write_debug_video(
    video_path: Path,
    out_path: Path,
    raw_img: np.ndarray,   # [N, 21, 2] 原始图像坐标（[0,1]）
    landmarks: np.ndarray, # [N, 21, 3] 手掌局部坐标（保留参数用于接口一致）
    labels: np.ndarray,    # [N]
    valid: np.ndarray,     # [N] bool
    fps: float,
) -> None:
    """生成骨架叠加调试视频，用于人工检查追踪质量和标签准确性。

    将原始视频逐帧读取，叠加 MediaPipe 检测到的骨架关键点（红色）
    和帧编号、标签、检测状态等 HUD 信息，输出为新的视频文件。

    参数:
        video_path: 原始视频文件路径。
        out_path: 调试视频输出路径。
        raw_img: [N, 21, 2] MediaPipe 原始图像归一化坐标 (x,y ∈ [0,1])。
        landmarks: [N, 21, 3] 手掌局部坐标系下的骨架坐标（本函数未使用，
                   保留参数以维持接口一致性，未来可能绘制归一化骨架）。
        labels: [N] 帧级标签数组。
        valid: [N] bool，每帧是否检测到手部。
        fps: 输出视频帧率。
    """
    cap = cv2.VideoCapture(str(video_path))
    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = make_writer(out_path, fps, fw, fh)

    for i in range(len(labels)):
        ok, frame = cap.read()
        if not ok:
            break

        label_name = LABEL_NAMES[int(labels[i])]
        is_valid   = valid[i]

        # ── 红色：原始 MediaPipe 关键点（[0,1] 图像坐标）
        if is_valid:
            _draw_skeleton(frame, raw_img[i], (60, 60, 255), True, fw, fh)

        # ── HUD 文字：帧号 + 标签名 + 检测状态
        status_color = (60, 220, 60) if is_valid else (60, 60, 255)
        status_text  = "OK" if is_valid else "MISS"
        # 先画黑色描边再画彩色文字，确保在任何背景上可读
        cv2.putText(frame, f"#{i:04d}  {label_name}  [{status_text}]",
                    (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"#{i:04d}  {label_name}  [{status_text}]",
                    (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, "RAW(red)  FEATURES=palm-local 21pts", (12, fh - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"    [debug] → {out_path}")


# ── 单个 session 处理 ───────────────────────────────────────────────────────

def process_session(
    video_path: Path,
    label_path: Path,
    out_path:   Path,
    lmkr,
    hand_side: Optional[str] = None,
    debug_video_path: Optional[Path] = None,
    show_progress: bool = True,
    pre_ignore_frames: int = PRE_IGNORE_FRAMES,
    post_ignore_seconds: float = POST_IGNORE_SECONDS,
) -> dict:
    """提取一个 session 的骨架序列和帧级标签。

    核心处理流程：
      1. 加载 JSON 标注，构建帧级标签数组和训练掩码
      2. 逐帧调用 MediaPipe 手部追踪，提取 21 关键点
      3. 归一化到手掌局部坐标系，构建运动特征
      4. 计算质量指标，保存为 .npz 文件

    帧索引约定：
      - JSON 标注使用 1-indexed 的 start_frame / end_frame
      - 本函数内部转换为 0-indexed
      - BOUNDARY_MARGIN 在标注起止帧两侧各裁掉若干过渡帧

    训练掩码 (train_mask) 的构建逻辑：
      - 默认全为 True（所有帧参与训练）
      - 手势标签区间内（裁掉边界后）的帧 train_mask=True，用于手势分类训练
      - 手势区间前的 pre_ignore_frames 帧 → train_mask=False
        （预备动作，既不像纯背景也不像目标手势）
      - 手势区间后的 post_ignore_frames 帧 → train_mask=False
        （回弹段，手指恢复原位的过程可能看起来像反向手势）
      - 纯背景区域 train_mask=True，作为背景类训练数据

    参数:
        video_path: 视频文件路径 (.mp4)。
        label_path: 对应的 JSON 标注文件路径。
        out_path: 输出 .npz 特征文件路径。
        lmkr: MediaPipe HandLandmarker 实例（VIDEO 模式）。
              注意：VIDEO 模式的 landmarker 要求时间戳单调递增，
              每个视频需要新建实例，不可跨视频复用。
        hand_side: 指定提取哪只手的关键点 ("Left"/"Right")。
                   None 表示取检测到的第一只手。单手数据集建议指定，避免选错手。
        debug_video_path: 调试视频输出路径，None 则不生成。
        show_progress: 是否在终端显示逐帧进度条。
        pre_ignore_frames: 动作开始前忽略的帧数，这些帧 train_mask=False。
        post_ignore_seconds: 动作结束后忽略的秒数，这些帧 train_mask=False。

    返回:
        包含处理统计信息的字典:
          n_frames: 总帧数
          n_gesture: 手势帧数（标签 < BACKGROUND_LABEL）
          n_detect_fail: 检测丢失帧数
          n_ignored_train_frames: train_mask=False 的帧数
          post_ignore_frames: 回弹屏蔽换算后的帧数
          detect_rate: 检测率字符串
          feature_channels: 特征通道数
          quality: 质量指标字典
    """
    # ── 加载标注 ────────────────────────────────────────────────────────────
    with label_path.open(encoding="utf-8") as f:
        meta = json.load(f)

    annotations = meta["annotations"]
    fps         = float(meta["fps"])
    # 将回弹屏蔽秒数换算为帧数
    post_ignore_frames = max(0, int(round(fps * post_ignore_seconds)))

    # ── 打开视频 ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        # 部分编码器无法预先知道帧数，先读完
        n_frames = None

    # ── 构建帧级标签 ────────────────────────────────────────────────────────
    # 先读完一遍以确认总帧数（若未知）
    if n_frames is None:
        total = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            total += 1
        n_frames = total
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 初始化所有帧标签为背景类(6)
    labels = np.full(n_frames, BACKGROUND_LABEL, dtype=np.int64)
    # 训练掩码：默认所有帧参与训练
    train_mask = np.ones(n_frames, dtype=bool)
    for ann in annotations:
        # JSON 标注中 start_frame/end_frame 是 1-indexed，转为 0-indexed
        raw_s = int(ann["start_frame"]) - 1
        raw_e = int(ann["end_frame"]) - 1
        # 向内裁掉边界过渡帧，得到可靠的标签区间
        s = raw_s + BOUNDARY_MARGIN   # 转 0-indexed
        e = raw_e - BOUNDARY_MARGIN
        # 在裁剪后的可靠区间内赋予手势标签
        if 0 <= s <= e < n_frames:
            labels[s : e + 1] = ann["label"]
        # 屏蔽按键边界、动作预备和动作结束后的恢复段，避免把"回弹反向动作"当背景训练。
        # 前导忽略区：动作起始帧(raw_s)前 pre_ignore_frames 帧到裁剪起点(s)前 1 帧
        ignore_s = max(0, raw_s - pre_ignore_frames)
        ignore_e = min(n_frames - 1, s - 1)
        if ignore_s <= ignore_e:
            train_mask[ignore_s : ignore_e + 1] = False
        # 后续忽略区：裁剪终点(e)后 1 帧到动作结束帧(raw_e)后 post_ignore_frames 帧
        ignore_s = max(0, e + 1)
        ignore_e = min(n_frames - 1, raw_e + post_ignore_frames)
        if ignore_s <= ignore_e:
            train_mask[ignore_s : ignore_e + 1] = False

    # ── 逐帧提取骨架 ────────────────────────────────────────────────────────
    landmarks_arr = np.zeros((n_frames, NUM_NODES, 3), dtype=np.float32)
    valid_arr     = np.zeros(n_frames, dtype=bool)
    # 上一帧的有效骨架，用于检测丢失时填充（保持时序连续性，避免 NaN）
    last_valid    = np.zeros((NUM_NODES, 3), dtype=np.float32)

    # debug 模式：额外保存原始图像坐标（[0,1] 归一化像素坐标）
    raw_img_arr: Optional[np.ndarray] = (
        np.zeros((n_frames, NUM_NODES, 2), dtype=np.float32)
        if debug_video_path else None
    )

    frame_idx = 0
    n_fail    = 0
    progress_started_at = time.monotonic()
    last_progress_at = 0.0
    if show_progress:
        _print_frame_progress(0, n_frames, progress_started_at)

    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= n_frames:
            break

        # VIDEO 模式要求时间戳单调递增，单位为毫秒
        timestamp_ms = int(round(frame_idx * 1000 / fps))
        lms = detect_video(frame, lmkr, timestamp_ms, hand_side=hand_side)
        if lms is not None:
            # 提取 21 个关键点的 x, y, z 坐标
            raw = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            # 只保留 KEEP_INDICES 指定的 21 个关键点（原始 MediaPipe 有更多点）
            sub = raw[KEEP_INDICES]              # [21, 3]
            if raw_img_arr is not None:
                raw_img_arr[frame_idx] = sub[:, :2]  # 保存 x,y 图像坐标
            # 归一化到手掌局部坐标系（以手腕为原点，消除平移）
            sub = normalize_landmarks(sub)
            landmarks_arr[frame_idx] = sub
            valid_arr[frame_idx]     = True
            last_valid               = sub
        else:
            # 检测丢失：用上一帧的骨架填充，保持序列连续性
            # valid_arr 标记为 False，训练时可通过 train_mask 或 valid 过滤
            landmarks_arr[frame_idx] = last_valid
            valid_arr[frame_idx]     = False
            n_fail += 1

        frame_idx += 1
        # 限频刷新进度条，约每 0.12 秒更新一次，避免终端闪烁
        now = time.monotonic()
        if show_progress and (now - last_progress_at >= 0.12 or frame_idx >= n_frames):
            _print_frame_progress(frame_idx, n_frames, progress_started_at, final=frame_idx >= n_frames)
            last_progress_at = now

    cap.release()
    # 确保进度条最终换行
    if show_progress and frame_idx < n_frames:
        _print_frame_progress(frame_idx, n_frames, progress_started_at, final=True)

    # 若视频实际帧数 < n_frames（罕见，某些编码器报告不准），裁剪到实际长度
    actual = frame_idx
    landmarks_arr = landmarks_arr[:actual]
    valid_arr     = valid_arr[:actual]
    labels        = labels[:actual]
    train_mask    = train_mask[:actual]
    # 基于骨架序列构建运动特征（速度、距离、方向等）
    features_arr  = build_motion_features(landmarks_arr, valid_arr)
    # 计算质量指标
    quality       = compute_quality_metrics(landmarks_arr, valid_arr, labels, fps)

    # ── 保存 ───────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        landmarks = landmarks_arr,
        features  = features_arr,
        feature_names = np.array(FEATURE_NAMES),
        labels    = labels,
        valid     = valid_arr,
        train_mask = train_mask,
        fps       = np.float32(fps),
        quality_json = np.array(json.dumps(quality, ensure_ascii=False)),
    )

    n_gesture = int((labels < BACKGROUND_LABEL).sum())

    # 生成调试视频（如果指定了 debug_video_path）
    if debug_video_path is not None and raw_img_arr is not None:
        _write_debug_video(
            video_path, debug_video_path,
            raw_img_arr[:actual],
            landmarks_arr[:actual],
            labels[:actual],
            valid_arr[:actual],
            fps,
        )

    return {
        "n_frames":       actual,
        "n_gesture":      n_gesture,
        "n_detect_fail":  n_fail,
        "n_ignored_train_frames": int((~train_mask).sum()),
        "post_ignore_frames": post_ignore_frames,
        "post_ignore_seconds": float(post_ignore_seconds),
        "detect_rate":    f"{(actual - n_fail) / max(actual, 1):.1%}",
        "feature_channels": int(features_arr.shape[2]),
        "quality": quality,
    }


# ── 批量处理 ────────────────────────────────────────────────────────────────

def match_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """将 video/*.mp4 和 labels/*.json 按 session_subject 名称配对。

    配对规则：视频和标注文件的 stem（不含扩展名的文件名）必须一致。
    例如 video/session01_subjectA.mp4 与 labels/session01_subjectA.json 配对。

    参数:
        data_dir: 数据根目录，应包含 video/ 和 labels/ 子目录。

    返回:
        (video_path, label_path) 元组列表，按文件名排序。

    异常:
        FileNotFoundError: 如果没有找到任何配对。
    """
    video_map = {p.stem: p for p in (data_dir / "video").glob("*.mp4")}
    label_map = {p.stem: p for p in (data_dir / "labels").glob("*.json")}
    # 取两个集合的交集，即同时有视频和标注的 session
    common    = sorted(set(video_map) & set(label_map))
    if not common:
        raise FileNotFoundError(
            f"No matching video+label pairs found in {data_dir}. "
            "Run finger-collect first."
        )
    return [(video_map[k], label_map[k]) for k in common]


# ── CLI 入口 ────────────────────────────────────────────────────────────────

def main() -> None:
    """finger-preprocess 命令行入口。

    扫描 data-dir 下的 video/ 和 labels/ 目录，将每个配对的 (视频, 标注)
    交给 process_session 处理，输出 .npz 特征文件到 data-dir/features/。

    关键逻辑:
      - 每个视频新建一个 HandLandmarker 实例（VIDEO 模式要求时间戳从 0 开始，
        不可跨视频复用同一个 landmarker）
      - 默认跳过已存在的 .npz 文件，--force 强制重提取
      - GPU delegate 先探测可用性，不可用则回退 CPU
    """
    ap = argparse.ArgumentParser(
        description="从采集视频中提取 ST-GCN 骨架特征"
    )
    ap.add_argument("--data-dir",        default="data",    help="数据根目录")
    ap.add_argument("--model-cache-dir", default=".models", help="MediaPipe 模型缓存目录")
    ap.add_argument("--force",           action="store_true",
                    help="强制重新提取（跳过已存在的 .npz）")
    ap.add_argument("--hand-side",       default=None, choices=["Left", "Right"],
                    help="只提取指定手的关键点（Left/Right），None 表示不过滤")
    ap.add_argument("--delegate", default="CPU", choices=["CPU", "GPU"],
                    help="MediaPipe delegate（默认 CPU；GPU 可用性取决于平台/wheel）")
    ap.add_argument("--show-mediapipe-logs", action="store_true",
                    help="显示 MediaPipe/TFLite 原生日志（默认屏蔽 noisy stderr）")
    ap.add_argument("--debug-video",     action="store_true",
                    help="输出骨架 HUD 调试视频到 data/debug/（红=原始，绿=归一化反投影）")
    ap.add_argument("--no-progress",     action="store_true",
                    help="关闭逐帧进度条输出")
    ap.add_argument("--pre-ignore-frames", type=int, default=PRE_IGNORE_FRAMES,
                    help="每段动作开始附近忽略帧数，不参与训练（默认 4）")
    ap.add_argument("--post-ignore-seconds", type=float, default=POST_IGNORE_SECONDS,
                    help="每段动作结束后恢复段忽略秒数，不参与训练（默认 1.0）")
    args = ap.parse_args()

    data_dir     = Path(args.data_dir)
    features_dir = data_dir / "features"
    debug_dir    = data_dir / "debug" if args.debug_video else None

    # 下载或定位 MediaPipe hand_landmarker.task 模型文件
    model_path = resolve_model(args.model_cache_dir)
    actual_delegate = args.delegate
    if args.delegate == "GPU":
        # 探测 GPU delegate 可用性。VIDEO 模式的 landmarker 不允许跨视频复用
        # （时间戳必须从 0 开始单调递增），所以真正的实例在每个 session 中创建。
        # 这里仅创建一个探测实例来验证 GPU 是否可用，然后立即关闭。
        try:
            with _suppress_native_stderr(not args.show_mediapipe_logs):
                probe = make_landmarker(model_path, running_mode="VIDEO", delegate="GPU")
            if probe is not None:
                probe.close()
        except Exception as exc:
            print(f"[warn] MediaPipe GPU delegate 不可用，回退 CPU：{exc}")
            actual_delegate = "CPU"

    # 按 session_subject 名称配对视频和标注文件
    pairs = match_pairs(data_dir)
    print(f"[info] 发现 {len(pairs)} 个 session，开始提取骨架特征...")

    for video_path, label_path in pairs:
        out_path = features_dir / (video_path.stem + ".npz")
        # 默认跳过已存在的特征文件，避免重复提取
        if out_path.exists() and not args.force:
            print(f"  [skip] {out_path.name} 已存在（--force 强制重提取）")
            continue

        print(f"  [proc] {video_path.name}")
        # 每个视频创建独立的 landmarker 实例
        lmkr = None
        try:
            debug_video_path = (
                debug_dir / (video_path.stem + "_debug.mp4") if debug_dir else None
            )
            with _suppress_native_stderr(not args.show_mediapipe_logs):
                # 创建 VIDEO 模式的 HandLandmarker，每个视频单独创建
                lmkr = make_landmarker(model_path, running_mode="VIDEO", delegate=actual_delegate)
                stats = process_session(video_path, label_path, out_path, lmkr,
                                        hand_side=args.hand_side,
                                        debug_video_path=debug_video_path,
                                        show_progress=not args.no_progress,
                                        pre_ignore_frames=args.pre_ignore_frames,
                                        post_ignore_seconds=args.post_ignore_seconds)
            print(
                f"✓  {stats['n_frames']} 帧  "
                f"手势帧 {stats['n_gesture']}  "
                f"忽略训练帧 {stats['n_ignored_train_frames']}  "
                f"回弹屏蔽 {stats['post_ignore_frames']} 帧  "
                f"检测率 {stats['detect_rate']}  "
                f"特征 {stats['feature_channels']}ch  "
                f"质量 {stats['quality']['quality_score']:.1f}"
            )
            print(
                f"      jitter_p95={stats['quality']['jitter_p95']:.4f}  "
                f"bg_jitter_p95={stats['quality']['background_jitter_p95']:.4f}  "
                f"worst={stats['quality']['worst_jitter_node']}"
            )
        except Exception as e:
            if not args.no_progress:
                # 异常时确保进度条换行，避免后续输出错位
                sys.stdout.write("\n")
                sys.stdout.flush()
            print(f"✗  错误：{e}")
        finally:
            # 无论成功与否，都必须关闭 landmarker 释放资源
            if lmkr is not None:
                lmkr.close()

    print(f"[done] 特征文件保存于 {features_dir}/")


if __name__ == "__main__":
    main()
