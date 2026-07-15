"""Review collected gesture sessions.

Reads the MP4/JSON pair produced by ``finger-collect`` and opens an OpenCV
player with annotation overlays and an action-window timeline.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from finger_ml.hand_tracking import (
    HAND_CONNECTIONS,
    HIGHLIGHT_COLORS,
    detect,
    make_landmarker,
    resolve_model,
)
from finger_ml.labels import GESTURE_ZH, LABEL_NAMES

try:
    from PIL import Image as PILImage, ImageDraw, ImageFont

    _PIL_OK = True
except ImportError:
    _PIL_OK = False


# 窗口标题
WIN_NAME = "Gesture Data Review"
# 初始窗口最大宽度（像素）
MAX_INIT_WIN_W = 1280
# 初始窗口最大高度（像素）
MAX_INIT_WIN_H = 720
# 中文字体候选路径列表（按优先级排序，优先使用微软雅黑）
FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
)
# 每种标签对应的颜色 (BGR)，用于动作窗着色
COLORS = (
    (80, 220, 80),
    (60, 170, 255),
    (255, 170, 60),
    (220, 90, 255),
    (255, 220, 80),
    (80, 210, 230),
)
# 抖动警告阈值：低于此值认为稳定（绿色）
JITTER_WARN = 0.06
# 抖动严重阈值：高于此值认为抖动严重（红色）
JITTER_BAD = 0.12


@dataclass(frozen=True)
class Annotation:
    """单条手势标注记录，对应一个动作窗（gesture window）。

    Attributes:
        gesture: 手势英文名称，如 "fist", "open_palm"
        label: 手势对应的整数标签索引
        rep: 该手势在第几组重复（rep编号）
        start_frame: 动作窗起始帧号（含），1-indexed
        end_frame: 动作窗结束帧号（含），1-indexed
        start_ms: 动作窗起始毫秒时间戳
        end_ms: 动作窗结束毫秒时间戳
    """

    gesture: str
    label: int
    rep: int
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int

    @property
    def duration_sec(self) -> float:
        """动作窗时长（秒）。"""
        return max(0, self.end_ms - self.start_ms) / 1000.0


@dataclass
class FeatureOverlay:
    """从预处理 NPZ 文件加载的特征叠加数据，用于时间轴抖动可视化。

    Attributes:
        path: NPZ 文件路径
        valid: 布尔数组，每帧是否检测到手部（valid[i]=True 表示第 i 帧检测成功）
        jitter: 浮点数组，每帧的抖动值（相邻帧关键点平均位移）
        quality: 预处理质量报告字典（从 quality_json 字段解析）
    """

    path: Path
    valid: np.ndarray
    jitter: np.ndarray
    quality: dict[str, Any]

    @property
    def n_frames(self) -> int:
        """特征数据覆盖的帧数。"""
        return int(len(self.valid))


def _resolve_font() -> str | None:
    """在系统字体目录中查找可用的中文字体文件路径。

    Returns:
        找到的字体文件绝对路径，若均不存在则返回 None
    """
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def _put_text(
    frame: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    size: int = 24,
    color: tuple[int, int, int] = (245, 245, 245),
    font_path: str | None = None,
) -> None:
    """在视频帧上绘制文字，优先使用 PIL 渲染中文，回退到 OpenCV putText。

    PIL 分支支持 TrueType 中文字体，渲染效果清晰；
    若 PIL 不可用，则退化为 cv2.putText（无法显示中文）。

    Args:
        frame: BGR 格式的视频帧，会被原地修改
        text: 要绘制的文字字符串
        xy: 文字左上角坐标 (x, y)
        size: 字体大小（像素）
        color: 文字颜色 (R, G, B) 格式（PIL 分支）或 (B, G, R)（cv2 分支内部转换）
        font_path: TrueType 字体文件路径，None 则使用默认字体
    """
    if _PIL_OK:
        # PIL 分支：先转 RGB 绘制，再转回 BGR 写回 frame
        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        draw.text(xy, text, font=font, fill=color)
        frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return

    # 回退分支：OpenCV putText，颜色从 RGB 翻转为 BGR
    cv2.putText(
        frame,
        text,
        (xy[0], xy[1] + size),
        cv2.FONT_HERSHEY_SIMPLEX,
        size / 32,
        color[::-1],
        1,
        cv2.LINE_AA,
    )


def _load_json(label_path: Path) -> dict[str, Any]:
    """从 JSON 标注文件加载元数据，并将原始标注字典转换为 Annotation 对象列表。

    Args:
        label_path: JSON 标注文件路径

    Returns:
        包含 "annotations" 键（Annotation 列表）及其他元数据的字典
    """
    with label_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    # 将每个原始标注字典转为 Annotation 数据类实例
    payload["annotations"] = [
        Annotation(
            gesture=str(a["gesture"]),
            label=int(a["label"]),
            rep=int(a.get("rep", 0)),
            start_frame=int(a["start_frame"]),
            end_frame=int(a["end_frame"]),
            start_ms=int(a.get("start_ms", 0)),
            end_ms=int(a.get("end_ms", 0)),
        )
        for a in payload.get("annotations", [])
    ]
    return payload


def _features_path_for(label_path: Path) -> Path:
    """根据标注文件路径推断对应的特征 NPZ 文件路径。

    约定：若标注文件位于 labels/ 子目录，则特征文件在同级的 features/ 子目录中；
    否则特征文件与标注文件同名但扩展名为 .npz。

    Args:
        label_path: JSON 标注文件路径

    Returns:
        推断出的 NPZ 特征文件路径
    """
    if label_path.parent.name == "labels":
        return label_path.parent.parent / "features" / f"{label_path.stem}.npz"
    return label_path.with_suffix(".npz")


def _load_feature_overlay(path: Path | None) -> FeatureOverlay | None:
    """从 NPZ 特征文件加载抖动可视化数据。

    读取 landmarks、valid 数组，计算逐帧抖动值（相邻帧关键点平均位移），
    并解析 quality_json 字段为质量报告字典。

    Args:
        path: NPZ 文件路径，None 或文件不存在则返回 None

    Returns:
        FeatureOverlay 实例，加载失败或路径无效时返回 None
    """
    if path is None or not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        # landmarks: (N, 21, 3) 手部关键点坐标
        landmarks = data["landmarks"].astype(np.float32, copy=False)
        # valid: (N,) 每帧是否检测到手部
        valid = data["valid"].astype(bool, copy=False)
        quality: dict[str, Any] = {}
        # quality_json 是预处理产出的质量报告字符串
        if "quality_json" in data.files:
            quality_raw = str(data["quality_json"])
            quality = json.loads(quality_raw) if quality_raw else {}
    except Exception as exc:
        print(f"[review] warning: cannot load feature overlay {path}: {exc}")
        return None

    # 计算逐帧抖动值：相邻帧间所有关键点的平均位移距离
    jitter = np.zeros(len(valid), dtype=np.float32)
    if len(landmarks) > 1:
        # 仅在前后帧都检测到手时才计算抖动
        pair_valid = valid[1:] & valid[:-1]
        # step: (N-1, 21) 每个关键点的位移距离
        step = np.linalg.norm(landmarks[1:] - landmarks[:-1], axis=2)
        # frame_jitter: (N-1,) 每帧所有关键点的平均位移
        frame_jitter = step.mean(axis=1).astype(np.float32)
        # 仅在两帧都 valid 时填入抖动值，否则为 0
        jitter[1:] = np.where(pair_valid, frame_jitter, 0.0)
    return FeatureOverlay(path=path, valid=valid, jitter=jitter, quality=quality)


def _resolve_pair(data_dir: Path, session: str | None, video: str | None, label: str | None) -> tuple[Path, Path]:
    """根据用户输入的参数解析视频文件与标注文件的路径对。

    优先级：
    1. 若显式指定了 --video，则用该路径，标注路径用 --label 或按命名约定推断
    2. 若只指定了 --label，则从中读取 video_file 字段或按命名约定推断视频路径
    3. 若只指定了 --session，则按 stem 拼接 labels/ 和 video/ 路径
    4. 否则自动选取 data/labels/ 下最新修改的 JSON 文件作为默认会话

    Args:
        data_dir: 数据根目录（默认 data/）
        session: 会话 stem 名称（如 20260425_153000_S01）
        video: 显式指定的 MP4 视频路径
        label: 显式指定的 JSON 标注路径

    Returns:
        (video_path, label_path) 路径元组

    Raises:
        FileNotFoundError: 标注文件或视频文件不存在时抛出
    """
    if video:
        # 分支 1：显式指定了视频路径
        video_path = Path(video)
        label_path = Path(label) if label else data_dir / "labels" / f"{video_path.stem}.json"
    elif label:
        # 分支 2：只指定了标注路径
        label_path = Path(label)
        video_file = _load_json(label_path).get("video_file")
        video_path = Path(video_file) if video_file else data_dir / "video" / f"{label_path.stem}.mp4"
    else:
        # 分支 3/4：用 session stem 或自动选择最新标注
        if session:
            stem = session
        else:
            # 按 mtime 降序排列，取最新修改的 JSON
            labels = sorted((data_dir / "labels").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not labels:
                raise FileNotFoundError(f"No labels found under {data_dir / 'labels'}")
            stem = labels[0].stem
        label_path = data_dir / "labels" / f"{stem}.json"
        video_path = data_dir / "video" / f"{stem}.mp4"

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    return video_path, label_path


def _active_annotation(annotations: list[Annotation], frame_idx: int) -> int | None:
    """查找当前帧所属的动作窗标注索引。

    遍历所有标注，返回第一个 start_frame <= frame_idx <= end_frame 的标注索引。
    若当前帧不在任何动作窗内（即背景帧），返回 None。

    Args:
        annotations: Annotation 列表
        frame_idx: 当前帧号（0-indexed）

    Returns:
        匹配的标注在列表中的索引，或 None（背景帧）
    """
    for i, ann in enumerate(annotations):
        if ann.start_frame <= frame_idx <= ann.end_frame:
            return i
    return None


def _draw_timeline(
    frame: np.ndarray,
    annotations: list[Annotation],
    frame_idx: int,
    total_frames: int,
    active_idx: int | None,
    features: FeatureOverlay | None,
    font_path: str | None,
) -> None:
    """在视频帧底部绘制时间轴面板，包含抖动条和动作窗时间线。

    面板布局（从上到下）：
    - 标题行："抖动/漏检 + 动作窗时间轴"
    - 抖动条（jitter bar）：绿色=稳定，黄色/红色=高抖动，蓝色=漏检（未检测到手部）
    - 动作窗条：彩色矩形表示每个标注的动作窗范围，白色边框高亮当前活跃窗口
    - 播放头竖线：白色竖线标记当前帧位置
    - 底部操作提示行

    Args:
        frame: 视频帧（原地修改）
        annotations: Annotation 列表
        frame_idx: 当前帧号
        total_frames: 视频总帧数
        active_idx: 当前活跃标注索引，None 表示背景帧
        features: FeatureOverlay 数据，None 则不绘制抖动条
        font_path: 中文字体路径
    """
    h, w = frame.shape[:2]
    # 面板总高度（像素）
    panel_h = 124
    # 面板起始 y 坐标
    y0 = h - panel_h
    # 半透明背景：先在副本上画深色矩形，再按权重混合
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # 时间轴左右边距
    left, right = 28, w - 28
    # 抖动条区域：y 坐标和高度
    jitter_y, jitter_h = y0 + 40, 12
    # 动作窗条区域：y 坐标和高度
    bar_y, bar_h = y0 + 60, 20
    # 绘制抖动条和动作窗条的底色
    cv2.rectangle(frame, (left, jitter_y), (right, jitter_y + jitter_h), (42, 42, 42), -1)
    cv2.rectangle(frame, (left, bar_y), (right, bar_y + bar_h), (58, 58, 58), -1)

    # 帧号到像素 x 的分母，避免除以 0
    denom = max(1, total_frames - 1)
    # ---- 抖动条绘制：根据特征数据着色 ----
    if features is not None and features.n_frames:
        n = min(features.n_frames, total_frames)
        # 根据像素宽度和帧数确定分桶数，避免过密绘制
        bucket_count = max(1, min(right - left, n))
        bucket_frames = max(1, int(np.ceil(n / bucket_count)))
        for start in range(0, n, bucket_frames):
            end = min(n, start + bucket_frames)
            x1 = left + int((right - left) * start / denom)
            x2 = left + int((right - left) * max(start + 1, end - 1) / denom)
            if not bool(np.any(features.valid[start:end])):
                # 蓝色：该区间全部漏检（未检测到手部）
                color = (60, 60, 235)
            else:
                # 取区间内最大抖动值作为代表
                value = float(np.max(features.jitter[start:end]))
                if value <= JITTER_WARN:
                    # 绿色到黄绿色的渐变：抖动值越高越偏黄
                    level = value / JITTER_WARN
                    color = (70, int(220 - 45 * level), int(80 + 130 * level))
                else:
                    # 黄绿色到红色的渐变：超过警告阈值后越来越红
                    level = float(np.clip((value - JITTER_WARN) / (JITTER_BAD - JITTER_WARN), 0.0, 1.0))
                    color = (int(70 - 20 * level), int(175 - 105 * level), int(210 + 45 * level))
            cv2.rectangle(frame, (x1, jitter_y), (max(x2, x1 + 1), jitter_y + jitter_h), color, -1)

    # ---- 动作窗条绘制：每个标注的帧范围用彩色矩形表示 ----
    for i, ann in enumerate(annotations):
        x1 = left + int((right - left) * ann.start_frame / denom)
        x2 = left + int((right - left) * ann.end_frame / denom)
        color = COLORS[ann.label % len(COLORS)]
        cv2.rectangle(frame, (x1, bar_y), (max(x2, x1 + 2), bar_y + bar_h), color, -1)
        if active_idx == i:
            # 白色边框高亮当前活跃的动作窗
            cv2.rectangle(frame, (x1, bar_y - 5), (max(x2, x1 + 2), bar_y + bar_h + 5), (255, 255, 255), 2)

    # ---- 播放头竖线：标记当前帧在时间轴上的位置 ----
    x = left + int((right - left) * min(max(frame_idx, 0), denom) / denom)
    cv2.line(frame, (x, jitter_y - 6), (x, bar_y + bar_h + 12), (245, 245, 245), 2, cv2.LINE_AA)
    _put_text(frame, "抖动/漏检 + 动作窗时间轴", (left, y0 + 12), size=21, font_path=font_path)
    _put_text(frame, "绿=稳定  黄/红=抖动高  蓝=漏检", (left + 260, y0 + 14), size=16, color=(210, 210, 210), font_path=font_path)
    _put_text(frame, "SPACE 播放/暂停   A/D 前后窗口   L 循环窗口   ←/→ 逐帧/跳转   Q 退出", (left, y0 + 92), size=18, color=(210, 210, 210), font_path=font_path)


def _draw_hud(
    frame: np.ndarray,
    meta: dict[str, Any],
    annotations: list[Annotation],
    frame_idx: int,
    fps: float,
    active_idx: int | None,
    features: FeatureOverlay | None,
    skeleton_ok: bool | None,
    paused: bool,
    loop_window: bool,
    font_path: str | None,
) -> None:
    """在视频帧顶部绘制 HUD 信息条。

    包含：
    - 当前手势标签名称和序号（背景帧显示"静息/background"）
    - 会话 ID、帧号、时间戳、播放/暂停状态
    - 骨架检测状态（OK/MISS）
    - 特征数据状态（valid/jitter 值）

    Args:
        frame: 视频帧（原地修改）
        meta: 会话元数据字典
        annotations: Annotation 列表
        frame_idx: 当前帧号
        fps: 视频帧率
        active_idx: 当前活跃标注索引，None 表示背景帧
        features: FeatureOverlay 数据，None 表示无特征文件
        skeleton_ok: 骨架检测结果，True=检测到，False=未检测到，None=未启用
        paused: 是否暂停
        loop_window: 是否开启窗口循环
        font_path: 中文字体路径
    """
    h, w = frame.shape[:2]
    # 半透明顶部背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 92), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # 根据是否在动作窗内，显示标签名称或背景提示
    if active_idx is None:
        label_text = "静息 / background"
        color = (210, 210, 210)
    else:
        ann = annotations[active_idx]
        zh = GESTURE_ZH.get(ann.gesture, ann.gesture)
        label_text = f"#{active_idx + 1:02d} {zh} / {ann.gesture}  rep={ann.rep}  {ann.duration_sec:.2f}s"
        color = COLORS[ann.label % len(COLORS)]
    # 播放状态文字
    status = "暂停" if paused else "播放"
    if loop_window:
        status += " | 窗口循环"
    # 骨架检测状态
    skeleton_text = "骨架=实时检测"
    if skeleton_ok is True:
        skeleton_text += " OK"
    elif skeleton_ok is False:
        skeleton_text += " MISS"
    # 特征数据状态文字
    metric_text = ""
    if features is not None and 0 <= frame_idx < features.n_frames:
        valid = "OK" if bool(features.valid[frame_idx]) else "MISS"
        jitter = float(features.jitter[frame_idx])
        metric_text = f"  feature_valid={valid}  jitter={jitter:.4f}"
    elif features is None:
        metric_text = "  no feature npz"
    # 绘制 HUD 三行信息
    _put_text(frame, label_text, (18, 16), size=26, color=color, font_path=font_path)
    _put_text(
        frame,
        f"{meta.get('session_id', '')}  frame={frame_idx}  t={frame_idx / max(fps, 1e-6):.2f}s  {status}",
        (18, 52),
        size=20,
        color=(230, 230, 230),
        font_path=font_path,
    )
    _put_text(
        frame,
        f"{skeleton_text}{metric_text}",
        (max(18, w - 520), 52),
        size=18,
        color=(210, 230, 230),
        font_path=font_path,
    )


def _draw_skeleton_overlay(frame: np.ndarray, lms) -> bool:
    """在视频帧上绘制手部骨架叠加层。

    绘制内容：
    - 所有关键点连接线（浅灰色）
    - 常规关键点：小圆（半径 3，浅灰色）
    - 高亮关键点：大圆（半径 6，彩色）+ 白色外框

    Args:
        frame: BGR 格式的视频帧（原地修改）
        lms: MediaPipe 检测返回的关键点列表，None 则不绘制

    Returns:
        True 表示检测到手部并已绘制，False 表示未检测到
    """
    if lms is None:
        return False

    h, w = frame.shape[:2]
    # 将归一化坐标转换为像素坐标
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    # 绘制所有关键点连接线
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (220, 220, 220), 1, cv2.LINE_AA)

    # 绘制各关键点圆点，高亮点用大圆+白框
    for i, pt in enumerate(pts):
        color = HIGHLIGHT_COLORS.get(i, (180, 180, 180))
        radius = 6 if i in HIGHLIGHT_COLORS else 3
        cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
        if i in HIGHLIGHT_COLORS:
            cv2.circle(frame, pt, radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
    return True


def _print_summary(
    video_path: Path,
    label_path: Path,
    meta: dict[str, Any],
    total_frames: int,
    features: FeatureOverlay | None,
) -> None:
    """在控制台打印会话摘要信息。

    包括视频路径、标注路径、帧数/帧率/标注数、
    特征文件信息和质量报告，以及每个动作窗的详细列表。

    Args:
        video_path: 视频文件路径
        label_path: 标注文件路径
        meta: 会话元数据字典
        total_frames: 视频总帧数
        features: FeatureOverlay 数据，可能为 None
    """
    annotations: list[Annotation] = meta["annotations"]
    fps = float(meta.get("fps") or 0)
    print(f"[review] video: {video_path}")
    print(f"[review] label: {label_path}")
    print(f"[review] frames={total_frames} fps={fps:.3f} annotations={len(annotations)}")
    if features is not None:
        print(f"[review] features: {features.path} frames={features.n_frames}")
        if features.quality:
            quality = features.quality
            print(
                "[review] quality="
                f"{quality.get('quality_score', '?')} "
                f"valid_rate={quality.get('valid_rate', '?')} "
                f"jitter_p95={quality.get('jitter_p95', '?')} "
                f"bg_jitter_p95={quality.get('background_jitter_p95', '?')} "
                f"worst={quality.get('worst_jitter_node', '?')}"
            )
    for i, ann in enumerate(annotations, start=1):
        name = LABEL_NAMES[ann.label] if 0 <= ann.label < len(LABEL_NAMES) else ann.gesture
        print(
            f"  {i:02d}. {GESTURE_ZH.get(name, name)} / {name} "
            f"rep={ann.rep} frames={ann.start_frame}-{ann.end_frame} "
            f"{ann.duration_sec:.2f}s"
        )


def _initial_window_size(cap: cv2.VideoCapture, meta: dict[str, Any]) -> tuple[int, int]:
    """根据视频原始分辨率计算初始窗口大小，不超过最大限制。

    若视频分辨率超过 MAX_INIT_WIN_W x MAX_INIT_WIN_H，则按等比缩小；
    若无法读取分辨率，则使用默认值。

    Args:
        cap: OpenCV 视频捕获对象
        meta: 会话元数据字典（可能包含 width/height 字段）

    Returns:
        (宽度, 高度) 像素元组，最小为 320x240
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(meta.get("width") or MAX_INIT_WIN_W)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(meta.get("height") or MAX_INIT_WIN_H)
    if width <= 0 or height <= 0:
        return MAX_INIT_WIN_W, MAX_INIT_WIN_H

    # 按最大限制等比缩放，不放大
    scale = min(MAX_INIT_WIN_W / width, MAX_INIT_WIN_H / height, 1.0)
    return max(320, int(width * scale)), max(240, int(height * scale))


def review(
    video_path: Path,
    label_path: Path,
    start_window: int | None = None,
    *,
    feature_path: Path | None = None,
    hand_side: str | None = "Right",
    draw_skeleton: bool = True,
    model_cache_dir: str = ".models",
) -> None:
    """主回放循环：加载视频和标注，进入 OpenCV 交互式播放窗口。

    播放控制逻辑：
    - 采用时钟锁定机制（clock_anchor_frame + clock_anchor_time）：记录某个时刻的帧号和系统时间，
      播放时根据 elapsed * fps 计算目标帧，跟踪视频解码位置，保证帧率稳定。
    - 暂停时停止帧推进，恢复时重置时钟锚点。
    - 循环窗口模式：当前动作窗播放完毕后自动跳回到窗口起始帧。

    Args:
        video_path: MP4 视频文件路径
        label_path: JSON 标注文件路径
        start_window: 从第几个动作窗开始（1-indexed），None 则从头开始
        feature_path: 指定特征 NPZ 路径，None 则自动推断
        hand_side: 骨架检测手性（"Right"/"Left"/None=第一只手）
        draw_skeleton: 是否绘制实时骨架叠加层
        model_cache_dir: MediaPipe 模型缓存目录
    """
    meta = _load_json(label_path)
    annotations: list[Annotation] = meta["annotations"]
    features = _load_feature_overlay(feature_path if feature_path is not None else _features_path_for(label_path))
    lmkr = None
    if draw_skeleton:
        # 加载 MediaPipe 手部检测模型
        model_path = resolve_model(model_cache_dir)
        lmkr = make_landmarker(model_path)
        if lmkr is None:
            print("[review] warning: MediaPipe is unavailable; skeleton overlay disabled")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(meta.get("source_frames") or 0)
    fps = float(meta.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0)
    font_path = _resolve_font()

    _print_summary(video_path, label_path, meta, total_frames, features)

    # 当前动作窗索引
    window_idx = 0
    if start_window is not None and annotations:
        window_idx = min(max(0, start_window - 1), len(annotations) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, annotations[window_idx].start_frame)
    # 暂停状态
    paused = False
    # 窗口循环模式：指定 start_window 时默认开启
    loop_window = start_window is not None
    # 时钟锚点：记录当前帧号和对应的系统单调时间
    clock_anchor_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    clock_anchor_time = time.monotonic()

    def reset_playback_clock(frame: int | None = None) -> None:
        # 重置时钟锚点：将当前帧号和系统时间重新绑定
        # 用于暂停恢复、跳帧、切换窗口等场景，避免帧号跳变导致播放速度异常
        nonlocal clock_anchor_frame, clock_anchor_time
        clock_anchor_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) if frame is None else frame)
        clock_anchor_time = time.monotonic()

    init_w, init_h = _initial_window_size(cap, meta)
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, init_w, init_h)

    # ---- 主播放循环 ----
    while True:
        # ---- 时钟锁定帧推进 ----
        # 播放状态下，根据经过的时间计算目标帧号，并将视频解码位置跳转到目标帧
        # 这样即使解码/绘制有延迟，也能保持与实际时间同步
        if not paused and fps > 0:
            elapsed = time.monotonic() - clock_anchor_time
            target_frame = clock_anchor_frame + int(elapsed * fps)
            pos_now = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if target_frame > pos_now:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(target_frame, max(0, total_frames - 1)))

        # 读取当前位置的帧
        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ok, frame = cap.read()
        if not ok:
            # 视频结束，跳回开头继续循环
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            reset_playback_clock(0)
            continue

        # cap.read() 会将位置推进一帧，所以当前帧号 = pos - 1
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        active_idx = _active_annotation(annotations, frame_idx)
        if active_idx is not None:
            # 更新当前动作窗索引
            window_idx = active_idx

        # ---- 实时骨架检测和绘制 ----
        skeleton_ok: bool | None = None
        if lmkr is not None:
            skeleton_ok = _draw_skeleton_overlay(frame, detect(frame, lmkr, hand_side=hand_side))

        # ---- 绘制 HUD 和时间轴 ----
        _draw_hud(frame, meta, annotations, frame_idx, fps, active_idx, features, skeleton_ok, paused, loop_window, font_path)
        _draw_timeline(frame, annotations, frame_idx, total_frames, active_idx, features, font_path)
        cv2.imshow(WIN_NAME, frame)

        # ---- 窗口循环检查 ----
        # 若开启循环且当前帧已超过动作窗结束帧，则准备跳回到窗口起始帧
        loop_restart_frame: int | None = None
        if loop_window and annotations:
            ann = annotations[window_idx]
            if frame_idx >= ann.end_frame:
                loop_restart_frame = ann.start_frame

        # ---- 计算等待时间 ----
        # 暂停时无限等待（wait_ms=0）；播放时根据下一帧的应到时间计算等待毫秒数
        if paused or fps <= 0:
            wait_ms = 0
        else:
            next_due = clock_anchor_time + ((frame_idx + 1 - clock_anchor_frame) / fps)
            wait_ms = max(1, int((next_due - time.monotonic()) * 1000))
        key = cv2.waitKeyEx(wait_ms)

        # ---- 窗口循环自动跳转 ----
        # 若无按键且需要循环跳转，则跳转到窗口起始帧并重置时钟
        if loop_restart_frame is not None and key == -1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, loop_restart_frame)
            reset_playback_clock(loop_restart_frame)
            continue

        # ---- 键盘事件处理 ----
        # Q / Esc: 退出播放
        if key in (ord("q"), ord("Q"), 27):
            break
        # 空格: 切换暂停/播放
        if key == ord(" "):
            if paused:
                # 恢复播放：以当前视频位置为锚点重置时钟
                paused = False
                reset_playback_clock()
            else:
                # 暂停：将视频位置回退到刚才读取的帧，避免暂停后丢帧
                paused = True
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        # L: 切换窗口循环模式
        elif key in (ord("l"), ord("L")):
            loop_window = not loop_window
            if loop_window and annotations:
                # 开启循环时跳到当前活跃窗口的起始帧
                target = active_idx if active_idx is not None else window_idx
                window_idx = target
                cap.set(cv2.CAP_PROP_POS_FRAMES, annotations[window_idx].start_frame)
                reset_playback_clock(annotations[window_idx].start_frame)
        # A: 切换到上一个动作窗，并暂停
        elif key in (ord("a"), ord("A")) and annotations:
            window_idx = max(0, window_idx - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, annotations[window_idx].start_frame)
            reset_playback_clock(annotations[window_idx].start_frame)
            paused = True
        # D: 切换到下一个动作窗，并暂停
        elif key in (ord("d"), ord("D")) and annotations:
            window_idx = min(len(annotations) - 1, window_idx + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, annotations[window_idx].start_frame)
            reset_playback_clock(annotations[window_idx].start_frame)
            paused = True
        # 左箭头 (keycode 81/2424832): 后退一帧，并暂停
        elif key in (81, 2424832):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 2))
            reset_playback_clock(max(0, pos - 2))
            paused = True
        # 右箭头 (keycode 83/2555904): 前进约 1 秒的帧数，并暂停
        elif key in (83, 2555904):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, pos + int(max(1, fps))))
            reset_playback_clock(min(total_frames - 1, pos + int(max(1, fps))))
            paused = True

    cap.release()
    cv2.destroyWindow(WIN_NAME)


def main() -> None:
    """命令行入口：解析参数并启动回放。"""
    ap = argparse.ArgumentParser(description="回放 finger-collect 采集数据，并展示每个手势动作窗")
    ap.add_argument("--data-dir", default="data", help="数据根目录")
    ap.add_argument("--session", default=None, help="session stem，例如 20260425_153000_S01；默认打开最新 label")
    ap.add_argument("--video", default=None, help="直接指定 MP4 路径")
    ap.add_argument("--label", default=None, help="直接指定 JSON 标注路径")
    ap.add_argument("--window", type=int, default=None, help="从第 N 个动作窗开始，并默认循环该窗口")
    ap.add_argument("--features", default=None, help="指定同名 NPZ 特征路径；默认读取 data/features/<session>.npz")
    ap.add_argument("--hand-side", default="Right", choices=("Left", "Right", "Any"), help="骨架检测手性，Any 表示第一只手")
    ap.add_argument("--no-skeleton", action="store_true", help="关闭实时骨架叠加")
    ap.add_argument("--model-cache-dir", default=".models", help="MediaPipe 模型缓存目录")
    args = ap.parse_args()

    video_path, label_path = _resolve_pair(Path(args.data_dir), args.session, args.video, args.label)
    review(
        video_path,
        label_path,
        args.window,
        feature_path=Path(args.features) if args.features else None,
        hand_side=None if args.hand_side == "Any" else args.hand_side,
        draw_skeleton=not args.no_skeleton,
        model_cache_dir=args.model_cache_dir,
    )


if __name__ == "__main__":
    main()
