"""
finger_ml.capture — 手势视频采集器

连续录制单条视频，用户按 SPACE 手动标注每段手势的起止时间。
静息状态自然保留在视频中，供模型学习背景/静息类别。

用法：
    uv run finger-collect --subject S01 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from finger_ml.hand_tracking import (
    HAND_CONNECTIONS,
    HIGHLIGHT_COLORS,
    MEDIAPIPE_AVAILABLE,
    detect,
    make_landmarker,
    resolve_model,
)
from finger_ml.labels import (
    GESTURE_EN,
    GESTURE_LABEL,
    GESTURE_ORDER,
    GESTURE_ZH,
)
from finger_ml.video_io import make_writer

try:
    from PIL import Image as PILImage, ImageDraw, ImageFont

    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# 窗口布局常量
WIN_W = 1280  # 窗口总宽度
WIN_H = 720   # 窗口总高度
PANEL_SPLIT = int(WIN_W * 0.65)  # 相机面板宽度 = 832（左侧 65%）
PANEL_R_W = WIN_W - PANEL_SPLIT  # 右侧信息面板宽度 = 448（右侧 35%）
WIN_NAME = "Gesture Collector"  # OpenCV 窗口名称

# 中文字体候选路径列表（覆盖 macOS / Windows 常见中文字体）
FONT_CANDIDATES = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/msyh.ttc",     # 微软雅黑 常规
    "C:/Windows/Fonts/msyhbd.ttc",   # 微软雅黑 粗体
    "C:/Windows/Fonts/simhei.ttf",   # 黑体
    "C:/Windows/Fonts/simsun.ttc",   # 宋体
]

# Data models — 数据模型


@dataclass
class AnnotationEntry:
    """单条手势标注记录。

    记录一次手势动作在视频中的起止位置，包含帧号和毫秒时间戳。
    所有帧号均为 1-indexed（与 JSON 标注文件一致），preprocess 阶段再转为 0-indexed。

    Attributes:
        gesture:     手势名称（如 "pinch_index"）
        label:       手势对应的整数标签（来自 GESTURE_LABEL 映射）
        rep:         当前手势的第几次重复（1-indexed）
        start_frame: 手势起始帧号（1-indexed）
        end_frame:   手势结束帧号（1-indexed）
        start_ms:    手势起始毫秒时间戳
        end_ms:      手势结束毫秒时间戳
    """
    gesture: str
    label: int
    rep: int
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int


@dataclass
class SessionMeta:
    """采集会话的元数据。

    记录本次采集的受试者信息、视频参数、手势顺序等，与标注数据一起保存为 JSON。
    source_fps / source_frames / duplicated_frames 在会话结束后由主循环填充，
    用于区分输出视频的恒定帧率与摄像头实际供帧能力。

    Attributes:
        subject_id:        受试者 ID（如 "S01"）
        session_id:        会话 ID（时间戳格式，如 "20260714_153000"）
        video_file:        输出视频文件路径
        fps:               输出视频帧率（CFR 恒定帧率）
        width:             视频宽度（像素）
        height:            视频高度（像素）
        gestures_order:    手势顺序列表（与 GESTURE_ORDER 一致）
        repeats:           每种手势重复次数
        created_at:        创建时间字符串
        source_fps:        摄像头实际供帧帧率（会话结束后计算，可选）
        source_frames:     摄像头实际产生的帧数（不含补帧）
        duplicated_frames: 为维持 CFR 时间轴而重复补帧的帧数
    """
    subject_id: str
    session_id: str
    video_file: str
    fps: float
    width: int
    height: int
    gestures_order: List[str]
    repeats: int
    created_at: str
    source_fps: Optional[float] = None
    source_frames: int = 0
    duplicated_frames: int = 0


# Utilities — 工具函数


def resolve_font() -> Optional[str]:
    """在系统中查找可用的中文字体文件路径。

    按优先级遍历 FONT_CANDIDATES 列表，返回第一个存在的字体路径。
    若系统无任何候选字体，返回 None（后续绘制回退到 OpenCV 默认字体）。

    Returns:
        可用字体文件路径，或 None
    """
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def save_session(
    label_path: Path,
    controller: "SessionController",
    meta: SessionMeta,
) -> None:
    """将当前会话的标注数据与元数据保存为 JSON 文件。

    每次用户按 SPACE 标注或按 Q 退出时调用，确保标注数据实时持久化，
    即使程序中途崩溃也能保留已标注的数据。

    Args:
        label_path:  标注 JSON 文件的输出路径
        controller:  当前会话控制器（包含标注列表和中止标志）
        meta:        会话元数据
    """
    payload = {
        "subject_id": meta.subject_id,
        "session_id": meta.session_id,
        "video_file": meta.video_file,
        "fps": meta.fps,
        "width": meta.width,
        "height": meta.height,
        "gestures_order": meta.gestures_order,
        "repeats": meta.repeats,
        "created_at": meta.created_at,
        "source_fps": meta.source_fps,
        "source_frames": meta.source_frames,
        "duplicated_frames": meta.duplicated_frames,
        "aborted": controller.aborted,
        "annotations": [asdict(a) for a in controller.annotations],
    }
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def draw_landmarks(
    panel: np.ndarray,
    lms,
    draw_w: int,
    draw_h: int,
    off_x: int = 0,
    off_y: int = 0,
) -> None:
    """Overlay hand skeleton on *panel* in-place.

    draw_w / draw_h are the pixel dimensions of the content area (after letterboxing).
    off_x / off_y are the top-left offsets of that content area within *panel*.

    在相机面板上绘制手部关键点骨架。只绘制模型使用的 13 个节点
    （腕部 + 拇指 + 食指 + 中指，索引 0-12），高亮节点用彩色大圆标注，
    其余节点用灰色小圆表示。

    Args:
        panel:   目标 BGR 图像数组（会被原地修改）
        lms:     MediaPipe 检测到的手部关键点列表（归一化坐标 0~1）
        draw_w:  内容区域的像素宽度（letterbox 后的缩放宽度）
        draw_h:  内容区域的像素高度
        off_x:   内容区域在 panel 中的 x 偏移
        off_y:   内容区域在 panel 中的 y 偏移
    """
    if lms is None:
        return
    pts = [(int(lm.x * draw_w) + off_x, int(lm.y * draw_h) + off_y) for lm in lms]
    # 只绘制模型使用的 13 个节点（腕部 + 拇指 + 食指 + 中指，索引 0-12）
    for a, b in HAND_CONNECTIONS:
        if a < 13 and b < 13:
            cv2.line(panel, pts[a], pts[b], (200, 200, 200), 1, cv2.LINE_AA)
    for i in range(13):
        pt = pts[i]
        color = HIGHLIGHT_COLORS.get(i)
        if color is not None:
            # 高亮关键点：彩色实心圆 + 白色描边
            cv2.circle(panel, pt, 7, color, -1, cv2.LINE_AA)
            cv2.circle(panel, pt, 9, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # 非高亮点：灰色小圆
            cv2.circle(panel, pt, 3, (180, 180, 180), -1, cv2.LINE_AA)


# UI rendering — 界面渲染


def letterbox(
    frame: np.ndarray,
    target_w: int,
    target_h: int,
) -> tuple[np.ndarray, int, int, int, int]:
    """Fit *frame* into target_w×target_h with black bars, preserving aspect ratio.

    将输入帧等比缩放到目标尺寸内，不足部分用黑边填充（letterbox）。
    用于在固定尺寸的面板中显示任意宽高比的摄像头画面。

    Returns:
        panel   — target_w×target_h BGR array
        draw_w  — pixel width of the scaled content
        draw_h  — pixel height of the scaled content
        off_x   — x offset of content within panel
        off_y   — y offset of content within panel
    """
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    draw_w = int(w * scale)
    draw_h = int(h * scale)
    off_x = (target_w - draw_w) // 2
    off_y = (target_h - draw_h) // 2
    panel = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    panel[off_y : off_y + draw_h, off_x : off_x + draw_w] = cv2.resize(
        frame, (draw_w, draw_h)
    )
    return panel, draw_w, draw_h, off_x, off_y


def draw_gesture_hint(
    panel: np.ndarray,
    gesture: str,
    x0: int,
    y0: int,
    w: int,
    h: int,
) -> None:
    """Draw a minimal schematic of the gesture using OpenCV primitives.

    在右侧信息面板中绘制当前手势的简易示意图，帮助用户理解需要做的动作。
    使用 OpenCV 基本图形（圆、线、箭头）绘制，无需中文字体支持。

    Args:
        panel:  目标 BGR 图像数组（会被原地修改）
        gesture: 手势名称（如 "pinch_index"、"thumb_slide_up" 等）
        x0:     绘制区域左上角 x 坐标
        y0:     绘制区域左上角 y 坐标
        w:      绘制区域宽度
        h:      绘制区域高度
    """
    cx = x0 + w // 2
    cy = y0 + h // 2

    if gesture == "pinch_index":
        # 食指捏合：两个圆（拇指+食指）+ 虚线表示靠近
        cv2.circle(panel, (cx - 32, cy - 8), 15, (255, 210, 50), 2)
        cv2.circle(panel, (cx + 32, cy - 8), 15, (50, 230, 80), 2)
        for dx in range(-22, 23, 8):
            cv2.circle(panel, (cx + dx, cy - 8), 2, (160, 160, 160), -1)
        cv2.putText(
            panel,
            "PINCH",
            (cx - 24, cy + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 160),
            1,
        )

    elif gesture == "pinch_middle":
        # 中指捏合：两个圆（拇指+中指）+ 虚线表示靠近
        cv2.circle(panel, (cx - 32, cy - 8), 15, (255, 210, 50), 2)
        cv2.circle(panel, (cx + 32, cy - 8), 15, (30, 140, 255), 2)
        for dx in range(-22, 23, 8):
            cv2.circle(panel, (cx + dx, cy - 8), 2, (160, 160, 160), -1)
        cv2.putText(
            panel,
            "PINCH",
            (cx - 24, cy + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 160),
            1,
        )

    elif gesture == "thumb_slide_up":
        # 拇指上滑：竖线 + 向上箭头
        cv2.line(panel, (cx, cy + 42), (cx, cy - 20), (200, 200, 200), 3)
        cv2.arrowedLine(
            panel,
            (cx + 28, cy + 22),
            (cx + 28, cy - 32),
            (255, 210, 50),
            2,
            tipLength=0.28,
        )
        cv2.putText(
            panel,
            "UP",
            (cx + 18, cy + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 160),
            1,
        )

    elif gesture == "thumb_slide_down":
        # 拇指下滑：竖线 + 向下箭头
        cv2.line(panel, (cx, cy - 42), (cx, cy + 20), (200, 200, 200), 3)
        cv2.arrowedLine(
            panel,
            (cx + 28, cy - 22),
            (cx + 28, cy + 32),
            (255, 210, 50),
            2,
            tipLength=0.28,
        )
        cv2.putText(
            panel,
            "DOWN",
            (cx + 10, cy + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 160),
            1,
        )

    elif gesture == "thumb_slide_left":
        # 拇指左滑：竖线 + 向左箭头
        cv2.line(panel, (cx - 8, cy - 42), (cx - 8, cy + 20), (200, 200, 200), 3)
        cv2.arrowedLine(
            panel, (cx + 32, cy), (cx - 32, cy), (255, 210, 50), 2, tipLength=0.28
        )
        cv2.putText(
            panel,
            "LEFT",
            (cx - 18, cy + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 160),
            1,
        )

    elif gesture == "thumb_slide_right":
        # 拇指右滑：竖线 + 向右箭头
        cv2.line(panel, (cx - 8, cy - 42), (cx - 8, cy + 20), (200, 200, 200), 3)
        cv2.arrowedLine(
            panel, (cx - 32, cy), (cx + 32, cy), (255, 210, 50), 2, tipLength=0.28
        )
        cv2.putText(
            panel,
            "RIGHT",
            (cx - 20, cy + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 160, 160),
            1,
        )


def _put_zh(
    canvas_or_draw: np.ndarray | ImageDraw.ImageDraw,
    text: str,
    xy: tuple,
    font_path: Optional[str],
    size: int,
    color_rgb: tuple,
) -> None:
    """Render text onto a PIL Draw object or a numpy BGR image.

    统一的中文文字渲染函数。优先使用 PIL + TrueType 字体渲染
    （支持中文和任意字号），PIL 不可用时回退到 OpenCV 的 putText
    （仅支持 ASCII，中文会显示为方框）。

    当 canvas_or_draw 为 ndarray 时，会临时转换为 PIL 图像进行渲染，
    渲染完成后再转回 BGR 格式写回原数组（原地修改）。

    Args:
        canvas_or_draw: PIL ImageDraw 对象 或 numpy BGR 数组
        text:           要绘制的文本内容
        xy:             文本左上角坐标 (x, y)
        font_path:      TrueType 字体文件路径，None 则使用默认字体
        size:           字号（像素）
        color_rgb:      文本颜色，RGB 格式元组
    """
    try:
        font = (
            ImageFont.truetype(font_path, size)
            if font_path
            else ImageFont.load_default()
        )
    except Exception:
        font = ImageFont.load_default()

    if isinstance(canvas_or_draw, ImageDraw.ImageDraw):
        canvas_or_draw.text(xy, text, font=font, fill=color_rgb)
    else:
        # 兼容模式：为 ndarray 创建临时的 PIL 环境
        if _PIL_OK:
            pil_img = PILImage.fromarray(cv2.cvtColor(canvas_or_draw, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            draw.text(xy, text, font=font, fill=color_rgb)
            canvas_or_draw[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            # PIL 不可用的降级方案：OpenCV putText（不支持中文）
            cv2.putText(
                canvas_or_draw,
                text,
                (int(xy[0]), int(xy[1]) + size),
                cv2.FONT_HERSHEY_SIMPLEX,
                size / 40.0,
                color_rgb[::-1],  # BGR
                1,
                cv2.LINE_AA,
            )


def build_canvas(
    raw_frame: np.ndarray,
    lms,
    controller: "SessionController",
    font_path: Optional[str],
    rec_fps: float = 0.0,
    det_fps: float = 0.0,
) -> np.ndarray:
    """Compose the 1280×720 display canvas from camera + right info panel.

    组合完整的显示画面：左侧为 letterbox 后的摄像头画面（带手部骨架叠加），
    右侧为信息面板（当前手势名称、示意图、状态提示、重复计数、FPS 等）。

    优化点：所有中文文字绘制合并到一次 PIL 转换中，避免反复 BGR<->RGB 交换的开销。

    Args:
        raw_frame:   摄像头原始 BGR 帧
        lms:         MediaPipe 检测到的手部关键点（可为 None）
        controller:  会话状态控制器
        font_path:   中文字体路径（可为 None）
        rec_fps:     当前录制帧率（用于状态栏显示）
        det_fps:     当前检测帧率（用于状态栏显示）

    Returns:
        1280x720 的 BGR 图像数组
    """
    # ── Camera panel (left): letterboxed to preserve aspect ratio ────────────
    cam_panel, draw_w, draw_h, off_x, off_y = letterbox(raw_frame, PANEL_SPLIT, WIN_H)
    draw_landmarks(cam_panel, lms, draw_w, draw_h, off_x, off_y)
    if controller.state == AppState.COUNTDOWN:
        # 大数字倒计时叠加在相机画面中央
        rem = controller.countdown_remaining
        num_str = str(math.ceil(rem)) if rem > 0 else "GO!"
        scale, thick = 7.0, 10
        (tw, th), _ = cv2.getTextSize(num_str, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        tx = (PANEL_SPLIT - tw) // 2
        ty = (WIN_H + th) // 2
        # 先画阴影（偏移 + 更粗），再画前景，形成描边效果
        cv2.putText(
            cam_panel,
            num_str,
            (tx + 5, ty + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thick + 6,
            cv2.LINE_AA,
        )
        cv2.putText(
            cam_panel,
            num_str,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (30, 190, 255),
            thick,
            cv2.LINE_AA,
        )
    elif controller.state == AppState.ANNOTATING:
        # 录制中：在相机画面边框绘制红色边框，提示正在录制
        cv2.rectangle(cam_panel, (0, 0), (PANEL_SPLIT - 1, WIN_H - 1), (0, 0, 220), 5)

    # ── Right info panel ─────────────────────────────────────────────────────
    rp = np.full((WIN_H, PANEL_R_W, 3), 28, dtype=np.uint8)

    # 优化点：合并所有文字绘制到一次 PIL 转换中
    if _PIL_OK and font_path:
        pil_rp = PILImage.fromarray(cv2.cvtColor(rp, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_rp)

        rx = 18
        gesture = controller.current_gesture
        zh_name = GESTURE_ZH.get(gesture, gesture)
        en_name = GESTURE_EN.get(gesture, gesture)

        # 手势名称（中文大字 + 英文小字）
        _put_zh(draw, zh_name, (rx, 22), font_path, 40, (255, 220, 50))
        _put_zh(draw, en_name, (rx, 76), font_path, 22, (150, 150, 150))

        # 状态提示文字 — 根据当前状态显示不同的提示和颜色
        if controller.state == AppState.WAIT:
            status_txt, status_color = "准备好后按 SPACE", (80, 220, 80)
        elif controller.state == AppState.COUNTDOWN:
            status_txt, status_color = (
                f"准备...  {controller.countdown_remaining:.1f}  秒",
                (230, 180, 40),
            )
        elif controller.state == AppState.ANNOTATING:
            status_txt, status_color = "录制中... 完成后按 SPACE", (230, 60, 60)
        elif controller.state == AppState.REST:
            status_txt, status_color = f"休息  {controller.rest_remaining:.0f}  秒", (
                230,
                180,
                40,
            )
        else:
            status_txt, status_color = "全部完成！", (255, 220, 50)

        # 重复次数 / 手势序号
        _put_zh(
            draw,
            f"{controller.rep} / {controller.repeats}  次",
            (rx, 298),
            font_path,
            20,
            (140, 140, 140),
        )
        _put_zh(
            draw,
            f"手势  {controller.g_idx + 1} / {len(controller.gestures)}",
            (rx, 326),
            font_path,
            20,
            (120, 120, 120),
        )
        _put_zh(draw, status_txt, (rx, 364), font_path, 26, status_color)

        # 操作提示说明文字
        hints = [
            ("[SPACE]", "标记开始 / 结束"),
            ("[R]", "撤销上一个标注"),
            ("[Q]", "退出保存"),
        ]
        hy = 424
        for _, desc in hints:
            _put_zh(draw, desc, (rx + 72, hy - 14), font_path, 18, (150, 150, 150))
            hy += 30

        # 已标注段数统计
        _put_zh(
            draw,
            f"已标注  {len(controller.annotations)}  段",
            (rx, WIN_H - 52),
            font_path,
            18,
            (90, 90, 90),
        )

        # PIL 绘制完毕，转回 BGR
        rp = cv2.cvtColor(np.array(pil_rp), cv2.COLOR_RGB2BGR)

    # 绘制非文字部分（手势示意图、圆点进度、快捷键标签、FPS 等）
    rx = 18
    # 手势简易示意图
    draw_gesture_hint(
        rp, controller.current_gesture, rx + 20, 112, PANEL_R_W - rx * 2 - 20, 130
    )
    # 重复次数圆点指示器：已完成=绿色实心，当前=白色实心，未完成=灰色空心
    dot_y, x_start = 270, rx + 8
    for i in range(controller.repeats):
        cx_d = x_start + i * 26
        if i + 1 < controller.rep:
            cv2.circle(rp, (cx_d, dot_y), 9, (50, 200, 80), -1)
        elif i + 1 == controller.rep:
            cv2.circle(rp, (cx_d, dot_y), 9, (240, 240, 240), -1)
        else:
            cv2.circle(rp, (cx_d, dot_y), 9, (90, 90, 90), 1)

    # 分隔线
    cv2.line(rp, (rx, 408), (PANEL_R_W - rx, 408), (65, 65, 65), 1)
    # 快捷键标签（英文部分用 OpenCV 绘制，与中文说明文字配对显示）
    hy = 424
    for key_str, _ in [("[SPACE]", ""), ("[R]", ""), ("[Q]", "")]:
        cv2.putText(
            rp,
            key_str,
            (rx, hy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (190, 190, 80),
            1,
            cv2.LINE_AA,
        )
        hy += 30

    # FPS 显示：录制帧率颜色根据性能变化（绿>50fps, 蓝>25fps, 红<25fps）
    rec_color = (
        (60, 220, 60)
        if rec_fps >= 50
        else (60, 160, 255) if rec_fps >= 25 else (60, 60, 220)
    )
    cv2.putText(
        rp,
        f"REC {rec_fps:4.1f} fps",
        (rx, WIN_H - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        rec_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        rp,
        f"DET {det_fps:4.1f} fps",
        (rx + 110, WIN_H - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    # 水平拼接：左侧相机面板 + 右侧信息面板
    return np.hstack([cam_panel, rp])



def draw_done_screen(
    controller: "SessionController",
    video_path: Path,
    label_path: Path,
    font_path: Optional[str],
) -> np.ndarray:
    """绘制采集结束后的完成/中止画面。

    显示采集结果摘要（标注段数、文件路径）和退出提示。

    Args:
        controller:  会话状态控制器（用于判断是否中止及获取标注数）
        video_path:  输出视频文件路径
        label_path:  输出标注文件路径
        font_path:   中文字体路径（可为 None）

    Returns:
        1280x720 的 BGR 图像数组
    """
    canvas = np.full((WIN_H, WIN_W, 3), 20, dtype=np.uint8)
    center_x = WIN_W // 2

    if controller.aborted:
        title, title_color = "已中止", (60, 100, 230)
    else:
        title, title_color = "采集完成！", (50, 220, 255)

    _put_zh(canvas, title, (center_x - 90, 160), font_path, 56, title_color)
    _put_zh(
        canvas,
        f"共标注  {len(controller.annotations)}  段",
        (center_x - 80, 268),
        font_path,
        30,
        (200, 200, 200),
    )
    _put_zh(
        canvas,
        f"视频：{video_path.name}",
        (center_x - 180, 336),
        font_path,
        22,
        (140, 140, 140),
    )
    _put_zh(
        canvas,
        f"标注：{label_path.name}",
        (center_x - 180, 372),
        font_path,
        22,
        (140, 140, 140),
    )
    _put_zh(
        canvas,
        "按 Q 或 SPACE 退出",
        (center_x - 110, 460),
        font_path,
        26,
        (170, 170, 170),
    )
    return canvas


# Threaded I/O — 线程化 I/O 架构
#
# 整体架构采用三个独立线程实现流水线并行：
#   1. CameraStream  — 摄像头采集线程：持续从摄像头读取新帧，主线程按需取用
#   2. AsyncVideoWriter — 视频写入线程：主线程将帧入队，写入线程异步编码写盘
#   3. _detect_worker — MediaPipe 检测线程：主线程提交帧，检测线程异步返回关键点
#
# 这样主循环只负责：取帧 -> 推送到写入/检测队列 -> 组合 UI -> 显示，
# 耗时的编码写盘和手部检测都在后台线程中并行完成，不阻塞帧率。


class CameraStream:
    """异步摄像头采集线程。

    在后台线程中持续从摄像头读取帧，主线程通过 read() 获取最新帧。
    使用线程锁保护帧数据的读写，使用 _seq 序号让主线程判断是否有新帧
    （主循环用它来区分"真正的摄像头新帧"和"重复读取同一帧"）。

    为什么需要这个类：OpenCV 的 VideoCapture.read() 是阻塞调用，
    如果在主循环中直接调用，等待摄像头 I/O 会拖慢整个帧循环。
    将采集放到独立线程后，主循环可以随时拿到最新帧，零等待。
    """
    def __init__(self, index: int, target_w: int, target_h: int, target_fps: int):
        """初始化摄像头采集器。

        尝试以指定参数打开摄像头，并读取第一帧作为初始值。

        Args:
            index:      摄像头索引（通常 0=内置，1=USB）
            target_w:   请求的采集宽度
            target_h:   请求的采集高度
            target_fps: 请求的采集帧率
        """
        self.cap = _open_camera(index)
        if self.cap is None:
            raise RuntimeError(f"Could not open camera {index}")

        # Try to set resolution/fps again just in case
        # 部分摄像头驱动在 _open_camera 中设置可能被覆盖，这里再设置一次
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)

        # 预读第一帧，确保 read() 不会返回 None
        self.ret, self.frame = self.cap.read()
        self.frame_ts = time.monotonic()
        self.stopped = False
        self.lock = threading.Lock()
        # 帧序号：每次摄像头产生新帧时自增，供主循环去重用
        self._seq: int = 0

    def start(self):
        """启动后台采集线程。返回 self 以支持链式调用 CameraStream(...).start()。"""
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
        """后台线程主循环：持续从摄像头读取帧并更新共享状态。

        读取成功时自增 _seq 序号，主循环通过比较序号判断是否为新帧。
        读取失败时设置 stopped 标志，通知主循环摄像头已断开。
        """
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame
                self.frame_ts = time.monotonic()
                self._seq += 1

    def read(self):
        """返回 (ret, frame_copy, seq, timestamp)。seq 每次摄像头产生新帧时自增。

        返回帧的副本而非引用，避免主线程处理帧时被后台线程覆盖。
        seq 用于主循环判断是否为新的摄像头帧（seq 变化 = 新帧到达）。
        """
        with self.lock:
            return (
                self.ret,
                self.frame.copy() if self.frame is not None else None,
                self._seq,
                self.frame_ts,
            )

    def stop(self):
        """停止采集线程并释放摄像头资源。"""
        self.stopped = True
        if self.cap:
            self.cap.release()

    def get(self, prop):
        """获取摄像头属性值（透传到 VideoCapture.get）。

        Args:
            prop: OpenCV 摄像头属性常量（如 cv2.CAP_PROP_FPS）

        Returns:
            属性值
        """
        return self.cap.get(prop)


class AsyncVideoWriter:
    """异步视频写入线程。

    主循环将帧放入有界队列，后台线程从队列取出帧并编码写入视频文件。
    队列容量为 512 帧（约 8.5 秒 @60fps），足够缓冲编码延迟。
    队列满时丢弃新帧并打印警告，避免内存溢出。

    为什么需要这个类：视频编码（尤其 H.264）的 write() 调用耗时不确定，
    可能在关键帧处突然变慢。如果主循环同步调用 writer.write()，
    编码延迟会直接导致帧率抖动。异步写入将编码耗时与帧循环解耦。
    """
    def __init__(self, path: Path, fps: float, w: int, h: int):
        """初始化异步视频写入器并立即启动后台线程。

        Args:
            path: 输出视频文件路径
            fps:  输出视频帧率
            w:    视频宽度
            h:    视频高度
        """
        self.writer = make_writer(path, fps, w, h)
        self.queue: queue.Queue = queue.Queue(maxsize=512)
        self.stopped = False
        self.frames_written = 0
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        """后台线程主循环：从队列取帧并写入视频文件。

        当 stopped=True 且队列清空后退出。使用 timeout 避免永久阻塞，
        允许定期检查 stopped 标志。
        """
        while not self.stopped or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                self.writer.write(frame)
                self.frames_written += 1
            except queue.Empty:
                continue

    def write(self, frame, *, block: bool = True):
        """将一帧推入写入队列。

        Args:
            frame: BGR 帧数组
            block: 是否阻塞等待队列有空位。默认 True（阻塞）；
                   False 时队列满则立即丢弃并打印警告。
        """
        try:
            if block:
                self.queue.put(frame)
            else:
                self.queue.put_nowait(frame)
        except queue.Full:
            print("[warn] Video writer queue full, dropping frame")

    def release(self):
        """停止写入线程并释放视频写入器资源。

        设置停止标志后等待线程处理完队列中的剩余帧，然后释放编码器。
        """
        self.stopped = True
        self.thread.join()
        self.writer.release()


# Session state machine — 会话状态机
#
# AppState 定义了采集会话的 6 个状态，状态转换流程如下：
#   WAIT -> COUNTDOWN -> ANNOTATING -> REST -> WAIT (下一手势) -> ... -> COMPLETE -> DONE
#
# 状态转换规则：
#   WAIT        — 等待用户按 SPACE 开始
#       SPACE -> COUNTDOWN（启动倒计时）
#       R     -> 撤销上一个标注（如果有），回退 g_idx 和 rep
#
#   COUNTDOWN   — 3 秒倒计时（给用户准备时间）
#       倒计时结束 -> ANNOTATING（自动，由 tick() 驱动）
#       R         -> 取消倒计时，回到 WAIT
#
#   ANNOTATING  — 正在录制手势动作
#       SPACE -> 提交标注 -> 自动推进（下一 rep 或下一手势）
#              -> 若还有 rep：COUNTDOWN（新倒计时）
#              -> 若手势做完但有下一个手势：REST（休息间隔）
#              -> 若全部手势完成：COMPLETE
#       R     -> 取消本次标注，回到 WAIT
#
#   REST        — 手势组间休息
#       休息结束 -> WAIT（自动，由 tick() 驱动，进入下一个手势）
#       SPACE  -> 提前结束休息，进入下一个手势
#
#   COMPLETE    — 所有手势采集完成
#       SPACE -> DONE
#
#   DONE        — 终态，主循环退出


class AppState(Enum):
    """采集会话状态枚举。

    定义状态机的所有合法状态，每个状态对应不同的 UI 显示和用户交互行为。
    """
    WAIT = "wait"                    # 等待用户按 SPACE 开始
    COUNTDOWN = "countdown"          # 按下 SPACE 后 3 秒倒计时，结束自动标记开始
    ANNOTATING = "annotating"        # 正在录制手势动作，按 SPACE 标记结束
    REST = "rest"                    # 手势组间休息，倒计时结束或按 SPACE 跳过
    COMPLETE = "complete"            # 所有手势采集完成，按 SPACE 确认
    DONE = "done"                    # 终态，主循环退出


class SessionController:
    """采集会话的状态机控制器。

    管理手势采集的完整生命周期：按顺序遍历每种手势的每次重复，
    维护当前状态、帧号、标注列表等。主循环通过 on_space() / on_redo() /
    on_quit() 响应用户输入，通过 tick() 驱动时间驱动的状态转换（倒计时结束、
    休息结束）。
    """
    def __init__(
        self,
        gestures: List[str],
        repeats: int,
        rest_sec: float,
        countdown_sec: float,
        fps: float,
    ) -> None:
        """初始化会话控制器。

        Args:
            gestures:      手势名称列表（按 GESTURE_ORDER 顺序）
            repeats:       每种手势重复次数
            rest_sec:      手势组间休息秒数
            countdown_sec: 录制前倒计时秒数
            fps:           输出视频帧率（用于帧号到毫秒转换）
        """
        self.gestures = gestures       # 手势名称列表
        self.repeats = repeats         # 每种手势重复次数
        self.rest_sec = rest_sec       # 手势组间休息秒数
        self.countdown_sec = countdown_sec  # 录制前倒计时秒数
        self.fps = fps                 # 输出视频帧率（帧号到毫秒转换用）

        self.state = AppState.WAIT            # 当前状态
        self.g_idx = 0                        # 当前手势索引（0-indexed）
        self.rep = 1                           # 当前重复次数（1-indexed）
        self.frame_num = 0                     # 当前输出帧号（由主循环自增）

        self.annot_start_frame: Optional[int] = None   # 当前标注起始帧号
        self.annot_start_ms: Optional[int] = None      # 当前标注起始毫秒
        self.annotations: List[AnnotationEntry] = []   # 已提交的标注列表

        self.countdown_deadline = 0.0   # 倒计时截止时间（绝对时间）
        self.rest_deadline = 0.0        # 休息截止时间（绝对时间）
        self.aborted = False            # 是否用户中止
        self.completed = False          # 是否所有手势已完成

    @property
    def current_gesture(self) -> str:
        """当前需要录制的手势名称。越界时返回列表最后一项。"""
        return self.gestures[min(self.g_idx, len(self.gestures) - 1)]

    @property
    def countdown_remaining(self) -> float:
        """倒计时剩余秒数。非负，倒计时未激活时返回 0。"""
        return max(0.0, self.countdown_deadline - time.time())

    @property
    def rest_remaining(self) -> float:
        """休息剩余秒数。非负，休息未激活时返回 0。"""
        return max(0.0, self.rest_deadline - time.time())

    def on_space(self, frame_num: int, ms: int) -> None:
        """处理 SPACE 键按下事件。

        根据当前状态执行不同的状态转换：
        - WAIT: 启动倒计时 -> COUNTDOWN
        - ANNOTATING: 提交标注并推进 -> COUNTDOWN / REST / COMPLETE
        - REST: 提前结束休息 -> 进入下一手势
        - COMPLETE: 确认完成 -> DONE

        Args:
            frame_num: 当前帧号
            ms:        当前毫秒时间戳
        """
        if self.state == AppState.WAIT:
            # 开始 3 秒倒计时
            self.countdown_deadline = time.time() + self.countdown_sec
            self.state = AppState.COUNTDOWN
        elif self.state == AppState.ANNOTATING:
            # 标记结束，自动触发下一轮倒计时（或休息/完成）
            self._commit_annotation(frame_num, ms)
            self._advance()
        elif self.state == AppState.REST:
            self._enter_next_gesture()
        elif self.state == AppState.COMPLETE:
            self.state = AppState.DONE

    def on_redo(self) -> None:
        """处理 R 键按下事件（撤销/回退）。

        根据当前状态执行不同的回退操作：
        - COUNTDOWN: 取消倒计时，回到 WAIT
        - ANNOTATING: 取消本次标注，回到 WAIT
        - WAIT: 撤销上一个已提交的标注，回退 g_idx 和 rep
        """
        if self.state == AppState.COUNTDOWN:
            # 取消倒计时，回到等待
            self.state = AppState.WAIT
        elif self.state == AppState.ANNOTATING:
            self.annot_start_frame = None
            self.annot_start_ms = None
            self.state = AppState.WAIT
        elif self.state == AppState.WAIT and self.annotations:
            last = self.annotations.pop()
            self.g_idx = self.gestures.index(last.gesture)
            self.rep = last.rep
            self.state = AppState.WAIT

    def on_quit(self) -> None:
        """处理 Q 键按下事件。标记中止并进入 DONE 状态。"""
        if not self.completed:
            self.aborted = True
        self.state = AppState.DONE

    def tick(self) -> None:
        """时间驱动的状态转换（由主循环每帧调用）。

        检查倒计时和休息是否到期，自动推进状态：
        - COUNTDOWN 到期 -> ANNOTATING（自动开始录制）
        - REST 到期 -> WAIT（进入下一个手势）
        """
        now = time.time()
        if self.state == AppState.COUNTDOWN and now >= self.countdown_deadline:
            self._start_annotation()
        elif self.state == AppState.REST and now >= self.rest_deadline:
            self._enter_next_gesture()

    def _start_annotation(self) -> None:
        """倒计时结束 -> 自动记录开始帧，进入录制状态。"""
        self.annot_start_frame = self.frame_num
        self.annot_start_ms = int(self.frame_num * 1000 / self.fps)
        self.state = AppState.ANNOTATING

    def _commit_annotation(self, end_frame: int, end_ms: int) -> None:
        """将当前标注提交到 annotations 列表。

        Args:
            end_frame: 手势结束帧号
            end_ms:    手势结束毫秒时间戳
        """
        self.annotations.append(
            AnnotationEntry(
                gesture=self.current_gesture,
                label=GESTURE_LABEL[self.current_gesture],
                rep=self.rep,
                start_frame=self.annot_start_frame,  # type: ignore[arg-type]
                end_frame=end_frame,
                start_ms=self.annot_start_ms,  # type: ignore[arg-type]
                end_ms=end_ms,
            )
        )
        self.annot_start_frame = None
        self.annot_start_ms = None

    def _advance(self) -> None:
        """提交标注后推进到下一轮（下一 rep / 下一手势 / 完成）。

        推进逻辑：
        - 还有同一手势的重复次数 -> rep+1, 新倒计时
        - 当前手势做完但有下一个手势 -> 进入 REST 休息
        - 所有手势完成 -> 进入 COMPLETE
        """
        if self.rep < self.repeats:
            # 下一 rep：自动开始新倒计时
            self.rep = self.rep + 1
            self.countdown_deadline = time.time() + self.countdown_sec
            self.state = AppState.COUNTDOWN
        elif self.g_idx + 1 >= len(self.gestures):
            self.completed = True
            self.state = AppState.COMPLETE
        else:
            self.rest_deadline = time.time() + self.rest_sec
            self.state = AppState.REST

    def _enter_next_gesture(self) -> None:
        """进入下一个手势（g_idx + 1），重置 rep=1。

        如果已无下一个手势，进入 COMPLETE 状态；否则进入 WAIT 等待用户准备。
        """
        self.g_idx += 1
        self.rep = 1
        if self.g_idx >= len(self.gestures):
            self.completed = True
            self.state = AppState.COMPLETE
        else:
            self.state = AppState.WAIT


# Entry point — 主入口


def _precise_sleep(target_time: float) -> None:
    """睡眠到 target_time（time.monotonic()）。

    Windows 下 time.sleep() 粒度约 15 ms，无法满足 60fps（16.7ms/帧）的精度要求。
    策略：先粗粒度 sleep 到距 deadline 约 1ms 处，再忙等消耗剩余时间。
    这样既避免了纯忙等的 CPU 浪费，又保证了帧对齐的精确度。

    Args:
        target_time: 目标唤醒时间（time.monotonic() 返回值）
    """
    remaining = target_time - time.monotonic()
    if remaining <= 0:
        return
    if remaining > 0.001:
        time.sleep(remaining - 0.001)
    # 忙等最后 ~1ms，保证精确对齐到 target_time
    while time.monotonic() < target_time:
        pass


def _open_camera(preferred_index: int) -> Optional[cv2.VideoCapture]:
    """Open the preferred camera index; fall back to 0 if it fails.

    打开指定索引的摄像头，失败时自动降级到索引 0。
    Windows 平台优先使用 MSMF 后端，macOS 优先使用 AVFoundation。
    Windows 下还会强制使用 MJPG 四字符码以获得更高的帧率支持。

    Args:
        preferred_index: 首选摄像头索引

    Returns:
        成功打开的 VideoCapture 对象，或 None
    """
    if sys.platform == "win32":
        backend_candidates = [("MSMF", cv2.CAP_MSMF)]
    elif sys.platform == "darwin":
        backend_candidates = [("AVFoundation", cv2.CAP_AVFOUNDATION), ("ANY", cv2.CAP_ANY)]
    else:
        backend_candidates = [("ANY", cv2.CAP_ANY)]

    for idx in dict.fromkeys([preferred_index, 0]):  # try preferred first, then 0
        for backend_name, backend in backend_candidates:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                if sys.platform == "win32":
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
                if idx != preferred_index:
                    print(
                        f"[warn] 摄像头 {preferred_index} 不可用，已降级到摄像头 {idx}"
                        f"（backend={backend_name}）"
                    )
                else:
                    print(f"[info] 使用摄像头 {idx}（backend={backend_name}）")
                return cap
            cap.release()
    print(f"[error] 无法打开任何摄像头（尝试了索引 {preferred_index} 和 0）")
    return None


def _probe_frame_shape(cap: cv2.VideoCapture) -> tuple[int, int]:
    """Return (height, width) as reported by the driver after setting resolution.

    获取摄像头驱动实际报告的分辨率（可能与请求值不同）。

    Args:
        cap: 已打开的 VideoCapture 对象

    Returns:
        (height, width) 元组
    """
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return h, w


def main() -> None:
    """手势视频采集器主函数。

    完整流程：初始化摄像头/检测器/写入器 -> 进入主循环 ->
    按帧率采集、显示、响应键盘事件 -> 会话结束后保存数据并清理资源。
    """
    ap = argparse.ArgumentParser(
        description="手势视频采集器 — 连续录制 + 手动标注起止时间"
    )
    ap.add_argument("--subject", default="S01", help="受试者 ID")
    ap.add_argument("--repeats", type=int, default=5, help="每种手势重复次数")
    ap.add_argument(
        "--camera",
        type=int,
        default=1,
        help="摄像头索引（默认 1 = USB webcam；失败时自动降级到 0）",
    )
    ap.add_argument(
        "--fps", type=float, default=60.0, help="目标帧率（取摄像头实际值优先）"
    )
    ap.add_argument(
        "--countdown-sec",
        type=float,
        default=3.0,
        help="每次动作前的倒计时秒数（默认 3）",
    )
    ap.add_argument("--rest-sec", type=float, default=3.0, help="手势组间休息秒数")
    ap.add_argument("--output-dir", default="data", help="数据输出根目录")
    ap.add_argument(
        "--model-cache-dir", default=".models", help="MediaPipe 模型缓存目录"
    )
    args = ap.parse_args()

    if not MEDIAPIPE_AVAILABLE:
        print("[error] mediapipe 未安装。请执行：uv sync")
        return

    font_path = resolve_font()
    model_path = resolve_model(args.model_cache_dir)
    lmkr = make_landmarker(model_path)

    # 使用异步摄像头流
    stream = CameraStream(args.camera, 1920, 1080, int(args.fps)).start()
    cam_h, cam_w = _probe_frame_shape(stream.cap)
    cam_fps = stream.get(cv2.CAP_PROP_FPS)
    fps = float(cam_fps) if cam_fps and cam_fps > 1 else args.fps

    ret, probe_frame, _, _ = stream.read()
    if not ret or probe_frame is None:
        print("[error] 无法读取摄像头画面")
        stream.stop()
        return
    cam_h, cam_w = probe_frame.shape[:2]
    print(f"[info] 实际分辨率：{cam_w}×{cam_h} @ {fps:.0f}fps")

    session_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    video_path = out_dir / "video" / f"{session_id}_{args.subject}.mp4"
    label_path = out_dir / "labels" / f"{session_id}_{args.subject}.json"

    # 使用异步视频写入
    writer = AsyncVideoWriter(video_path, fps, cam_w, cam_h)
    meta = SessionMeta(
        subject_id=args.subject,
        session_id=session_id,
        video_file=str(video_path),
        fps=fps,
        width=cam_w,
        height=cam_h,
        gestures_order=list(GESTURE_ORDER),
        repeats=args.repeats,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    controller = SessionController(
        gestures=list(GESTURE_ORDER),
        repeats=args.repeats,
        rest_sec=args.rest_sec,
        countdown_sec=args.countdown_sec,
        fps=fps,
    )

    controller.frame_num = 0

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)

    print(f"[info] 视频  → {video_path}")
    print(f"[info] 标注  → {label_path}")
    print("[info] SPACE=标记起止  R=撤销  Q=退出保存")

    # ── 异步 MediaPipe 检测线程 ────────────────────────────────────────────────
    detect_queue: queue.Queue = queue.Queue(maxsize=1)  # 检测帧队列（只保留最新一帧）
    latest_lms: list = [None]                           # 最新检测结果（列表包装以便闭包修改）
    det_fps_val: list = [0.0]                           # 检测帧率（指数移动平均）
    stop_event = threading.Event()                      # 线程停止信号

    def _detect_worker():
        """MediaPipe 检测线程函数：循环取帧 -> 检测 -> 更新最新结果。"""
        t_last = time.monotonic()
        while not stop_event.is_set():
            try:
                frame = detect_queue.get(timeout=0.05)
                # 检查 landmarker 是否仍然可用（可能已被其他线程关闭）
                if lmkr is None: break
                res = detect(frame, lmkr)
                latest_lms[0] = res
            except (queue.Empty, Exception):
                continue
            # 检测帧率计算：指数移动平均（EMA），系数 0.8/0.2，平滑显示
            now = time.monotonic()
            dt = now - t_last
            if dt > 0:
                det_fps_val[0] = det_fps_val[0] * 0.8 + (1.0 / dt) * 0.2
            t_last = now

    detect_thread = threading.Thread(target=_detect_worker, daemon=True)
    detect_thread.start()

    # ── 视频时间轴控制（CFR 恒定帧率 + 帧补齐） ────────────────────────────
    # 核心策略：输出视频保持恒定 fps（CFR, Constant Frame Rate）。
    # 真实时间过去了多少，就补齐多少个输出帧；
    # 摄像头或主循环变慢时重复最新画面，避免文件播放速度被压快。
    # 这保证了输出视频在任何播放器中都以正确速度回放，
    # 同时标注的帧号-时间对应关系保持精确。
    frame_interval = 1.0 / fps       # 每帧的标称时间间隔
    t_start = time.monotonic()       # 会话起始时间基准
    t_end = t_start                  # 最近一次循环的时间
    t_prev_source_frame = t_start    # 上一次摄像头新帧到达时间
    rec_fps_val = 0.0               # 录制帧率（EMA 平滑）
    last_source_seq = -1            # 上一次摄像头帧序号（用于检测新帧）
    last_video_seq = -1             # 上一次写入视频的帧序号（用于统计补帧）
    source_frame_count = 0          # 摄像头实际产生的帧数
    duplicated_frame_count = 0      # 为维持 CFR 而重复补帧的帧数

    # ── 主循环：帧率控制 + 采集 + 显示 + 键盘响应 ─────────────────────────
    while True:
        if controller.state == AppState.DONE:
            break

        # 精确睡眠到下一个视频帧时间点；如果已经落后，后面会批量补帧。
        _precise_sleep(t_start + controller.frame_num * frame_interval)

        ok, raw_frame, cur_seq, _ = stream.read()
        if not ok or raw_frame is None:
            continue

        t_end = time.monotonic()
        # 检测是否收到摄像头的真正新帧（seq 变化 = 新帧到达）
        if cur_seq != last_source_seq:
            # 更新录制帧率（EMA，系数 0.9/0.1）
            dt_frame = t_end - t_prev_source_frame
            if dt_frame > 0:
                rec_fps_val = rec_fps_val * 0.9 + (1.0 / dt_frame) * 0.1
            t_prev_source_frame = t_end
            last_source_seq = cur_seq

            # 将新帧推送到检测队列（队列满则跳过，只保留最新帧）
            try:
                detect_queue.put_nowait(raw_frame)
            except queue.Full:
                pass

        # 计算当前时间点应该到达的帧号（CFR 时间轴对齐）
        # 如果主循环落后了，while 循环会批量写入重复帧来补齐
        target_frame_num = max(1, int((t_end - t_start) * fps) + 1)
        while controller.frame_num < target_frame_num:
            writer.write(raw_frame)
            controller.frame_num += 1
            # 统计：同一摄像头帧被写入多次 = 补帧；首次写入 = 真实帧
            if cur_seq == last_video_seq:
                duplicated_frame_count += 1
            else:
                source_frame_count += 1
                last_video_seq = cur_seq

        # 当前帧对应的毫秒时间戳（用于标注记录）
        frame_ms = int(controller.frame_num * 1000 / fps)

        # 驱动时间相关的状态转换（倒计时结束 -> 开始录制，休息结束 -> 下一手势）
        controller.tick()

        # 组合并显示 UI 画面（左侧摄像头 + 右侧信息面板）
        cv2.imshow(
            WIN_NAME,
            build_canvas(
                raw_frame,
                latest_lms[0],
                controller,
                font_path,
                rec_fps=rec_fps_val,
                det_fps=det_fps_val[0],
            ),
        )

        # 键盘事件处理：Q/ESC=退出，SPACE=标注，R=撤销
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            controller.on_quit()
            save_session(label_path, controller, meta)  # 退出时立即保存，防数据丢失
        elif key == ord(" "):
            controller.on_space(controller.frame_num, frame_ms)
            save_session(label_path, controller, meta)  # 每次标注后立即保存
        elif key in (ord("r"), ord("R")):
            controller.on_redo()

    # ── 清理资源 ───────────────────────────────────────────────────────────
    stop_event.set()                    # 通知检测线程停止
    detect_thread.join(timeout=1.0)     # 等待检测线程退出

    writer.release()                    # 刷新并关闭视频写入器
    stream.stop()                       # 停止摄像头采集并释放设备

    if lmkr is not None:
        lmkr.close()                    # 关闭 MediaPipe landmarker
    cv2.destroyAllWindows()             # 关闭 OpenCV 窗口

    # ── 统计并输出会话摘要 ────────────────────────────────────────────────
    # 输出视频是固定 fps 的 CFR 时间轴；source_fps 只反映摄像头真实供帧能力。
    elapsed_total = t_end - t_start
    if elapsed_total > 0 and controller.frame_num > 1:
        meta.fps = fps
        meta.source_fps = source_frame_count / elapsed_total
        meta.source_frames = source_frame_count
        meta.duplicated_frames = duplicated_frame_count
        print(
            f"[info] 视频时间轴：{meta.fps:.1f} fps CFR；"
            f"摄像头供帧约 {meta.source_fps:.1f} fps；"
            f"补帧 {meta.duplicated_frames} 帧"
        )

    save_session(label_path, controller, meta)  # 最终保存完整元数据（含 source_fps 等）
    status = "中止" if controller.aborted else "完成"
    print(f"[{status}] 标注数：{len(controller.annotations)}")
    print(f"[{status}] 视频  ：{video_path}")
    print(f"[{status}] 标注  ：{label_path}")


if __name__ == "__main__":
    main()
