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

WIN_W = 1280
WIN_H = 720
PANEL_SPLIT = int(WIN_W * 0.65)  # camera panel width = 832
PANEL_R_W = WIN_W - PANEL_SPLIT  # right panel width  = 448
WIN_NAME = "Gesture Collector"

FONT_CANDIDATES = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]

# Data models


@dataclass
class AnnotationEntry:
    gesture: str
    label: int
    rep: int
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int


@dataclass
class SessionMeta:
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


# Utilities


def resolve_font() -> Optional[str]:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def save_session(
    label_path: Path,
    controller: "SessionController",
    meta: SessionMeta,
) -> None:
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
            cv2.circle(panel, pt, 7, color, -1, cv2.LINE_AA)
            cv2.circle(panel, pt, 9, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(panel, pt, 3, (180, 180, 180), -1, cv2.LINE_AA)


# UI rendering


def letterbox(
    frame: np.ndarray,
    target_w: int,
    target_h: int,
) -> tuple[np.ndarray, int, int, int, int]:
    """Fit *frame* into target_w×target_h with black bars, preserving aspect ratio.

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
    """Draw a minimal schematic of the gesture using OpenCV primitives."""
    cx = x0 + w // 2
    cy = y0 + h // 2

    if gesture == "pinch_index":
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
    """Render text onto a PIL Draw object or a numpy BGR image."""
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
    """Compose the 1280×720 display canvas from camera + right info panel."""
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

        _put_zh(draw, zh_name, (rx, 22), font_path, 40, (255, 220, 50))
        _put_zh(draw, en_name, (rx, 76), font_path, 22, (150, 150, 150))

        # Status text
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

        hints = [
            ("[SPACE]", "标记开始 / 结束"),
            ("[R]", "撤销上一个标注"),
            ("[Q]", "退出保存"),
        ]
        hy = 424
        for _, desc in hints:
            _put_zh(draw, desc, (rx + 72, hy - 14), font_path, 18, (150, 150, 150))
            hy += 30

        _put_zh(
            draw,
            f"已标注  {len(controller.annotations)}  段",
            (rx, WIN_H - 52),
            font_path,
            18,
            (90, 90, 90),
        )

        rp = cv2.cvtColor(np.array(pil_rp), cv2.COLOR_RGB2BGR)

    # 绘制非文字部分
    rx = 18
    draw_gesture_hint(
        rp, controller.current_gesture, rx + 20, 112, PANEL_R_W - rx * 2 - 20, 130
    )
    dot_y, x_start = 270, rx + 8
    for i in range(controller.repeats):
        cx_d = x_start + i * 26
        if i + 1 < controller.rep:
            cv2.circle(rp, (cx_d, dot_y), 9, (50, 200, 80), -1)
        elif i + 1 == controller.rep:
            cv2.circle(rp, (cx_d, dot_y), 9, (240, 240, 240), -1)
        else:
            cv2.circle(rp, (cx_d, dot_y), 9, (90, 90, 90), 1)

    cv2.line(rp, (rx, 408), (PANEL_R_W - rx, 408), (65, 65, 65), 1)
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

    # FPS
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

    return np.hstack([cam_panel, rp])



def draw_done_screen(
    controller: "SessionController",
    video_path: Path,
    label_path: Path,
    font_path: Optional[str],
) -> np.ndarray:
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


# Threaded I/O


class CameraStream:
    def __init__(self, index: int, target_w: int, target_h: int, target_fps: int):
        self.cap = _open_camera(index)
        if self.cap is None:
            raise RuntimeError(f"Could not open camera {index}")

        # Try to set resolution/fps again just in case
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)

        self.ret, self.frame = self.cap.read()
        self.frame_ts = time.monotonic()
        self.stopped = False
        self.lock = threading.Lock()
        # 帧序号：每次摄像头产生新帧时自增，供主循环去重用
        self._seq: int = 0

    def start(self):
        t = threading.Thread(target=self.update, args=(), daemon=True)
        t.start()
        return self

    def update(self):
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
        """返回 (ret, frame_copy, seq, timestamp)。seq 每次摄像头产生新帧时自增。"""
        with self.lock:
            return (
                self.ret,
                self.frame.copy() if self.frame is not None else None,
                self._seq,
                self.frame_ts,
            )

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()

    def get(self, prop):
        return self.cap.get(prop)


class AsyncVideoWriter:
    def __init__(self, path: Path, fps: float, w: int, h: int):
        self.writer = make_writer(path, fps, w, h)
        self.queue: queue.Queue = queue.Queue(maxsize=512)
        self.stopped = False
        self.frames_written = 0
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while not self.stopped or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                self.writer.write(frame)
                self.frames_written += 1
            except queue.Empty:
                continue

    def write(self, frame, *, block: bool = True):
        try:
            if block:
                self.queue.put(frame)
            else:
                self.queue.put_nowait(frame)
        except queue.Full:
            print("[warn] Video writer queue full, dropping frame")

    def release(self):
        self.stopped = True
        self.thread.join()
        self.writer.release()


# Session state machine


class AppState(Enum):
    WAIT = "wait"
    COUNTDOWN = "countdown"  # 按下 SPACE 后 3 秒倒计时，结束自动标记开始
    ANNOTATING = "annotating"
    REST = "rest"
    COMPLETE = "complete"
    DONE = "done"


class SessionController:
    def __init__(
        self,
        gestures: List[str],
        repeats: int,
        rest_sec: float,
        countdown_sec: float,
        fps: float,
    ) -> None:
        self.gestures = gestures
        self.repeats = repeats
        self.rest_sec = rest_sec
        self.countdown_sec = countdown_sec
        self.fps = fps

        self.state = AppState.WAIT
        self.g_idx = 0
        self.rep = 1
        self.frame_num = 0

        self.annot_start_frame: Optional[int] = None
        self.annot_start_ms: Optional[int] = None
        self.annotations: List[AnnotationEntry] = []

        self.countdown_deadline = 0.0
        self.rest_deadline = 0.0
        self.aborted = False
        self.completed = False

    @property
    def current_gesture(self) -> str:
        return self.gestures[min(self.g_idx, len(self.gestures) - 1)]

    @property
    def countdown_remaining(self) -> float:
        return max(0.0, self.countdown_deadline - time.time())

    @property
    def rest_remaining(self) -> float:
        return max(0.0, self.rest_deadline - time.time())

    def on_space(self, frame_num: int, ms: int) -> None:
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
        if not self.completed:
            self.aborted = True
        self.state = AppState.DONE

    def tick(self) -> None:
        now = time.time()
        if self.state == AppState.COUNTDOWN and now >= self.countdown_deadline:
            self._start_annotation()
        elif self.state == AppState.REST and now >= self.rest_deadline:
            self._enter_next_gesture()

    def _start_annotation(self) -> None:
        """倒计时结束 → 自动记录开始帧，进入录制状态。"""
        self.annot_start_frame = self.frame_num
        self.annot_start_ms = int(self.frame_num * 1000 / self.fps)
        self.state = AppState.ANNOTATING

    def _commit_annotation(self, end_frame: int, end_ms: int) -> None:
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
        self.g_idx += 1
        self.rep = 1
        if self.g_idx >= len(self.gestures):
            self.completed = True
            self.state = AppState.COMPLETE
        else:
            self.state = AppState.WAIT


# Entry point


def _precise_sleep(target_time: float) -> None:
    """睡眠到 target_time（time.monotonic()）。

    Windows 下 time.sleep() 粒度约 15 ms，无法满足 60fps（16.7ms/帧）的精度要求。
    策略：先粗粒度 sleep 到距 deadline 约 1ms 处，再忙等消耗剩余时间。
    """
    remaining = target_time - time.monotonic()
    if remaining <= 0:
        return
    if remaining > 0.001:
        time.sleep(remaining - 0.001)
    # 忙等最后 ~1ms，保证精确对齐
    while time.monotonic() < target_time:
        pass


def _open_camera(preferred_index: int) -> Optional[cv2.VideoCapture]:
    """Open the preferred camera index; fall back to 0 if it fails."""
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
    """Return (height, width) as reported by the driver after setting resolution."""
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return h, w


def main() -> None:
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
    detect_queue: queue.Queue = queue.Queue(maxsize=1)
    latest_lms: list = [None]
    det_fps_val: list = [0.0]
    stop_event = threading.Event()

    def _detect_worker():
        t_last = time.monotonic()
        while not stop_event.is_set():
            try:
                frame = detect_queue.get(timeout=0.05)
                # 检查 lmkr 是否仍然可用
                if lmkr is None: break
                res = detect(frame, lmkr)
                latest_lms[0] = res
            except (queue.Empty, Exception):
                continue
            
            now = time.monotonic()
            dt = now - t_last
            if dt > 0:
                det_fps_val[0] = det_fps_val[0] * 0.8 + (1.0 / dt) * 0.2
            t_last = now

    detect_thread = threading.Thread(target=_detect_worker, daemon=True)
    detect_thread.start()

    # ── 视频时间轴控制 ────────────────────────────────────────────────────────
    # 输出视频保持恒定 fps。真实时间过去了多少，就补齐多少个输出帧；
    # 摄像头或主循环变慢时重复最新画面，避免文件播放速度被压快。
    frame_interval = 1.0 / fps
    t_start = time.monotonic()
    t_end = t_start
    t_prev_source_frame = t_start
    rec_fps_val = 0.0
    last_source_seq = -1
    last_video_seq = -1
    source_frame_count = 0
    duplicated_frame_count = 0

    while True:
        if controller.state == AppState.DONE:
            break

        # 等到下一个视频帧时间点；如果已经落后，后面会批量补帧。
        _precise_sleep(t_start + controller.frame_num * frame_interval)

        ok, raw_frame, cur_seq, _ = stream.read()
        if not ok or raw_frame is None:
            continue

        t_end = time.monotonic()
        if cur_seq != last_source_seq:
            dt_frame = t_end - t_prev_source_frame
            if dt_frame > 0:
                rec_fps_val = rec_fps_val * 0.9 + (1.0 / dt_frame) * 0.1
            t_prev_source_frame = t_end
            last_source_seq = cur_seq

            try:
                detect_queue.put_nowait(raw_frame)
            except queue.Full:
                pass

        target_frame_num = max(1, int((t_end - t_start) * fps) + 1)
        while controller.frame_num < target_frame_num:
            writer.write(raw_frame)
            controller.frame_num += 1
            if cur_seq == last_video_seq:
                duplicated_frame_count += 1
            else:
                source_frame_count += 1
                last_video_seq = cur_seq

        frame_ms = int(controller.frame_num * 1000 / fps)

        controller.tick()

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

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            controller.on_quit()
            save_session(label_path, controller, meta)
        elif key == ord(" "):
            controller.on_space(controller.frame_num, frame_ms)
            save_session(label_path, controller, meta)
        elif key in (ord("r"), ord("R")):
            controller.on_redo()

    stop_event.set()
    detect_thread.join(timeout=1.0)

    writer.release()
    stream.stop()

    if lmkr is not None:
        lmkr.close()
    cv2.destroyAllWindows()

    # 输出视频是固定 fps 的时间轴；source_fps 只反映摄像头真实供帧能力。
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

    save_session(label_path, controller, meta)
    status = "中止" if controller.aborted else "完成"
    print(f"[{status}] 标注数：{len(controller.annotations)}")
    print(f"[{status}] 视频  ：{video_path}")
    print(f"[{status}] 标注  ：{label_path}")


if __name__ == "__main__":
    main()
