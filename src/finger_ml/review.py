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


WIN_NAME = "Gesture Data Review"
MAX_INIT_WIN_W = 1280
MAX_INIT_WIN_H = 720
FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
)
COLORS = (
    (80, 220, 80),
    (60, 170, 255),
    (255, 170, 60),
    (220, 90, 255),
    (255, 220, 80),
    (80, 210, 230),
)
JITTER_WARN = 0.06
JITTER_BAD = 0.12


@dataclass(frozen=True)
class Annotation:
    gesture: str
    label: int
    rep: int
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int

    @property
    def duration_sec(self) -> float:
        return max(0, self.end_ms - self.start_ms) / 1000.0


@dataclass
class FeatureOverlay:
    path: Path
    valid: np.ndarray
    jitter: np.ndarray
    quality: dict[str, Any]

    @property
    def n_frames(self) -> int:
        return int(len(self.valid))


def _resolve_font() -> str | None:
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
    if _PIL_OK:
        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        draw.text(xy, text, font=font, fill=color)
        frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return

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
    with label_path.open(encoding="utf-8") as f:
        payload = json.load(f)
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
    if label_path.parent.name == "labels":
        return label_path.parent.parent / "features" / f"{label_path.stem}.npz"
    return label_path.with_suffix(".npz")


def _load_feature_overlay(path: Path | None) -> FeatureOverlay | None:
    if path is None or not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        landmarks = data["landmarks"].astype(np.float32, copy=False)
        valid = data["valid"].astype(bool, copy=False)
        quality: dict[str, Any] = {}
        if "quality_json" in data.files:
            quality_raw = str(data["quality_json"])
            quality = json.loads(quality_raw) if quality_raw else {}
    except Exception as exc:
        print(f"[review] warning: cannot load feature overlay {path}: {exc}")
        return None

    jitter = np.zeros(len(valid), dtype=np.float32)
    if len(landmarks) > 1:
        pair_valid = valid[1:] & valid[:-1]
        step = np.linalg.norm(landmarks[1:] - landmarks[:-1], axis=2)
        frame_jitter = step.mean(axis=1).astype(np.float32)
        jitter[1:] = np.where(pair_valid, frame_jitter, 0.0)
    return FeatureOverlay(path=path, valid=valid, jitter=jitter, quality=quality)


def _resolve_pair(data_dir: Path, session: str | None, video: str | None, label: str | None) -> tuple[Path, Path]:
    if video:
        video_path = Path(video)
        label_path = Path(label) if label else data_dir / "labels" / f"{video_path.stem}.json"
    elif label:
        label_path = Path(label)
        video_file = _load_json(label_path).get("video_file")
        video_path = Path(video_file) if video_file else data_dir / "video" / f"{label_path.stem}.mp4"
    else:
        if session:
            stem = session
        else:
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
    h, w = frame.shape[:2]
    panel_h = 124
    y0 = h - panel_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    left, right = 28, w - 28
    jitter_y, jitter_h = y0 + 40, 12
    bar_y, bar_h = y0 + 60, 20
    cv2.rectangle(frame, (left, jitter_y), (right, jitter_y + jitter_h), (42, 42, 42), -1)
    cv2.rectangle(frame, (left, bar_y), (right, bar_y + bar_h), (58, 58, 58), -1)

    denom = max(1, total_frames - 1)
    if features is not None and features.n_frames:
        n = min(features.n_frames, total_frames)
        bucket_count = max(1, min(right - left, n))
        bucket_frames = max(1, int(np.ceil(n / bucket_count)))
        for start in range(0, n, bucket_frames):
            end = min(n, start + bucket_frames)
            x1 = left + int((right - left) * start / denom)
            x2 = left + int((right - left) * max(start + 1, end - 1) / denom)
            if not bool(np.any(features.valid[start:end])):
                color = (60, 60, 235)
            else:
                value = float(np.max(features.jitter[start:end]))
                if value <= JITTER_WARN:
                    level = value / JITTER_WARN
                    color = (70, int(220 - 45 * level), int(80 + 130 * level))
                else:
                    level = float(np.clip((value - JITTER_WARN) / (JITTER_BAD - JITTER_WARN), 0.0, 1.0))
                    color = (int(70 - 20 * level), int(175 - 105 * level), int(210 + 45 * level))
            cv2.rectangle(frame, (x1, jitter_y), (max(x2, x1 + 1), jitter_y + jitter_h), color, -1)

    for i, ann in enumerate(annotations):
        x1 = left + int((right - left) * ann.start_frame / denom)
        x2 = left + int((right - left) * ann.end_frame / denom)
        color = COLORS[ann.label % len(COLORS)]
        cv2.rectangle(frame, (x1, bar_y), (max(x2, x1 + 2), bar_y + bar_h), color, -1)
        if active_idx == i:
            cv2.rectangle(frame, (x1, bar_y - 5), (max(x2, x1 + 2), bar_y + bar_h + 5), (255, 255, 255), 2)

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
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 92), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    if active_idx is None:
        label_text = "静息 / background"
        color = (210, 210, 210)
    else:
        ann = annotations[active_idx]
        zh = GESTURE_ZH.get(ann.gesture, ann.gesture)
        label_text = f"#{active_idx + 1:02d} {zh} / {ann.gesture}  rep={ann.rep}  {ann.duration_sec:.2f}s"
        color = COLORS[ann.label % len(COLORS)]
    status = "暂停" if paused else "播放"
    if loop_window:
        status += " | 窗口循环"
    skeleton_text = "骨架=实时检测"
    if skeleton_ok is True:
        skeleton_text += " OK"
    elif skeleton_ok is False:
        skeleton_text += " MISS"
    metric_text = ""
    if features is not None and 0 <= frame_idx < features.n_frames:
        valid = "OK" if bool(features.valid[frame_idx]) else "MISS"
        jitter = float(features.jitter[frame_idx])
        metric_text = f"  feature_valid={valid}  jitter={jitter:.4f}"
    elif features is None:
        metric_text = "  no feature npz"
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
    if lms is None:
        return False

    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (220, 220, 220), 1, cv2.LINE_AA)

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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(meta.get("width") or MAX_INIT_WIN_W)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(meta.get("height") or MAX_INIT_WIN_H)
    if width <= 0 or height <= 0:
        return MAX_INIT_WIN_W, MAX_INIT_WIN_H

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
    meta = _load_json(label_path)
    annotations: list[Annotation] = meta["annotations"]
    features = _load_feature_overlay(feature_path if feature_path is not None else _features_path_for(label_path))
    lmkr = None
    if draw_skeleton:
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

    window_idx = 0
    if start_window is not None and annotations:
        window_idx = min(max(0, start_window - 1), len(annotations) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, annotations[window_idx].start_frame)
    paused = False
    loop_window = start_window is not None
    clock_anchor_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    clock_anchor_time = time.monotonic()

    def reset_playback_clock(frame: int | None = None) -> None:
        nonlocal clock_anchor_frame, clock_anchor_time
        clock_anchor_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES) if frame is None else frame)
        clock_anchor_time = time.monotonic()

    init_w, init_h = _initial_window_size(cap, meta)
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, init_w, init_h)
    while True:
        if not paused and fps > 0:
            elapsed = time.monotonic() - clock_anchor_time
            target_frame = clock_anchor_frame + int(elapsed * fps)
            pos_now = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if target_frame > pos_now:
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(target_frame, max(0, total_frames - 1)))

        pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            reset_playback_clock(0)
            continue

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        active_idx = _active_annotation(annotations, frame_idx)
        if active_idx is not None:
            window_idx = active_idx

        skeleton_ok: bool | None = None
        if lmkr is not None:
            skeleton_ok = _draw_skeleton_overlay(frame, detect(frame, lmkr, hand_side=hand_side))

        _draw_hud(frame, meta, annotations, frame_idx, fps, active_idx, features, skeleton_ok, paused, loop_window, font_path)
        _draw_timeline(frame, annotations, frame_idx, total_frames, active_idx, features, font_path)
        cv2.imshow(WIN_NAME, frame)

        loop_restart_frame: int | None = None
        if loop_window and annotations:
            ann = annotations[window_idx]
            if frame_idx >= ann.end_frame:
                loop_restart_frame = ann.start_frame

        if paused or fps <= 0:
            wait_ms = 0
        else:
            next_due = clock_anchor_time + ((frame_idx + 1 - clock_anchor_frame) / fps)
            wait_ms = max(1, int((next_due - time.monotonic()) * 1000))
        key = cv2.waitKeyEx(wait_ms)
        if loop_restart_frame is not None and key == -1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, loop_restart_frame)
            reset_playback_clock(loop_restart_frame)
            continue
        if key in (ord("q"), ord("Q"), 27):
            break
        if key == ord(" "):
            if paused:
                paused = False
                reset_playback_clock()
            else:
                paused = True
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        elif key in (ord("l"), ord("L")):
            loop_window = not loop_window
            if loop_window and annotations:
                target = active_idx if active_idx is not None else window_idx
                window_idx = target
                cap.set(cv2.CAP_PROP_POS_FRAMES, annotations[window_idx].start_frame)
                reset_playback_clock(annotations[window_idx].start_frame)
        elif key in (ord("a"), ord("A")) and annotations:
            window_idx = max(0, window_idx - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, annotations[window_idx].start_frame)
            reset_playback_clock(annotations[window_idx].start_frame)
            paused = True
        elif key in (ord("d"), ord("D")) and annotations:
            window_idx = min(len(annotations) - 1, window_idx + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, annotations[window_idx].start_frame)
            reset_playback_clock(annotations[window_idx].start_frame)
            paused = True
        elif key in (81, 2424832):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos - 2))
            reset_playback_clock(max(0, pos - 2))
            paused = True
        elif key in (83, 2555904):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, pos + int(max(1, fps))))
            reset_playback_clock(min(total_frames - 1, pos + int(max(1, fps))))
            paused = True

    cap.release()
    cv2.destroyWindow(WIN_NAME)


def main() -> None:
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
