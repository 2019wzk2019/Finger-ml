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
    """Suppress noisy C++ stderr logs from MediaPipe/TFLite.

    MediaPipe emits INFO/WARNING lines from native code, so Python's warnings
    filters do not affect them. Redirecting fd=2 keeps our stdout progress
    visible while hiding those backend messages.
    """
    if not enabled:
        yield
        return
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)

# ── 常量 ─────────────────────────────────────────────────────────────────────

_DEBUG_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

BOUNDARY_MARGIN  = 2    # 手势边界各裁掉 2 帧（过渡帧标签模糊）
PRE_IGNORE_FRAMES = 4   # 动作开始附近的预备/过渡帧不参与训练
POST_IGNORE_SECONDS = 1.0 # 动作结束后的回弹段可能像反向手势，默认屏蔽 1 秒


def _longest_false_run(valid: np.ndarray) -> int:
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
    """Return per-node displacement between consecutive valid frames."""
    if len(landmarks) < 2:
        return np.empty((0, NUM_NODES), dtype=np.float32)
    pair_valid = valid[1:] & valid[:-1]
    if mask is not None:
        pair_valid &= mask[1:] & mask[:-1]
    if not np.any(pair_valid):
        return np.empty((0, NUM_NODES), dtype=np.float32)
    diff = landmarks[1:] - landmarks[:-1]
    return np.linalg.norm(diff[pair_valid], axis=2).astype(np.float32)


def _p95(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 95))


def _score_high_is_good(value: float, bad_at: float) -> float:
    if bad_at <= 0:
        return 1.0
    return float(np.clip(1.0 - value / bad_at, 0.0, 1.0))


def compute_quality_metrics(
    landmarks: np.ndarray,
    valid: np.ndarray,
    labels: np.ndarray,
    fps: float,
) -> dict:
    """Compute session-level skeleton quality metrics.

    Jitter is measured in palm-local normalized units per frame. Background
    jitter is the most useful stability signal because intentional gesture
    motion is mostly excluded.
    """
    n_frames = int(len(valid))
    valid_rate = float(valid.mean()) if n_frames else 0.0
    longest_gap_frames = _longest_false_run(valid)
    longest_gap_sec = float(longest_gap_frames / fps) if fps > 0 else 0.0

    bg_mask = labels == BACKGROUND_LABEL
    all_steps = _consecutive_step_norms(landmarks, valid)
    bg_steps = _consecutive_step_norms(landmarks, valid, bg_mask)
    jitter_steps = bg_steps if bg_steps.size else all_steps

    per_node_p95 = (
        np.percentile(jitter_steps, 95, axis=0).astype(np.float32)
        if jitter_steps.size
        else np.zeros(NUM_NODES, dtype=np.float32)
    )
    worst_node_idx = int(per_node_p95.argmax()) if len(per_node_p95) else 0

    counts = np.bincount(labels, minlength=BACKGROUND_LABEL + 1)
    present_gesture_classes = int((counts[:BACKGROUND_LABEL] > 0).sum())
    class_coverage = present_gesture_classes / BACKGROUND_LABEL

    detect_score = valid_rate
    gap_score = _score_high_is_good(longest_gap_sec, bad_at=0.5)
    # 0.02 normalized units/frame is very stable; 0.12 is visibly noisy.
    jitter_p95 = _p95(jitter_steps)
    jitter_score = 1.0 - float(np.clip((jitter_p95 - 0.02) / 0.10, 0.0, 1.0))
    coverage_score = class_coverage
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


# ── Debug 视频渲染 ────────────────────────────────────────────────────────────

def _render_progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[?]"
    ratio = min(max(done / total, 0.0), 1.0)
    filled = int(width * ratio)
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
    """在 frame 上绘制 21 节点骨架。"""
    if is_normalized:
        px = (pts[:, 0] * fw).astype(int)
        py = (pts[:, 1] * fh).astype(int)
    else:
        px, py = pts[:, 0].astype(int), pts[:, 1].astype(int)

    for i, j in _DEBUG_EDGES:
        cv2.line(frame, (px[i], py[i]), (px[j], py[j]), color, 1, cv2.LINE_AA)
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

        # ── HUD 文字
        status_color = (60, 220, 60) if is_valid else (60, 60, 255)
        status_text  = "OK" if is_valid else "MISS"
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


# ── 单个 session 处理 ─────────────────────────────────────────────────────────

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

    Returns:
        stats dict with n_frames, n_gesture_frames, n_detect_fail
    """
    # ── 加载标注 ──────────────────────────────────────────────────────────────
    with label_path.open(encoding="utf-8") as f:
        meta = json.load(f)

    annotations = meta["annotations"]
    fps         = float(meta["fps"])
    post_ignore_frames = max(0, int(round(fps * post_ignore_seconds)))

    # ── 打开视频 ──────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        # 部分编码器无法预先知道帧数，先读完
        n_frames = None

    # ── 构建帧级标签 ──────────────────────────────────────────────────────────
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

    labels = np.full(n_frames, BACKGROUND_LABEL, dtype=np.int64)
    train_mask = np.ones(n_frames, dtype=bool)
    for ann in annotations:
        raw_s = int(ann["start_frame"]) - 1
        raw_e = int(ann["end_frame"]) - 1
        s = raw_s + BOUNDARY_MARGIN   # 转 0-indexed
        e = raw_e - BOUNDARY_MARGIN
        if 0 <= s <= e < n_frames:
            labels[s : e + 1] = ann["label"]
        # 屏蔽按键边界、动作预备和动作结束后的恢复段，避免把“回弹反向动作”当背景训练。
        ignore_s = max(0, raw_s - pre_ignore_frames)
        ignore_e = min(n_frames - 1, s - 1)
        if ignore_s <= ignore_e:
            train_mask[ignore_s : ignore_e + 1] = False
        ignore_s = max(0, e + 1)
        ignore_e = min(n_frames - 1, raw_e + post_ignore_frames)
        if ignore_s <= ignore_e:
            train_mask[ignore_s : ignore_e + 1] = False

    # ── 逐帧提取骨架 ──────────────────────────────────────────────────────────
    landmarks_arr = np.zeros((n_frames, NUM_NODES, 3), dtype=np.float32)
    valid_arr     = np.zeros(n_frames, dtype=bool)
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

        timestamp_ms = int(round(frame_idx * 1000 / fps))
        lms = detect_video(frame, lmkr, timestamp_ms, hand_side=hand_side)
        if lms is not None:
            raw = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            sub = raw[KEEP_INDICES]              # [21, 3]
            if raw_img_arr is not None:
                raw_img_arr[frame_idx] = sub[:, :2]  # 保存 x,y 图像坐标
            sub = normalize_landmarks(sub)
            landmarks_arr[frame_idx] = sub
            valid_arr[frame_idx]     = True
            last_valid               = sub
        else:
            # 填充上一帧（保持序列连续性）
            landmarks_arr[frame_idx] = last_valid
            valid_arr[frame_idx]     = False
            n_fail += 1

        frame_idx += 1
        now = time.monotonic()
        if show_progress and (now - last_progress_at >= 0.12 or frame_idx >= n_frames):
            _print_frame_progress(frame_idx, n_frames, progress_started_at, final=frame_idx >= n_frames)
            last_progress_at = now

    cap.release()
    if show_progress and frame_idx < n_frames:
        _print_frame_progress(frame_idx, n_frames, progress_started_at, final=True)

    # 若视频实际帧数 < n_frames（罕见），裁剪
    actual = frame_idx
    landmarks_arr = landmarks_arr[:actual]
    valid_arr     = valid_arr[:actual]
    labels        = labels[:actual]
    train_mask    = train_mask[:actual]
    features_arr  = build_motion_features(landmarks_arr, valid_arr)
    quality       = compute_quality_metrics(landmarks_arr, valid_arr, labels, fps)

    # ── 保存 ─────────────────────────────────────────────────────────────────
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


# ── 批量处理 ──────────────────────────────────────────────────────────────────

def match_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """将 video/*.mp4 和 labels/*.json 按 session_subject 名称配对。"""
    video_map = {p.stem: p for p in (data_dir / "video").glob("*.mp4")}
    label_map = {p.stem: p for p in (data_dir / "labels").glob("*.json")}
    common    = sorted(set(video_map) & set(label_map))
    if not common:
        raise FileNotFoundError(
            f"No matching video+label pairs found in {data_dir}. "
            "Run finger-collect first."
        )
    return [(video_map[k], label_map[k]) for k in common]


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

def main() -> None:
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

    model_path = resolve_model(args.model_cache_dir)
    actual_delegate = args.delegate
    if args.delegate == "GPU":
        # Probe once. A VIDEO-mode landmarker cannot be reused across videos
        # whose timestamps restart at 0, so the real instance is created per
        # session below.
        try:
            with _suppress_native_stderr(not args.show_mediapipe_logs):
                probe = make_landmarker(model_path, running_mode="VIDEO", delegate="GPU")
            if probe is not None:
                probe.close()
        except Exception as exc:
            print(f"[warn] MediaPipe GPU delegate 不可用，回退 CPU：{exc}")
            actual_delegate = "CPU"

    pairs = match_pairs(data_dir)
    print(f"[info] 发现 {len(pairs)} 个 session，开始提取骨架特征...")

    for video_path, label_path in pairs:
        out_path = features_dir / (video_path.stem + ".npz")
        if out_path.exists() and not args.force:
            print(f"  [skip] {out_path.name} 已存在（--force 强制重提取）")
            continue

        print(f"  [proc] {video_path.name}")
        lmkr = None
        try:
            debug_video_path = (
                debug_dir / (video_path.stem + "_debug.mp4") if debug_dir else None
            )
            with _suppress_native_stderr(not args.show_mediapipe_logs):
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
                sys.stdout.write("\n")
                sys.stdout.flush()
            print(f"✗  错误：{e}")
        finally:
            if lmkr is not None:
                lmkr.close()

    print(f"[done] 特征文件保存于 {features_dir}/")


if __name__ == "__main__":
    main()
