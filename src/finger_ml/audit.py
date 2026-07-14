"""finger_ml.audit — Inspect dataset coverage and feature quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from finger_ml.labels import BACKGROUND_LABEL, LABEL_NAMES, NUM_CLASSES

DEFAULT_TARGET_EVENTS_PER_CLASS = 500
DEFAULT_MIN_SUBJECTS = 5
DEFAULT_MIN_SESSIONS_PER_SUBJECT = 3
DEFAULT_MIN_QUALITY = 90.0
DEFAULT_MIN_VALID_RATE = 0.98
DEFAULT_MAX_BG_JITTER_P95 = 0.06
DEFAULT_MAX_DETECT_GAP_FRAMES = 5


def _subject_from_stem(stem: str) -> str:
    return stem.rsplit("_", 1)[-1] if "_" in stem else ""


def _label_stats(label_path: Path) -> dict:
    with label_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    durations = []
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    event_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    fps = float(meta.get("fps", 0) or 0)
    for ann in meta.get("annotations", []):
        label = int(ann["label"])
        frames = max(0, int(ann["end_frame"]) - int(ann["start_frame"]) + 1)
        counts[label] += frames
        event_counts[label] += 1
        durations.append(frames / fps if fps > 0 else 0.0)
    return {
        "fps": fps,
        "n_annotations": len(meta.get("annotations", [])),
        "mean_event_sec": float(np.mean(durations)) if durations else 0.0,
        "label_frames": counts.tolist(),
        "event_counts": event_counts.tolist(),
        "source_fps": meta.get("source_fps"),
        "duplicated_frames": meta.get("duplicated_frames", 0),
    }


def _feature_stats(feature_path: Path) -> dict:
    with np.load(feature_path) as data:
        labels = data["labels"].astype(np.int64)
        valid = data["valid"].astype(bool)
        arr = data["features"] if "features" in data else data["landmarks"]
        fps = float(data["fps"])
        quality = None
        if "quality_json" in data:
            raw_quality = data["quality_json"]
            quality = json.loads(str(raw_quality.item() if raw_quality.shape == () else raw_quality))
        ignored_train_frames = int((~data["train_mask"].astype(bool)).sum()) if "train_mask" in data else 0
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    row = {
        "frames": int(len(labels)),
        "fps": fps,
        "valid_rate": float(valid.mean()) if len(valid) else 0.0,
        "nodes": int(arr.shape[1]),
        "feature_channels": int(arr.shape[2]),
        "counts": counts.tolist(),
        "gesture_frames": int(counts[:BACKGROUND_LABEL].sum()),
        "background_frames": int(counts[BACKGROUND_LABEL]),
        "ignored_train_frames": ignored_train_frames,
    }
    if quality is not None:
        row["quality"] = quality
    return row


def audit(data_dir: Path) -> dict:
    video_map = {p.stem: p for p in (data_dir / "video").glob("*.mp4")}
    label_map = {p.stem: p for p in (data_dir / "labels").glob("*.json")}
    feature_map = {p.stem: p for p in (data_dir / "features").glob("*.npz")}
    stems = sorted(set(video_map) | set(label_map) | set(feature_map))

    sessions = []
    subject_totals: dict[str, np.ndarray] = {}
    label_subject_totals: dict[str, np.ndarray] = {}
    sessions_per_subject: dict[str, int] = {}
    label_event_totals = np.zeros(NUM_CLASSES, dtype=np.int64)
    for stem in stems:
        subject = _subject_from_stem(stem)
        row = {
            "session": stem,
            "subject": subject,
            "has_video": stem in video_map,
            "has_label": stem in label_map,
            "has_feature": stem in feature_map,
        }
        if stem in label_map:
            label = _label_stats(label_map[stem])
            row["label"] = label
            events = np.array(label["event_counts"], dtype=np.int64)
            label_event_totals += events
            label_subject_totals.setdefault(subject, np.zeros(NUM_CLASSES, dtype=np.int64))
            label_subject_totals[subject] += events
            sessions_per_subject[subject] = sessions_per_subject.get(subject, 0) + 1
        if stem in feature_map:
            feat = _feature_stats(feature_map[stem])
            row["feature"] = feat
            subject_totals.setdefault(row["subject"], np.zeros(NUM_CLASSES, dtype=np.int64))
            subject_totals[row["subject"]] += np.array(feat["counts"], dtype=np.int64)
        sessions.append(row)

    totals = {
        "videos": len(video_map),
        "labels": len(label_map),
        "features": len(feature_map),
        "missing_labels": sorted(set(video_map) - set(label_map)),
        "missing_videos": sorted(set(label_map) - set(video_map)),
        "missing_features": sorted((set(video_map) & set(label_map)) - set(feature_map)),
        "subjects": {
            subject: dict(zip(LABEL_NAMES, counts.tolist()))
            for subject, counts in sorted(subject_totals.items())
        },
        "label_events": dict(zip(LABEL_NAMES, label_event_totals.tolist())),
        "label_subject_events": {
            subject: dict(zip(LABEL_NAMES, counts.tolist()))
            for subject, counts in sorted(label_subject_totals.items())
        },
        "sessions_per_subject": dict(sorted(sessions_per_subject.items())),
    }
    return {"totals": totals, "sessions": sessions}


def evaluate_readiness(
    report: dict,
    target_events_per_class: int = DEFAULT_TARGET_EVENTS_PER_CLASS,
    min_subjects: int = DEFAULT_MIN_SUBJECTS,
    min_sessions_per_subject: int = DEFAULT_MIN_SESSIONS_PER_SUBJECT,
    min_quality: float = DEFAULT_MIN_QUALITY,
    min_valid_rate: float = DEFAULT_MIN_VALID_RATE,
    max_bg_jitter_p95: float = DEFAULT_MAX_BG_JITTER_P95,
    max_detect_gap_frames: int = DEFAULT_MAX_DETECT_GAP_FRAMES,
) -> dict:
    totals = report["totals"]
    failures = []
    bad_sessions = []

    for stem in totals["missing_labels"]:
        failures.append(f"{stem}: video exists but label is missing")
    for stem in totals["missing_videos"]:
        failures.append(f"{stem}: label exists but video is missing")
    for stem in totals["missing_features"]:
        failures.append(f"{stem}: feature is missing; run finger-preprocess")

    label_events = totals["label_events"]
    class_events = {}
    for name in LABEL_NAMES[:BACKGROUND_LABEL]:
        count = int(label_events.get(name, 0))
        passed = count >= target_events_per_class
        class_events[name] = {
            "count": count,
            "target": target_events_per_class,
            "passed": passed,
        }
        if not passed:
            failures.append(f"{name}: events {count} < {target_events_per_class}")

    n_subjects = len(totals["label_subject_events"])
    if n_subjects < min_subjects:
        failures.append(f"subjects {n_subjects} < {min_subjects}")

    subject_sessions = {}
    for subject, n_sessions in totals["sessions_per_subject"].items():
        n_sessions = int(n_sessions)
        passed = n_sessions >= min_sessions_per_subject
        subject_sessions[subject] = {
            "count": n_sessions,
            "target": min_sessions_per_subject,
            "passed": passed,
        }
        if not passed:
            failures.append(f"{subject}: sessions {n_sessions} < {min_sessions_per_subject}")

    for row in report["sessions"]:
        reasons = []
        session = row["session"]
        if not row["has_video"]:
            reasons.append("missing video")
        if not row["has_label"]:
            reasons.append("missing label")
        if not row["has_feature"]:
            reasons.append("missing feature")

        feat = row.get("feature")
        if feat:
            if feat["nodes"] != 21:
                reasons.append(f"nodes {feat['nodes']} != 21; run finger-preprocess --force")
            if feat["feature_channels"] < 12:
                reasons.append(f"feature channels {feat['feature_channels']} < 12")

            quality = feat.get("quality")
            if quality is None:
                reasons.append("missing quality_json; run finger-preprocess --force")
            else:
                q = float(quality.get("quality_score", 0.0))
                valid_rate = float(quality.get("valid_rate", feat.get("valid_rate", 0.0)))
                bg_jitter = float(quality.get("background_jitter_p95", 0.0))
                max_gap = int(quality.get("longest_detect_fail_run_frames", 0))
                if q < min_quality:
                    reasons.append(f"quality {q:.1f} < {min_quality:.1f}")
                if valid_rate < min_valid_rate:
                    reasons.append(f"valid_rate {valid_rate:.1%} < {min_valid_rate:.1%}")
                if bg_jitter > max_bg_jitter_p95:
                    reasons.append(f"background_jitter_p95 {bg_jitter:.4f} > {max_bg_jitter_p95:.4f}")
                if max_gap > max_detect_gap_frames:
                    reasons.append(f"max detect gap {max_gap}f > {max_detect_gap_frames}f")

        if reasons:
            bad_sessions.append(
                {
                    "session": session,
                    "subject": row.get("subject", ""),
                    "reasons": reasons,
                }
            )

    readiness = {
        "passed": not failures and not bad_sessions,
        "failures": failures,
        "bad_sessions": bad_sessions,
        "thresholds": {
            "target_events_per_class": target_events_per_class,
            "min_subjects": min_subjects,
            "min_sessions_per_subject": min_sessions_per_subject,
            "min_quality": min_quality,
            "min_valid_rate": min_valid_rate,
            "max_bg_jitter_p95": max_bg_jitter_p95,
            "max_detect_gap_frames": max_detect_gap_frames,
        },
        "class_events": class_events,
        "subject_sessions": subject_sessions,
        "subject_count": {
            "count": n_subjects,
            "target": min_subjects,
            "passed": n_subjects >= min_subjects,
        },
    }
    report["readiness"] = readiness
    return readiness


def _print_readiness(readiness: dict) -> None:
    print("\n[readiness]")
    for name in LABEL_NAMES[:BACKGROUND_LABEL]:
        row = readiness["class_events"][name]
        status = "OK" if row["passed"] else "LOW"
        print(f"  {name:<20} events={row['count']:>4} / {row['target']:<4} {status}")

    subject_count = readiness["subject_count"]
    subject_status = "OK" if subject_count["passed"] else "LOW"
    print(f"  subjects             {subject_count['count']:>4} / {subject_count['target']:<4} {subject_status}")

    for subject, row in readiness["subject_sessions"].items():
        status = "OK" if row["passed"] else "LOW"
        print(f"  {subject:<20} sessions={row['count']:>4} / {row['target']:<4} {status}")

    gate = "PASS" if readiness["passed"] else "FAIL"
    print(f"\n[gate] {gate}")
    if not readiness["passed"]:
        for failure in readiness["failures"]:
            print(f"  - {failure}")


def _print_bad_sessions(readiness: dict) -> None:
    print("\n[bad sessions]")
    if not readiness["bad_sessions"]:
        print("  none")
        return
    for row in readiness["bad_sessions"]:
        print(f"  {row['session']} ({row['subject']})")
        for reason in row["reasons"]:
            print(f"    - {reason}")


def main() -> None:
    ap = argparse.ArgumentParser(description="审计 finger-ml 数据覆盖与特征质量")
    ap.add_argument("--data-dir", default="data", help="数据根目录")
    ap.add_argument("--out-json", default=None, help="可选：保存完整审计 JSON")
    ap.add_argument("--target-events-per-class", type=int, default=DEFAULT_TARGET_EVENTS_PER_CLASS,
                    help="建议每类有效事件数下限（默认 500）")
    ap.add_argument("--min-subjects", type=int, default=DEFAULT_MIN_SUBJECTS,
                    help="建议 subject 数下限；只做单人右手模型时可设为 1")
    ap.add_argument("--min-sessions-per-subject", type=int, default=DEFAULT_MIN_SESSIONS_PER_SUBJECT,
                    help="建议每个 subject 的 session 数下限")
    ap.add_argument("--min-quality", type=float, default=DEFAULT_MIN_QUALITY,
                    help="单 session 综合质量分下限（默认 90）")
    ap.add_argument("--min-valid-rate", type=float, default=DEFAULT_MIN_VALID_RATE,
                    help="单 session 骨架检出率下限，0-1（默认 0.98）")
    ap.add_argument("--max-bg-jitter-p95", type=float, default=DEFAULT_MAX_BG_JITTER_P95,
                    help="单 session 背景骨架抖动 p95 上限（默认 0.06）")
    ap.add_argument("--max-detect-gap-frames", type=int, default=DEFAULT_MAX_DETECT_GAP_FRAMES,
                    help="单 session 最长连续漏检帧数上限（默认 5）")
    ap.add_argument("--list-bad", action="store_true",
                    help="列出不达标 session 及原因")
    ap.add_argument("--fail-on-miss", action="store_true",
                    help="readiness 未通过时返回非零退出码，适合训练前门禁")
    ap.add_argument("--skip-readiness", action="store_true",
                    help="不打印采集量 readiness 检查")
    args = ap.parse_args()

    report = audit(Path(args.data_dir))
    readiness = evaluate_readiness(
        report,
        target_events_per_class=args.target_events_per_class,
        min_subjects=args.min_subjects,
        min_sessions_per_subject=args.min_sessions_per_subject,
        min_quality=args.min_quality,
        min_valid_rate=args.min_valid_rate,
        max_bg_jitter_p95=args.max_bg_jitter_p95,
        max_detect_gap_frames=args.max_detect_gap_frames,
    )
    totals = report["totals"]
    print(
        f"[audit] videos={totals['videos']} labels={totals['labels']} "
        f"features={totals['features']}"
    )
    if totals["missing_features"]:
        print("[warn] 缺少 feature：")
        for stem in totals["missing_features"]:
            print(f"  {stem}")
    if totals["missing_labels"]:
        print("[warn] 有视频但无 label：")
        for stem in totals["missing_labels"]:
            print(f"  {stem}")
    if totals["missing_videos"]:
        print("[warn] 有 label 但无视频：")
        for stem in totals["missing_videos"]:
            print(f"  {stem}")

    print("\n[subjects]")
    for subject, counts in totals["subjects"].items():
        detail = ", ".join(f"{name}:{counts[name]}" for name in LABEL_NAMES)
        print(f"  {subject}: {detail}")

    if totals["label_subject_events"]:
        print("\n[label events]")
        for subject, counts in totals["label_subject_events"].items():
            detail = ", ".join(f"{name}:{counts[name]}" for name in LABEL_NAMES[:BACKGROUND_LABEL])
            print(f"  {subject}: {detail}")

    print("\n[sessions]")
    for row in report["sessions"]:
        feat = row.get("feature")
        if feat:
            print(
                f"  {row['session']}: frames={feat['frames']} "
                f"valid={feat['valid_rate']:.1%} nodes={feat['nodes']} "
                    f"ch={feat['feature_channels']} "
                    f"gesture={feat['gesture_frames']} bg={feat['background_frames']} "
                    f"ignored={feat['ignored_train_frames']}"
                )
            quality = feat.get("quality")
            if quality:
                print(
                    f"    quality={quality['quality_score']:.1f} "
                    f"jitter_p95={quality['jitter_p95']:.4f} "
                    f"bg_jitter_p95={quality['background_jitter_p95']:.4f} "
                    f"worst={quality['worst_jitter_node']} "
                    f"max_gap={quality['longest_detect_fail_run_frames']}f"
                )
            if feat["nodes"] != 21 or feat["feature_channels"] < 12:
                print("    [stale] 建议运行 finger-preprocess --force 重新生成 21 节点/12 通道特征")

    if not args.skip_readiness:
        _print_readiness(readiness)

    if args.list_bad:
        _print_bad_sessions(readiness)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[save] {out}")

    if args.fail_on_miss and not readiness["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
