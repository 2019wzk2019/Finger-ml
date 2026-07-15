"""finger_ml.audit — Inspect dataset coverage and feature quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from finger_ml.labels import BACKGROUND_LABEL, LABEL_NAMES, NUM_CLASSES

# ========== 审计默认阈值常量 ==========

# 每个手势类别建议的最少有效事件(标注片段)数量
# 低于此值可能导致该类别训练不充分,模型泛化能力不足
DEFAULT_TARGET_EVENTS_PER_CLASS = 500

# 建议的最少受试者(被采集者)人数
# 多受试者能提高模型对不同手型、手势习惯的鲁棒性
DEFAULT_MIN_SUBJECTS = 5

# 每个受试者建议的最少采集会话(session)数
# 多会话可覆盖不同时间、环境下的手势变化
DEFAULT_MIN_SESSIONS_PER_SUBJECT = 3

# 单个会话综合质量分的下限(0-100)
# 综合质量分综合了骨架检出率、抖动等指标
DEFAULT_MIN_QUALITY = 90.0

# 单个会话骨架检出率(valid_rate)的下限(0-1)
# 检出率 = 成功检测到手部关键点的帧数 / 总帧数
# 低于此值意味着大量帧缺少骨架数据,特征不连续
DEFAULT_MIN_VALID_RATE = 0.98

# 背景帧骨架抖动的 P95 百分位上限(归一化坐标)
# 背景帧应无手部动作,抖动过大说明检测不稳定或噪声高
# 0.06 约等于归一化坐标下 6% 的偏移量
DEFAULT_MAX_BG_JITTER_P95 = 0.06

# 最长连续漏检帧数上限
# 连续漏检会导致特征出现空洞,插值无法可靠补全
# 5 帧以内的短暂漏检可通过前后帧插值容忍
DEFAULT_MAX_DETECT_GAP_FRAMES = 5


def _subject_from_stem(stem: str) -> str:
    """从文件名主干中提取受试者标识。

    约定文件名格式为 '<前缀>_<受试者ID>', 例如 'session01_ZS' 提取出 'ZS'。
    若文件名中无下划线则返回空字符串。

    参数:
        stem: 文件名主干(不含扩展名)

    返回:
        受试者标识字符串,无下划线时返回空串
    """
    return stem.rsplit("_", 1)[-1] if "_" in stem else ""


def _label_stats(label_path: Path) -> dict:
    """解析单个标注 JSON 文件,统计各类别的帧数与事件数。

    遍历标注文件中的所有 annotations, 对每个标注:
    - 累计该类别占用的帧数 (end_frame - start_frame + 1)
    - 累计该类别的事件数 (每个 annotation 记为一个事件)
    - 计算事件持续时长(秒), 用于后续平均时长统计

    参数:
        label_path: 标注 JSON 文件路径

    返回:
        包含以下键的字典:
        - fps: 视频帧率
        - n_annotations: 标注条目总数
        - mean_event_sec: 平均事件持续秒数
        - label_frames: 各类别帧数列表(长度 NUM_CLASSES)
        - event_counts: 各类别事件数列表(长度 NUM_CLASSES)
        - source_fps: 原始视频帧率(可能为 None)
        - duplicated_frames: 重复帧数(用于处理可变帧率视频)
    """
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
    """解析单个特征 .npz 文件,统计帧数、类别分布与质量指标。

    从特征文件中加载 labels(每帧类别)、valid(每帧骨架是否有效)、
    features/landmarks(特征张量)以及可选的 quality_json 和 train_mask。

    算法:
    1. 加载 npz 中的 labels、valid、features/landmarks、fps
    2. 若存在 quality_json,解析为字典(包含质量分、抖动等指标)
    3. 若存在 train_mask,统计被忽略的训练帧数(边界帧标记为 IGNORE_INDEX)
    4. 用 bincount 统计各类别帧数分布

    参数:
        feature_path: 特征 .npz 文件路径

    返回:
        包含以下键的字典:
        - frames: 总帧数
        - fps: 帧率
        - valid_rate: 骨架检出率(有效帧占比)
        - nodes: 关键点节点数(应为 21,即 MediaPipe 21 个手部关键点)
        - feature_channels: 特征通道数(应 >= 12,含 x/y/z/visibility 等)
        - counts: 各类别帧数列表(长度 NUM_CLASSES)
        - gesture_frames: 手势帧总数(非背景帧)
        - background_frames: 背景帧总数
        - ignored_train_frames: 因边界裁剪被排除的训练帧数
        - quality: 质量指标字典(若存在)
    """
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
    """对数据目录执行全面审计,汇总各会话的标注与特征统计。

    扫描 data_dir 下的 video/、labels/、features/ 三个子目录,
    按文件名主干(stem)对齐三者, 逐会话收集标注统计与特征统计,
    并按受试者汇总各类别的事件数与帧数。

    算法:
    1. 分别扫描 video/、labels/、features/ 目录,建立 stem -> 路径的映射
    2. 取三者 stem 的并集,确定所有会话
    3. 对每个 stem: 提取受试者ID,若有标注则累计事件数,若有特征则累计帧数
    4. 汇总缺失文件、各受试者类别分布、每受试者会话数等全局统计

    参数:
        data_dir: 数据根目录,应包含 video/、labels/、features/ 子目录

    返回:
        包含以下键的字典:
        - totals: 全局汇总统计(文件数、缺失列表、各受试者分布等)
        - sessions: 每个会话的详细统计列表
    """
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
    """评估数据集是否已达到训练就绪状态。

    基于多维阈值对数据集进行全面门禁检查, 只有所有检查项均通过
    才判定为 ready。检查分为两个层次:

    一、全局级检查(failures 列表):
      1. 文件完整性: 视频-标注-特征三者是否一一对应
      2. 采集量充足性: 每个手势类别的事件数是否达到目标
      3. 受试者多样性: 受试者人数是否达标
      4. 会话充分性: 每个受试者的会话数是否达标

    二、会话级检查(bad_sessions 列表):
      1. 文件完整性: 该会话是否缺少视频/标注/特征
      2. 特征维度: 关键点数是否为 21, 通道数是否 >= 12
      3. 质量指标(若存在 quality_json):
         - 综合质量分是否 >= min_quality
         - 骨架检出率是否 >= min_valid_rate
         - 背景抖动 P95 是否 <= max_bg_jitter_p95
         - 最长连续漏检是否 <= max_detect_gap_frames

    参数:
        report: audit() 返回的完整审计报告
        target_events_per_class: 每类手势建议的最少事件数
        min_subjects: 建议的最少受试者数
        min_sessions_per_subject: 每个受试者建议的最少会话数
        min_quality: 单会话综合质量分下限
        min_valid_rate: 单会话骨架检出率下限(0-1)
        max_bg_jitter_p95: 背景骨架抖动 P95 上限(归一化坐标)
        max_detect_gap_frames: 最长连续漏检帧数上限

    返回:
        就绪评估字典, 包含:
        - passed: 布尔值, 是否全部通过(无 failures 且无 bad_sessions)
        - failures: 全局级失败原因列表
        - bad_sessions: 不达标会话列表(含具体原因)
        - thresholds: 本次评估使用的阈值参数
        - class_events: 各类别事件数与达标情况
        - subject_sessions: 各受试者会话数与达标情况
        - subject_count: 受试者总数与达标情况

    副作用:
        将 readiness 字典写入 report["readiness"]
    """
    totals = report["totals"]
    failures = []
    bad_sessions = []

    # --- 检查1: 文件完整性(视频/标注/特征三方对齐) ---
    for stem in totals["missing_labels"]:
        failures.append(f"{stem}: video exists but label is missing")
    for stem in totals["missing_videos"]:
        failures.append(f"{stem}: label exists but video is missing")
    for stem in totals["missing_features"]:
        failures.append(f"{stem}: feature is missing; run finger-preprocess")

    # --- 检查2: 各手势类别的事件采集量是否达标 ---
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

    # --- 检查3: 受试者总数是否达标 ---
    n_subjects = len(totals["label_subject_events"])
    if n_subjects < min_subjects:
        failures.append(f"subjects {n_subjects} < {min_subjects}")

    # --- 检查4: 每个受试者的会话数是否达标 ---
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

    # --- 检查5: 逐会话质量检查 ---
    for row in report["sessions"]:
        reasons = []
        session = row["session"]
        # 5a: 检查文件是否齐全
        if not row["has_video"]:
            reasons.append("missing video")
        if not row["has_label"]:
            reasons.append("missing label")
        if not row["has_feature"]:
            reasons.append("missing feature")

        # 5b: 检查特征维度是否正确(21 节点 / 12+ 通道)
        feat = row.get("feature")
        if feat:
            if feat["nodes"] != 21:
                reasons.append(f"nodes {feat['nodes']} != 21; run finger-preprocess --force")
            if feat["feature_channels"] < 12:
                reasons.append(f"feature channels {feat['feature_channels']} < 12")

            # 5c: 检查质量指标(综合质量分、检出率、背景抖动、连续漏检)
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
    """以可读格式打印就绪评估结果。

    输出内容包括:
    - 各手势类别的事件采集量与达标状态(OK/LOW)
    - 受试者总数与达标状态
    - 各受试者的会话数与达标状态
    - 最终门禁判定(PASS/FAIL)及失败原因

    参数:
        readiness: evaluate_readiness() 返回的评估字典
    """
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
    """打印不达标会话的详细列表及原因。

    每个不达标会话列出其 session 名称、受试者 ID,
    以及具体未通过的质量检查项(如检出率不足、抖动过大等)。

    参数:
        readiness: evaluate_readiness() 返回的评估字典
    """
    print("\n[bad sessions]")
    if not readiness["bad_sessions"]:
        print("  none")
        return
    for row in readiness["bad_sessions"]:
        print(f"  {row['session']} ({row['subject']})")
        for reason in row["reasons"]:
            print(f"    - {reason}")


def main() -> None:
    """审计命令行入口。

    执行流程:
    1. 解析命令行参数(数据目录、各阈值、输出选项)
    2. 调用 audit() 对数据目录执行全面审计
    3. 调用 evaluate_readiness() 进行就绪评估
    4. 打印审计摘要(文件数、缺失列表、受试者分布、会话详情)
    5. 可选打印就绪评估结果与不达标会话
    6. 可选将完整审计报告保存为 JSON
    7. 若启用 --fail-on-miss 且评估未通过, 以非零退出码退出
    """
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
