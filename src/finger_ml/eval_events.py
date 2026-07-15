"""finger_ml.eval_events — Event-level evaluation for finger-detect predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from finger_ml.labels import BACKGROUND_LABEL, LABEL_NAMES, NUM_CLASSES

# 验收目标阈值：整体事件 F1 需达到此值
TARGET_EVENT_F1 = 0.98
# 验收目标阈值：每类精确率下限
TARGET_CLASS_PRECISION = 0.97
# 验收目标阈值：每类召回率下限
TARGET_CLASS_RECALL = 0.97


def _load_gt(label_path: Path) -> list[dict]:
    """从标注 JSON 加载真值事件列表。

    将 JSON 中 1-indexed 的 start_frame/end_frame 转换为 0-indexed，
    以便与预测结果对齐比较。

    参数:
        label_path: 采集标注 JSON 文件路径

    返回:
        真值事件字典列表，每个字典包含 label、start_frame(0-indexed)、end_frame(0-indexed)
    """
    with label_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    events = []
    for ann in meta.get("annotations", []):
        events.append(
            {
                "label": int(ann["label"]),
                # 标注 JSON 使用 1-indexed 帧号，此处转换为 0-indexed
                "start_frame": int(ann["start_frame"]) - 1,
                "end_frame": int(ann["end_frame"]) - 1,
            }
        )
    return events


def _load_pred(pred_path: Path) -> list[dict]:
    """从预测 JSON 加载预测事件列表。

    预测 JSON 中的帧号已经是 0-indexed，无需转换。
    额外提取每个事件的平均置信度 mean_conf。

    参数:
        pred_path: finger-detect 输出的预测 JSON 文件路径

    返回:
        预测事件字典列表，每个字典包含 label、start_frame、end_frame、mean_conf
    """
    with pred_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    events = []
    for ev in payload.get("events", []):
        events.append(
            {
                "label": int(ev["label"]),
                # 预测结果已是 0-indexed，无需偏移
                "start_frame": int(ev["start_frame"]),
                "end_frame": int(ev["end_frame"]),
                "mean_conf": float(ev.get("mean_conf", 0.0)),
            }
        )
    return events


def _iou(a: dict, b: dict) -> float:
    """计算两个事件之间的帧级 IoU（交并比）。

    将事件视为帧区间 [start_frame, end_frame]（闭区间），
    先求两个区间的交集帧数和并集帧数，再相除得到 IoU。

    算法:
        intersection = max(0, min(a.end, b.end) - max(a.start, b.start) + 1)
        union = |a| + |b| - intersection
        IoU = intersection / union

    参数:
        a: 事件字典，含 start_frame 和 end_frame
        b: 事件字典，含 start_frame 和 end_frame

    返回:
        IoU 值，范围 [0.0, 1.0]；并集为 0 时返回 0.0
    """
    # 交集区间的左端点（两个起始帧的较大者）
    lo = max(a["start_frame"], b["start_frame"])
    # 交集区间的右端点（两个结束帧的较小者）
    hi = min(a["end_frame"], b["end_frame"])
    # 交集帧数（闭区间，所以 +1）
    inter = max(0, hi - lo + 1)
    # 并集帧数 = 两个区间长度之和 - 交集
    union = (
        max(0, a["end_frame"] - a["start_frame"] + 1)
        + max(0, b["end_frame"] - b["start_frame"] + 1)
        - inter
    )
    return inter / union if union > 0 else 0.0


def evaluate(gt_events: list[dict], pred_events: list[dict], iou_threshold: float) -> dict:
    """基于 IoU 的事件级匹配评估核心函数。

    匹配算法（贪心匈牙利策略）：
        对每个真值事件，在所有同类别的未匹配预测事件中寻找 IoU 最大的，
        若 IoU >= iou_threshold 则匹配成功（TP），否则该真值事件为漏检（FN）。
        所有未被匹配的预测事件为误检（FP）。

    对匹配成功的事件对，计算起止帧的时间误差用于分析定位精度。

    参数:
        gt_events: 真值事件列表（0-indexed 帧号）
        pred_events: 预测事件列表（0-indexed 帧号）
        iou_threshold: 事件匹配所需的最低 IoU 阈值

    返回:
        评估报告字典，包含:
            overall   — 汇总的 TP/FP/FN/precision/recall/F1
            per_class — 每个手势类别的 TP/FP/FN/precision/recall/F1
            confusion — 混淆矩阵（行=真值，列=预测）、漏检数、误检数
            matches   — 匹配详情列表（真值索引、预测索引、标签、IoU）
            timing_errors — 匹配事件对的起止帧误差详情
    """
    # 已匹配的预测事件索引集合，确保一个预测只被一个真值匹配
    matched_pred: set[int] = set()
    # 匹配详情行
    rows = []
    # 每类的 TP/FP/FN 计数器
    class_tp = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_fp = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_fn = np.zeros(NUM_CLASSES, dtype=np.int64)
    # 匹配成功事件对的时间误差记录
    timing_errors = []

    # 第一轮：遍历每个真值事件，寻找最佳同类预测匹配
    for gi, gt in enumerate(gt_events):
        # best = (预测索引, 最大IoU)，初始为未匹配状态
        best = (-1, 0.0)
        for pi, pred in enumerate(pred_events):
            # 跳过已匹配的预测，以及类别不同的预测（同类匹配策略）
            if pi in matched_pred or pred["label"] != gt["label"]:
                continue
            score = _iou(gt, pred)
            if score > best[1]:
                best = (pi, score)

        # 匹配成功条件：找到了候选预测 且 IoU 达到阈值
        if best[0] >= 0 and best[1] >= iou_threshold:
            pred = pred_events[best[0]]
            # 标记该预测已被占用，不再参与后续匹配
            matched_pred.add(best[0])
            # 该类别 TP 计数 +1
            class_tp[gt["label"]] += 1
            # 记录时间误差：预测起止帧与真值的偏差（正值=预测偏晚，负值=偏早）
            timing_errors.append(
                {
                    "label": gt["label"],
                    "start_error_frames": pred["start_frame"] - gt["start_frame"],
                    "end_error_frames": pred["end_frame"] - gt["end_frame"],
                    "iou": best[1],
                }
            )
            rows.append({"gt": gi, "pred": best[0], "label": gt["label"], "iou": best[1]})
        else:
            # 未匹配到合格预测 → 漏检（FN）
            class_fn[gt["label"]] += 1

    # 第二轮：未被匹配的预测事件均为误检（FP）
    for pi, pred in enumerate(pred_events):
        if pi not in matched_pred:
            class_fp[pred["label"]] += 1

    # 计算每个手势类别的 precision / recall / F1
    per_class = {}
    for c, name in enumerate(LABEL_NAMES):
        # 跳过背景类，不纳入手势评估
        if c == BACKGROUND_LABEL:
            continue
        tp, fp, fn = int(class_tp[c]), int(class_fp[c]), int(class_fn[c])
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class[name] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # 汇总所有手势类别的 TP/FP/FN（排除背景类）
    tp = int(class_tp[:BACKGROUND_LABEL].sum())
    fp = int(class_fp[:BACKGROUND_LABEL].sum())
    fn = int(class_fn[:BACKGROUND_LABEL].sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    # 计算跨类别混淆矩阵（不考虑标签是否一致，仅按时间重叠匹配）
    confusion = _event_confusion(gt_events, pred_events, iou_threshold)

    return {
        "overall": {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1},
        "per_class": per_class,
        "confusion": confusion,
        "matches": rows,
        "timing_errors": timing_errors,
    }


def _event_confusion(gt_events: list[dict], pred_events: list[dict], iou_threshold: float) -> dict:
    """Match events by time overlap regardless of label to expose class confusions.

    计算跨类别混淆矩阵。与 evaluate() 中的同类匹配不同，此处按时间重叠
    匹配时忽略标签约束，从而揭示"真值 A 被误识别为 B"的混淆情况。

    混淆矩阵 matrix[gt_label, pred_label] 记录了真值类别 gt_label
    被预测为 pred_label 的次数。此外还统计了漏检数（missed）和误检数（spurious）。

    参数:
        gt_events: 真值事件列表
        pred_events: 预测事件列表
        iou_threshold: 匹配所需最低 IoU 阈值

    返回:
        字典，包含:
            labels   — 类别名称列表（仅手势类别，不含背景）
            matrix   — 混淆矩阵 (N x N)，行=真值，列=预测
            missed   — 每个真值类别的漏检事件数
            spurious — 每个预测类别的误检事件数
    """
    # 已匹配预测索引集合
    matched_pred: set[int] = set()
    # 混淆矩阵：行=真值标签，列=预测标签（仅手势类别，不含背景）
    matrix = np.zeros((BACKGROUND_LABEL, BACKGROUND_LABEL), dtype=np.int64)
    # 漏检计数：真值事件未被任何预测匹配
    missed = np.zeros(BACKGROUND_LABEL, dtype=np.int64)
    # 误检计数：预测事件未匹配到任何真值
    spurious = np.zeros(BACKGROUND_LABEL, dtype=np.int64)

    # 遍历每个真值事件，按时间重叠寻找最佳预测（不限制类别）
    for gt in gt_events:
        gt_label = int(gt["label"])
        # 忽略超出有效类别范围的标签
        if not 0 <= gt_label < BACKGROUND_LABEL:
            continue

        best = (-1, 0.0)
        for pi, pred in enumerate(pred_events):
            # 跳过已匹配的预测
            if pi in matched_pred:
                continue
            pred_label = int(pred["label"])
            # 忽略超出有效类别范围的预测标签
            if not 0 <= pred_label < BACKGROUND_LABEL:
                continue
            # 计算时间 IoU（不考虑标签是否一致）
            score = _iou(gt, pred)
            if score > best[1]:
                best = (pi, score)

        # 匹配成功：在混淆矩阵对应位置 +1
        if best[0] >= 0 and best[1] >= iou_threshold:
            pred_label = int(pred_events[best[0]]["label"])
            matched_pred.add(best[0])
            # matrix[真值标签, 预测标签] += 1，对角线为正确分类
            matrix[gt_label, pred_label] += 1
        else:
            # 漏检：真值事件无任何时间重叠达标的预测
            missed[gt_label] += 1

    # 遍历未被匹配的预测，统计误检
    for pi, pred in enumerate(pred_events):
        if pi in matched_pred:
            continue
        pred_label = int(pred["label"])
        if 0 <= pred_label < BACKGROUND_LABEL:
            # 该预测没有对应的真值事件 → 误检
            spurious[pred_label] += 1

    labels = list(LABEL_NAMES[:BACKGROUND_LABEL])
    return {
        "labels": labels,
        "matrix": matrix.tolist(),
        "missed": missed.tolist(),
        "spurious": spurious.tolist(),
    }


def check_targets(
    report: dict,
    target_f1: float,
    target_class_precision: float,
    target_class_recall: float,
) -> tuple[bool, list[str]]:
    """验收门禁检查：判断评估结果是否达到预设目标。

    检查逻辑：
        1. 整体事件 F1 是否 >= target_f1
        2. 每个手势类别的 precision 是否 >= target_class_precision
        3. 每个手势类别的 recall 是否 >= target_class_recall
    任一条件不满足即视为验收失败，返回所有不满足条件的具体描述。

    参数:
        report: evaluate() 返回的评估报告字典
        target_f1: 整体事件 F1 的验收下限
        target_class_precision: 每类精确率的验收下限
        target_class_recall: 每类召回率的验收下限

    返回:
        (是否通过, 失败原因列表) — 通过时列表为空
    """
    failures = []
    overall = report["overall"]
    # 检查整体 F1
    if overall["f1"] < target_f1:
        failures.append(f"overall F1 {overall['f1']:.3f} < {target_f1:.3f}")

    # 逐类检查 precision 和 recall
    for name, row in report["per_class"].items():
        if row["precision"] < target_class_precision:
            failures.append(
                f"{name} precision {row['precision']:.3f} < {target_class_precision:.3f}"
            )
        if row["recall"] < target_class_recall:
            failures.append(f"{name} recall {row['recall']:.3f} < {target_class_recall:.3f}")

    # 无失败项则通过
    return not failures, failures


def print_report(report: dict) -> None:
    """将评估报告以可读格式打印到标准输出。

    输出包含:
        - 整体事件级 P/R/F1/TP/FP/FN
        - 每类别的 P/R/F1/TP/FP/FN
        - 混淆矩阵（行=真值，列=预测）及漏检/误检统计
        - 匹配事件对的起止帧时间误差统计（均值+标准差）

    参数:
        report: evaluate() 返回的评估报告字典
    """
    overall = report["overall"]
    # 打印整体指标
    print(
        f"[event] P={overall['precision']:.3f} R={overall['recall']:.3f} "
        f"F1={overall['f1']:.3f} TP={overall['tp']} FP={overall['fp']} FN={overall['fn']}"
    )
    # 打印每类别指标
    for name, row in report["per_class"].items():
        print(
            f"  {name:<20} P={row['precision']:.3f} R={row['recall']:.3f} "
            f"F1={row['f1']:.3f} TP={row['tp']} FP={row['fp']} FN={row['fn']}"
        )

    # 打印混淆矩阵
    confusion = report["confusion"]
    print("\n[confusion] rows=GT cols=pred")
    # 表头：类别名缩写
    header = "GT\\P".ljust(20) + " ".join(name[:6].rjust(6) for name in confusion["labels"])
    print(header)
    # 每行：真值类别名 + 各预测类别的计数 + 漏检数
    for name, row, missed in zip(confusion["labels"], confusion["matrix"], confusion["missed"]):
        cells = " ".join(str(v).rjust(6) for v in row)
        print(f"  {name:<18}{cells}  missed={missed}")
    # 误检统计
    spurious = ", ".join(
        f"{name}:{count}" for name, count in zip(confusion["labels"], confusion["spurious"]) if count
    )
    print(f"  spurious: {spurious or 'none'}")

    # 打印时间误差统计（仅在有匹配事件时显示）
    if report["timing_errors"]:
        # 起始帧误差：正值=预测起始偏晚，负值=偏早
        starts = np.array([e["start_error_frames"] for e in report["timing_errors"]])
        # 结束帧误差：正值=预测结束偏晚，负值=偏早
        ends = np.array([e["end_error_frames"] for e in report["timing_errors"]])
        ious = np.array([e["iou"] for e in report["timing_errors"]])
        print(
            f"\n[timing] start_err={starts.mean():.1f}+/-{starts.std():.1f} frames  "
            f"end_err={ends.mean():.1f}+/-{ends.std():.1f} frames  IoU={ious.mean():.3f}"
        )


def main() -> None:
    """CLI 入口：加载真值和预测，执行事件级评估，可选验收门禁。"""
    ap = argparse.ArgumentParser(description="事件级评估 finger-detect 输出")
    ap.add_argument("--label", required=True, help="采集 label JSON")
    ap.add_argument("--pred-json", required=True, help="finger-detect 输出 JSON")
    # IoU 匹配阈值，低于此值的事件对不被视为匹配
    ap.add_argument("--iou-threshold", type=float, default=0.30, help="事件匹配 IoU 阈值")
    # 以下三个参数为验收门禁的目标阈值
    ap.add_argument("--target-f1", type=float, default=TARGET_EVENT_F1,
                    help="验收目标：overall event F1（默认 0.98）")
    ap.add_argument("--target-class-precision", type=float, default=TARGET_CLASS_PRECISION,
                    help="验收目标：每类 precision 下限（默认 0.97）")
    ap.add_argument("--target-class-recall", type=float, default=TARGET_CLASS_RECALL,
                    help="验收目标：每类 recall 下限（默认 0.97）")
    # --fail-on-miss 用于 CI 门禁：验收不通过时返回非零退出码
    ap.add_argument("--fail-on-miss", action="store_true",
                    help="未达到验收目标时返回非零退出码，适合 CI/脚本门禁")
    ap.add_argument("--out-json", default=None, help="可选：保存评估 JSON")
    args = ap.parse_args()

    # 加载真值和预测，执行评估
    report = evaluate(_load_gt(Path(args.label)), _load_pred(Path(args.pred_json)), args.iou_threshold)
    print_report(report)

    # 执行验收门禁检查
    passed, failures = check_targets(
        report,
        args.target_f1,
        args.target_class_precision,
        args.target_class_recall,
    )
    # 将门禁结果写入报告
    report["gate"] = {
        "passed": passed,
        "target_f1": args.target_f1,
        "target_class_precision": args.target_class_precision,
        "target_class_recall": args.target_class_recall,
        "failures": failures,
    }
    # 打印门禁结论
    if passed:
        print(
            f"\n[gate] PASS  F1>={args.target_f1:.3f}, "
            f"class P/R>={args.target_class_precision:.3f}/{args.target_class_recall:.3f}"
        )
    else:
        print("\n[gate] FAIL")
        for failure in failures:
            print(f"  - {failure}")

    # 可选：将完整评估报告保存为 JSON
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[save] {out}")

    # CI 门禁模式：验收不通过时以非零退出码退出
    if args.fail_on_miss and not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
