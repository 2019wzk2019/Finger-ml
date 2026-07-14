"""finger_ml.eval_events — Event-level evaluation for finger-detect predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from finger_ml.labels import BACKGROUND_LABEL, LABEL_NAMES, NUM_CLASSES

TARGET_EVENT_F1 = 0.98
TARGET_CLASS_PRECISION = 0.97
TARGET_CLASS_RECALL = 0.97


def _load_gt(label_path: Path) -> list[dict]:
    with label_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    events = []
    for ann in meta.get("annotations", []):
        events.append(
            {
                "label": int(ann["label"]),
                "start_frame": int(ann["start_frame"]) - 1,
                "end_frame": int(ann["end_frame"]) - 1,
            }
        )
    return events


def _load_pred(pred_path: Path) -> list[dict]:
    with pred_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    events = []
    for ev in payload.get("events", []):
        events.append(
            {
                "label": int(ev["label"]),
                "start_frame": int(ev["start_frame"]),
                "end_frame": int(ev["end_frame"]),
                "mean_conf": float(ev.get("mean_conf", 0.0)),
            }
        )
    return events


def _iou(a: dict, b: dict) -> float:
    lo = max(a["start_frame"], b["start_frame"])
    hi = min(a["end_frame"], b["end_frame"])
    inter = max(0, hi - lo + 1)
    union = (
        max(0, a["end_frame"] - a["start_frame"] + 1)
        + max(0, b["end_frame"] - b["start_frame"] + 1)
        - inter
    )
    return inter / union if union > 0 else 0.0


def evaluate(gt_events: list[dict], pred_events: list[dict], iou_threshold: float) -> dict:
    matched_pred: set[int] = set()
    rows = []
    class_tp = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_fp = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_fn = np.zeros(NUM_CLASSES, dtype=np.int64)
    timing_errors = []

    for gi, gt in enumerate(gt_events):
        best = (-1, 0.0)
        for pi, pred in enumerate(pred_events):
            if pi in matched_pred or pred["label"] != gt["label"]:
                continue
            score = _iou(gt, pred)
            if score > best[1]:
                best = (pi, score)
        if best[0] >= 0 and best[1] >= iou_threshold:
            pred = pred_events[best[0]]
            matched_pred.add(best[0])
            class_tp[gt["label"]] += 1
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
            class_fn[gt["label"]] += 1

    for pi, pred in enumerate(pred_events):
        if pi not in matched_pred:
            class_fp[pred["label"]] += 1

    per_class = {}
    for c, name in enumerate(LABEL_NAMES):
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

    tp = int(class_tp[:BACKGROUND_LABEL].sum())
    fp = int(class_fp[:BACKGROUND_LABEL].sum())
    fn = int(class_fn[:BACKGROUND_LABEL].sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    confusion = _event_confusion(gt_events, pred_events, iou_threshold)

    return {
        "overall": {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1},
        "per_class": per_class,
        "confusion": confusion,
        "matches": rows,
        "timing_errors": timing_errors,
    }


def _event_confusion(gt_events: list[dict], pred_events: list[dict], iou_threshold: float) -> dict:
    """Match events by time overlap regardless of label to expose class confusions."""
    matched_pred: set[int] = set()
    matrix = np.zeros((BACKGROUND_LABEL, BACKGROUND_LABEL), dtype=np.int64)
    missed = np.zeros(BACKGROUND_LABEL, dtype=np.int64)
    spurious = np.zeros(BACKGROUND_LABEL, dtype=np.int64)

    for gt in gt_events:
        gt_label = int(gt["label"])
        if not 0 <= gt_label < BACKGROUND_LABEL:
            continue

        best = (-1, 0.0)
        for pi, pred in enumerate(pred_events):
            if pi in matched_pred:
                continue
            pred_label = int(pred["label"])
            if not 0 <= pred_label < BACKGROUND_LABEL:
                continue
            score = _iou(gt, pred)
            if score > best[1]:
                best = (pi, score)

        if best[0] >= 0 and best[1] >= iou_threshold:
            pred_label = int(pred_events[best[0]]["label"])
            matched_pred.add(best[0])
            matrix[gt_label, pred_label] += 1
        else:
            missed[gt_label] += 1

    for pi, pred in enumerate(pred_events):
        if pi in matched_pred:
            continue
        pred_label = int(pred["label"])
        if 0 <= pred_label < BACKGROUND_LABEL:
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
    failures = []
    overall = report["overall"]
    if overall["f1"] < target_f1:
        failures.append(f"overall F1 {overall['f1']:.3f} < {target_f1:.3f}")

    for name, row in report["per_class"].items():
        if row["precision"] < target_class_precision:
            failures.append(
                f"{name} precision {row['precision']:.3f} < {target_class_precision:.3f}"
            )
        if row["recall"] < target_class_recall:
            failures.append(f"{name} recall {row['recall']:.3f} < {target_class_recall:.3f}")

    return not failures, failures


def print_report(report: dict) -> None:
    overall = report["overall"]
    print(
        f"[event] P={overall['precision']:.3f} R={overall['recall']:.3f} "
        f"F1={overall['f1']:.3f} TP={overall['tp']} FP={overall['fp']} FN={overall['fn']}"
    )
    for name, row in report["per_class"].items():
        print(
            f"  {name:<20} P={row['precision']:.3f} R={row['recall']:.3f} "
            f"F1={row['f1']:.3f} TP={row['tp']} FP={row['fp']} FN={row['fn']}"
        )

    confusion = report["confusion"]
    print("\n[confusion] rows=GT cols=pred")
    header = "GT\\P".ljust(20) + " ".join(name[:6].rjust(6) for name in confusion["labels"])
    print(header)
    for name, row, missed in zip(confusion["labels"], confusion["matrix"], confusion["missed"]):
        cells = " ".join(str(v).rjust(6) for v in row)
        print(f"  {name:<18}{cells}  missed={missed}")
    spurious = ", ".join(
        f"{name}:{count}" for name, count in zip(confusion["labels"], confusion["spurious"]) if count
    )
    print(f"  spurious: {spurious or 'none'}")

    if report["timing_errors"]:
        starts = np.array([e["start_error_frames"] for e in report["timing_errors"]])
        ends = np.array([e["end_error_frames"] for e in report["timing_errors"]])
        ious = np.array([e["iou"] for e in report["timing_errors"]])
        print(
            f"\n[timing] start_err={starts.mean():.1f}+/-{starts.std():.1f} frames  "
            f"end_err={ends.mean():.1f}+/-{ends.std():.1f} frames  IoU={ious.mean():.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="事件级评估 finger-detect 输出")
    ap.add_argument("--label", required=True, help="采集 label JSON")
    ap.add_argument("--pred-json", required=True, help="finger-detect 输出 JSON")
    ap.add_argument("--iou-threshold", type=float, default=0.30, help="事件匹配 IoU 阈值")
    ap.add_argument("--target-f1", type=float, default=TARGET_EVENT_F1,
                    help="验收目标：overall event F1（默认 0.98）")
    ap.add_argument("--target-class-precision", type=float, default=TARGET_CLASS_PRECISION,
                    help="验收目标：每类 precision 下限（默认 0.97）")
    ap.add_argument("--target-class-recall", type=float, default=TARGET_CLASS_RECALL,
                    help="验收目标：每类 recall 下限（默认 0.97）")
    ap.add_argument("--fail-on-miss", action="store_true",
                    help="未达到验收目标时返回非零退出码，适合 CI/脚本门禁")
    ap.add_argument("--out-json", default=None, help="可选：保存评估 JSON")
    args = ap.parse_args()

    report = evaluate(_load_gt(Path(args.label)), _load_pred(Path(args.pred_json)), args.iou_threshold)
    print_report(report)

    passed, failures = check_targets(
        report,
        args.target_f1,
        args.target_class_precision,
        args.target_class_recall,
    )
    report["gate"] = {
        "passed": passed,
        "target_f1": args.target_f1,
        "target_class_precision": args.target_class_precision,
        "target_class_recall": args.target_class_recall,
        "failures": failures,
    }
    if passed:
        print(
            f"\n[gate] PASS  F1>={args.target_f1:.3f}, "
            f"class P/R>={args.target_class_precision:.3f}/{args.target_class_recall:.3f}"
        )
    else:
        print("\n[gate] FAIL")
        for failure in failures:
            print(f"  - {failure}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[save] {out}")

    if args.fail_on_miss and not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
