"""Train an offline gesture temporal segmentation model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from finger_ml.dataset import (
    IGNORE_INDEX,
    GestureSegmentationDataset,
    split_feature_files,
)
from finger_ml.labels import LABEL_NAMES, NUM_CLASSES
from finger_ml.model import GestureSegmenter


def select_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
        print(f"[info] device: cuda ({torch.cuda.get_device_name(0)})")
        return device
    if torch.backends.mps.is_available():
        print("[info] device: mps")
        return torch.device("mps")
    print("[info] device: cpu")
    return torch.device("cpu")


def class_weights(dataset: GestureSegmentationDataset) -> torch.Tensor:
    counts = dataset.class_counts().astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.from_numpy(weights.astype(np.float32))


def tmse_loss(logits: torch.Tensor, mask: torch.Tensor, tau: float = 4.0) -> torch.Tensor:
    if logits.shape[-1] < 2:
        return logits.new_tensor(0.0)
    valid = (mask[:, 1:] & mask[:, :-1]).unsqueeze(1)
    diff = F.log_softmax(logits[:, :, 1:], dim=1) - F.log_softmax(logits.detach()[:, :, :-1], dim=1)
    loss = torch.clamp(diff.pow(2), max=tau * tau)
    if not valid.any():
        return logits.new_tensor(0.0)
    return loss.masked_select(valid).mean()


def masked_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    valid = target != IGNORE_INDEX
    if not valid.any():
        return 0.0
    pred = logits.argmax(dim=1)
    return float((pred[valid] == target[valid]).float().mean().item())


def run_epoch(
    model: GestureSegmenter,
    loader: DataLoader,
    ce_loss: nn.Module,
    bce_loss: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    lambda_boundary: float,
    lambda_smooth: float,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    with torch.set_grad_enabled(is_train):
        for x, y, boundary, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            boundary = boundary.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            logits, boundary_logits, stages = model(x)
            loss = sum(ce_loss(stage, y) for stage in stages) / len(stages)
            if lambda_boundary > 0:
                b_loss = bce_loss(boundary_logits, boundary)
                loss = loss + lambda_boundary * b_loss.masked_select(mask.unsqueeze(1)).mean()
            if lambda_smooth > 0:
                loss = loss + lambda_smooth * tmse_loss(logits, mask)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss += float(loss.item())
            total_acc += masked_accuracy(logits, y)
            batches += 1
    return total_loss / max(1, batches), total_acc / max(1, batches)


@torch.no_grad()
def per_class_recall(model: GestureSegmenter, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    total = np.zeros(NUM_CLASSES, dtype=np.int64)
    for x, y, _, _ in loader:
        logits, _, _ = model(x.to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
        target = y.numpy()
        valid = target != IGNORE_INDEX
        for c in range(NUM_CLASSES):
            mask = (target == c) & valid
            correct[c] += int((pred[mask] == c).sum())
            total[c] += int(mask.sum())
    return {
        LABEL_NAMES[c]: (float(correct[c] / total[c]) if total[c] else float("nan"))
        for c in range(NUM_CLASSES)
    }


def train(args: argparse.Namespace) -> None:
    device = select_device()
    feature_dir = Path(args.data_dir) / "features"
    subjects = args.subjects.split(",") if args.subjects else None
    train_files, val_files = split_feature_files(feature_dir, val_ratio=args.val_ratio, subjects=subjects)
    train_ds = GestureSegmentationDataset(
        train_files,
        chunk_len=args.chunk_len,
        hop=args.train_hop,
        augment=True,
        boundary_radius=args.boundary_radius,
    )
    val_ds = GestureSegmentationDataset(
        val_files,
        chunk_len=args.chunk_len,
        hop=args.val_hop,
        augment=False,
        boundary_radius=args.boundary_radius,
    )
    input_channels = max(train_ds.input_channels, val_ds.input_channels)
    train_ds.input_channels = input_channels
    val_ds.input_channels = input_channels
    print(train_ds)
    print(val_ds)
    print(f"[info] input_channels={input_channels}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    model = GestureSegmenter(
        input_channels=input_channels,
        hidden_dim=args.hidden_dim,
        temporal_channels=args.temporal_channels,
        temporal_layers=args.temporal_layers,
        temporal_stages=args.temporal_stages,
        dropout=args.dropout,
    ).to(device)
    print(f"[info] parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    ce = nn.CrossEntropyLoss(weight=class_weights(train_ds).to(device), ignore_index=IGNORE_INDEX)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([8.0, 8.0], device=device).view(1, 2, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=6, factor=0.5)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(
            model,
            train_loader,
            ce,
            bce,
            optimizer,
            device,
            lambda_boundary=args.lambda_boundary,
            lambda_smooth=args.lambda_smooth,
        )
        va_loss, va_acc = run_epoch(
            model,
            val_loader,
            ce,
            bce,
            None,
            device,
            lambda_boundary=args.lambda_boundary,
            lambda_smooth=args.lambda_smooth,
        )
        scheduler.step(va_acc)
        row = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_frame_acc": round(tr_acc, 4),
            "val_loss": round(va_loss, 4),
            "val_frame_acc": round(va_acc, 4),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        marker = ""
        if va_acc > best_score:
            best_score = va_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_frame_acc": va_acc,
                    "args": vars(args) | {"input_channels": input_channels},
                },
                ckpt_dir / "best.pt",
            )
            marker = "  <- best"
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"loss {tr_loss:.4f}/{va_loss:.4f} "
            f"frame_acc {tr_acc:.3f}/{va_acc:.3f} "
            f"{time.time() - t0:.1f}s{marker}"
        )
        if epoch % args.report_every == 0 or epoch == args.epochs:
            for name, recall in per_class_recall(model, val_loader, device).items():
                value = "N/A" if np.isnan(recall) else f"{recall:.3f}"
                print(f"    recall {name:<20}: {value}")

    torch.save(
        {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "val_frame_acc": va_acc,
            "args": vars(args) | {"input_channels": input_channels},
        },
        ckpt_dir / "last.pt",
    )
    (ckpt_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] best val_frame_acc={best_score:.4f}")
    print(f"[done] checkpoint={ckpt_dir / 'best.pt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train offline gesture temporal segmentation model")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--subjects", default=None)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--chunk-len", type=int, default=256)
    ap.add_argument("--train-hop", type=int, default=64)
    ap.add_argument("--val-hop", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--temporal-channels", type=int, default=128)
    ap.add_argument("--temporal-layers", type=int, default=6)
    ap.add_argument("--temporal-stages", type=int, default=2)
    ap.add_argument("--boundary-radius", type=int, default=2)
    ap.add_argument("--lambda-boundary", type=float, default=0.2)
    ap.add_argument("--lambda-smooth", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--report-every", type=int, default=10)
    train(ap.parse_args())


if __name__ == "__main__":
    main()
