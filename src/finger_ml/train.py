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
    """自动选择最优计算设备。

    优先级: CUDA > MPS (Apple Silicon) > CPU。
    若选择 CUDA，会启用 cuDNN benchmark 模式以加速卷积运算。

    Returns:
        torch.device: 选中的计算设备
    """
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
    """根据类别频率计算带权重的类别权重向量，用于交叉熵损失。

    算法:
        1. 统计每个类别的样本数量
        2. 将零计数替换为 1.0，避免除零
        3. 权重 = 1 / sqrt(计数)，使用平方根而非倒数，
           防止极稀少类别获得过大权重导致训练不稳定
        4. 归一化权重，使其总和等于 NUM_CLASSES，
           保证加权损失与无权损失的量级一致

    Args:
        dataset: 手势分割数据集，需实现 class_counts() 方法

    Returns:
        torch.Tensor: 形状为 (NUM_CLASSES,) 的 float32 权重张量
    """
    counts = dataset.class_counts().astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.from_numpy(weights.astype(np.float32))


def tmse_loss(logits: torch.Tensor, mask: torch.Tensor, tau: float = 4.0) -> torch.Tensor:
    """截断均方平滑损失 (Truncated Mean Squared Error Smoothness Loss)。

    作用: 惩罚相邻帧之间预测类别的剧烈跳变，鼓励模型输出在时间维度上平滑过渡。
    这是时序分割任务中关键的辅助损失，因为手势动作在时间上具有连续性，
    不应出现帧级别的剧烈跳变。

    算法:
        1. 找到相邻两帧都有效的位置 (mask[:, 1:] & mask[:, :-1])
        2. 计算相邻帧 log_softmax 概率分布的差异
           (使用 detach 截断梯度，仅对当前帧梯度传播，防止双向耦合)
        3. 对差异取平方并用 tau 进行截断，防止个别异常帧产生过大梯度
        4. 在有效位置上取均值

    Args:
        logits: 模型输出 logits, 形状 (B, C, T)，C 为类别数，T 为时间步数
        mask: 有效帧掩码, 形状 (B, T)，True 表示该帧有效
        tau: 截断阈值，差异平方超过 tau^2 时截断。默认 4.0

    Returns:
        torch.Tensor: 标量损失值；若无有效帧则返回 0.0
    """
    if logits.shape[-1] < 2:
        return logits.new_tensor(0.0)
    valid = (mask[:, 1:] & mask[:, :-1]).unsqueeze(1)
    diff = F.log_softmax(logits[:, :, 1:], dim=1) - F.log_softmax(logits.detach()[:, :, :-1], dim=1)
    loss = torch.clamp(diff.pow(2), max=tau * tau)
    if not valid.any():
        return logits.new_tensor(0.0)
    return loss.masked_select(valid).mean()


def masked_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    """计算忽略 IGNORE_INDEX 帧后的分类准确率。

    仅统计有效帧（非边界/过渡帧）的预测准确率，
    因为 IGNORE_INDEX 标记的帧不参与训练，也不应计入评估指标。

    Args:
        logits: 模型输出 logits, 形状 (B, C, T)
        target: 真实标签, 形状 (B, T)，其中 IGNORE_INDEX 表示忽略的帧

    Returns:
        float: 有效帧的分类准确率，若无有效帧则返回 0.0
    """
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
    """执行一个训练或验证 epoch。

    训练循环流程:
        1. 根据是否有优化器决定训练/验证模式
        2. 前向传播: 模型输出 logits, boundary_logits, stages
        3. 计算总损失 = 主损失(CE) + 边界损失(BCE) + 平滑损失(TMSE)
           - 主损失: 多阶段深度监督 CE 损失的均值，利用模型中间层输出进行辅助监督
           - 边界损失: BCE 损失，监督手势起止边界的二分类预测
           - 平滑损失: TMSE 损失，鼓励时序预测的平滑性
        4. 训练时: 梯度裁剪 -> 参数更新
        5. 累计损失和准确率

    损失函数角色说明:
        - CE (交叉熵): 主分类损失，将每帧分类为7个类别(6种手势+背景)，
          使用类别权重缓解样本不均衡，忽略 IGNORE_INDEX 帧
        - BCE (二元交叉熵): 边界检测损失，预测每帧是否为手势起止点，
          使用 pos_weight=8 缓解正负样本不平衡(边界帧远少于非边界帧)
        - TMSE (截断均方平滑): 时序平滑正则化，防止帧间预测跳变

    Args:
        model: 手势分割模型
        loader: 数据加载器
        ce_loss: 带权重的交叉熵损失函数
        bce_loss: 带权重的二元交叉熵损失函数
        optimizer: 优化器，None 表示验证阶段
        device: 计算设备
        lambda_boundary: 边界损失权重系数
        lambda_smooth: 平滑损失权重系数

    Returns:
        tuple[float, float]: (平均损失, 平均帧级准确率)
    """
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
            # 前向传播: 获取主输出 logits、边界预测 boundary_logits、各中间阶段 stages
            logits, boundary_logits, stages = model(x)
            # 主损失: 对所有中间阶段输出计算 CE 损失并取均值 (深度监督)
            # 利用中间层监督信号加速收敛，改善梯度回传
            loss = sum(ce_loss(stage, y) for stage in stages) / len(stages)
            # 边界损失: 监督手势起止边界预测，仅在有效帧上计算
            if lambda_boundary > 0:
                b_loss = bce_loss(boundary_logits, boundary)
                loss = loss + lambda_boundary * b_loss.masked_select(mask.unsqueeze(1)).mean()
            # 平滑损失: 鼓励相邻帧的预测分布平滑过渡
            if lambda_smooth > 0:
                loss = loss + lambda_smooth * tmse_loss(logits, mask)

            if is_train:
                # 梯度裁剪: 限制梯度范数不超过 5.0，防止梯度爆炸
                # 这在 RNN/Transformer 类模型中尤其重要
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
    """计算验证集上每个类别的召回率 (Recall)。

    召回率 = 正确预测为该类的帧数 / 该类真实帧数。
    忽略标记为 IGNORE_INDEX 的帧。该指标用于诊断模型是否对
    某些手势类别存在遗漏(漏检)，是类别级评估的关键指标。

    Args:
        model: 手势分割模型
        loader: 验证集数据加载器
        device: 计算设备

    Returns:
        dict[str, float]: 类别名到召回率的映射，若某类无样本则值为 NaN
    """
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
    """主训练流程。

    整体流程:
        1. 设备选择与数据准备
        2. 数据集构建: 训练集(含数据增强) + 验证集(无增强)
        3. 模型初始化: GestureSegmenter
        4. 损失函数与优化器配置
        5. 训练循环: 每个 epoch 包含训练 + 验证
        6. 检查点保存: 保存最优和最后模型
        7. 训练历史记录导出

    关键设计决策:
        - 类别权重: 使用 sqrt 频率倒数为 CE 损失加权，缓解类别不均衡
        - 梯度裁剪: clip_grad_norm_ 限制梯度范数为 5.0，稳定训练
        - 学习率调度: ReduceLROnPlateau，验证准确率停滞 6 epoch 后将学习率减半
        - AdamW 优化器: 解耦权重衰减，比标准 Adam 更适合正则化训练
        - BCE pos_weight=8: 边界帧远少于非边界帧，8 倍权重补偿正样本稀少
        - 深度监督: 多阶段 CE 损失均值，改善梯度流和收敛速度

    检查点保存逻辑:
        - best.pt: 验证帧级准确率最高时保存，包含模型参数、epoch、准确率、训练配置
        - last.pt: 训练结束时保存最后一个 epoch 的模型，用于断点续训
        - history.json: 每个 epoch 的训练/验证损失和准确率记录

    Args:
        args: 命令行参数命名空间
    """
    device = select_device()
    feature_dir = Path(args.data_dir) / "features"
    subjects = args.subjects.split(",") if args.subjects else None
    # 按受试者划分训练/验证集，保证同一受试者的数据不出现在两个集合中
    train_files, val_files = split_feature_files(feature_dir, val_ratio=args.val_ratio, subjects=subjects)
    # 训练集: 启用数据增强 (随机裁剪、时间扰动等)，较小的 hop 增加样本多样性
    train_ds = GestureSegmentationDataset(
        train_files,
        chunk_len=args.chunk_len,
        hop=args.train_hop,
        augment=True,
        boundary_radius=args.boundary_radius,
    )
    # 验证集: 不启用数据增强，较大 hop 减少冗余评估
    val_ds = GestureSegmentationDataset(
        val_files,
        chunk_len=args.chunk_len,
        hop=args.val_hop,
        augment=False,
        boundary_radius=args.boundary_radius,
    )
    # 确保训练集和验证集的输入通道数一致，取较大值
    input_channels = max(train_ds.input_channels, val_ds.input_channels)
    train_ds.input_channels = input_channels
    val_ds.input_channels = input_channels
    print(train_ds)
    print(val_ds)
    print(f"[info] input_channels={input_channels}")

    # CUDA 设备启用 pin_memory 加速 CPU->GPU 数据传输
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

    # 构建模型并移至目标设备
    model = GestureSegmenter(
        input_channels=input_channels,
        hidden_dim=args.hidden_dim,
        temporal_channels=args.temporal_channels,
        temporal_layers=args.temporal_layers,
        temporal_stages=args.temporal_stages,
        dropout=args.dropout,
    ).to(device)
    print(f"[info] parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # CE 损失: 使用类别权重缓解不均衡，ignore_index 忽略边界/过渡帧
    ce = nn.CrossEntropyLoss(weight=class_weights(train_ds).to(device), ignore_index=IGNORE_INDEX)
    # BCE 损失: 逐元素计算 (reduction=none)，pos_weight=8 补偿边界正样本稀少
    # pos_weight 形状 (1, 2, 1) 对应 (batch, 2类边界, time)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor([8.0, 8.0], device=device).view(1, 2, 1))
    # AdamW: 解耦权重衰减的 Adam 变体，适合带正则化的训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # ReduceLROnPlateau: 验证准确率连续 6 epoch 无改善时，学习率乘以 0.5
    # mode="max" 表示监控指标越大越好
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=6, factor=0.5)

    # 检查点目录与训练状态初始化
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0  # 历史最优验证帧级准确率
    history = []       # 训练历史记录，每个 epoch 一条

    # 主训练循环
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # 训练阶段: 前向+反向+参数更新
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
        # 验证阶段: 仅前向，无梯度计算
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
        # 根据验证准确率调整学习率
        scheduler.step(va_acc)
        # 记录本 epoch 的训练指标
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
        # 检查点保存: 当验证帧级准确率超过历史最优时保存 best.pt
        # 保存内容包括: epoch号、模型参数、验证准确率、完整训练配置(含input_channels)
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
        # 定期打印各类别召回率，监控类别级性能
        if epoch % args.report_every == 0 or epoch == args.epochs:
            for name, recall in per_class_recall(model, val_loader, device).items():
                value = "N/A" if np.isnan(recall) else f"{recall:.3f}"
                print(f"    recall {name:<20}: {value}")

    # 训练结束: 保存最后一个 epoch 的模型 (last.pt)
    # 用途: 若需从最后状态续训，或对比 best 与 last 的泛化差异
    torch.save(
        {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "val_frame_acc": va_acc,
            "args": vars(args) | {"input_channels": input_channels},
        },
        ckpt_dir / "last.pt",
    )
    # 导出训练历史为 JSON，便于后续绘制学习曲线和超参数分析
    (ckpt_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] best val_frame_acc={best_score:.4f}")
    print(f"[done] checkpoint={ckpt_dir / 'best.pt'}")


def main() -> None:
    """命令行入口: 解析参数并启动训练。"""
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
