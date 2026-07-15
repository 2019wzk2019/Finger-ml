"""Datasets for offline hand-gesture temporal segmentation."""

# 离线手势时序分割数据集模块
# 本模块定义了用于训练/验证的帧级手势分割数据集，将连续视频特征切分为固定长度的片段（chunk），
# 并为每个帧提供类别标签、边界检测目标和训练掩码。

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from finger_ml.features import NUM_NODES
from finger_ml.labels import BACKGROUND_LABEL, NUM_CLASSES

# 忽略索引：用于 PyTorch 交叉熵损失中的 ignore_index 参数
# 标记为 IGNORE_INDEX 的帧不参与损失计算，包括：
#   1. 填充（padding）帧 —— chunk 长度超出序列长度时的填充部分
#   2. 手势过渡/边界帧 —— train_mask 中为 False 的帧（由预处理阶段的 BOUNDARY_MARGIN 产生）
# 值 -100 是 PyTorch CrossEntropyLoss 的默认 ignore_index
IGNORE_INDEX = -100


@dataclass(frozen=True)
class Session:
    """单个视频会话的特征与标注数据。

    由 load_session() 从 .npz 文件加载并构造，是数据集的基本存储单元。

    Attributes:
        path: .npz 特征文件的路径，用于溯源和日志输出
        features: 帧特征数组，形状 [T, 21, C]，T 为帧数，21 为手部关键点数，C 为特征通道数
                  通道包括：归一化坐标(x,y,z) + 速度 + 角度等
        labels: 帧级标签数组，形状 [T]，值为 0~5(手势类别) 或 6(背景)
        train_mask: 训练掩码数组，形状 [T]，bool 类型
                    True 表示该帧参与训练损失计算
                    False 表示该帧为过渡/边界帧，标记为 IGNORE_INDEX 后不参与损失
        fps: 视频帧率，用于时间相关的计算（如速度估计），默认 30.0
    """

    path: Path
    features: np.ndarray
    labels: np.ndarray
    train_mask: np.ndarray
    fps: float


def subject_from_stem(stem: str) -> str:
    """从文件名主干提取被试标识符。

    文件名约定为 <前缀>_<被试ID>.npz，本函数取最后一个下划线后的部分作为被试 ID。
    例如 "session_P001" -> "p001"，"video_Alice" -> "alice"

    Args:
        stem: 文件名主干（不含扩展名），如 "session_P001"

    Returns:
        被试标识符（小写），若无下划线则返回空字符串
    """
    return stem.rsplit("_", 1)[-1].lower() if "_" in stem else ""


def split_feature_files(
    features_dir: str | Path,
    *,
    val_ratio: float = 0.2,
    subjects: list[str] | None = None,
) -> tuple[list[Path], list[Path]]:
    """按帧数比例将特征文件划分为训练集和验证集。

    采用贪心策略：按文件名排序后，从尾部（通常为最后录制的会话）开始选取文件，
    累加帧数直到达到总帧数的 val_ratio 比例，选出的文件归入验证集，其余为训练集。
    这样可以尽量保持训练集和验证集的帧数比例接近 val_ratio。

    Args:
        features_dir: 包含 .npz 特征文件的目录路径
        val_ratio: 验证集占总帧数的比例，默认 0.2（20%）
        subjects: 可选的被试 ID 列表，若提供则只保留文件名中包含这些被试 ID 的文件

    Returns:
        (train_files, val_files): 训练集和验证集的文件路径列表，均按文件名排序
        特殊情况：若只有 1 个文件，训练集和验证集使用同一文件

    Raises:
        FileNotFoundError: 若指定目录下无 .npz 特征文件
    """
    features_dir = Path(features_dir)
    # 按文件名排序，确保划分结果可复现
    files = sorted(features_dir.glob("*.npz"))
    if subjects:
        allowed = {s.lower() for s in subjects}
        files = [p for p in files if subject_from_stem(p.stem) in allowed]
    if not files:
        raise FileNotFoundError(f"No feature .npz files found in {features_dir}")
    if len(files) == 1:
        return files, files

    # 统计每个文件的帧数，用于按帧数比例划分
    frame_counts = []
    for path in files:
        with np.load(path) as data:
            frame_counts.append(int(data["labels"].shape[0]))
    # 计算验证集应包含的目标帧数
    target = sum(frame_counts) * val_ratio
    # 从尾部开始贪心选取，直到验证集帧数达到目标
    val: list[Path] = []
    total = 0
    for path, count in zip(reversed(files), reversed(frame_counts)):
        val.append(path)
        total += count
        if total >= target:
            break
    val_set = set(val)
    # 训练集 = 全部文件 - 验证集
    train = [p for p in files if p not in val_set]
    # 边界情况：若验证集占满所有文件，则至少保留 1 个训练文件
    if not train:
        train = val[:1]
        val = val[1:] or val[:1]
    return train, sorted(val)


def load_session(path: Path) -> Session:
    """从 .npz 文件加载一个视频会话的特征与标注数据。

    .npz 文件由 finger-preprocess 步骤生成，包含以下数组：
      - "features" 或 "landmarks": 帧特征 [T, 21, C]
      - "labels": 帧级标签 [T]
      - "train_mask": 训练掩码 [T]（可选，默认全 True）
      - "fps": 帧率标量（可选，默认 30.0）

    Args:
        path: .npz 特征文件路径

    Returns:
        Session 数据对象

    Raises:
        ValueError: 若特征维度不符合预期（应为 [T, 21, C]，21 = NUM_NODES）
    """
    with np.load(path) as data:
        # 兼容旧版文件：字段名可能为 "landmarks" 或 "features"
        features = (data["features"] if "features" in data else data["landmarks"]).astype(np.float32)
        labels = data["labels"].astype(np.int64)
        # train_mask 缺失时默认全 True（所有帧均参与训练）
        train_mask = (
            data["train_mask"].astype(bool)
            if "train_mask" in data
            else np.ones_like(labels, dtype=bool)
        )
        fps = float(data["fps"]) if "fps" in data else 30.0
    # 校验特征形状：必须为 [T, 21, C]，其中 21 = 手部关键点数（NUM_NODES）
    if features.ndim != 3 or features.shape[1] != NUM_NODES:
        raise ValueError(f"{path}: expected features [T, 21, C], got {features.shape}")
    return Session(path, features, labels, train_mask, fps)


def boundary_targets(labels: np.ndarray, radius: int = 2) -> np.ndarray:
    """从帧级标签生成手势起止边界检测目标。

    输出形状 [2, T]，两个通道分别表示"手势起始"和"手势结束"的概率。
    在手势从背景切换到非背景的帧（起始点），以及从非背景切换到背景的帧（结束点）周围
    radius 帧范围内，对应通道标记为 1.0，其余为 0.0。

    Args:
        labels: 帧级标签数组，形状 [T]，值为类别索引或 BACKGROUND_LABEL(6)
        radius: 边界扩散半径，在边界点前后各 radius 帧内标记为正样本，默认 2

    Returns:
        边界目标数组，形状 [2, T]，float32
        - out[0] = 手势起始通道
        - out[1] = 手势结束通道
    """
    """Return [2, T] start/end targets from frame labels."""
    n = len(labels)
    out = np.zeros((2, n), dtype=np.float32)
    # 构造前一帧标签数组，用于检测标签变化（边界）
    prev = np.full(n, BACKGROUND_LABEL, dtype=np.int64)
    prev[1:] = labels[:-1]
    # 起始点：当前帧为手势，前一帧为背景
    starts = np.where((labels != BACKGROUND_LABEL) & (prev == BACKGROUND_LABEL))[0]
    # 结束点：当前帧为背景，前一帧为手势（取前一个帧索引）
    ends = np.where((labels == BACKGROUND_LABEL) & (prev != BACKGROUND_LABEL))[0] - 1
    # 边界情况：序列末尾仍为手势，则最后一帧为结束点
    if n and labels[-1] != BACKGROUND_LABEL:
        ends = np.append(ends, n - 1)
    # 在边界点周围 radius 范围内标记为正样本
    for channel, indices in enumerate((starts, ends)):
        for idx in indices:
            lo = max(0, int(idx) - radius)
            hi = min(n, int(idx) + radius + 1)
            out[channel, lo:hi] = 1.0
    return out


class GestureSegmentationDataset(Dataset):
    """Fixed-length chunks for dense frame labeling.

    Each item is ``x [C,T,21]``, ``y [T]``, ``boundary [2,T]`` and ``mask [T]``.
    Padding and ignored annotation-transition frames use ``IGNORE_INDEX``.
    """

    """手势时序分割数据集：将视频特征切分为固定长度的片段用于密集帧标注。

    数据流程：
      1. 加载所有 .npz 文件为 Session 对象
      2. 按 chunk_len 和 hop 将每个 Session 切分为多个固定长度片段（chunk）
      3. 对每个片段：
         - 提取特征 x: [C, T, 21]（转置自 [T, 21, C]）
         - 提取标签 y: [T]，填充帧和 train_mask=False 的帧标记为 IGNORE_INDEX
         - 提取边界目标 b: [2, T]
         - 提取有效帧掩码 mask: [T]（bool，True 表示真实帧）

    切分策略：
      - 若 Session 长度 <= chunk_len，只生成 1 个片段（起始位置 0）
      - 否则从起始位置 0 开始，以 hop 为步长滑动窗口，直到窗口终点超出序列
      - 训练时可启用 augment，对起始位置施加随机抖动

    Args:
        files: .npz 特征文件路径列表
        chunk_len: 每个片段的帧数，默认 256
        hop: 滑动窗口步长，默认等于 chunk_len（无重叠），设为较小值可产生重叠片段
        augment: 是否启用数据增强，默认 False
        boundary_radius: 边界检测目标的扩散半径，传递给 boundary_targets()

    Returns (per __getitem__):
        x: 特征张量，形状 [C, chunk_len, 21]
        y: 标签张量，形状 [chunk_len]，填充/过渡帧值为 IGNORE_INDEX(-100)
        b: 边界目标张量，形状 [2, chunk_len]
        mask: 有效帧掩码，形状 [chunk_len]，bool
    """

    def __init__(
        self,
        files: list[Path],
        *,
        chunk_len: int = 256,
        hop: int | None = None,
        augment: bool = False,
        boundary_radius: int = 2,
    ) -> None:
        # 加载所有会话数据
        self.sessions = [load_session(p) for p in files]
        self.chunk_len = int(chunk_len)
        # hop 默认等于 chunk_len（无重叠切片），设为较小值可产生重叠片段以增加训练数据量
        self.hop = int(hop if hop is not None else chunk_len)
        self.augment = augment
        self.boundary_radius = int(boundary_radius)
        # 取所有会话中最大的特征通道数，用于统一填充
        self.input_channels = max(int(s.features.shape[2]) for s in self.sessions)
        # 构建片段索引：(session_index, start_frame) 列表
        self.index: list[tuple[int, int]] = []
        for si, session in enumerate(self.sessions):
            n = len(session.labels)
            # 短序列：只生成 1 个从位置 0 开始的片段
            if n <= self.chunk_len:
                self.index.append((si, 0))
                continue
            # 长序列：以 hop 为步长滑动窗口切分
            for start in range(0, n, self.hop):
                self.index.append((si, start))
                # 当窗口终点已超出序列时，停止生成新片段
                if start + self.chunk_len >= n:
                    break

    def __len__(self) -> int:
        """返回数据集中片段的总数。"""
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取指定索引的片段数据。

        数据处理流程：
          1. 根据索引定位 Session 和起始帧
          2. 若启用增强，对起始帧施加随机抖动
          3. 提取特征片段，不足 chunk_len 的部分用零填充
          4. 提取标签片段，填充部分标记为 IGNORE_INDEX
          5. 将 train_mask=False 的帧标签也标记为 IGNORE_INDEX
          6. 提取边界目标片段
          7. 若启用增强，施加高斯噪声和帧丢弃
          8. 转换为 PyTorch 张量，特征维度从 [T, 21, C] 转置为 [C, T, 21]

        Args:
            idx: 片段索引

        Returns:
            (x, y, b, mask) 四元组:
            - x: 特征 [C, chunk_len, 21]
            - y: 标签 [chunk_len]，填充/过渡帧 = IGNORE_INDEX(-100)
            - b: 边界目标 [2, chunk_len]
            - mask: 有效帧掩码 [chunk_len]，bool
        """
        # 定位片段所属的 Session 和起始帧位置
        si, start = self.index[idx]
        session = self.sessions[si]
        n = len(session.labels)
        # 训练增强：对起始位置施加随机抖动，使同一片段在不同 epoch 中覆盖不同帧范围
        if self.augment and n > self.chunk_len:
            jitter = random.randint(-self.hop // 2, self.hop // 2)
            start = min(max(0, start + jitter), max(0, n - self.chunk_len))

        # 计算实际可用的帧范围
        end = min(n, start + self.chunk_len)
        length = end - start  # 真实帧数，可能小于 chunk_len
        c = self.input_channels
        # 初始化特征数组，形状 [chunk_len, 21, C]，填充零
        x = np.zeros((self.chunk_len, NUM_NODES, c), dtype=np.float32)
        # 将真实特征填入前 length 帧（特征通道数不足 C 的部分保持为零）
        x[:length, :, : session.features.shape[2]] = session.features[start:end]
        # 初始化标签数组，全部填充 IGNORE_INDEX（填充帧和过渡帧的默认值）
        y = np.full(self.chunk_len, IGNORE_INDEX, dtype=np.int64)
        # 初始化有效帧掩码，全部为 False
        mask = np.zeros(self.chunk_len, dtype=bool)
        # 填入真实标签和掩码
        y[:length] = session.labels[start:end]
        mask[:length] = session.train_mask[start:end]
        # train_mask=False 的帧为过渡/边界帧，标签标记为 IGNORE_INDEX 以排除损失计算
        y[~mask] = IGNORE_INDEX

        # 生成边界检测目标，从完整序列的标签中提取后再切片
        b_all = boundary_targets(session.labels, self.boundary_radius)
        b = np.zeros((2, self.chunk_len), dtype=np.float32)
        b[:, :length] = b_all[:, start:end]

        # 数据增强
        if self.augment:
            # 对前 6 个通道（坐标 + 速度通道，归一化后的手掌局部值）添加高斯噪声
            # Coordinate and velocity channels are normalized palm-local values.
            noise_channels = min(6, c)
            x[:, :, :noise_channels] += np.random.normal(
                0.0, 0.008, x[:, :, :noise_channels].shape
            ).astype(np.float32)
            # 15% 概率执行帧丢弃：每帧 4% 概率被置零，模拟遮挡
            if random.random() < 0.15:
                drop = np.random.rand(self.chunk_len) < 0.04
                x[drop, :, :] = 0.0

        # 转换为张量，特征维度转置：[T, 21, C] -> [C, T, 21] 以适配 Conv1d 输入格式
        return (
            torch.from_numpy(x).permute(2, 0, 1),
            torch.from_numpy(y),
            torch.from_numpy(b),
            torch.from_numpy(mask),
        )

    def class_counts(self) -> np.ndarray:
        """统计数据集中各类别（含背景）的有效帧数。

        仅统计 train_mask=True 的帧，排除过渡/边界帧。

        Returns:
            长度为 NUM_CLASSES(7) 的计数数组，counts[i] 为类别 i 的帧数
        """
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for session in self.sessions:
            valid = session.train_mask
            counts += np.bincount(session.labels[valid], minlength=NUM_CLASSES)
        return counts

    def __repr__(self) -> str:
        names = ", ".join(s.path.stem for s in self.sessions)
        return (
            f"GestureSegmentationDataset(n_chunks={len(self)}, "
            f"chunk_len={self.chunk_len}, files=[{names}])"
        )
