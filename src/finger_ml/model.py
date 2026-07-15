"""ST-GCN + MS-TCN style model for offline gesture segmentation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from finger_ml.features import NUM_NODES
from finger_ml.labels import BACKGROUND_LABEL, GESTURE_ORDER, NUM_CLASSES

# 手部骨架的边定义，基于 MediaPipe 21 关键点拓扑结构。
# 每条边 (i, j) 表示关键点 i 与关键点 j 之间有骨骼连接。
# 前四行：拇指(0-4)、食指(0,5-8)、中指(0,9-12)、无名指(0,13-16)、小指(0,17-20)。
# 第五行：掌根之间的横向连接(5-9, 9-13, 13-17)。
# 第六行：指尖之间的横向连接(4-8, 4-12, 4-16, 4-20)，用于捕捉手指展开/并拢等手势。
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
    (4, 8), (4, 12), (4, 16), (4, 20),
]


def build_adjacency(num_nodes: int = NUM_NODES, edges: list[tuple[int, int]] | None = None) -> torch.Tensor:
    """根据骨架边构建对称归一化邻接矩阵。

    使用 D^{-1/2} A D^{-1/2} 归一化方式，使得每个节点的特征聚合
    考虑了其邻居数量，避免度数大的节点特征值过大。

    Args:
        num_nodes: 关键点数量，默认为 NUM_NODES (21)。
        edges: 边列表，每条边为 (i, j) 元组；默认使用 EDGES。
    Returns:
        归一化邻接矩阵 [V, V]，V = num_nodes。
    """
    edges = EDGES if edges is None else edges
    # 初始化为单位矩阵（自环），保证每个节点至少聚合自身特征
    a = np.eye(num_nodes, dtype=np.float32)
    # 添加无向边，邻接矩阵对称
    for i, j in edges:
        a[i, j] = 1.0
        a[j, i] = 1.0
    # 度向量：每行求和
    d = a.sum(axis=1)
    # D^{-1/2}，度为0的位置（不应出现）设为0
    d_inv = np.where(d > 0, d ** -0.5, 0.0)
    # 对称归一化：D^{-1/2} A D^{-1/2}
    return torch.from_numpy(np.diag(d_inv) @ a @ np.diag(d_inv))


class AdaptiveGraphConv(nn.Module):
    """自适应图卷积层：固定解剖学图 + 可学习残差图。

    在固定的手部骨架邻接矩阵基础上，增加一个可学习的残差邻接矩阵，
    允许模型发现数据驱动的非物理连接关系（例如跨手指的关联）。
    残差图通过 tanh 压缩到 [-1, 1] 后乘以 0.25 缩放，
    确保其不会大幅偏离原始解剖学结构。

    输入形状: [B, C_in, T, V]
    输出形状: [B, C_out, T, V]
    其中 B=批次, C=通道数, T=时间帧数, V=节点数
    """

    def __init__(self, c_in: int, c_out: int, adjacency: torch.Tensor) -> None:
        """
        Args:
            c_in: 输入通道数。
            c_out: 输出通道数。
            adjacency: 归一化邻接矩阵 [V, V]。
        """
        super().__init__()
        # 将固定邻接矩阵注册为 buffer（不参与梯度更新，但会随模型移动到 GPU）
        self.register_buffer("a_fixed", adjacency)
        # 可学习残差邻接矩阵，初始化为0，训练中逐渐发现有用的非物理连接
        self.a_residual = nn.Parameter(torch.zeros_like(adjacency))
        # 1x1 卷积实现通道维度投影（即逐节点的线性变换）
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：先投影通道，再沿节点维度进行图卷积。

        Args:
            x: 输入张量 [B, C_in, T, V]。
        Returns:
            图卷积输出 [B, C_out, T, V]。
        """
        # 自适应邻接矩阵 = 固定图 + tanh(残差) * 0.25
        a = self.a_fixed + torch.tanh(self.a_residual) * 0.25
        # 通道投影
        x = self.proj(x)
        # 图卷积：对节点维度做加权聚合
        # einsum "bctv,vw->bctw"：对 v 维度用邻接矩阵加权求和，得到新的节点特征
        return torch.einsum("bctv,vw->bctw", x, a)


class STGCNBlock(nn.Module):
    """时空图卷积块：图卷积 + 时间卷积 + 残差连接。

    结构：GCN -> BN -> ReLU -> 时间卷积(kernel=9) -> BN -> Dropout -> 残差加 -> ReLU
    - 图卷积（AdaptiveGraphConv）：沿空间维度（节点）聚合邻域信息
    - 时间卷积（9帧核）：沿时间维度捕获局部时序模式
    - 残差连接：输入直通到输出，缓解梯度消失，允许训练更深的网络

    输入形状: [B, C_in, T, V]
    输出形状: [B, C_out, T, V]
    """

    def __init__(self, c_in: int, c_out: int, adjacency: torch.Tensor, dropout: float = 0.0) -> None:
        """
        Args:
            c_in: 输入通道数。
            c_out: 输出通道数。
            adjacency: 归一化邻接矩阵 [V, V]。
            dropout: Dropout 概率，默认0（无正则化）。
        """
        super().__init__()
        # 图卷积：空间维度聚合
        self.gcn = AdaptiveGraphConv(c_in, c_out, adjacency)
        # 时间卷积：沿时间轴用9帧核捕获局部时序模式，padding=4保持时间长度不变
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(c_out),
            nn.Dropout(dropout),
        )
        # 残差分支：通道数不同时用1x1卷积对齐，否则直通
        self.residual = (
            nn.Sequential(nn.Conv2d(c_in, c_out, 1), nn.BatchNorm2d(c_out))
            if c_in != c_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [B, C_in, T, V]。
        Returns:
            输出张量 [B, C_out, T, V]。
        """
        return F.relu(self.tcn(self.gcn(x)) + self.residual(x), inplace=True)


class STGCNEncoder(nn.Module):
    """ST-GCN 编码器：三个逐步升维的 STGCNBlock，最后对节点维度取均值。

    将骨架序列从 [B, C, T, V] 编码为 [B, D, T] 的时序特征。
    通过对 V（节点）维度取均值（全局平均池化），得到与节点无关的全局手势表示。

    架构流程：
        输入 [B, input_channels, T, V]
        -> STGCNBlock(input_channels -> 48)    无dropout
        -> STGCNBlock(48 -> 96)                轻度dropout
        -> STGCNBlock(96 -> hidden_dim)         正常dropout
        -> 节点维度均值池化
        -> 输出 [B, hidden_dim, T]
    """

    def __init__(self, input_channels: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        """
        Args:
            input_channels: 输入通道数（每节点的特征维度，如 x,y 坐标为 2，加上速度等为更多）。
            hidden_dim: 最终编码维度，默认128。
            dropout: 基础dropout概率，逐层递增。
        """
        super().__init__()
        a = build_adjacency()
        self.blocks = nn.Sequential(
            STGCNBlock(input_channels, 48, a, dropout=0.0),
            STGCNBlock(48, 96, a, dropout=dropout * 0.5),
            STGCNBlock(96, hidden_dim, a, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入骨架序列 [B, C, T, V]。
        Returns:
            时序编码 [B, D, T]，D = hidden_dim。
        """
        # [B,C,T,V] -> [B,D,T]，对节点维度V取均值，聚合所有关节信息为全局表示
        return self.blocks(x).mean(dim=-1)


class DilatedResidualLayer(nn.Module):
    """膨胀残差层：膨胀因果卷积 + 残差连接。

    使用膨胀卷积（dilation）在不增加参数量的情况下扩大时间感受野。
    残差连接保证梯度直通，训练稳定。

    结构：Conv1d(dilation) -> ReLU -> Conv1d(1x1) -> Dropout -> 残差加

    感受野计算：每层感受野增量为 2 * dilation，堆叠多层的感受野指数增长。
    例如 dilation=1,2,4,8,16,32 的6层，感受野覆盖约64帧。
    """

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        """
        Args:
            channels: 通道数（输入输出相同）。
            dilation: 膨胀率。
            dropout: Dropout 概率。
        """
        super().__init__()
        self.net = nn.Sequential(
            # padding=dilation 保证输出时间长度与输入相同（因果等效）
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True),
            # 1x1 卷积做逐点非线性变换
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, C, T]。
        Returns:
            残差输出 [B, C, T]。
        """
        return x + self.net(x)


class TemporalStage(nn.Module):
    """MS-TCN 风格的时间建模阶段：膨胀残差层堆叠 + 分类头。

    每个阶段包含多个不同膨胀率的 DilatedResidualLayer，
    捕获从短期到长期的时间依赖关系。
    最后通过 1x1 卷积分类头输出逐帧类别 logits。

    输入形状: [B, in_channels, T]
    输出: (logits [B, num_classes, T], feat [B, channels, T])
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        """
        Args:
            in_channels: 输入通道数。
            channels: 中间特征通道数。
            num_layers: 膨胀残差层数量。
            num_classes: 分类类别数。
            dropout: Dropout 概率。
        """
        super().__init__()
        # 输入投影：1x1卷积对齐通道维度
        self.in_proj = nn.Conv1d(in_channels, channels, 1)
        # 膨胀残差层：膨胀率按 2^i 指数增长，感受野逐层翻倍
        self.layers = nn.ModuleList(
            DilatedResidualLayer(channels, 2 ** i, dropout)
            for i in range(num_layers)
        )
        # 分类头：1x1卷积将特征映射到类别空间
        self.classifier = nn.Conv1d(channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入特征 [B, in_channels, T]。
        Returns:
            logits: 逐帧分类 logits [B, num_classes, T]。
            feat: 中间特征 [B, channels, T]，供边界检测头使用。
        """
        feat = self.in_proj(x)
        for layer in self.layers:
            feat = layer(feat)
        return self.classifier(feat), feat


class GestureSegmenter(nn.Module):
    """Dense frame-label model.

    整体模型架构：ST-GCN 编码器 + 多阶段 MS-TCN 时间建模 + 边界检测头。

    数据流：
        1. 输入骨架序列 [B, input_channels, T, V]
        2. STGCNEncoder 编码为时序特征 [B, hidden_dim, T]
        3. 第0阶段 (stage0) 从编码特征预测初始 logits [B, num_classes, T]
        4. 精炼阶段 (refiners) 以前一阶段 softmax 概率为输入，迭代精炼 logits
        5. 边界检测头从 stage0 的中间特征预测起止边界概率 [B, 2, T]

    多阶段精炼的思路来自 MS-TCN：后续阶段以前一阶段的预测概率为输入，
    专注于修正错误，类似"自纠正"过程。每个精炼阶段结构相同，
    但输入是概率而非原始特征。

    Returns:
        logits: [B, num_classes, T]
        boundary_logits: [B, 2, T] for start/end boundary probabilities
        stage_logits: list of refinement stage logits
    """

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 128,
        temporal_channels: int = 128,
        temporal_layers: int = 6,
        temporal_stages: int = 2,
        dropout: float = 0.25,
    ) -> None:
        """
        Args:
            input_channels: 每节点输入特征维度（如 x,y,z,velocity 等组成12维）。
            num_classes: 分类类别数，默认7（6种手势 + 背景类）。
            hidden_dim: ST-GCN 编码器输出维度。
            temporal_channels: 时间建模阶段的中间特征通道数。
            temporal_layers: 每个阶段的膨胀残差层数量。
            temporal_stages: 时间建模阶段总数（含stage0），至少1。
            dropout: Dropout 概率。
        """
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        # ST-GCN 编码器：将骨架序列编码为时序特征
        self.encoder = STGCNEncoder(input_channels, hidden_dim, dropout)
        # 第0阶段：从编码特征直接预测
        self.stage0 = TemporalStage(hidden_dim, temporal_channels, temporal_layers, num_classes, dropout)
        # 精炼阶段：以前一阶段的概率为输入迭代修正
        self.refiners = nn.ModuleList(
            TemporalStage(num_classes, temporal_channels, temporal_layers, num_classes, dropout)
            for _ in range(max(0, temporal_stages - 1))
        )
        # 边界检测头：预测每个时间帧是否为手势的起始/结束边界
        # 输入为 stage0 的中间特征，输出 [B, 2, T]，2个通道分别为起始和结束概率
        self.boundary_head = nn.Sequential(
            nn.Conv1d(temporal_channels, temporal_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(temporal_channels // 2, 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """权重初始化：Conv 层使用 Kaiming 正态初始化，BN 层权重为1偏置为0。"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        前向传播：编码 -> 多阶段时间建模 -> 边界检测。

        Args:
            x: 输入骨架序列 [B, input_channels, T, V]。
        Returns:
            logits: 最终精炼后的分类 logits [B, num_classes, T]。
            boundary_logits: 边界检测 logits [B, 2, T]，通道0=起始概率，通道1=结束概率。
            stage_logits: 各阶段的 logits 列表（用于多阶段损失计算）。
        """
        # 编码：[B, C, T, V] -> [B, hidden_dim, T]
        encoded = self.encoder(x)
        # 第0阶段：从编码特征预测初始 logits
        logits, feat = self.stage0(encoded)
        stages = [logits]
        # 精炼阶段：将前一阶段 softmax 概率作为输入，迭代优化
        for refiner in self.refiners:
            logits, _ = refiner(F.softmax(logits, dim=1))
            stages.append(logits)
        # 边界检测：从 stage0 的中间特征预测起止边界
        boundary = self.boundary_head(feat)
        return logits, boundary, stages


def probabilities_to_events(
    probs: np.ndarray,
    fps: float,
    *,
    boundary_probs: np.ndarray | None = None,
    conf_threshold: float = 0.55,
    min_event_ms: int = 120,
    max_gap_ms: int = 120,
    smooth: int = 7,
) -> list[dict]:
    """Convert frame probabilities ``[T,C]`` to gesture event dictionaries.

    后处理管线流程：
        1. 取每帧 argmax 得到硬标签，低于置信度阈值的帧标记为背景
        2. 对硬标签进行类中值平滑，消除短时抖动
        3. 提取连续同标签片段，过滤过短片段
        4. 用边界概率精炼每个片段的起止帧位置
        5. 合并相邻同类别短间隔片段

    Args:
        probs: 逐帧分类概率 [T, C]，C=类别数。
        fps: 视频帧率，用于帧与毫秒的转换。
        boundary_probs: 边界概率 [2, T]，通道0=起始概率，通道1=结束概率；可选。
        conf_threshold: 置信度阈值，低于此阈值的帧归为背景类，默认0.55。
        min_event_ms: 最短事件时长（毫秒），短于此的片段被丢弃，默认120ms。
        max_gap_ms: 最大合并间隔（毫秒），相邻同类别片段间隔不超过此值则合并，默认120ms。
        smooth: 类中值平滑窗口宽度（帧数），1则不平滑，默认7。
    Returns:
        手势事件字典列表，每个字典包含 gesture, label, start_frame, end_frame,
        start_ms, end_ms, duration_ms, mean_conf 等字段。
    """
    if probs.size == 0:
        return []
    # 步骤1：取每帧 argmax 得到硬标签
    labels = probs.argmax(axis=1).astype(np.int64)
    # 取每帧对应类别的置信度
    confs = probs[np.arange(len(labels)), labels]
    # 置信度低于阈值的帧标记为背景
    labels = np.where(confs >= conf_threshold, labels, BACKGROUND_LABEL)
    # 步骤2：类中值平滑，消除短时标签抖动
    if smooth > 1:
        labels = _median_like_smooth(labels, probs, smooth)
    # 将毫秒阈值转换为帧数
    min_frames = max(1, int(round(min_event_ms * fps / 1000.0)))
    max_gap = max(0, int(round(max_gap_ms * fps / 1000.0)))

    # 步骤3+4：提取连续同标签片段，过滤过短片段，精炼边界
    raw: list[dict] = []
    i = 0
    while i < len(labels):
        label = int(labels[i])
        # 跳过背景帧
        if label == BACKGROUND_LABEL:
            i += 1
            continue
        # 找到连续同标签片段的结束位置
        j = i + 1
        while j < len(labels) and int(labels[j]) == label:
            j += 1
        # 过滤过短片段
        if j - i >= min_frames:
            # 用边界概率精炼起止帧位置
            start, end = _refine_boundaries(i, j - 1, boundary_probs)
            # 计算片段内该类别的平均置信度
            mean_conf = float(probs[i:j, label].mean())
            raw.append(_event(label, start, end, fps, mean_conf))
        i = j
    # 步骤5：合并相邻同类别短间隔片段
    return _merge_events(raw, max_gap, fps)


def _median_like_smooth(labels: np.ndarray, probs: np.ndarray, width: int) -> np.ndarray:
    """类中值平滑：在滑动窗口内按概率加权投票决定中心帧标签。

    与普通中值滤波不同，此方法考虑了概率强度：窗口内每个非背景标签
    的"票数"等于其概率值，而非简单计数。这样低置信度的预测权重更小。

    Args:
        labels: 硬标签序列 [T]。
        probs: 概率矩阵 [T, C]。
        width: 窗口宽度（帧数）。
    Returns:
        平滑后的标签序列 [T]。
    """
    half = width // 2
    out = labels.copy()
    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)
        # 按类别累加概率作为投票权重
        votes: dict[int, float] = {}
        for k in range(lo, hi):
            label = int(labels[k])
            if label == BACKGROUND_LABEL:
                continue
            votes[label] = votes.get(label, 0.0) + float(probs[k, label])
        # 取得票最多的标签；若窗口内无手势标签，保留背景
        out[i] = max(votes, key=votes.get) if votes else BACKGROUND_LABEL
    return out


def _refine_boundaries(start: int, end: int, boundary_probs: np.ndarray | None) -> tuple[int, int]:
    """使用边界检测概率精炼事件起止帧位置。

    在原始起止位置附近搜索边界概率峰值，将边界对齐到最可能的
    手势起始/结束时刻，提高事件时间定位的精度。

    搜索范围：原始位置前后 pad 帧，pad 大小与片段长度成正比（3~12帧）。

    Args:
        start: 原始起始帧索引。
        end: 原始结束帧索引。
        boundary_probs: 边界概率 [2, T]，通道0=起始概率，通道1=结束概率。
    Returns:
        精炼后的 (start, end) 帧索引。
    """
    if boundary_probs is None or boundary_probs.size == 0:
        return start, end
    n = boundary_probs.shape[1]
    # 搜索范围：原始位置前后 pad 帧
    pad = max(3, min(12, (end - start + 1) // 2))
    # 起始帧搜索区间
    s0, s1 = max(0, start - pad), min(n, start + pad + 1)
    # 结束帧搜索区间
    e0, e1 = max(0, end - pad), min(n, end + pad + 1)
    # 在搜索区间内找起始概率最大值的位置
    if s1 > s0:
        start = s0 + int(np.argmax(boundary_probs[0, s0:s1]))
    # 在搜索区间内找结束概率最大值的位置
    if e1 > e0:
        end = e0 + int(np.argmax(boundary_probs[1, e0:e1]))
    # 安全检查：确保结束帧不早于起始帧
    if end < start:
        end = start
    return start, end


def _event(label: int, start: int, end: int, fps: float, confidence: float) -> dict:
    """构建单个手势事件字典，包含帧索引和毫秒时间信息。

    Args:
        label: 手势类别索引。
        start: 起始帧索引（0-based）。
        end: 结束帧索引（0-based，包含）。
        fps: 视频帧率。
        confidence: 片段内平均置信度。
    Returns:
        事件字典，包含 gesture, label, start_frame, end_frame,
        start_ms, end_ms, duration_ms, mean_conf 字段。
    """
    name = GESTURE_ORDER[label] if 0 <= label < len(GESTURE_ORDER) else str(label)
    return {
        "gesture": name,
        "label": int(label),
        "start_frame": int(start),
        "end_frame": int(end),
        "start_ms": int(round(start * 1000 / fps)),
        "end_ms": int(round(end * 1000 / fps)),
        "duration_ms": int(round((end - start + 1) * 1000 / fps)),
        "mean_conf": round(float(confidence), 4),
    }


def _merge_events(events: list[dict], max_gap: int, fps: float) -> list[dict]:
    """合并相邻同类别且间隔不超过阈值的片段。

    手势执行中可能因短暂置信度下降而被分割为多个片段，
    此函数将间隔不超过 max_gap 帧的同类别片段合并为一个。
    合并时按帧数加权平均更新置信度。

    Args:
        events: 已排序的事件字典列表。
        max_gap: 最大允许间隔帧数。
        fps: 视频帧率，用于重新计算毫秒时间。
    Returns:
        合并后的事件列表。
    """
    if not events:
        return []
    merged = [events[0].copy()]
    for ev in events[1:]:
        prev = merged[-1]
        # 计算两片段之间的间隔帧数
        gap = ev["start_frame"] - prev["end_frame"] - 1
        # 同类别且间隔不超过阈值则合并
        if ev["label"] == prev["label"] and gap <= max_gap:
            # 按帧数加权计算合并后的平均置信度
            prev_len = prev["end_frame"] - prev["start_frame"] + 1
            cur_len = ev["end_frame"] - ev["start_frame"] + 1
            # 更新结束帧和时间信息
            prev["end_frame"] = ev["end_frame"]
            prev["end_ms"] = ev["end_ms"]
            prev["duration_ms"] = int(round((prev["end_frame"] - prev["start_frame"] + 1) * 1000 / fps))
            # 加权平均置信度
            prev["mean_conf"] = round(
                (prev["mean_conf"] * prev_len + ev["mean_conf"] * cur_len) / (prev_len + cur_len),
                4,
            )
        else:
            merged.append(ev.copy())
    return merged
