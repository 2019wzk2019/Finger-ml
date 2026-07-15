"""手部骨架归一化和运动特征工程。

将 MediaPipe 原始 21 点骨架坐标转换为手掌局部坐标系下的归一化表示，
并构建包含坐标、速度、距离、方向和有效性标记的扩展特征矩阵。

特征通道（共 12 通道，每个节点重复）：
    0-2  x, y, z         — 归一化坐标
    3-5  dx, dy, dz      — 帧间速度（相邻帧差值）
    6    thumb_index_dist — 拇指尖到食指尖距离（标量，所有节点共享）
    7    thumb_middle_dist— 拇指尖到中指尖距离（标量，所有节点共享）
    8-10 thumb_index_dx/dy/dz — 拇指到食指方向向量（所有节点共享）
    11   valid            — MediaPipe 检测有效性标记（1=检测到，0=未检测到）
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# 骨架节点数量，MediaPipe Hand Landmarker 固定输出 21 个关键点
NUM_NODES = 21

# 保留的节点索引列表（当前保留全部 21 个节点）
KEEP_INDICES = list(range(NUM_NODES))

# 以下为常用节点索引常量，方便代码引用
WRIST = 0          # 腕部（坐标原点）
THUMB_TIP = 4      # 拇指尖
INDEX_MCP = 5      # 食指掌指关节（用于构建手掌局部坐标系 x 轴）
INDEX_TIP = 8      # 食指尖
MIDDLE_TIP = 12    # 中指尖
PINKY_MCP = 17     # 小指掌指关节（用于构建手掌局部坐标系 z 轴）

# 21 个节点的英文名称，用于质量报告中的抖动诊断
NODE_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)

# 特征通道名称，与 build_motion_features() 输出对应
FEATURE_NAMES: tuple[str, ...] = (
    "x", "y", "z",               # 归一化坐标
    "dx", "dy", "dz",            # 帧间速度
    "thumb_index_dist",          # 拇指-食指距离
    "thumb_middle_dist",         # 拇指-中指距离
    "thumb_index_dx",            # 拇指-食指方向 x 分量
    "thumb_index_dy",            # 拇指-食指方向 y 分量
    "thumb_index_dz",            # 拇指-食指方向 z 分量
    "valid",                     # 有效性标记
)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """将 MediaPipe 21 点骨架转换为手掌局部坐标系下的归一化坐标。

    坐标系构建方法：
        1. 以腕部(WRIST)为原点平移
        2. x 轴 = 腕部→食指MCP 方向（手掌展开方向）
        3. z 轴 = x 轴 × 腕部→小指MCP（手掌法线方向）
        4. y 轴 = z 轴 × x 轴（正交化）
        5. 以腕部→食指MCP 距离为尺度做归一化

    Args:
        landmarks: [21, 3] 原始 MediaPipe 归一化坐标

    Returns:
        [21, 3] 手掌局部坐标系下的归一化坐标，范围约 [-1, 1]
    """
    landmarks = landmarks.astype(np.float32, copy=False)
    # 以腕部为原点平移
    centered = landmarks - landmarks[WRIST]

    # 构建 x 轴：腕部 → 食指 MCP 方向
    x_axis = centered[INDEX_MCP].copy()
    # 取小指 MCP 用于计算法线
    pinky_vec = centered[PINKY_MCP].copy()
    # 尺度 = 腕部到食指MCP的距离，作为归一化分母
    scale = float(np.linalg.norm(x_axis))
    if scale > 1e-6:
        x_axis = x_axis / scale
    else:
        # 退化情况：手掌过小或检测异常，使用默认方向
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        scale = 1.0

    # 构建 z 轴：x × 腕部→小指MCP，得到手掌法线方向
    z_axis = np.cross(x_axis, pinky_vec)
    z_norm = float(np.linalg.norm(z_axis))
    if z_norm > 1e-6:
        z_axis = z_axis / z_norm
    else:
        # 退化情况：手指共线，使用默认法线
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # 构建 y 轴：z × x，保证正交右手系
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if y_norm > 1e-6:
        y_axis = y_axis / y_norm
    else:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # 组装旋转矩阵 [x_axis, y_axis, z_axis]，每列一个基向量
    basis = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
    # 旋转 + 尺度归一化：将坐标投影到局部坐标系并除以手掌尺度
    return ((centered @ basis) / scale).astype(np.float32)


def build_motion_features(
    landmarks: np.ndarray,
    valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """构建坐标、速度、接触/方向和有效性特征。

    Args:
        landmarks: [T, 21, 3] 归一化后的骨架坐标序列
        valid: [T] bool 数组，标记每帧 MediaPipe 是否检测成功；
               None 表示全部有效

    Returns:
        [T, 21, 12] 扩展特征矩阵，12 个通道定义见 FEATURE_NAMES
    """
    landmarks = landmarks.astype(np.float32, copy=False)
    n, v, _ = landmarks.shape

    # 帧间速度：相邻帧坐标差值，第一帧速度为 0
    delta = np.zeros_like(landmarks, dtype=np.float32)
    if n > 1:
        delta[1:] = landmarks[1:] - landmarks[:-1]

    # 提取拇指、食指、中指指尖坐标，用于计算捏合距离和方向
    thumb_tip = landmarks[:, THUMB_TIP]
    index_tip = landmarks[:, INDEX_TIP]
    middle_tip = landmarks[:, MIDDLE_TIP]

    # 拇指→食指 向量和距离
    thumb_index_vec = thumb_tip - index_tip
    thumb_middle_vec = thumb_tip - middle_tip
    thumb_index_dist = np.linalg.norm(thumb_index_vec, axis=1, keepdims=True)   # [T, 1]
    thumb_middle_dist = np.linalg.norm(thumb_middle_vec, axis=1, keepdims=True) # [T, 1]

    # 全局特征：距离(2) + 方向向量(3) = 5 维，广播到所有 21 个节点
    global_feats = np.concatenate(
        [thumb_index_dist, thumb_middle_dist, thumb_index_vec],
        axis=1,
    ).astype(np.float32)  # [T, 5]
    # 复制到每个节点：[T, 5] → [T, 21, 5]
    global_feats = np.repeat(global_feats[:, None, :], v, axis=1)

    # 有效性标记通道：1=检测到，0=未检测到
    if valid is None:
        valid_channel = np.ones((n, v, 1), dtype=np.float32)
    else:
        # [T] → [T, 21, 1]
        valid_channel = np.repeat(valid.astype(np.float32)[:, None, None], v, axis=1)

    # 拼接所有通道：坐标(3) + 速度(3) + 全局特征(5) + 有效性(1) = 12
    return np.concatenate([landmarks, delta, global_feats, valid_channel], axis=2).astype(np.float32)
