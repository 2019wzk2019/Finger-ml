# Finger-ml 项目详细说明文档

## 一、项目概述

Finger-ml 是一个**面向录制视频的离线手势事件检测**系统。它从输入视频中检测手势片段，并输出结构化事件：手势类别、开始帧/时间、结束帧/时间。

项目使用 **MediaPipe Hand Landmarker** 提取 21 点手部骨架，再在骨架序列上训练**稠密时序分割模型**（ST-GCN + MS-TCN），实现帧级手势分类和边界检测。

> 本项目**不是**按实时当前帧分类器优化的。核心指标是事件 F1、边界时间误差、误检和漏检。

---

## 二、系统架构

```
视频 (MP4) + 标注 (JSON)
        │
        ▼
  finger-collect ──→ data/video/*.mp4 + data/labels/*.json
        │
        ▼
  finger-review   ──→ 人工检查标注窗和骨架质量
        │
        ▼
  finger-preprocess ──→ data/features/*.npz
        │                     ├── landmarks [T, 21, 3]   手掌局部坐标系骨架
        │                     ├── features  [T, 21, 12]  扩展特征（坐标/速度/距离/有效性）
        │                     ├── labels    [T]          帧级标签 0-6
        │                     ├── train_mask [T]         过渡帧屏蔽标记
        │                     └── valid     [T]          MediaPipe 检测成功标记
        │
        ▼
  finger-audit    ──→ 数据覆盖和质量门禁检查
        │
        ▼
  finger-train    ──→ checkpoints/best.pt
        │
        ▼
  finger-detect   ──→ results/*.events.json (+ 可选叠加视频)
        │
        ▼
  finger-eval     ──→ 事件级 precision/recall/F1 + 时间误差报告
```

---

## 三、环境安装

### 3.1 推荐方式：uv

```bash
# 安装 uv（如尚未安装）
pip install uv

# 基础运行环境（不含 PyTorch，仅采集/预处理/检测/评估）
uv sync

# 加上 PyTorch CPU 版（训练用）
uv sync --group train

# 加上 PyTorch CUDA 版（需 NVIDIA GPU + CUDA 12.1+ 驱动）
uv sync --group train-cuda
```

### 3.2 没有 uv 的情况

项目是 `pyproject.toml` 驱动的 hatchling 包，无法直接 `pip install -e .`。替代方法：

```powershell
# 设置包搜索路径（每次新终端都需要）
$env:PYTHONPATH = "src"

# 手动安装运行时依赖
pip install opencv-python numpy mediapipe pillow

# 训练还需要
pip install torch torchvision tqdm scikit-learn

# 然后以模块方式运行
python -m finger_ml.capture --subject S01 --repeats 5
python -m finger_ml.preprocess --data-dir data --hand-side Right
python -m finger_ml.train --data-dir data --epochs 80
# ...以此类推
```

### 3.3 常见依赖冲突

| 问题 | 原因 | 解决 |
|------|------|------|
| `AttributeError: _ARRAY_API not found` | Anaconda 旧版 matplotlib 是 numpy 1.x 编译，mediapipe 装了 numpy 2.x | `pip install "matplotlib>=3.8"` |
| `opencv-contrib-python 5.x` 要求 numpy>=2 | 与旧 matplotlib 冲突 | `pip install "opencv-contrib-python<5"` |
| `numba/scipy` 要求 numpy<1.23 | 与新 mediapipe/opencv 不兼容 | 不影响本项目运行，可忽略 |

---

## 四、手势类别定义

定义在 `src/finger_ml/labels.py`，**这是全项目唯一的手势标签权威来源**：

| 标签值 | 键名 | 中文含义 | 英文全称 |
|--------|------|----------|----------|
| 0 | `pinch_index` | 拇指捏食指尖 | Pinch Index Tip |
| 1 | `pinch_middle` | 拇指捏中指尖 | Pinch Middle Tip |
| 2 | `thumb_slide_up` | 拇指向上滑动 | Thumb Slide Up |
| 3 | `thumb_slide_down` | 拇指向下滑动 | Thumb Slide Down |
| 4 | `thumb_slide_left` | 拇指向左滑动 | Thumb Slide Left |
| 5 | `thumb_slide_right` | 拇指向右滑动 | Thumb Slide Right |
| 6 | `background` | 无手势/静息 | Background |

关键常量：
- `GESTURE_ORDER`：6 个手势名称的有序元组（索引 0-5）
- `BACKGROUND_LABEL = 6`
- `NUM_CLASSES = 7`
- `LABEL_NAMES = (*GESTURE_ORDER, "background")`：7 个名称的完整有序元组

---

## 五、完整流水线详解

### 5.1 采集：`finger-collect`

```bash
uv run finger-collect --subject S01 --repeats 5 --camera 1
```

**功能**：打开摄像头，连续录制 MP4 视频，用户按 SPACE 手动标注每段手势的起止时间。

**工作流程**：
1. 打开摄像头，启动 MediaPipe IMAGE 模式实时骨架检测（仅供画面叠加，不保存）
2. 启动异步视频写入线程，以摄像头实际帧率恒定写入 MP4
3. 右侧面板显示当前手势名称、示意图、重复次数、状态提示
4. 用户按 SPACE → 3 秒倒计时 → 自动标记开始帧 → 录制中再按 SPACE 标记结束帧
5. 自动切换到下一个手势/下一个 rep，或进入休息倒计时
6. 所有手势完成后，保存视频和标注 JSON

**操作按键**：
- `SPACE`：标记开始/结束（自动倒计时后开始）
- `R`：撤销上一个标注
- `Q`：退出保存

**输出文件**：
- `data/video/<session_id>_<subject>.mp4`
- `data/labels/<session_id>_<subject>.json`

**JSON 标注格式**（注意：`start_frame`/`end_frame` 为 **1-indexed**）：
```json
{
  "subject_id": "S01",
  "session_id": "20260429_151254",
  "fps": 60.0,
  "width": 1920,
  "height": 1080,
  "gestures_order": ["pinch_index", "pinch_middle", ...],
  "repeats": 5,
  "annotations": [
    {
      "gesture": "pinch_index",
      "label": 0,
      "rep": 1,
      "start_frame": 318,
      "end_frame": 362,
      "start_ms": 5300,
      "end_ms": 6033
    }
  ]
}
```

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--subject` | `S01` | 受试者 ID |
| `--repeats` | `5` | 每种手势重复次数 |
| `--camera` | `1` | 摄像头索引，失败自动降级到 0 |
| `--fps` | `60.0` | 目标帧率 |
| `--countdown-sec` | `3.0` | 动作前倒计时秒数 |
| `--rest-sec` | `3.0` | 手势组间休息秒数 |
| `--output-dir` | `data` | 输出根目录 |

**技术细节**：
- 使用 `_precise_sleep()` 解决 Windows 下 `time.sleep()` 精度约 15ms 的问题（先粗 sleep 再忙等最后 ~1ms）
- 视频时间轴采用 CFR（恒定帧率）：如果主循环变慢，会重复最新画面补帧，保证输出文件帧率稳定
- 摄像头读取和视频写入都在独立线程中运行，避免阻塞 UI

---

### 5.2 回放检查：`finger-review`

```bash
# 查看最新 session
uv run finger-review --data-dir data

# 指定 session
uv run finger-review --data-dir data --session 20260429_151254_S01

# 循环播放第 3 个动作窗口
uv run finger-review --data-dir data --window 3
```

**功能**：回放采集数据，展示标注窗口、骨架叠加、抖动/漏检时间轴。

**操作按键**：
- `SPACE`：播放/暂停
- `A`/`D`：前/后一个动作窗口
- `L`：循环当前窗口
- `←`/`→`：逐帧/跳帧
- `Q`：退出

**时间轴说明**：
- 上方色条：抖动/漏检（绿=稳定、黄/红=抖动高、蓝=漏检）
- 下方色条：动作窗口（不同颜色代表不同手势类别）
- 如果 `data/features/` 下有对应 `.npz`，还会显示特征质量信息

---

### 5.3 预处理：`finger-preprocess`

```bash
# 基本用法
uv run finger-preprocess --data-dir data --force --hand-side Right

# 尝试 GPU 加速 MediaPipe
uv run finger-preprocess --data-dir data --force --hand-side Right --delegate GPU

# 调整边界屏蔽参数
uv run finger-preprocess --data-dir data --hand-side Right --pre-ignore-frames 4 --post-ignore-seconds 1.0
```

**功能**：从 MP4+JSON 对中提取手部骨架特征，生成 `.npz` 文件。

**处理流程**：
1. 按 session 名称匹配 `data/video/*.mp4` 和 `data/labels/*.json`
2. 构建 0-indexed 帧级标签，应用边界裁剪（`BOUNDARY_MARGIN = 2`）
3. 用 MediaPipe VIDEO 模式逐帧提取 21 点骨架
4. 对骨架做手掌局部坐标系归一化（`features.py:normalize_landmarks()`）
5. 构建扩展特征：坐标、速度、拇指-食指/中指距离和方向、有效性标记
6. 计算 `train_mask`：标记动作边界附近的过渡/回弹帧，训练时用 `IGNORE_INDEX = -100` 屏蔽
7. 计算质量指标（检出率、抖动、覆盖度等）
8. 保存压缩 `.npz`

**输出 NPZ 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `landmarks` | float32 [T, 21, 3] | 手掌局部坐标系下的归一化坐标 |
| `features` | float32 [T, 21, 12] | 扩展特征（坐标 3 + 速度 3 + 距离 2 + 方向 3 + 有效性 1） |
| `feature_names` | str array | 特征通道名称 |
| `labels` | int64 [T] | 帧级标签 (0-6) |
| `train_mask` | bool [T] | true=参与训练，false=忽略（过渡/回弹帧） |
| `valid` | bool [T] | MediaPipe 是否检测到手 |
| `fps` | float32 | 视频帧率 |
| `quality_json` | str | JSON 编码的质量指标 |

**`train_mask` 和边界屏蔽**：

这是本项目的一个关键设计。动作边界附近的帧标签模糊，直接用于训练会引入噪声：

- **边界裁剪**：`BOUNDARY_MARGIN = 2`，1-indexed 标注的起止各裁掉 2 帧后作为有效手势标签
- **前置忽略**：`PRE_IGNORE_FRAMES = 4`，动作开始前 4 帧屏蔽
- **后置忽略**：`POST_IGNORE_SECONDS = 1.0`，动作结束后 1 秒内的回弹帧屏蔽
- 被屏蔽的帧仍然有标签，但在训练时通过 `IGNORE_INDEX = -100` 排除出 loss 计算

**`--hand-side` 的重要性**：

默认 `None` 表示取 MediaPipe 检测到的第一只手。如果你的数据集是单手操作（例如只看右手），**必须**传 `--hand-side Right` 或 `--hand-side Left`，否则 MediaPipe 可能偶尔取到错误的手，导致特征混乱。

**帧索引转换**：

JSON 标注中的 `start_frame`/`end_frame` 是 **1-indexed**（采集器按人类习惯编号），`preprocess.py` 在构建标签时转换为 0-indexed：
```python
raw_s = int(ann["start_frame"]) - 1   # 1-indexed → 0-indexed
raw_e = int(ann["end_frame"]) - 1
s = raw_s + BOUNDARY_MARGIN           # 裁掉边界
e = raw_e - BOUNDARY_MARGIN
```

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `data` | 数据根目录 |
| `--force` | `False` | 强制重新提取（跳过已存在的 .npz） |
| `--hand-side` | `None` | 手性过滤（Left/Right），建议单手数据集必须指定 |
| `--delegate` | `CPU` | MediaPipe delegate（GPU 可用性取决于平台） |
| `--pre-ignore-frames` | `4` | 动作开始前忽略帧数 |
| `--post-ignore-seconds` | `1.0` | 动作结束后忽略秒数 |
| `--debug-video` | `False` | 输出骨架 HUD 调试视频 |

> **重要**：预处理只做一次！之后训练/评估直接读取 `.npz`。不要每次训练实验都重新提取 landmarks。

---

### 5.4 数据审计：`finger-audit`

```bash
uv run finger-audit \
  --data-dir data \
  --min-subjects 1 \
  --target-events-per-class 50 \
  --list-bad
```

**功能**：检查数据集覆盖率和特征质量，为训练前提供门禁。

**检查项**：
- 视频/标注/特征文件配对完整性
- 每类手势事件数量是否达标
- 受试者数量和每个受试者的 session 数
- 每个 session 的质量分、检出率、抖动、漏检
- 过时特征（节点数 ≠ 21 或通道数 < 12 → 需 `--force` 重新生成）

**质量指标**（来自 `preprocess.py:compute_quality_metrics()`）：

| 指标 | 含义 | 权重 |
|------|------|------|
| `detect_score` | 骨架检出率（valid_rate） | 45% |
| `jitter_score` | 背景段骨架抖动 p95 | 25% |
| `gap_score` | 最长连续漏检段 | 15% |
| `coverage_score` | 出现的手势类别比例 | 15% |
| **`quality_score`** | **综合加权分（0-100）** | - |

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--target-events-per-class` | `500` | 建议每类事件数下限 |
| `--min-subjects` | `5` | 建议受试者数下限（单人模型可设 1） |
| `--min-quality` | `90.0` | 单 session 综合质量分下限 |
| `--min-valid-rate` | `0.98` | 检出率下限 |
| `--max-bg-jitter-p95` | `0.06` | 背景抖动 p95 上限 |
| `--fail-on-miss` | `False` | 不达标时返回非零退出码（CI 用） |
| `--list-bad` | `False` | 列出不达标 session 详情 |

---

### 5.5 训练：`finger-train`

```bash
# 基本训练
uv run finger-train --data-dir data --epochs 80

# 指定受试者
uv run finger-train --data-dir data --subjects c10

# 调整模型和训练参数
uv run finger-train --data-dir data --chunk-len 512 --train-hop 128 --lambda-boundary 0.3
```

**模型结构**：

```
输入: [B, 12, T, 21]   (12 通道特征 × T 帧 × 21 节点)
        │
        ▼
  Adaptive ST-GCN Encoder
  ├── STGCNBlock(12 → 48)    无 dropout
  ├── STGCNBlock(48 → 96)    dropout * 0.5
  └── STGCNBlock(96 → 128)   dropout
        │
        ▼  空间维度池化: mean over 21 nodes → [B, 128, T]
        │
  ├── Temporal Stage 0 (MS-TCN 风格)
  │   ├── 6 层 DilatedResidualLayer (dilation = 1,2,4,8,16,32)
  │   ├── classifier → [B, 7, T]  帧级类别 logits
  │   └── boundary_head → [B, 2, T]  起止边界概率
  │
  └── Refinement Stage 1
      └── 以 stage0 的 softmax 输出作为输入，再次精炼
```

**损失函数**：

1. **加权帧级交叉熵**：类别权重 = `1/sqrt(count)` 归一化，`IGNORE_INDEX = -100` 屏蔽过渡帧
2. **边界 BCE**：起止边界的二分类损失，正样本权重 8:1
3. **TMSE 平滑损失**：相邻帧 log-softmax 差的 L2，鼓励预测平滑变化

总损失 = CE + λ_boundary × BCE + λ_smooth × TMSE

**数据加载**（`dataset.py`）：

- 将每个 session 的特征切分为固定长度 chunk（默认 256 帧）
- 训练集 chunk 间有重叠（hop=64），验证集 hop=128
- 数据增强：高斯噪声（σ=0.008）加到前 6 个通道，4% 概率随机丢帧
- 自动按帧数比例划分训练/验证集，或按 `--subjects` 过滤

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `data` | 数据根目录 |
| `--checkpoint-dir` | `checkpoints` | 检查点保存目录 |
| `--subjects` | `None` | 指定受试者（逗号分隔），None 为全部 |
| `--val-ratio` | `0.2` | 验证集比例 |
| `--chunk-len` | `256` | 训练 chunk 长度（帧） |
| `--train-hop` | `64` | 训练 chunk 步长 |
| `--batch-size` | `8` | 批大小 |
| `--epochs` | `80` | 训练轮数 |
| `--lr` | `3e-4` | 学习率 |
| `--lambda-boundary` | `0.2` | 边界损失权重 |
| `--lambda-smooth` | `0.15` | 平滑损失权重 |
| `--hidden-dim` | `128` | ST-GCN 隐藏维度 |
| `--temporal-layers` | `6` | MS-TCN 每阶段膨胀卷积层数 |
| `--temporal-stages` | `2` | MS-TCN 精炼阶段数 |
| `--dropout` | `0.25` | Dropout 率 |

**输出**：
- `checkpoints/best.pt`：验证集帧级准确率最高的模型
- `checkpoints/last.pt`：最后一轮模型
- `checkpoints/history.json`：训练历史

---

### 5.6 检测：`finger-detect`

```bash
# 基本检测
uv run finger-detect \
  --video data/video/session.mp4 \
  --checkpoint checkpoints/best.pt \
  --out-json results/session.events.json \
  --hand-side Right

# 同时输出叠加预测的视频
uv run finger-detect \
  --video data/video/session.mp4 \
  --checkpoint checkpoints/best.pt \
  --out-json results/pred.json \
  --out-video results/pred.mp4
```

**处理流程**：
1. 用 MediaPipe VIDEO 模式从视频提取骨架 → 归一化 → 构建扩展特征
2. 将特征按 chunk_len 分块，带 overlap 送入模型推理
3. 多次重叠区域的预测取平均
4. 后处理：置信度过滤 → 加权平滑 → 连续同类合并 → 短片段移除 → 边界精修
5. 输出事件列表和（可选的）叠加视频

**后处理参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--conf-threshold` | `0.55` | 低于此置信度的帧转为背景 |
| `--min-event-ms` | `120` | 移除短于此的误检片段 |
| `--max-gap-ms` | `120` | 合并同类碎片的最大间隔 |
| `--smooth` | `7` | 时序平滑窗口宽度 |
| `--chunk-len` | `512` | 推理分块长度 |
| `--overlap` | `128` | 分块重叠长度（避免块边界伪影） |
| `--delegate` | `CPU` | MediaPipe delegate |

**输出 JSON 格式**：
```json
{
  "task": "offline_gesture_event_detection",
  "video": "data/video/session.mp4",
  "fps": 60.0,
  "total_frames": 3600,
  "events": [
    {
      "gesture": "pinch_index",
      "label": 0,
      "start_frame": 318,
      "end_frame": 362,
      "start_ms": 5300,
      "end_ms": 6033,
      "duration_ms": 733,
      "mean_conf": 0.91
    }
  ],
  "frames": [ ... ]
}
```

---

### 5.7 评估：`finger-eval`

```bash
uv run finger-eval \
  --label data/labels/session.json \
  --pred-json results/session.events.json
```

**功能**：事件级评估，基于 IoU 匹配。

**评估指标**：
- 整体事件 precision、recall、F1
- 每类 precision、recall、F1、TP/FP/FN
- 事件混淆矩阵（GT × Pred）
- 起止帧级时间误差（均值 ± 标准差）
- IoU 均值

**验收门禁**：
- `TARGET_EVENT_F1 = 0.98`
- `TARGET_CLASS_PRECISION = 0.97`
- `TARGET_CLASS_RECALL = 0.97`

使用 `--fail-on-miss` 可在不达标时返回非零退出码，适合 CI 脚本。

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--iou-threshold` | `0.30` | 事件匹配 IoU 阈值 |
| `--target-f1` | `0.98` | 整体 F1 验收线 |
| `--target-class-precision` | `0.97` | 每类 precision 验收线 |
| `--target-class-recall` | `0.97` | 每类 recall 验收线 |
| `--fail-on-miss` | `False` | 不达标时 exit(1) |
| `--out-json` | `None` | 保存完整评估 JSON |

---

## 六、源码文件说明

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `__init__.py` | 包初始化 | `__version__` |
| `labels.py` | 手势标签定义（全项目唯一权威） | `GESTURE_ORDER`, `BACKGROUND_LABEL`, `NUM_CLASSES`, `LABEL_NAMES`, `GESTURE_ZH` |
| `features.py` | 骨架归一化和特征工程 | `normalize_landmarks()`, `build_motion_features()`, `NUM_NODES=21` |
| `hand_tracking.py` | MediaPipe 手部检测封装 | `make_landmarker()`, `detect()`, `detect_video()`, `resolve_model()` |
| `preprocess.py` | MP4+JSON → 骨架特征 NPZ | `process_session()`, `compute_quality_metrics()`, `BOUNDARY_MARGIN=2` |
| `dataset.py` | PyTorch 数据集和划分 | `GestureSegmentationDataset`, `IGNORE_INDEX=-100`, `split_feature_files()` |
| `model.py` | ST-GCN + MS-TCN 模型 | `GestureSegmenter`, `AdaptiveGraphConv`, `probabilities_to_events()` |
| `train.py` | 训练循环 | `train()`, `tmse_loss()`, `masked_accuracy()` |
| `detect.py` | 离线视频事件检测 | `detect_events_in_video()`, `extract_video_features()`, `predict_probabilities()` |
| `eval_events.py` | 事件级评估 | `evaluate()`, `check_targets()`, `print_report()` |
| `audit.py` | 数据审计 | `audit()`, `evaluate_readiness()` |
| `capture.py` | 数据采集 UI | `SessionController`, `CameraStream`, `AsyncVideoWriter` |
| `review.py` | 标注回放检查 | `review()`, `_draw_timeline()`, `_draw_hud()` |
| `video_io.py` | 视频写出工具 | `make_writer()`, `FFMPEGWriter` |

---

## 七、关键设计决策和注意事项

### 7.1 MediaPipe VIDEO 模式

`hand_tracking.py` 中 `detect_video()` 使用 `RunningMode.VIDEO`，不是逐帧 `IMAGE` 模式。

- VIDEO 模式允许 MediaPipe 在帧间复用 tracking 状态，减少重复 palm detection，提高吞吐
- VIDEO 模式的 landmarker 是**单次使用的**：timestamps 必须单调递增。每个 session 创建新的 landmarker，处理完后 close()
- `IMAGE` 模式把每帧当独立图片处理，对视频更慢且无法跟踪

### 7.2 预处理只做一次

`finger-preprocess` 是流水线中最慢的步骤（视频解码 + MediaPipe 提取）。生成的 `.npz` 特征文件可被训练反复读取。不要每次训练实验都重新提取。

### 7.3 train_mask 和 IGNORE_INDEX

`preprocess.py` 生成 `train_mask` 数组，动作边界附近（过渡/回弹）的帧被标记为 `False`。在 `dataset.py` 中，这些帧的标签被替换为 `IGNORE_INDEX = -100`，PyTorch 的 `CrossEntropyLoss` 会自动忽略这些位置。

**不要直接用 `labels` 数组训练！** 必须结合 `train_mask` 处理，否则回弹帧（看起来像反向手势）会严重干扰模型。

### 7.4 手性过滤

MediaPipe 默认返回检测到的第一只手。如果数据集是单手操作：
- 预处理时传 `--hand-side Right` 或 `--hand-side Left`
- 检测时也传对应参数
- 否则可能偶尔取到错误的手，导致特征不一致

### 7.5 视频写出

`video_io.py` 优先使用 ffmpeg/libx264 写 MP4（更好兼容性），如果系统没有 ffmpeg 则回退到 OpenCV 的 mp4v 编码器。

### 7.6 模型加载兼容性

`detect.py:load_model()` 从 checkpoint 的 `args` 字典中读取模型配置，保证加载旧 checkpoint 时参数默认值与当前代码一致。

---

## 八、Gitignored 目录

以下目录不在版本库中，由流水线各步骤创建：

| 目录 | 产生步骤 | 内容 |
|------|----------|------|
| `data/video/` | finger-collect | 采集的 MP4 视频 |
| `data/labels/` | finger-collect | 标注 JSON |
| `data/features/` | finger-preprocess | 骨架特征 NPZ |
| `data/debug/` | finger-preprocess --debug-video | 调试视频 |
| `checkpoints/` | finger-train | 模型检查点 |
| `results/` | finger-detect / finger-eval | 检测结果和评估报告 |
| `.models/` | 自动下载 | MediaPipe hand_landmarker.task |
