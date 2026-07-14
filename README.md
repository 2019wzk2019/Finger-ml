# finger-ml

面向录制视频的离线手势事件检测。

本项目从输入视频中检测手势片段，并输出结构化事件：手势类别、开始帧/时间、结束帧/时间。项目使用 MediaPipe Hand Landmarker 提取 21 点手部骨架，再在骨架序列上训练稠密时序分割模型。

```text
finger-collect      -> data/video/*.mp4 + data/labels/*.json
finger-review       -> 检查动作标注窗和骨架质量
finger-preprocess   -> data/features/*.npz
finger-audit        -> 数据覆盖和质量检查
finger-train        -> checkpoints/best.pt
finger-detect       -> video -> events JSON
finger-eval         -> 事件级 precision/recall/F1 和起止时间误差
```

## 目标

输入：

```text
data/video/example.mp4
```

输出：

```json
{
  "events": [
    {
      "gesture": "pinch_index",
      "label": 0,
      "start_frame": 318,
      "end_frame": 362,
      "start_ms": 5300,
      "end_ms": 6033,
      "mean_conf": 0.91
    }
  ]
}
```

本项目不是按实时当前帧分类器优化的。核心指标是事件 F1、边界时间误差、误检和漏检。

## 安装

```bash
uv sync
uv sync --group train

# 可选：CUDA 版 PyTorch
uv sync --group train-cuda
```

首次使用时，MediaPipe 会自动下载 `hand_landmarker.task` 到 `.models/`。

## 手势类别

| Label | Key | 含义 |
| --- | --- | --- |
| 0 | `pinch_index` | 拇指捏食指尖 |
| 1 | `pinch_middle` | 拇指捏中指尖 |
| 2 | `thumb_slide_up` | 拇指向上滑动 |
| 3 | `thumb_slide_down` | 拇指向下滑动 |
| 4 | `thumb_slide_left` | 拇指向左滑动 |
| 5 | `thumb_slide_right` | 拇指向右滑动 |
| 6 | `background` | 无手势 / 过渡段 |

## 流程

### 1. 采集

```bash
uv run finger-collect --subject c10 --repeats 50 --camera 1
```

采集器会录制连续 MP4，并把动作窗口标注写入 JSON。请尽量保持开始/结束按键标准一致；边界标注质量会直接影响训练出的检测器。

### 2. 回放检查

```bash
uv run finger-review --data-dir data
uv run finger-review --data-dir data --session 20260429_151254_c10 --window 3
```

### 3. 预处理

```bash
uv run finger-preprocess --data-dir data --force --hand-side Right
```

尝试 MediaPipe GPU delegate：

```bash
uv run finger-preprocess --data-dir data --force --hand-side Right --delegate GPU
```

如果当前 Python MediaPipe wheel 或平台无法创建 GPU delegate，请使用 CPU。实测 RTX 4060 笔记本上，VIDEO 模式 CPU 提取 1080p 视频约 80 fps，已经快于 60 fps 录制视频的实时速度。

预处理使用 `RunningMode.VIDEO`，不是逐帧 `IMAGE` 模式。VIDEO 模式允许 MediaPipe 在帧间复用 tracking 状态，减少重复 palm detection，提高吞吐。

特征文件内容：

```text
landmarks     float32 [T, 21, 3]   手掌局部坐标系下的归一化坐标
features      float32 [T, 21, C]   坐标、速度、接触/方向、有效性标记
labels        int64   [T]          帧级标签
train_mask    bool    [T]          false 表示忽略的过渡/回弹帧
valid         bool    [T]          MediaPipe 是否检测成功
fps           float32
quality_json  str
```

### 4. 数据审计

```bash
uv run finger-audit \
  --data-dir data \
  --min-subjects 1 \
  --target-events-per-class 50 \
  --list-bad
```

### 5. 训练

```bash
uv run finger-train --data-dir data --epochs 80 --checkpoint-dir checkpoints
```

模型结构：

```text
21 点骨架序列
-> adaptive ST-GCN encoder
-> MS-TCN-style dense temporal segmentation head
-> 帧级类别 logits [T, 7]
-> 边界 head [start/end, T]
```

训练使用加权帧级交叉熵、边界 BCE 和 TMSE 平滑损失。

常用参数：

```bash
uv run finger-train --data-dir data --subjects c10
uv run finger-train --data-dir data --chunk-len 512 --train-hop 128
uv run finger-train --data-dir data --lambda-boundary 0.3
```

### 6. 检测事件

```bash
uv run finger-detect \
  --video data/video/20260429_151254_c10.mp4 \
  --checkpoint checkpoints/best.pt \
  --out-json results/20260429_151254_c10.events.json \
  --hand-side Right
```

可选输出叠加预测的视频：

```bash
uv run finger-detect \
  --video data/video/20260429_151254_c10.mp4 \
  --checkpoint checkpoints/best.pt \
  --out-json results/pred.json \
  --out-video results/pred.mp4
```

后处理参数：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--conf-threshold` | `0.55` | 低置信度帧转为背景 |
| `--min-event-ms` | `120` | 移除很短的误检片段 |
| `--max-gap-ms` | `120` | 合并同类碎片 |
| `--smooth` | `7` | 按置信度加权的时序平滑 |
| `--chunk-len` | `512` | 模型推理分块长度 |
| `--overlap` | `128` | 分块重叠，避免块边界伪影 |
| `--delegate` | `CPU` | MediaPipe CPU/GPU delegate |

### 7. 评估

```bash
uv run finger-eval \
  --label data/labels/20260429_151254_c10.json \
  --pred-json results/20260429_151254_c10.events.json
```

事件评估器会输出：

- 整体事件 precision、recall 和 F1
- 每类 precision、recall 和 F1
- 事件混淆矩阵
- start/end 帧级时间误差

请使用 held-out session 或 held-out subject 得到有意义的泛化指标。

## MediaPipe 和速度说明

处理视频文件时使用 `RunningMode.VIDEO`。MediaPipe Tasks 的 VIDEO 模式面向按顺序输入、timestamp 单调递增的视频帧，并且可以在帧间使用 tracking。`RunningMode.IMAGE` 会把每一帧都当成独立图片处理，对视频更慢。

当前主要瓶颈：

1. 视频解码 + MediaPipe landmark 提取
2. 骨架 chunk 上的模型推理
3. 可选的叠加视频编码

加速选项：

- 使用 MediaPipe VIDEO 模式，已实现。
- 当已安装的 MediaPipe 包和平台支持时，可以使用 `--delegate GPU`；GPU delegate 创建失败时，代码会回退 CPU。
- 通过 `uv sync --group train-cuda` 让 Torch 训练/推理使用 CUDA。
- 批处理时跳过 `--out-video`；只输出 JSON 会快很多。
- 预处理只做一次，之后训练/评估直接读取 `.npz`；不要每次实验都重新提取 landmarks。

## 项目结构

```text
src/finger_ml/
  capture.py       # 数据采集 UI
  review.py        # 标注和骨架质量回放检查
  preprocess.py    # MP4 + JSON -> 骨架特征
  features.py      # 骨架归一化和运动/接触特征工程
  dataset.py       # session/chunk 时序分割数据集
  model.py         # adaptive ST-GCN + MS-TCN 分割模型
  train.py         # 稠密帧级标注训练循环
  detect.py        # 离线视频事件检测
  eval_events.py   # 事件级评估
  audit.py         # 数据质量和覆盖门禁
  hand_tracking.py # MediaPipe 手部检测/跟踪封装
  video_io.py      # 标准 MP4/H.264 视频写出
```
