"""共享手势标签定义 — 采集、训练、检测全流程统一使用。

本模块是项目内手势标签的唯一权威来源（single source of truth）。
所有需要手势名称或标签编号的代码都必须从这里导入，不要硬编码。

7 个类别：
    标签 0-5 = 6 种手势（GESTURE_ORDER 定义）
    标签 6   = 背景/静息（BACKGROUND_LABEL）
"""

from __future__ import annotations

# 6 种手势名称的有序元组，索引即为标签编号 0-5
GESTURE_ORDER: tuple[str, ...] = (
    "pinch_index",       # 0: 拇指捏食指尖
    "pinch_middle",      # 1: 拇指捏中指尖
    "thumb_slide_up",    # 2: 拇指向上滑动
    "thumb_slide_down",  # 3: 拇指向下滑动
    "thumb_slide_left",  # 4: 拇指向左滑动
    "thumb_slide_right", # 5: 拇指向右滑动
)

# 手势名称 → 标签编号 的映射字典
GESTURE_LABEL: dict[str, int] = {name: idx for idx, name in enumerate(GESTURE_ORDER)}

# 背景/静息标签编号，紧接在手势标签之后
BACKGROUND_LABEL = len(GESTURE_ORDER)  # = 6

# 全部 7 个类别的名称有序元组（手势 0-5 + 背景 6）
LABEL_NAMES: tuple[str, ...] = (*GESTURE_ORDER, "background")

# 分类总数，模型输出层和损失函数的类别维度
NUM_CLASSES = len(LABEL_NAMES)  # = 7

# 中文名称映射，用于 UI 显示
GESTURE_ZH: dict[str, str] = {
    "pinch_index": "捏食指",
    "pinch_middle": "捏中指",
    "thumb_slide_up": "拇指上滑",
    "thumb_slide_down": "拇指下滑",
    "thumb_slide_left": "拇指左滑",
    "thumb_slide_right": "拇指右滑",
    "background": "静息",
}

# 英文全称映射，用于 HUD 显示
GESTURE_EN: dict[str, str] = {
    "pinch_index": "Pinch Index Tip",
    "pinch_middle": "Pinch Middle Tip",
    "thumb_slide_up": "Thumb Slide Up",
    "thumb_slide_down": "Thumb Slide Down",
    "thumb_slide_left": "Thumb Slide Left",
    "thumb_slide_right": "Thumb Slide Right",
    "background": "Background",
}
