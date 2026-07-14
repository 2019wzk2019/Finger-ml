"""Shared gesture labels used by collection, training and event detection."""

from __future__ import annotations

GESTURE_ORDER: tuple[str, ...] = (
    "pinch_index",
    "pinch_middle",
    "thumb_slide_up",
    "thumb_slide_down",
    "thumb_slide_left",
    "thumb_slide_right",
)

GESTURE_LABEL: dict[str, int] = {name: idx for idx, name in enumerate(GESTURE_ORDER)}

BACKGROUND_LABEL = len(GESTURE_ORDER)
LABEL_NAMES: tuple[str, ...] = (*GESTURE_ORDER, "background")
NUM_CLASSES = len(LABEL_NAMES)

GESTURE_ZH: dict[str, str] = {
    "pinch_index": "捏食指",
    "pinch_middle": "捏中指",
    "thumb_slide_up": "拇指上滑",
    "thumb_slide_down": "拇指下滑",
    "thumb_slide_left": "拇指左滑",
    "thumb_slide_right": "拇指右滑",
    "background": "静息",
}

GESTURE_EN: dict[str, str] = {
    "pinch_index": "Pinch Index Tip",
    "pinch_middle": "Pinch Middle Tip",
    "thumb_slide_up": "Thumb Slide Up",
    "thumb_slide_down": "Thumb Slide Down",
    "thumb_slide_left": "Thumb Slide Left",
    "thumb_slide_right": "Thumb Slide Right",
    "background": "Background",
}
