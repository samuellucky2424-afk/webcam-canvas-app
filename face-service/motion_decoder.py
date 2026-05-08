from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any


def _num(data: dict[str, Any], *names: str, default: float = 0.0, lo: float | None = None, hi: float | None = None) -> float:
    value: Any = default
    for name in names:
        if name in data:
            value = data[name]
            break
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    if lo is not None:
        out = max(lo, out)
    if hi is not None:
        out = min(hi, out)
    return out


@dataclass(slots=True)
class MotionState:
    t: float
    yaw: float
    pitch: float
    roll: float
    blink_left: float
    blink_right: float
    mouth_open: float
    brow_raise: float
    smile: float
    pupil_x: float
    pupil_y: float
    head_x: float
    head_y: float
    shoulder_x: float
    shoulder_y: float
    confidence: float
    received_at: float


class MotionDecoder:
    """Decodes compact semantic JSON into clamped LivePortrait controls."""

    def decode(self, data: dict[str, Any]) -> MotionState:
        return MotionState(
            t=_num(data, "t", default=time() * 1000.0),
            yaw=_num(data, "yaw", "y", default=0.0, lo=-35.0, hi=35.0),
            pitch=_num(data, "pitch", "p", default=0.0, lo=-25.0, hi=25.0),
            roll=_num(data, "roll", "r", default=0.0, lo=-30.0, hi=30.0),
            blink_left=_num(data, "blinkLeft", "blinkL", "bl", default=0.0, lo=0.0, hi=1.0),
            blink_right=_num(data, "blinkRight", "blinkR", "br", default=0.0, lo=0.0, hi=1.0),
            mouth_open=_num(data, "mouthOpen", "mouth", "mo", default=0.0, lo=0.0, hi=1.0),
            brow_raise=_num(data, "browRaise", "brow", default=0.0, lo=0.0, hi=1.0),
            smile=_num(data, "smile", "sm", default=0.0, lo=0.0, hi=1.0),
            pupil_x=_num(data, "pupilX", "ex", default=0.0, lo=-1.0, hi=1.0),
            pupil_y=_num(data, "pupilY", "ey", default=0.0, lo=-1.0, hi=1.0),
            head_x=_num(data, "headX", default=0.5, lo=0.0, hi=1.0),
            head_y=_num(data, "headY", default=0.35, lo=0.0, hi=1.0),
            shoulder_x=_num(data, "shoulderX", default=0.5, lo=0.0, hi=1.0),
            shoulder_y=_num(data, "shoulderY", default=0.6, lo=0.0, hi=1.0),
            confidence=_num(data, "confidence", "c", default=0.0, lo=0.0, hi=1.0),
            received_at=time(),
        )
