from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from time import time

from motion_decoder import MotionState


def _clamp(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return lo
    return min(max(value, lo), hi)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smooth(prev: float, target: float, dt: float, alpha: float, deadzone: float, max_step_per_s: float) -> float:
    if abs(target - prev) < deadzone:
        return prev
    step = _clamp(target - prev, -max_step_per_s * dt, max_step_per_s * dt)
    return prev + step * alpha


@dataclass(slots=True)
class SemanticMapperConfig:
    rotation_alpha: float = 0.42
    expression_alpha: float = 0.62
    deadzone_angle: float = 0.18
    deadzone_unit: float = 0.004
    max_angle_step_per_s: float = 420.0
    max_unit_step_per_s: float = 14.0
    max_yaw: float = 25.0
    max_pitch: float = 15.0
    max_roll: float = 15.0
    stale_hold_s: float = 0.22
    expression_decay_s: float = 0.65
    pose_decay_delay_s: float = 1.0
    pose_decay_s: float = 2.0
    neutral_eye_ratio: float = 0.32
    neutral_lip_ratio: float = 0.02


@dataclass(slots=True)
class SemanticControls:
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    blink_left: float = 0.0
    blink_right: float = 0.0
    mouth_open: float = 0.0
    smile: float = 0.0
    brow_raise: float = 0.0
    pupil_x: float = 0.0
    pupil_y: float = 0.0
    head_x: float = 0.5
    head_y: float = 0.35
    shoulder_x: float = 0.5
    shoulder_y: float = 0.6
    confidence: float = 0.0
    eye_open_ratio: float = 0.32
    lip_open_ratio: float = 0.02
    smile_slider: float = 0.0
    wink_slider: float = 0.0
    eyebrow_slider: float = 0.0
    gaze_x_slider: float = 0.0
    gaze_y_slider: float = 0.0
    lip_variation_zero: float = 0.0
    lip_variation_one: float = 0.0
    lip_variation_two: float = 0.0
    lip_variation_three: float = 0.0
    mov_x: float = 0.0
    mov_y: float = 0.0
    mov_z: float = 1.0
    stale_ms: float = 0.0
    updated_at: float = 0.0

    @property
    def blink_avg(self) -> float:
        return (self.blink_left + self.blink_right) * 0.5

    def debug_dict(self) -> dict:
        data = asdict(self)
        data["blink_avg"] = self.blink_avg
        return data


class SemanticLivePortraitMapper:
    """Maps browser MediaPipe semantics to LivePortrait pose/expression controls."""

    def __init__(self, config: SemanticMapperConfig | None = None):
        self.config = config or SemanticMapperConfig()
        self._target: SemanticControls | None = None
        self._current: SemanticControls | None = None
        self._last_packet_at = 0.0
        self._last_sample_at = 0.0
        self._source_eye_ratio = self.config.neutral_eye_ratio
        self._source_lip_ratio = self.config.neutral_lip_ratio

    @property
    def has_motion(self) -> bool:
        return self._current is not None or self._target is not None

    def reset(self) -> None:
        self._target = None
        self._current = None
        self._last_packet_at = 0.0
        self._last_sample_at = 0.0
        self._source_eye_ratio = self.config.neutral_eye_ratio
        self._source_lip_ratio = self.config.neutral_lip_ratio

    def set_source_ratios(self, eye_ratio: float | None, lip_ratio: float | None) -> None:
        if eye_ratio is not None and math.isfinite(eye_ratio):
            self._source_eye_ratio = _clamp(eye_ratio, 0.18, 0.48)
        if lip_ratio is not None and math.isfinite(lip_ratio):
            self._source_lip_ratio = _clamp(lip_ratio, 0.0, 0.18)

    def update(self, motion: MotionState, now: float | None = None) -> SemanticControls:
        now = now if now is not None else time()
        self._target = self._from_motion(motion)
        self._last_packet_at = now
        return self.sample(now)

    def sample(self, now: float | None = None) -> SemanticControls | None:
        now = now if now is not None else time()
        if self._target is None:
            return self._current

        target = self._target_with_fallback(now)
        if self._current is None:
            self._current = target
            self._last_sample_at = now
            return self._current

        dt = _clamp(now - self._last_sample_at, 1 / 120, 0.2)
        self._current = self._smooth_controls(self._current, target, dt)
        self._last_sample_at = now
        return self._current

    def metrics(self) -> dict:
        controls = self._current.debug_dict() if self._current else {}
        return {
            "smoothing_alpha": self.config.rotation_alpha,
            "expression_alpha": self.config.expression_alpha,
            "has_motion": self.has_motion,
            "source_eye_ratio": round(self._source_eye_ratio, 4),
            "source_lip_ratio": round(self._source_lip_ratio, 4),
            "controls": {k: round(v, 4) if isinstance(v, float) else v for k, v in controls.items()},
        }

    def _from_motion(self, motion: MotionState) -> SemanticControls:
        blink_left = _clamp(motion.blink_left, 0.0, 1.0)
        blink_right = _clamp(motion.blink_right, 0.0, 1.0)
        blink_avg = (blink_left + blink_right) * 0.5
        mouth = _clamp(motion.mouth_open, 0.0, 1.0)
        smile = _clamp(motion.smile, 0.0, 1.0)
        brow = _clamp(motion.brow_raise, 0.0, 1.0)
        pupil_x = _clamp(motion.pupil_x, -1.0, 1.0)
        pupil_y = _clamp(motion.pupil_y, -1.0, 1.0)

        eye_open_ratio = _clamp(self._source_eye_ratio * (1.0 - blink_avg * 0.94), 0.015, 0.8)
        lip_open_ratio = _clamp(self._source_lip_ratio + mouth * 0.56, 0.0, 0.8)

        return SemanticControls(
            yaw=_clamp(motion.yaw, -self.config.max_yaw, self.config.max_yaw),
            pitch=_clamp(motion.pitch, -self.config.max_pitch, self.config.max_pitch),
            roll=_clamp(motion.roll, -self.config.max_roll, self.config.max_roll),
            blink_left=blink_left,
            blink_right=blink_right,
            mouth_open=mouth,
            smile=smile,
            brow_raise=brow,
            pupil_x=pupil_x,
            pupil_y=pupil_y,
            head_x=_clamp(motion.head_x, 0.0, 1.0),
            head_y=_clamp(motion.head_y, 0.0, 1.0),
            shoulder_x=_clamp(motion.shoulder_x, 0.0, 1.0),
            shoulder_y=_clamp(motion.shoulder_y, 0.0, 1.0),
            confidence=_clamp(motion.confidence, 0.0, 1.0),
            eye_open_ratio=eye_open_ratio,
            lip_open_ratio=lip_open_ratio,
            smile_slider=_clamp(smile * 1.25, -0.3, 1.3),
            wink_slider=_clamp((blink_left - blink_right) * 36.0, -39.0, 39.0),
            eyebrow_slider=_clamp(brow * 28.0, -30.0, 30.0),
            gaze_x_slider=_clamp(pupil_x * 24.0, -30.0, 30.0),
            gaze_y_slider=_clamp(pupil_y * 36.0, -63.0, 63.0),
            lip_variation_zero=0.0,
            lip_variation_one=_clamp(-mouth * 5.0 + smile * 3.0, -20.0, 15.0),
            lip_variation_two=_clamp(smile * 8.0, 0.0, 15.0),
            lip_variation_three=_clamp(mouth * 95.0, -90.0, 120.0),
            mov_x=_clamp((motion.head_x - 0.5) * 0.12, -0.19, 0.19),
            mov_y=_clamp((0.35 - motion.head_y) * 0.10, -0.19, 0.19),
            mov_z=1.0,
            stale_ms=0.0,
            updated_at=motion.received_at,
        )

    def _target_with_fallback(self, now: float) -> SemanticControls:
        target = self._target or SemanticControls()
        stale_s = max(0.0, now - self._last_packet_at)
        if stale_s <= self.config.stale_hold_s:
            target.stale_ms = stale_s * 1000.0
            return target

        expression_keep = math.exp(-(stale_s - self.config.stale_hold_s) / self.config.expression_decay_s)
        pose_keep = 1.0
        if stale_s > self.config.pose_decay_delay_s:
            pose_keep = math.exp(-(stale_s - self.config.pose_decay_delay_s) / self.config.pose_decay_s)

        neutral = SemanticControls(
            eye_open_ratio=self._source_eye_ratio,
            lip_open_ratio=self._source_lip_ratio,
            stale_ms=stale_s * 1000.0,
            updated_at=target.updated_at,
        )
        return SemanticControls(
            yaw=_lerp(neutral.yaw, target.yaw, pose_keep),
            pitch=_lerp(neutral.pitch, target.pitch, pose_keep),
            roll=_lerp(neutral.roll, target.roll, pose_keep),
            blink_left=_lerp(neutral.blink_left, target.blink_left, expression_keep),
            blink_right=_lerp(neutral.blink_right, target.blink_right, expression_keep),
            mouth_open=_lerp(neutral.mouth_open, target.mouth_open, expression_keep),
            smile=_lerp(neutral.smile, target.smile, expression_keep),
            brow_raise=_lerp(neutral.brow_raise, target.brow_raise, expression_keep),
            pupil_x=_lerp(neutral.pupil_x, target.pupil_x, expression_keep),
            pupil_y=_lerp(neutral.pupil_y, target.pupil_y, expression_keep),
            head_x=_lerp(neutral.head_x, target.head_x, pose_keep),
            head_y=_lerp(neutral.head_y, target.head_y, pose_keep),
            shoulder_x=target.shoulder_x,
            shoulder_y=target.shoulder_y,
            confidence=_lerp(0.0, target.confidence, expression_keep),
            eye_open_ratio=_lerp(neutral.eye_open_ratio, target.eye_open_ratio, expression_keep),
            lip_open_ratio=_lerp(neutral.lip_open_ratio, target.lip_open_ratio, expression_keep),
            smile_slider=_lerp(neutral.smile_slider, target.smile_slider, expression_keep),
            wink_slider=_lerp(neutral.wink_slider, target.wink_slider, expression_keep),
            eyebrow_slider=_lerp(neutral.eyebrow_slider, target.eyebrow_slider, expression_keep),
            gaze_x_slider=_lerp(neutral.gaze_x_slider, target.gaze_x_slider, expression_keep),
            gaze_y_slider=_lerp(neutral.gaze_y_slider, target.gaze_y_slider, expression_keep),
            lip_variation_zero=_lerp(neutral.lip_variation_zero, target.lip_variation_zero, expression_keep),
            lip_variation_one=_lerp(neutral.lip_variation_one, target.lip_variation_one, expression_keep),
            lip_variation_two=_lerp(neutral.lip_variation_two, target.lip_variation_two, expression_keep),
            lip_variation_three=_lerp(neutral.lip_variation_three, target.lip_variation_three, expression_keep),
            mov_x=_lerp(neutral.mov_x, target.mov_x, pose_keep),
            mov_y=_lerp(neutral.mov_y, target.mov_y, pose_keep),
            mov_z=target.mov_z,
            stale_ms=stale_s * 1000.0,
            updated_at=target.updated_at,
        )

    def _smooth_controls(self, prev: SemanticControls, target: SemanticControls, dt: float) -> SemanticControls:
        angle_alpha = _clamp(self.config.rotation_alpha, 0.01, 1.0)
        expr_alpha = _clamp(self.config.expression_alpha, 0.01, 1.0)
        return SemanticControls(
            yaw=_smooth(prev.yaw, target.yaw, dt, angle_alpha, self.config.deadzone_angle, self.config.max_angle_step_per_s),
            pitch=_smooth(prev.pitch, target.pitch, dt, angle_alpha, self.config.deadzone_angle, self.config.max_angle_step_per_s),
            roll=_smooth(prev.roll, target.roll, dt, angle_alpha, self.config.deadzone_angle, self.config.max_angle_step_per_s),
            blink_left=_smooth(prev.blink_left, target.blink_left, dt, expr_alpha, self.config.deadzone_unit, self.config.max_unit_step_per_s),
            blink_right=_smooth(prev.blink_right, target.blink_right, dt, expr_alpha, self.config.deadzone_unit, self.config.max_unit_step_per_s),
            mouth_open=_smooth(prev.mouth_open, target.mouth_open, dt, expr_alpha, self.config.deadzone_unit, self.config.max_unit_step_per_s),
            smile=_smooth(prev.smile, target.smile, dt, expr_alpha, self.config.deadzone_unit, self.config.max_unit_step_per_s),
            brow_raise=_smooth(prev.brow_raise, target.brow_raise, dt, expr_alpha, self.config.deadzone_unit, self.config.max_unit_step_per_s),
            pupil_x=_smooth(prev.pupil_x, target.pupil_x, dt, expr_alpha, self.config.deadzone_unit, self.config.max_unit_step_per_s),
            pupil_y=_smooth(prev.pupil_y, target.pupil_y, dt, expr_alpha, self.config.deadzone_unit, self.config.max_unit_step_per_s),
            head_x=_smooth(prev.head_x, target.head_x, dt, angle_alpha, 0.001, 3.0),
            head_y=_smooth(prev.head_y, target.head_y, dt, angle_alpha, 0.001, 3.0),
            shoulder_x=target.shoulder_x,
            shoulder_y=target.shoulder_y,
            confidence=_smooth(prev.confidence, target.confidence, dt, expr_alpha, self.config.deadzone_unit, self.config.max_unit_step_per_s),
            eye_open_ratio=_smooth(prev.eye_open_ratio, target.eye_open_ratio, dt, expr_alpha, self.config.deadzone_unit, self.config.max_unit_step_per_s),
            lip_open_ratio=_smooth(prev.lip_open_ratio, target.lip_open_ratio, dt, expr_alpha, self.config.deadzone_unit, self.config.max_unit_step_per_s),
            smile_slider=_smooth(prev.smile_slider, target.smile_slider, dt, expr_alpha, 0.01, 18.0),
            wink_slider=_smooth(prev.wink_slider, target.wink_slider, dt, expr_alpha, 0.08, 260.0),
            eyebrow_slider=_smooth(prev.eyebrow_slider, target.eyebrow_slider, dt, expr_alpha, 0.08, 220.0),
            gaze_x_slider=_smooth(prev.gaze_x_slider, target.gaze_x_slider, dt, expr_alpha, 0.08, 220.0),
            gaze_y_slider=_smooth(prev.gaze_y_slider, target.gaze_y_slider, dt, expr_alpha, 0.08, 320.0),
            lip_variation_zero=_smooth(prev.lip_variation_zero, target.lip_variation_zero, dt, expr_alpha, 0.001, 1.0),
            lip_variation_one=_smooth(prev.lip_variation_one, target.lip_variation_one, dt, expr_alpha, 0.08, 120.0),
            lip_variation_two=_smooth(prev.lip_variation_two, target.lip_variation_two, dt, expr_alpha, 0.08, 120.0),
            lip_variation_three=_smooth(prev.lip_variation_three, target.lip_variation_three, dt, expr_alpha, 0.2, 700.0),
            mov_x=_smooth(prev.mov_x, target.mov_x, dt, angle_alpha, 0.001, 1.0),
            mov_y=_smooth(prev.mov_y, target.mov_y, dt, angle_alpha, 0.001, 1.0),
            mov_z=_smooth(prev.mov_z, target.mov_z, dt, angle_alpha, 0.001, 1.0),
            stale_ms=target.stale_ms,
            updated_at=target.updated_at,
        )
