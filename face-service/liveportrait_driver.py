from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from inference import EngineConfig, LivePortraitEngine
from motion_decoder import MotionState
from semantic_mapper import SemanticControls, SemanticLivePortraitMapper

logger = logging.getLogger(__name__)


class LivePortraitSemanticDriver:
    """LivePortrait renderer driven directly by semantic controls."""

    def __init__(self, config: EngineConfig):
        self.engine = LivePortraitEngine(config)
        self.mapper = SemanticLivePortraitMapper()
        self.avatar_id: str | None = None
        self.last_render_ms = 0.0

    @property
    def device(self) -> torch.device:
        return self.engine.device

    @property
    def has_source(self) -> bool:
        return self.engine._source_info is not None  # noqa: SLF001

    def load_source(self, path: Path, avatar_id: str) -> None:
        self.engine.load_source(path)
        self.avatar_id = avatar_id
        self.mapper.reset()
        self._cache_source_ratios()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Semantic avatar %s cached on %s", avatar_id, self.device)

    def gpu_memory_mb(self) -> dict[str, float]:
        if self.device.type != "cuda":
            return {"allocated": 0.0, "reserved": 0.0}
        return {
            "allocated": round(torch.cuda.memory_allocated(self.device) / (1024 * 1024), 1),
            "reserved": round(torch.cuda.memory_reserved(self.device) / (1024 * 1024), 1),
        }

    @property
    def has_motion(self) -> bool:
        return self.mapper.has_motion

    def semantic_metrics(self) -> dict:
        return self.mapper.metrics()

    @torch.inference_mode()
    def render(self, motion: MotionState | None) -> np.ndarray:
        if not self.has_source:
            raise RuntimeError("No avatar source has been uploaded.")

        t0 = time.perf_counter()
        controls = self.mapper.update(motion) if motion is not None else self.mapper.sample()
        if controls is None:
            raise RuntimeError("No semantic motion has been received yet.")

        source = self.engine._source_info  # noqa: SLF001
        wrapper = self.engine._pipeline.live_portrait_wrapper  # noqa: SLF001
        inf_cfg = wrapper.inference_cfg

        x_s_info = source["x_s_info"]
        x_c_s = source["x_c_s"]
        x_s = source["x_s"]
        f_s = source["f_s"]

        pitch = x_s_info["pitch"] + self._tensor([[controls.pitch]])
        yaw = x_s_info["yaw"] + self._tensor([[controls.yaw]])
        roll = x_s_info["roll"] + self._tensor([[controls.roll]])
        r_new = self.engine._get_rotation_matrix(pitch, yaw, roll)  # noqa: SLF001

        delta_new = x_s_info["exp"].clone()
        self._apply_expression_delta(delta_new, controls)

        scale_new = x_s_info["scale"]
        t_new = x_s_info["t"].clone()
        t_new[..., 0] += controls.mov_x
        t_new[..., 1] += controls.mov_y
        t_new[..., 2].fill_(0)

        x_d_new = controls.mov_z * scale_new * (x_c_s @ r_new + delta_new) + t_new
        x_d_new = self._apply_ratio_retargeting(wrapper, source, x_s, x_d_new, controls)

        if inf_cfg.flag_stitching:
            x_d_new = wrapper.stitching(x_s, x_d_new)

        x_d_new = x_s + (x_d_new - x_s) * inf_cfg.driving_multiplier
        out = wrapper.warp_decode(f_s, x_s, x_d_new)
        out_rgb = wrapper.parse_output(out["out"])[0]
        self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    def _tensor(self, value: list[list[float]]) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float32, device=self.engine.device)

    def _apply_expression_delta(self, delta: torch.Tensor, controls: SemanticControls) -> None:
        gaze_x = controls.gaze_x_slider
        gaze_y = controls.gaze_y_slider
        smile = controls.smile_slider
        wink = controls.wink_slider
        brow = controls.eyebrow_slider
        lip_zero = controls.lip_variation_zero
        lip_one = controls.lip_variation_one
        lip_two = controls.lip_variation_two
        lip_three = controls.lip_variation_three

        if gaze_x != 0 or gaze_y != 0:
            if gaze_x > 0:
                delta[0, 11, 0] += gaze_x * 0.0007
                delta[0, 15, 0] += gaze_x * 0.001
            else:
                delta[0, 11, 0] += gaze_x * 0.001
                delta[0, 15, 0] += gaze_x * 0.0007
            delta[0, 11, 1] += gaze_y * -0.001
            delta[0, 15, 1] += gaze_y * -0.001
            gaze_blink = -gaze_y / 2.0
            delta[0, 11, 1] += gaze_blink * -0.001
            delta[0, 13, 1] += gaze_blink * 0.0003
            delta[0, 15, 1] += gaze_blink * -0.001
            delta[0, 16, 1] += gaze_blink * 0.0003

        if smile != 0:
            delta[0, 20, 1] += smile * -0.01
            delta[0, 14, 1] += smile * -0.02
            delta[0, 17, 1] += smile * 0.0065
            delta[0, 17, 2] += smile * 0.003
            delta[0, 13, 1] += smile * -0.00275
            delta[0, 16, 1] += smile * -0.00275
            delta[0, 3, 1] += smile * -0.0035
            delta[0, 7, 1] += smile * -0.0035

        if wink != 0:
            delta[0, 11, 1] += wink * 0.001
            delta[0, 13, 1] += wink * -0.0003
            delta[0, 17, 0] += wink * 0.0003
            delta[0, 17, 1] += wink * 0.0003
            delta[0, 3, 1] += wink * -0.0003

        if brow != 0:
            if brow > 0:
                delta[0, 1, 1] += brow * 0.001
                delta[0, 2, 1] += brow * -0.001
            else:
                delta[0, 1, 0] += brow * -0.001
                delta[0, 2, 0] += brow * 0.001
                delta[0, 1, 1] += brow * 0.0003
                delta[0, 2, 1] += brow * -0.0003

        if lip_zero != 0:
            delta[0, 19, 0] += lip_zero
        if lip_one != 0:
            delta[0, 14, 1] += lip_one * 0.001
            delta[0, 3, 1] += lip_one * -0.0005
            delta[0, 7, 1] += lip_one * -0.0005
            delta[0, 17, 2] += lip_one * -0.0005
        if lip_two != 0:
            delta[0, 20, 2] += lip_two * -0.001
            delta[0, 20, 1] += lip_two * -0.001
            delta[0, 14, 1] += lip_two * -0.001
        if lip_three != 0:
            delta[0, 19, 1] += lip_three * 0.001
            delta[0, 19, 2] += lip_three * 0.0001
            delta[0, 17, 1] += lip_three * -0.0001

    def _apply_ratio_retargeting(
        self,
        wrapper,
        source: dict,
        x_s: torch.Tensor,
        x_d_new: torch.Tensor,
        controls: SemanticControls,
    ) -> torch.Tensor:
        source_lmk = source.get("source_lmk")
        if source_lmk is None:
            return x_d_new

        eyes_delta = None
        lip_delta = None
        try:
            combined_eye = wrapper.calc_combined_eye_ratio([[float(controls.eye_open_ratio)]], source_lmk)
            eyes_delta = wrapper.retarget_eye(x_s, combined_eye)
        except Exception:  # noqa: BLE001
            logger.debug("Eye retargeting failed.", exc_info=True)

        try:
            combined_lip = wrapper.calc_combined_lip_ratio([[float(controls.lip_open_ratio)]], source_lmk)
            lip_delta = wrapper.retarget_lip(x_s, combined_lip)
        except Exception:  # noqa: BLE001
            logger.debug("Lip retargeting failed.", exc_info=True)

        if eyes_delta is not None:
            x_d_new = x_d_new + eyes_delta
        if lip_delta is not None:
            x_d_new = x_d_new + lip_delta
        return x_d_new

    def _cache_source_ratios(self) -> None:
        source = self.engine._source_info  # noqa: SLF001
        if source is None or source.get("source_lmk") is None:
            self.mapper.set_source_ratios(None, None)
            return

        wrapper = self.engine._pipeline.live_portrait_wrapper  # noqa: SLF001
        try:
            eye_ratios, lip_ratios = wrapper.calc_ratio([source["source_lmk"]])
            eye_ratio = float(np.asarray(eye_ratios[0]).mean())
            lip_ratio = float(np.asarray(lip_ratios[0]).mean())
            self.mapper.set_source_ratios(eye_ratio, lip_ratio)
            logger.info("Cached source ratios eye=%.3f lip=%.3f", eye_ratio, lip_ratio)
        except Exception:  # noqa: BLE001
            logger.debug("Could not cache source eye/lip ratios.", exc_info=True)
