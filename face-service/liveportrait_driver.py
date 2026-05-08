from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from inference import EngineConfig, LivePortraitEngine
from motion_decoder import MotionState

logger = logging.getLogger(__name__)


class LivePortraitSemanticDriver:
    """LivePortrait renderer driven directly by semantic controls."""

    def __init__(self, config: EngineConfig):
        self.engine = LivePortraitEngine(config)
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

    @torch.inference_mode()
    def render(self, motion: MotionState) -> np.ndarray:
        if not self.has_source:
            raise RuntimeError("No avatar source has been uploaded.")

        t0 = time.perf_counter()
        source = self.engine._source_info  # noqa: SLF001
        wrapper = self.engine._pipeline.live_portrait_wrapper  # noqa: SLF001
        inf_cfg = wrapper.inference_cfg

        x_s_info = source["x_s_info"]
        x_c_s = source["x_c_s"]
        x_s = source["x_s"]
        f_s = source["f_s"]

        pitch = x_s_info["pitch"] + self._tensor([[motion.pitch]])
        yaw = x_s_info["yaw"] + self._tensor([[motion.yaw]])
        roll = x_s_info["roll"] + self._tensor([[motion.roll]])
        r_new = self.engine._get_rotation_matrix(pitch, yaw, roll)  # noqa: SLF001

        delta_new = x_s_info["exp"].clone()
        self._apply_expression_delta(delta_new, motion)

        scale_new = x_s_info["scale"]
        t_new = x_s_info["t"].clone()
        t_new[..., 0] += (motion.head_x - 0.5) * 0.035
        t_new[..., 1] += (motion.head_y - 0.35) * 0.028
        t_new[..., 2].fill_(0)

        x_d_new = scale_new * (x_c_s @ r_new + delta_new) + t_new

        if inf_cfg.flag_stitching:
            x_d_new = wrapper.stitching(x_s, x_d_new)

        x_d_new = x_s + (x_d_new - x_s) * inf_cfg.driving_multiplier
        out = wrapper.warp_decode(f_s, x_s, x_d_new)
        out_rgb = wrapper.parse_output(out["out"])[0]
        self.last_render_ms = (time.perf_counter() - t0) * 1000.0
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    def _tensor(self, value: list[list[float]]) -> torch.Tensor:
        return torch.tensor(value, dtype=torch.float32, device=self.engine.device)

    def _apply_expression_delta(self, delta: torch.Tensor, motion: MotionState) -> None:
        pupil_x = self._tensor([[motion.pupil_x]]).squeeze()
        pupil_y = self._tensor([[motion.pupil_y]]).squeeze()
        smile = self._tensor([[motion.smile]]).squeeze()
        brow = self._tensor([[motion.brow_raise]]).squeeze()
        mouth = self._tensor([[motion.mouth_open]]).squeeze()
        blink = self._tensor([[(motion.blink_left + motion.blink_right) * 0.5]]).squeeze()
        wink = self._tensor([[motion.blink_left - motion.blink_right]]).squeeze()

        if pupil_x != 0 or pupil_y != 0:
            if pupil_x > 0:
                delta[0, 11, 0] += pupil_x * 0.0007
                delta[0, 15, 0] += pupil_x * 0.001
            else:
                delta[0, 11, 0] += pupil_x * 0.001
                delta[0, 15, 0] += pupil_x * 0.0007
            delta[0, 11, 1] += pupil_y * -0.001
            delta[0, 15, 1] += pupil_y * -0.001

        if smile != 0:
            delta[0, 20, 1] += smile * -0.01
            delta[0, 14, 1] += smile * -0.02
            delta[0, 17, 1] += smile * 0.0065
            delta[0, 17, 2] += smile * 0.003
            delta[0, 13, 1] += smile * -0.00275
            delta[0, 16, 1] += smile * -0.00275
            delta[0, 3, 1] += smile * -0.0035
            delta[0, 7, 1] += smile * -0.0035

        if brow != 0:
            delta[0, 1, 1] += brow * 0.001
            delta[0, 2, 1] += brow * -0.001

        if mouth != 0:
            delta[0, 14, 1] += mouth * 0.005
            delta[0, 17, 1] += mouth * -0.0015
            delta[0, 19, 1] += mouth * 0.001
            delta[0, 20, 2] += mouth * -0.001

        if blink != 0:
            delta[0, 11, 1] += blink * 0.001
            delta[0, 13, 1] += blink * -0.0003
            delta[0, 15, 1] += blink * 0.001
            delta[0, 16, 1] += blink * -0.0003

        if wink != 0:
            delta[0, 11, 1] += wink * 0.0006
            delta[0, 13, 1] += wink * -0.00025
            delta[0, 17, 0] += wink * 0.00025
