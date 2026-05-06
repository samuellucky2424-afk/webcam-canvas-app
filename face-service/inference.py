"""
LivePortrait inference wrapper.

Loads a source avatar image once, then animates it frame-by-frame using a
stream of driving frames (the user's webcam). Each call to `animate(frame)`
returns a single rendered RGB frame matching the avatar identity but driven
by the expression/pose of the input frame.

This module isolates the LivePortrait dependency so the rest of the service
(WebSocket transport, preprocessing, FPS pacing) doesn't need to know about
the model internals. Swap it for a different face-animation model by
implementing the same `LivePortraitEngine` interface.

Setup
-----
LivePortrait is not on PyPI. You must clone the official repo and download
its checkpoints next to this file:

    git clone https://github.com/KwaiVGI/LivePortrait third_party/LivePortrait
    cd third_party/LivePortrait
    bash scripts/download_models.sh

Then run the service from this directory so `third_party/LivePortrait` is on
the Python path (the server adds it automatically — see `server.py`).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """Runtime knobs for the inference engine."""

    # Square side length used for *both* the source crop and the driving
    # frame. LivePortrait was trained at 256; 384 trades a little speed for
    # noticeably crisper output. Keep it within [256, 384] per the spec.
    inference_size: int = 256
    # Force device. None → auto-detect CUDA, fall back to CPU.
    device: Optional[str] = None
    # Use half precision on CUDA. ~1.6× speedup on RTX cards, no measurable
    # quality loss for face animation.
    use_fp16: bool = True
    # Path to LivePortrait checkpoints. The official `download_models.sh`
    # script puts everything under `pretrained_weights/`.
    checkpoint_dir: Path = Path("third_party/LivePortrait/pretrained_weights")


def _select_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LivePortraitEngine:
    """
    Thin wrapper around LivePortrait's pipeline.

    Lifecycle:
        engine = LivePortraitEngine(config)
        engine.load_source(image_path_or_array)   # once, at start-up
        out = engine.animate(driving_bgr_frame)   # per frame
    """

    def __init__(self, config: EngineConfig):
        self.config = config
        self.device = _select_device(config.device)
        self.dtype = torch.float16 if (config.use_fp16 and self.device.type == "cuda") else torch.float32
        self._pipeline = None
        self._source_info = None
        self._source_path = None
        self._driving_anchor = None
        self._motion_multiplier = None
        self._warned_stream_retargeting = False
        self._last_inference_ms = 0.0

        logger.info(
            "Engine init: device=%s dtype=%s size=%d",
            self.device, self.dtype, config.inference_size,
        )

        self._build_pipeline()

    # ---- model wiring ------------------------------------------------------

    def _build_pipeline(self) -> None:
        """
        Constructs the underlying LivePortrait pipeline.

        Pulled into its own method so a different backbone (e.g. a quantised
        or distilled variant) can be dropped in without touching `animate`.
        """
        try:
            # Imports are deferred so the module can be imported (and tested
            # for shape) on machines without the LivePortrait checkpoints.
            from src.config.argument_config import ArgumentConfig
            from src.config.inference_config import InferenceConfig
            from src.config.crop_config import CropConfig
            from src.live_portrait_pipeline import LivePortraitPipeline
            from src.utils.camera import get_rotation_matrix
            from src.utils.helper import calc_motion_multiplier
            from src.utils.io import resize_to_limit
        except ImportError as exc:
            raise RuntimeError(
                "LivePortrait not on sys.path. Clone the repo into "
                "third_party/LivePortrait and ensure the server adds it to "
                "PYTHONPATH (see server.py)."
            ) from exc

        args = ArgumentConfig()
        # Map our config onto LivePortrait's nested configs.
        inference_cfg = InferenceConfig(
            flag_use_half_precision=(self.dtype == torch.float16),
            flag_force_cpu=(self.device.type == "cpu"),
            device_id=0 if self.device.type == "cuda" else -1,
        )
        crop_cfg = CropConfig(
            flag_force_cpu=(self.device.type == "cpu"),
            device_id=0 if self.device.type == "cuda" else -1,
        )

        # LivePortrait's InferenceConfig defaults already point at
        # `<LivePortrait>/pretrained_weights/liveportrait/...` via
        # `make_abs_path`, which is exactly where `download_models.py`
        # stages weights. Only override if the user supplied a custom
        # checkpoint root that differs from the bundled location.
        ckpt_root = self.config.checkpoint_dir
        if not ckpt_root.exists():
            raise FileNotFoundError(
                f"LivePortrait checkpoints not found at {ckpt_root}. "
                "Run `python download_models.py` from face-service/."
            )

        self._pipeline = LivePortraitPipeline(inference_cfg, crop_cfg)
        self._args = args
        self._get_rotation_matrix = get_rotation_matrix
        self._calc_motion_multiplier = calc_motion_multiplier
        self._resize_to_limit = resize_to_limit
        logger.info("LivePortrait pipeline ready.")

    # ---- source management -------------------------------------------------

    def load_source(self, source: str | Path | np.ndarray) -> None:
        """
        Pre-process and cache the source avatar. Call once before streaming.

        Accepts either a path or an already-decoded BGR ndarray. The pipeline
        extracts identity features here so per-frame inference doesn't have
        to repeat them.
        """
        if isinstance(source, (str, Path)):
            self._source_path = str(Path(source).resolve())
            img = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Could not read source image: {source}")
        else:
            self._source_path = "<array>"
            img = source

        wrapper = self._pipeline.live_portrait_wrapper
        cropper = self._pipeline.cropper
        inf_cfg = wrapper.inference_cfg

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = self._resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)

        crop_info = None
        if inf_cfg.flag_do_crop:
            crop_info = cropper.crop_source_image(img_rgb, cropper.crop_cfg)
            if crop_info is None:
                raise RuntimeError(
                    "No face was detected in the LivePortrait source image. "
                    "Use a clear front-facing portrait for --source."
                )
            source_lmk = crop_info["lmk_crop"]
            img_crop_256x256 = crop_info["img_crop_256x256"]
        else:
            source_lmk = cropper.calc_lmk_from_cropped_image(img_rgb)
            img_crop_256x256 = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)

        source_prepared = wrapper.prepare_source(img_crop_256x256)
        x_s_info = wrapper.get_kp_info(source_prepared)
        x_c_s = x_s_info["kp"]
        R_s = self._get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
        f_s = wrapper.extract_feature_3d(source_prepared)
        x_s = wrapper.transform_keypoint(x_s_info)

        lip_delta_before_animation = None
        if inf_cfg.flag_normalize_lip and inf_cfg.flag_relative_motion and source_lmk is not None:
            combined_lip_ratio = wrapper.calc_combined_lip_ratio([0.0], source_lmk)
            if combined_lip_ratio[0][0].item() >= inf_cfg.lip_normalize_threshold:
                lip_delta_before_animation = wrapper.retarget_lip(x_s, combined_lip_ratio)

        self._source_info = {
            "crop_info": crop_info,
            "source_lmk": source_lmk,
            "x_s_info": x_s_info,
            "x_c_s": x_c_s,
            "R_s": R_s,
            "f_s": f_s,
            "x_s": x_s,
            "lip_delta_before_animation": lip_delta_before_animation,
        }
        self._driving_anchor = None
        self._motion_multiplier = None
        logger.info(
            "Source loaded from %s (shape=%s, crop=%s).",
            self._source_path,
            img.shape,
            img_crop_256x256.shape,
        )

    # ---- per-frame inference ----------------------------------------------

    @torch.inference_mode()
    def animate(self, driving_bgr: np.ndarray) -> np.ndarray:
        """
        Animate the cached source using one driving frame.

        :param driving_bgr: HxWx3 uint8 BGR (OpenCV convention).
        :returns: HxWx3 uint8 BGR animated face crop, sized
                  `(inference_size, inference_size)`.
        """
        if self._source_info is None:
            raise RuntimeError("Call load_source() before animate().")

        size = self.config.inference_size
        # Resize once on the CPU before handing to the model. Bilinear is
        # plenty for face driving frames and avoids expensive INTER_AREA on
        # the hot path.
        if driving_bgr.shape[0] != size or driving_bgr.shape[1] != size:
            driving_bgr = cv2.resize(driving_bgr, (size, size), interpolation=cv2.INTER_LINEAR)

        t0 = time.perf_counter()
        out = self._animate_one_frame(driving_bgr)
        self._last_inference_ms = (time.perf_counter() - t0) * 1000.0

        return out

    def _clone_tensor_dict(self, data: dict) -> dict:
        cloned = {}
        for key, value in data.items():
            cloned[key] = value.clone() if isinstance(value, torch.Tensor) else value
        return cloned

    def _animate_one_frame(self, driving_bgr: np.ndarray) -> np.ndarray:
        wrapper = self._pipeline.live_portrait_wrapper
        inf_cfg = wrapper.inference_cfg
        source = self._source_info

        driving_rgb = cv2.cvtColor(driving_bgr, cv2.COLOR_BGR2RGB)
        I_d = wrapper.prepare_source(driving_rgb)
        x_d_i_info = wrapper.get_kp_info(I_d)
        R_d_i = self._get_rotation_matrix(
            x_d_i_info["pitch"],
            x_d_i_info["yaw"],
            x_d_i_info["roll"],
        )

        if self._driving_anchor is None:
            self._driving_anchor = {
                "R_d_0": R_d_i.clone(),
                "x_d_0_info": self._clone_tensor_dict(x_d_i_info),
                "x_d_0_new": None,
            }

        x_s_info = source["x_s_info"]
        x_c_s = source["x_c_s"]
        R_s = source["R_s"]
        x_s = source["x_s"]
        R_d_0 = self._driving_anchor["R_d_0"]
        x_d_0_info = self._driving_anchor["x_d_0_info"]

        delta_new = x_s_info["exp"].clone()
        if inf_cfg.flag_relative_motion:
            if inf_cfg.animation_region in ("all", "pose"):
                R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s
            else:
                R_new = R_s

            if inf_cfg.animation_region in ("all", "exp"):
                delta_new = x_s_info["exp"] + (x_d_i_info["exp"] - x_d_0_info["exp"])
            elif inf_cfg.animation_region == "lip":
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new[:, lip_idx, :] = (
                        x_s_info["exp"] + (x_d_i_info["exp"] - x_d_0_info["exp"])
                    )[:, lip_idx, :]
            elif inf_cfg.animation_region == "eyes":
                for eyes_idx in [11, 13, 15, 16, 18]:
                    delta_new[:, eyes_idx, :] = (
                        x_s_info["exp"] + (x_d_i_info["exp"] - x_d_0_info["exp"])
                    )[:, eyes_idx, :]

            scale_new = (
                x_s_info["scale"] * (x_d_i_info["scale"] / x_d_0_info["scale"])
                if inf_cfg.animation_region == "all"
                else x_s_info["scale"]
            )
            t_new = (
                x_s_info["t"] + (x_d_i_info["t"] - x_d_0_info["t"])
                if inf_cfg.animation_region in ("all", "pose")
                else x_s_info["t"]
            )
        else:
            R_new = R_d_i if inf_cfg.animation_region in ("all", "pose") else R_s
            if inf_cfg.animation_region in ("all", "exp"):
                for idx in [1, 2, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
                    delta_new[:, idx, :] = x_d_i_info["exp"][:, idx, :]
                delta_new[:, 3:5, 1] = x_d_i_info["exp"][:, 3:5, 1]
                delta_new[:, 5, 2] = x_d_i_info["exp"][:, 5, 2]
                delta_new[:, 8, 2] = x_d_i_info["exp"][:, 8, 2]
                delta_new[:, 9, 1:] = x_d_i_info["exp"][:, 9, 1:]
            elif inf_cfg.animation_region == "lip":
                for lip_idx in [6, 12, 14, 17, 19, 20]:
                    delta_new[:, lip_idx, :] = x_d_i_info["exp"][:, lip_idx, :]
            elif inf_cfg.animation_region == "eyes":
                for eyes_idx in [11, 13, 15, 16, 18]:
                    delta_new[:, eyes_idx, :] = x_d_i_info["exp"][:, eyes_idx, :]
            scale_new = x_s_info["scale"]
            t_new = x_d_i_info["t"] if inf_cfg.animation_region in ("all", "pose") else x_s_info["t"]

        t_new = t_new.clone()
        t_new[..., 2].fill_(0)
        x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

        if inf_cfg.flag_relative_motion and inf_cfg.driving_option == "expression-friendly":
            if self._driving_anchor["x_d_0_new"] is None:
                self._driving_anchor["x_d_0_new"] = x_d_i_new.clone()
                self._motion_multiplier = self._calc_motion_multiplier(x_s, x_d_i_new)
            x_d_diff = (x_d_i_new - self._driving_anchor["x_d_0_new"]) * self._motion_multiplier
            x_d_i_new = x_d_diff + x_s

        lip_delta = source["lip_delta_before_animation"]
        has_retargeting = inf_cfg.flag_eye_retargeting or inf_cfg.flag_lip_retargeting
        if not inf_cfg.flag_stitching and not has_retargeting:
            if lip_delta is not None:
                x_d_i_new += lip_delta
        elif inf_cfg.flag_stitching and not has_retargeting:
            x_d_i_new = wrapper.stitching(x_s, x_d_i_new)
            if lip_delta is not None:
                x_d_i_new += lip_delta
        elif inf_cfg.flag_stitching:
            if not self._warned_stream_retargeting:
                logger.warning(
                    "Live retargeting flags are enabled, but streaming mode does not compute driving landmarks; using stitching only."
                )
                self._warned_stream_retargeting = True
            x_d_i_new = wrapper.stitching(x_s, x_d_i_new)

        x_d_i_new = x_s + (x_d_i_new - x_s) * inf_cfg.driving_multiplier
        out = wrapper.warp_decode(source["f_s"], x_s, x_d_i_new)
        out_rgb = wrapper.parse_output(out["out"])[0]
        return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

    # ---- diagnostics -------------------------------------------------------

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    @property
    def source_path(self) -> str | None:
        return self._source_path
