from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from inference import EngineConfig
from liveportrait_driver import LivePortraitSemanticDriver
from motion_decoder import MotionState


class AvatarManager:
    """Owns persistent LivePortrait GPU state for uploaded avatars."""

    def __init__(self, upload_dir: Path, engine_config: EngineConfig):
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.driver = LivePortraitSemanticDriver(engine_config)
        self.lock = asyncio.Lock()
        self.current_avatar_id: str | None = None
        self.last_upload_at = 0.0
        self.last_render_at = 0.0

    async def upload(self, filename: str, payload: bytes) -> dict:
        avatar_id = uuid.uuid4().hex
        suffix = Path(filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".png"
        path = self.upload_dir / f"{avatar_id}{suffix}"

        await asyncio.to_thread(path.write_bytes, payload)

        async with self.lock:
            await asyncio.to_thread(self.driver.load_source, path, avatar_id)
            self.current_avatar_id = avatar_id
            self.last_upload_at = time.time()

        return {
            "ok": True,
            "avatar_id": avatar_id,
            "filename": path.name,
            "device": str(self.driver.device),
        }

    @property
    def has_motion(self) -> bool:
        return self.driver.has_motion

    async def render(self, motion: MotionState | None, avatar_id: str | None = None):
        async with self.lock:
            if avatar_id and self.current_avatar_id and avatar_id != self.current_avatar_id:
                raise RuntimeError("Requested avatar is not loaded on this worker.")
            frame = await asyncio.to_thread(self.driver.render, motion)
            self.last_render_at = time.time()
            return frame

    def metrics(self) -> dict:
        return {
            "avatar_id": self.current_avatar_id,
            "avatar_loaded": self.driver.has_source,
            "last_render_ms": round(self.driver.last_render_ms, 2),
            "gpu_memory_mb": self.driver.gpu_memory_mb(),
            "last_upload_at": self.last_upload_at,
            "last_render_at": self.last_render_at,
            "semantic_mapper": self.driver.semantic_metrics(),
        }
