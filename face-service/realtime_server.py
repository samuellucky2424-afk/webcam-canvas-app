"""
Realtime LivePortrait WebSocket server.

Run from this directory:

    uvicorn realtime_server:app --host 0.0.0.0 --port 8765

Environment knobs:

    LIVEPORTRAIT_SOURCE=../assets/person.jpg
    LIVEPORTRAIT_DEVICE=cuda
    LIVEPORTRAIT_REQUIRE_CUDA=1
    LIVEPORTRAIT_SIZE=256
    LIVEPORTRAIT_TARGET_FPS=25
    LIVEPORTRAIT_JPEG_QUALITY=45
    LIVEPORTRAIT_OUTPUT_SIZE=160
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

try:
    import onnxruntime as ort
except Exception as exc:  # noqa: BLE001
    ort = None
    ORT_IMPORT_ERROR: Exception | None = exc
else:
    ORT_IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent
THIRD_PARTY = BASE_DIR / "third_party" / "LivePortrait"
DEFAULT_SOURCE = BASE_DIR.parent / "assets" / "person.jpg"
DEFAULT_CHECKPOINT_DIR = THIRD_PARTY / "pretrained_weights"

# Append so local face-service/inference.py wins over upstream inference.py.
if THIRD_PARTY.exists():
    sys.path.append(str(THIRD_PARTY))

from inference import EngineConfig, LivePortraitEngine  # noqa: E402


logging.basicConfig(
    level=os.getenv("LIVEPORTRAIT_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("liveportrait-realtime")

MODEL_NAMES = (
    "appearance_feature_extractor",
    "motion_extractor",
    "warping_module",
    "spade_generator",
    "stitching_retargeting_module",
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _resolve_path(raw: str | None, default: Path) -> Path:
    path = Path(raw) if raw else default
    if not path.is_absolute():
        path = (BASE_DIR / path).resolve()
    return path


@dataclass(frozen=True)
class RuntimeConfig:
    source: Path
    checkpoint_dir: Path
    device: str
    require_cuda: bool
    use_fp16: bool
    inference_size: int
    target_fps: int
    jpeg_quality: int
    output_size: int
    warmup: bool

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        size = _env_int("LIVEPORTRAIT_SIZE", 256)
        if size < 256 or size > 384:
            raise ValueError("LIVEPORTRAIT_SIZE must be between 256 and 384.")

        target_fps = _env_int("LIVEPORTRAIT_TARGET_FPS", 25)
        if target_fps < 1:
            raise ValueError("LIVEPORTRAIT_TARGET_FPS must be at least 1.")

        jpeg_quality = _env_int("LIVEPORTRAIT_JPEG_QUALITY", 45)
        jpeg_quality = max(35, min(jpeg_quality, 65))

        output_size = _env_int("LIVEPORTRAIT_OUTPUT_SIZE", 160)
        output_size = max(128, min(output_size, 224))

        return cls(
            source=_resolve_path(os.getenv("LIVEPORTRAIT_SOURCE"), DEFAULT_SOURCE),
            checkpoint_dir=_resolve_path(
                os.getenv("LIVEPORTRAIT_CHECKPOINT_DIR"),
                DEFAULT_CHECKPOINT_DIR,
            ),
            device=os.getenv("LIVEPORTRAIT_DEVICE", "cuda"),
            require_cuda=_env_bool("LIVEPORTRAIT_REQUIRE_CUDA", True),
            use_fp16=_env_bool("LIVEPORTRAIT_FP16", True),
            inference_size=size,
            target_fps=target_fps,
            jpeg_quality=jpeg_quality,
            output_size=output_size,
            warmup=_env_bool("LIVEPORTRAIT_WARMUP", True),
        )


@dataclass
class QueuedFrame:
    payload: bytes
    received_at: float
    sequence: int
    pose: dict | None = None


class LatestFrameSlot:
    """A one-frame queue: every put replaces any unsent frame."""

    def __init__(self) -> None:
        self._latest: QueuedFrame | None = None
        self._event = asyncio.Event()
        self._lock = asyncio.Lock()
        self._sequence = 0

    async def put(self, payload: bytes, pose: dict | None = None) -> bool:
        async with self._lock:
            dropped_stale = self._latest is not None
            self._sequence += 1
            self._latest = QueuedFrame(
                payload=payload,
                received_at=time.perf_counter(),
                sequence=self._sequence,
                pose=pose,
            )
            self._event.set()
            return dropped_stale

    async def next(self, stop: asyncio.Event) -> QueuedFrame | None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(self._event.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                continue

            async with self._lock:
                frame = self._latest
                self._latest = None
                self._event.clear()

            if frame is not None:
                return frame
        return None

    async def depth(self) -> int:
        async with self._lock:
            return 1 if self._latest is not None else 0


@dataclass
class RuntimeMetrics:
    active_connections: int = 0
    rx_total: int = 0
    tx_total: int = 0
    stale_dropped: int = 0
    decode_errors: int = 0
    inference_errors: int = 0
    send_errors: int = 0
    rx_bytes: int = 0
    tx_bytes: int = 0
    pose_total: int = 0
    pose_bytes: int = 0
    last_pose_frame_id: int = -1
    queue_depth: int = 0
    last_latency_ms: float = 0.0
    last_inference_ms: float = 0.0
    sent_at: Deque[float] = field(default_factory=lambda: deque(maxlen=120))

    @property
    def inference_fps(self) -> float:
        if len(self.sent_at) < 2:
            return 0.0
        elapsed = self.sent_at[-1] - self.sent_at[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.sent_at) - 1) / elapsed


class RealtimeState:
    def __init__(self) -> None:
        self.config: RuntimeConfig | None = None
        self.engine: LivePortraitEngine | None = None
        self.onnx_providers: list[str] = []
        self.metrics = RuntimeMetrics()
        self.infer_lock = asyncio.Lock()
        self.started_at = 0.0

    def start(self) -> None:
        self.config = RuntimeConfig.from_env()
        logger.info(
            "Starting LivePortrait realtime server source=%s device=%s size=%d target_fps=%d",
            self.config.source,
            self.config.device,
            self.config.inference_size,
            self.config.target_fps,
        )
        logger.info("Expected bind command: uvicorn realtime_server:app --host 0.0.0.0 --port 8765")

        _configure_cuda(self.config)
        self.onnx_providers = _validate_onnx_runtime(self.config)

        cfg = EngineConfig(
            inference_size=self.config.inference_size,
            device=self.config.device,
            use_fp16=self.config.use_fp16,
            checkpoint_dir=self.config.checkpoint_dir,
        )

        try:
            self.engine = LivePortraitEngine(cfg)
            self._assert_models_loaded()
            self.engine.load_source(self.config.source)
            if self.config.warmup:
                self._warmup_engine()
        except Exception:
            logger.exception("LivePortrait startup failed.")
            self.cleanup_cuda()
            raise

        self.started_at = time.time()
        logger.info("LivePortrait realtime server ready.")

    def _assert_models_loaded(self) -> None:
        if self.engine is None:
            raise RuntimeError("LivePortrait engine has not been created.")
        wrapper = self.engine._pipeline.live_portrait_wrapper  # noqa: SLF001
        missing = [name for name in MODEL_NAMES if getattr(wrapper, name, None) is None]
        if missing:
            raise RuntimeError(f"LivePortrait model(s) failed to initialize: {', '.join(missing)}")
        logger.info("LivePortrait models initialized once at startup: %s", ", ".join(MODEL_NAMES))

    def _warmup_engine(self) -> None:
        if self.engine is None or self.config is None:
            return
        warmup_frame = cv2.imread(str(self.config.source), cv2.IMREAD_COLOR)
        if warmup_frame is None:
            logger.warning("Skipping warmup; could not read source image %s", self.config.source)
            return
        logger.info("Warming LivePortrait CUDA buffers.")
        with torch.inference_mode():
            _ = self.engine.animate(warmup_frame)
        self.engine._driving_anchor = None  # noqa: SLF001
        self.engine._motion_multiplier = None  # noqa: SLF001

    def cleanup_cuda(self) -> None:
        if torch.cuda.is_available():
            with suppress(Exception):
                torch.cuda.empty_cache()
            with suppress(Exception):
                torch.cuda.ipc_collect()
            logger.info("CUDA memory cache released.")

    def health(self) -> dict:
        cfg = self.config
        engine = self.engine
        return {
            "ok": engine is not None,
            "route": "/ws",
            "bind": "0.0.0.0:8765",
            "compact_pose": True,
            "device": str(engine.device) if engine else (cfg.device if cfg else None),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "onnx_providers": self.onnx_providers,
            "source_path": str(cfg.source) if cfg else None,
            "size": cfg.inference_size if cfg else None,
            "output_size": cfg.output_size if cfg else None,
            "jpeg_quality": cfg.jpeg_quality if cfg else None,
            "target_fps": cfg.target_fps if cfg else None,
            "active_connections": self.metrics.active_connections,
            "rx_total": self.metrics.rx_total,
            "tx_total": self.metrics.tx_total,
            "rx_bytes": self.metrics.rx_bytes,
            "tx_bytes": self.metrics.tx_bytes,
            "pose_total": self.metrics.pose_total,
            "pose_bytes": self.metrics.pose_bytes,
            "last_pose_frame_id": self.metrics.last_pose_frame_id,
            "avg_rx_frame_bytes": round(self.metrics.rx_bytes / self.metrics.rx_total, 1)
            if self.metrics.rx_total
            else 0,
            "avg_tx_frame_bytes": round(self.metrics.tx_bytes / self.metrics.tx_total, 1)
            if self.metrics.tx_total
            else 0,
            "stale_dropped": self.metrics.stale_dropped,
            "decode_errors": self.metrics.decode_errors,
            "inference_errors": self.metrics.inference_errors,
            "queue_depth": self.metrics.queue_depth,
            "latency_ms": round(self.metrics.last_latency_ms, 2),
            "last_inference_ms": round(self.metrics.last_inference_ms, 2),
            "inference_fps": round(self.metrics.inference_fps, 2),
            "uptime_s": round(time.time() - self.started_at, 1) if self.started_at else 0,
        }


def _configure_cuda(config: RuntimeConfig) -> None:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    cuda_available = torch.cuda.is_available()
    if config.require_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA is required but torch.cuda.is_available() is false. "
            "Install a CUDA PyTorch wheel or set LIVEPORTRAIT_REQUIRE_CUDA=0 for CPU debugging."
        )

    if cuda_available:
        torch.backends.cudnn.benchmark = True
        with suppress(Exception):
            torch.set_float32_matmul_precision("high")
        if config.device.startswith("cuda:"):
            torch.cuda.set_device(int(config.device.split(":", 1)[1]))
        else:
            torch.cuda.set_device(0)
        logger.info("CUDA ready: %s", torch.cuda.get_device_name(torch.cuda.current_device()))


def _validate_onnx_runtime(config: RuntimeConfig) -> list[str]:
    if ort is None:
        if config.require_cuda:
            raise RuntimeError("onnxruntime could not be imported.") from ORT_IMPORT_ERROR
        logger.warning("onnxruntime could not be imported: %s", ORT_IMPORT_ERROR)
        return []

    providers = list(ort.get_available_providers())
    logger.info("ONNX Runtime providers available: %s", providers)
    if config.require_cuda and "CUDAExecutionProvider" not in providers:
        raise RuntimeError(
            "CUDAExecutionProvider is not available in ONNX Runtime. "
            f"Available providers: {providers}. Install/fix onnxruntime-gpu and CUDA DLLs."
        )
    return providers


def _process_jpeg(
    engine: LivePortraitEngine,
    jpeg_in: bytes,
    jpeg_quality: int,
    output_size: int,
) -> tuple[bytes, float]:
    arr = np.frombuffer(jpeg_in, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode incoming JPEG frame.")

    with torch.inference_mode():
        out = engine.animate(frame)

    if output_size and (out.shape[0] != output_size or out.shape[1] != output_size):
        out = cv2.resize(out, (output_size, output_size), interpolation=cv2.INTER_AREA)

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    progressive_flag = getattr(cv2, "IMWRITE_JPEG_PROGRESSIVE", None)
    if progressive_flag is not None:
        encode_params.extend([int(progressive_flag), 0])
    optimize_flag = getattr(cv2, "IMWRITE_JPEG_OPTIMIZE", None)
    if optimize_flag is not None:
        encode_params.extend([int(optimize_flag), 1])

    ok, jpeg_out = cv2.imencode(
        ".jpg",
        out,
        encode_params,
    )
    if not ok:
        raise RuntimeError("Could not encode LivePortrait output JPEG.")
    return jpeg_out.tobytes(), engine.last_inference_ms


state = RealtimeState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.start()
    try:
        yield
    finally:
        logger.info("Shutting down LivePortrait realtime server.")
        state.engine = None
        state.cleanup_cuda()


app = FastAPI(title="Realtime LivePortrait Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz() -> dict:
    return state.health()


def _query_int(websocket: WebSocket, names: tuple[str, ...], default: int, low: int, high: int) -> int:
    for name in names:
        raw = websocket.query_params.get(name)
        if raw is None:
            continue
        try:
            value = int(float(raw))
        except ValueError:
            continue
        return max(low, min(value, high))
    return default


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    client = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
    logger.info("Incoming WebSocket connection attempt path=%s client=%s", websocket.url.path, client)
    try:
        await websocket.accept()
    except Exception:
        logger.exception("WebSocket handshake failed path=%s client=%s", websocket.url.path, client)
        raise

    logger.info("WebSocket client connected")
    state.metrics.active_connections += 1

    slot = LatestFrameSlot()
    stop = asyncio.Event()
    min_interval = 1.0 / max(state.config.target_fps if state.config else 25, 1)
    session_quality = _query_int(websocket, ("q", "quality", "jpeg_quality"), state.config.jpeg_quality if state.config else 45, 35, 65)
    session_output_size = _query_int(websocket, ("out", "output", "output_size"), state.config.output_size if state.config else 160, 128, 224)
    requested_driver_size = _query_int(websocket, ("driver", "driver_size"), 0, 0, 224)
    logger.info(
        "WebSocket realtime payload config client=%s driver=%s output=%d jpeg_quality=%d progressive=off",
        client,
        requested_driver_size or "-",
        session_output_size,
        session_quality,
    )
    rx_since = 0
    tx_since = 0
    rx_bytes_since = 0
    tx_bytes_since = 0
    pose_since = 0
    pose_bytes_since = 0
    latest_pose: dict | None = None
    last_report = time.perf_counter()

    async def receiver() -> None:
        nonlocal rx_since, rx_bytes_since, pose_since, pose_bytes_since, latest_pose
        try:
            while not stop.is_set():
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect

                text = message.get("text")
                if text is not None:
                    text_bytes = len(text.encode("utf-8"))
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        logger.debug("Ignoring non-JSON websocket text message from %s.", client)
                        continue
                    if isinstance(data, dict) and data.get("type") == "pose":
                        latest_pose = data
                        state.metrics.pose_total += 1
                        state.metrics.pose_bytes += text_bytes
                        with suppress(TypeError, ValueError):
                            state.metrics.last_pose_frame_id = int(data.get("frameId", -1))
                        pose_since += 1
                        pose_bytes_since += text_bytes
                    else:
                        logger.debug("Ignoring unsupported websocket text message from %s.", client)
                    continue

                payload = message.get("bytes")
                if not payload:
                    continue
                dropped = await slot.put(payload, latest_pose)
                state.metrics.rx_total += 1
                state.metrics.rx_bytes += len(payload)
                rx_since += 1
                rx_bytes_since += len(payload)
                if dropped:
                    state.metrics.stale_dropped += 1
                state.metrics.queue_depth = await slot.depth()
        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected during receive.")
        except Exception:
            logger.exception("WebSocket receive loop failed.")
        finally:
            stop.set()

    async def worker() -> None:
        nonlocal tx_since, tx_bytes_since
        last_send = 0.0
        while not stop.is_set():
            wait = min_interval - (time.perf_counter() - last_send)
            if wait > 0:
                await asyncio.sleep(wait)

            queued = await slot.next(stop)
            if queued is None:
                continue

            state.metrics.queue_depth = await slot.depth()
            if state.engine is None or state.config is None:
                logger.error("LivePortrait engine is not ready.")
                stop.set()
                return

            try:
                async with state.infer_lock:
                    out_jpeg, inference_ms = await asyncio.to_thread(
                        _process_jpeg,
                        state.engine,
                        queued.payload,
                        session_quality,
                        session_output_size,
                    )
            except ValueError:
                state.metrics.decode_errors += 1
                logger.exception("Incoming JPEG decode failed.")
                continue
            except Exception:
                state.metrics.inference_errors += 1
                logger.exception("LivePortrait inference failed.")
                continue

            try:
                await websocket.send_bytes(out_jpeg)
            except (WebSocketDisconnect, RuntimeError):
                state.metrics.send_errors += 1
                logger.info("WebSocket send failed because the client disconnected.")
                break

            now = time.perf_counter()
            latency_ms = (now - queued.received_at) * 1000.0
            state.metrics.tx_total += 1
            state.metrics.tx_bytes += len(out_jpeg)
            state.metrics.last_latency_ms = latency_ms
            state.metrics.last_inference_ms = inference_ms
            state.metrics.sent_at.append(now)
            tx_since += 1
            tx_bytes_since += len(out_jpeg)
            last_send = now

    async def reporter() -> None:
        nonlocal rx_since, tx_since, rx_bytes_since, tx_bytes_since, pose_since, pose_bytes_since, last_report
        while not stop.is_set():
            await asyncio.sleep(1.0)
            now = time.perf_counter()
            elapsed = max(now - last_report, 1e-6)
            queue_depth = await slot.depth()
            state.metrics.queue_depth = queue_depth
            logger.info(
                "LivePortrait ws stats inference_fps=%.1f rx=%.1f/s tx=%.1f/s queue_depth=%d "
                "up=%.1fKB/s pose=%.1f/s pose_up=%.1fKB/s down=%.1fKB/s "
                "latency=%.1fms inference=%.1fms stale_dropped=%d",
                state.metrics.inference_fps,
                rx_since / elapsed,
                tx_since / elapsed,
                queue_depth,
                (rx_bytes_since / 1024) / elapsed,
                pose_since / elapsed,
                (pose_bytes_since / 1024) / elapsed,
                (tx_bytes_since / 1024) / elapsed,
                state.metrics.last_latency_ms,
                state.metrics.last_inference_ms,
                state.metrics.stale_dropped,
            )
            rx_since = 0
            tx_since = 0
            rx_bytes_since = 0
            tx_bytes_since = 0
            pose_since = 0
            pose_bytes_since = 0
            last_report = now

    tasks = [
        asyncio.create_task(receiver(), name="liveportrait-ws-receiver"),
        asyncio.create_task(worker(), name="liveportrait-ws-worker"),
        asyncio.create_task(reporter(), name="liveportrait-ws-reporter"),
    ]

    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        state.metrics.queue_depth = 0
        state.metrics.active_connections = max(0, state.metrics.active_connections - 1)
        logger.info("WebSocket client disconnected")
        with suppress(Exception):
            await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "realtime_server:app",
        host=os.getenv("LIVEPORTRAIT_HOST", "0.0.0.0"),
        port=_env_int("LIVEPORTRAIT_PORT", 8765),
        log_level=os.getenv("LIVEPORTRAIT_UVICORN_LOG_LEVEL", "info"),
    )
