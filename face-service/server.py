"""
WebSocket inference server for LivePortrait.

Wire format (binary frames only):

    Client → server:  raw JPEG bytes of a driving frame.
    Server → client:  raw JPEG bytes of the animated source frame.

Why JPEG over the wire? It compresses ~10× better than raw RGB for face crops,
the browser already has hardware-accelerated JPEG encode/decode, and decoding
is the cheapest part of the per-frame budget. WebSocket is preferred over
WebRTC here for simplicity — there's only one stream per client and we don't
need P2P NAT traversal.

Pacing:
    The server is throttled to `--target-fps` (default 20). Driving frames
    arriving faster than that are dropped (only the latest is animated) so
    the client always sees fresh output and the GPU never queues up stale
    work. This is the single most important latency knob.

Run:
    python server.py --source path/to/avatar.png --target-fps 20

Then connect the browser to ws://host:8765/ws.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Make LivePortrait importable when checkpoints live in third_party/.
# Append (not insert) so our local `inference.py` wrapper wins over the
# upstream `inference.py` script that ships with LivePortrait.
THIRD_PARTY = Path(__file__).resolve().parent / "third_party" / "LivePortrait"
if THIRD_PARTY.exists():
    sys.path.append(str(THIRD_PARTY))

from inference import EngineConfig, LivePortraitEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("face-service")


def lower_process_priority_for_cpu_preview() -> None:
    """Keep CPU inference from starving the browser/MediaPipe process."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        below_normal_priority = 0x00004000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()
        if kernel32.SetPriorityClass(handle, below_normal_priority):
            logger.info("Set process priority to below normal for CPU preview.")
    except Exception:
        logger.exception("Could not lower process priority.")


def build_app(engine: LivePortraitEngine, target_fps: int) -> FastAPI:
    app = FastAPI(title="LivePortrait Face Service")

    # Browser served from a different origin (the static webcam app on
    # :3000) needs CORS to upgrade WebSockets across origins.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    min_interval = 1.0 / max(target_fps, 1)

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "ok": True,
            "device": str(engine.device),
            "size": engine.config.inference_size,
            "target_fps": target_fps,
            "last_inference_ms": round(engine.last_inference_ms, 2),
            "source_path": getattr(engine, "source_path", None),
        }

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        client = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"
        logger.info("Incoming WebSocket connection attempt path=%s client=%s", websocket.url.path, client)
        try:
            await websocket.accept()
        except Exception:
            logger.exception("WebSocket handshake failed path=%s client=%s", websocket.url.path, client)
            raise
        logger.info("WebSocket client connected")

        # Latest-wins queue: producer (recv loop) overwrites; consumer
        # (inference loop) takes whatever is current. Prevents head-of-line
        # blocking when the GPU is slower than the camera.
        latest: dict[str, bytes | None] = {"frame": None}
        stop = asyncio.Event()
        rx_total = 0
        tx_total = 0
        rx_since = 0
        tx_since = 0
        decode_fail_since = 0
        infer_fail_since = 0
        last_rate_t = time.perf_counter()

        async def receiver() -> None:
            nonlocal rx_total, rx_since
            try:
                while not stop.is_set():
                    frame_bytes = await websocket.receive_bytes()
                    if frame_bytes:
                        latest["frame"] = frame_bytes
                        rx_total += 1
                        rx_since += 1
            except WebSocketDisconnect:
                pass
            finally:
                stop.set()

        async def reporter() -> None:
            nonlocal rx_since, tx_since, decode_fail_since, infer_fail_since, last_rate_t
            try:
                while not stop.is_set():
                    await asyncio.sleep(1.0)
                    now = time.perf_counter()
                    dt = now - last_rate_t
                    if dt <= 0:
                        continue
                    rx_fps = rx_since / dt
                    tx_fps = tx_since / dt
                    decode_fps = decode_fail_since / dt
                    infer_fps = infer_fail_since / dt
                    rx_since = 0
                    tx_since = 0
                    decode_fail_since = 0
                    infer_fail_since = 0
                    last_rate_t = now
                    logger.info(
                        "WS rates rx=%.1f fps tx=%.1f fps decode_fail=%.1f/s infer_fail=%.1f/s",
                        rx_fps,
                        tx_fps,
                        decode_fps,
                        infer_fps,
                    )
            finally:
                stop.set()

        async def worker() -> None:
            nonlocal tx_total, tx_since, decode_fail_since, infer_fail_since
            last_send = 0.0
            try:
                while not stop.is_set():
                    # FPS pacing: never run inference faster than target.
                    now = time.perf_counter()
                    wait = min_interval - (now - last_send)
                    if wait > 0:
                        await asyncio.sleep(wait)

                    frame_bytes = latest["frame"]
                    if frame_bytes is None:
                        await asyncio.sleep(0.005)
                        continue
                    latest["frame"] = None  # consume

                    try:
                        logger.info("Processing LivePortrait frame rx_total=%d", rx_total)
                        out_jpeg = await asyncio.to_thread(_process, engine, frame_bytes)
                    except Exception:
                        logger.exception("LivePortrait inference failed")
                        infer_fail_since += 1
                        continue
                    if out_jpeg is None:
                        decode_fail_since += 1
                        continue
                    if stop.is_set():
                        logger.info("Dropping processed frame because the client disconnected.")
                        break

                    try:
                        await websocket.send_bytes(out_jpeg)
                    except (WebSocketDisconnect, RuntimeError):
                        logger.info("Could not send LivePortrait frame; client is already disconnected.")
                        break
                    last_send = time.perf_counter()
                    tx_total += 1
                    tx_since += 1
                    logger.info("Sent LivePortrait frame tx_total=%d bytes=%d", tx_total, len(out_jpeg))
            except WebSocketDisconnect:
                pass
            finally:
                stop.set()

        try:
            await asyncio.gather(receiver(), worker(), reporter())
        except Exception:
            logger.exception("Session error")
        finally:
            logger.info("Client disconnected.")
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001
                pass

    return app


def _process(engine: LivePortraitEngine, jpeg_in: bytes) -> bytes | None:
    """Decode → animate → encode. Runs in a thread so it doesn't block asyncio."""
    arr = np.frombuffer(jpeg_in, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    out = engine.animate(frame)
    ok, jpeg_out = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None
    return jpeg_out.tobytes()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, type=Path,
                   help="Avatar source image to animate.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--target-fps", type=int, default=20,
                   help="Hard cap on output frame rate (15–24 recommended).")
    p.add_argument("--size", type=int, default=256, choices=range(256, 385),
                   metavar="[256-384]",
                   help="Inference resolution. 256=fast, 384=crisp.")
    p.add_argument("--device", default=None, help="cuda | cpu (auto if unset).")
    p.add_argument("--no-fp16", action="store_true",
                   help="Disable half precision on CUDA (debugging).")
    p.add_argument("--checkpoint-dir", default=None, type=Path,
                   help="Override LivePortrait checkpoint directory.")
    args = p.parse_args()
    lower_process_priority_for_cpu_preview()

    cfg_kwargs: dict = {
        "inference_size": args.size,
        "device": args.device,
        "use_fp16": not args.no_fp16,
    }
    if args.checkpoint_dir:
        cfg_kwargs["checkpoint_dir"] = args.checkpoint_dir
    cfg = EngineConfig(**cfg_kwargs)

    engine = LivePortraitEngine(cfg)
    engine.load_source(args.source)

    app = build_app(engine, args.target_fps)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
