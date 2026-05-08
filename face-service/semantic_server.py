from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

THIRD_PARTY = Path(__file__).resolve().parent / "third_party" / "LivePortrait"
if THIRD_PARTY.exists():
    sys.path.append(str(THIRD_PARTY))

from avatar_manager import AvatarManager  # noqa: E402
from inference import EngineConfig  # noqa: E402
from motion_decoder import MotionDecoder, MotionState  # noqa: E402

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("semantic-liveportrait")


def build_app(manager: AvatarManager, target_fps: int, jpeg_quality: int) -> FastAPI:
    app = FastAPI(title="Semantic LivePortrait Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    decoder = MotionDecoder()
    min_interval = 1.0 / max(target_fps, 1)
    metrics = {
        "packets": 0,
        "dropped_packets": 0,
        "sent_frames": 0,
        "packets_since": 0,
        "frames_since": 0,
        "last_rate_t": time.perf_counter(),
        "packet_rate": 0.0,
        "inference_fps": 0.0,
        "last_ws_latency_ms": None,
    }

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "target_fps": target_fps, **manager.metrics(), **_rate_snapshot(metrics)}

    @app.post("/avatar/upload")
    async def upload_avatar(image: UploadFile = File(...)) -> dict:
        if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WebP portrait.")
        payload = await image.read()
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Avatar image must be 10 MB or smaller.")
        if not payload:
            raise HTTPException(status_code=400, detail="Empty upload.")

        started = time.perf_counter()
        try:
            result = await manager.upload(image.filename or "avatar.png", payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Avatar upload failed.")
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        result["preprocess_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        logger.info("Avatar uploaded id=%s preprocess_ms=%.1f", result["avatar_id"], result["preprocess_ms"])
        return result

    @app.websocket("/ws/semantic")
    async def semantic_ws(websocket: WebSocket) -> None:
        avatar_id = websocket.query_params.get("avatar_id")
        await websocket.accept()
        latest: dict[str, MotionState | None] = {"motion": None}
        stop = asyncio.Event()
        last_send = 0.0
        session_drops = 0

        async def receiver() -> None:
            nonlocal session_drops
            try:
                while not stop.is_set():
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        raise WebSocketDisconnect
                    text = message.get("text")
                    data_bytes = message.get("bytes")
                    if text is None and data_bytes is not None:
                        text = data_bytes.decode("utf-8", errors="ignore")
                    if not text:
                        continue
                    try:
                        packet = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    if packet.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong", "t": packet.get("t")}))
                        continue
                    motion = decoder.decode(packet)
                    if latest["motion"] is not None:
                        session_drops += 1
                        metrics["dropped_packets"] += 1
                    latest["motion"] = motion
                    metrics["packets"] += 1
                    metrics["packets_since"] += 1
                    if motion.t:
                        metrics["last_ws_latency_ms"] = max(0.0, time.time() * 1000.0 - motion.t)
            except WebSocketDisconnect:
                pass
            finally:
                stop.set()

        async def renderer() -> None:
            nonlocal last_send
            try:
                while not stop.is_set():
                    wait = min_interval - (time.perf_counter() - last_send)
                    if wait > 0:
                        await asyncio.sleep(wait)

                    motion = latest["motion"]
                    if motion is None:
                        await asyncio.sleep(0.004)
                        continue
                    latest["motion"] = None

                    try:
                        frame = await manager.render(motion, avatar_id)
                        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                        if not ok:
                            continue
                        await websocket.send_bytes(encoded.tobytes())
                    except (WebSocketDisconnect, RuntimeError):
                        break
                    except Exception:  # noqa: BLE001
                        logger.exception("Semantic render failed.")
                        continue

                    last_send = time.perf_counter()
                    metrics["sent_frames"] += 1
                    metrics["frames_since"] += 1
                    _refresh_rates(metrics)
            finally:
                stop.set()

        async def reporter() -> None:
            try:
                while not stop.is_set():
                    await asyncio.sleep(1.0)
                    rates = _refresh_rates(metrics)
                    gpu = manager.metrics()["gpu_memory_mb"]
                    logger.info(
                        "semantic ws packets=%.1f/s render=%.1f/s dropped=%d render_ms=%.1f ws_latency=%s gpu=%sMB",
                        rates["packet_rate"],
                        rates["inference_fps"],
                        metrics["dropped_packets"],
                        manager.driver.last_render_ms,
                        "-" if metrics["last_ws_latency_ms"] is None else f"{metrics['last_ws_latency_ms']:.0f}ms",
                        gpu,
                    )
            finally:
                stop.set()

        try:
            await asyncio.gather(receiver(), renderer(), reporter())
        finally:
            logger.info("Semantic websocket closed avatar=%s session_drops=%d", avatar_id, session_drops)
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001
                pass

    return app


def _refresh_rates(metrics: dict) -> dict:
    now = time.perf_counter()
    elapsed = now - metrics["last_rate_t"]
    if elapsed >= 1.0:
        metrics["packet_rate"] = metrics["packets_since"] / elapsed
        metrics["inference_fps"] = metrics["frames_since"] / elapsed
        metrics["packets_since"] = 0
        metrics["frames_since"] = 0
        metrics["last_rate_t"] = now
    return _rate_snapshot(metrics)


def _rate_snapshot(metrics: dict) -> dict:
    return {
        "packet_rate": round(metrics["packet_rate"], 2),
        "inference_fps": round(metrics["inference_fps"], 2),
        "dropped_packets": metrics["dropped_packets"],
        "sent_frames": metrics["sent_frames"],
        "last_ws_latency_ms": None
        if metrics["last_ws_latency_ms"] is None
        else round(metrics["last_ws_latency_ms"], 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--target-fps", type=int, default=8)
    parser.add_argument("--jpeg-quality", type=int, default=55)
    parser.add_argument("--size", type=int, default=256, choices=range(256, 385), metavar="[256-384]")
    parser.add_argument("--device", default=None, help="cuda | cpu (auto if unset)")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None, type=Path)
    parser.add_argument("--upload-dir", default=Path(".uploads/avatars"), type=Path)
    args = parser.parse_args()

    cfg_kwargs = {
        "inference_size": args.size,
        "device": args.device,
        "use_fp16": not args.no_fp16,
    }
    if args.checkpoint_dir:
        cfg_kwargs["checkpoint_dir"] = args.checkpoint_dir

    manager = AvatarManager(args.upload_dir, EngineConfig(**cfg_kwargs))
    app = build_app(manager, target_fps=args.target_fps, jpeg_quality=args.jpeg_quality)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
