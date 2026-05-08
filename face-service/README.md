# Semantic LivePortrait Server

This service keeps LivePortrait rendering on the GPU server while the browser sends only compact semantic motion packets.

Endpoints:

- `POST /avatar/upload` uploads one JPG, PNG, or WebP portrait, preprocesses it once, and caches LivePortrait source tensors on GPU.
- `WS /ws/semantic?avatar_id=...` receives semantic motion JSON and returns rendered avatar JPEG frames.
- `GET /healthz` reports render timing, packet rates, dropped packets, and GPU memory.

Run:

```powershell
cd face-service
.\.venv\Scripts\Activate.ps1
python server.py --host 0.0.0.0 --port 8765 --target-fps 8 --jpeg-quality 55
```

The browser should point to `http://127.0.0.1:8765` for local testing. On a remote RTX 3090 server, expose the same HTTP/WebSocket origin and set the frontend server field to that base URL.

The browser must not send webcam frames. It uploads the avatar image once, then sends packets shaped like:

```json
{
  "t": 1715000000,
  "yaw": 12.4,
  "pitch": -3.2,
  "roll": 1.1,
  "blinkLeft": 0.9,
  "blinkRight": 0.8,
  "mouthOpen": 0.22,
  "browRaise": 0.12,
  "smile": 0.41,
  "pupilX": 0.1,
  "pupilY": -0.05,
  "headX": 0.5,
  "headY": 0.35,
  "shoulderX": 0.5,
  "shoulderY": 0.62,
  "confidence": 0.94
}
```

Latency policy:

- latest motion packet wins
- stale packets are discarded
- no source preprocessing per frame
- source appearance tensors remain cached until a new avatar is uploaded
