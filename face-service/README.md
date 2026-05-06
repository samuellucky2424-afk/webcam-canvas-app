# LivePortrait Face Service

Real-time face animation server. Streams **driving frames** from the browser
(your webcam, captured by [`src/main.js`](../src/main.js)) into a GPU-hosted
LivePortrait pipeline that animates a single **source avatar** to mimic the
driver's expression and head pose. The animated frames stream back over the
same WebSocket and can be fed into the avatar's `setHeadSource` to replace
the drawn humanoid head.

## Architecture

```
┌──────────────────┐   JPEG over WS   ┌────────────────────┐
│ Browser (canvas) │ ───────────────▶ │ FastAPI server     │
│ webcam frames    │                  │  ├─ inference.py   │
│ + decoded face   │ ◀─────────────── │  └─ LivePortrait   │
└──────────────────┘   JPEG over WS   └────────────────────┘
```

- **Transport:** WebSocket binary frames. Each direction is a single JPEG.
  Simpler than WebRTC, fine for one upstream + one downstream stream per
  client.
- **Pacing:** the server caps inference at `--target-fps` (default 20). Newly
  arriving driving frames overwrite the pending one (latest-wins) so the GPU
  never works on stale input.
- **Modular:** `server.py` knows nothing about LivePortrait internals. Swap
  the engine in [inference.py](inference.py) for any model that accepts a
  source image and per-frame driving inputs.

## Setup

> **Python 3.10, 3.11, or 3.12 required.** PyTorch does not ship wheels for
> 3.13 or 3.14 yet. If `python --version` shows 3.13/3.14, install
> [Python 3.11](https://www.python.org/downloads/release/python-3119/) and
> use `py -3.11` below.

### 1. Create a Python env and install dependencies

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# (Optional, for CUDA) install the matching torch wheel first. Pick the
# index URL that matches your installed CUDA toolkit:
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# Then the rest:
pip install -r requirements.txt
```

CPU-only works for testing but won't hit 15 FPS.

### 2. Clone LivePortrait and download checkpoints

```powershell
mkdir third_party
git clone https://github.com/KwaiVGI/LivePortrait third_party/LivePortrait

# Windows-friendly downloader (no bash needed):
python download_models.py
```

This pulls the checkpoint snapshot from HuggingFace into
`third_party/LivePortrait/pretrained_weights/` (~2 GB). On Linux/macOS you
can use the upstream `bash scripts/download_models.sh` instead — both end up
in the same place.

### 3. Run the server

```powershell
python server.py --source path\to\avatar.png --target-fps 20 --size 256
```

| Flag             | Default | Notes                                          |
|------------------|---------|------------------------------------------------|
| `--source`       | —       | Path to the avatar image. Required.            |
| `--target-fps`   | `20`    | Hard cap. Use 15 on weaker GPUs, 24 on strong. |
| `--size`         | `256`   | Inference resolution, 256–384.                 |
| `--device`       | auto    | `cuda` / `cpu`. Auto-detect by default.        |
| `--no-fp16`      | off     | Disable half precision (debugging only).       |
| `--port`         | `8765`  | WebSocket port.                                |

Verify it's up: open `http://host:8765/healthz`.

```json
{ "ok": true, "device": "cuda", "size": 256, "target_fps": 20, "last_inference_ms": 0.0 }
```

## Performance notes

- **CUDA + FP16 + 256 px** lands around 35–60 ms/frame on an RTX 3060,
  comfortably above the 15 FPS floor.
- **256 vs 384 px** roughly doubles inference time. Stay at 256 unless the
  GPU has spare headroom.
- The browser should send frames at most as fast as `--target-fps`; sending
  faster just wastes upload bandwidth (the server drops them).
- JPEG quality 80 on the way back is the sweet spot — quality 95 is barely
  visible and adds 2× bytes; quality 60 visibly blocks at 256 px.

## Wiring into the avatar (browser side)

Once the server is running, connect a hidden `<video>` element to a
`MediaStream` synthesised from the inbound JPEGs and pass it to the avatar's
sprite head renderer:

```js
import { createSpriteHeadRenderer } from "./head.js";

// ... after avatarRenderer is created ...
const aiCanvas = document.createElement("canvas");
aiCanvas.width = aiCanvas.height = 256;
const aiCtx = aiCanvas.getContext("2d");
const ws = new WebSocket("ws://localhost:8765/ws");
ws.binaryType = "arraybuffer";

ws.onmessage = async (ev) => {
  const blob = new Blob([ev.data], { type: "image/jpeg" });
  const bmp = await createImageBitmap(blob);
  aiCtx.drawImage(bmp, 0, 0, aiCanvas.width, aiCanvas.height);
  bmp.close();
};

avatarRenderer.setHeadRenderer(
  createSpriteHeadRenderer({ source: aiCanvas, clipOval: true })
);

// In your render loop, send the most recent webcam frame as JPEG:
function sendDriver() {
  if (ws.readyState !== WebSocket.OPEN) return;
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  c.getContext("2d").drawImage(video, 0, 0, 256, 256);
  c.toBlob((blob) => blob.arrayBuffer().then((b) => ws.send(b)),
           "image/jpeg", 0.7);
}
setInterval(sendDriver, 1000 / 20); // match --target-fps
```

The body skeleton, hand tracker, and head placement are unchanged — only the
content drawn inside the head silhouette is now coming from the GPU service.
