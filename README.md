# Webcam Canvas App

Low-bandwidth realtime avatar pipeline.

Browser pipeline:

1. Webcam capture stays local through `getUserMedia`.
2. MediaPipe Pose, FaceMesh, and optional Hands run in the browser.
3. Landmarks are converted into normalized semantic controls: head rotation, blinks, mouth/jaw, smile, brow, gaze, shoulders, torso, and neck.
4. Only compact semantic JSON packets may be sent over WebSocket.
5. The avatar renders locally with `requestAnimationFrame`; rendering never waits for network traffic.
6. A portrait image can be uploaded once to the LivePortrait server, then animated by semantic packets only.
7. `canvas.captureStream()` exposes a local virtual camera as `window.__virtualCameraStream`.

Realtime webcam frames are not uploaded. The semantic LivePortrait server is documented in `face-service/README.md`.

Run the frontend:

```powershell
npm start
```

Then open `http://127.0.0.1:3000`.

The LivePortrait panel defaults to `http://127.0.0.1:8765`, uploads a portrait to `POST /avatar/upload`, then streams semantic motion to `WS /ws/semantic`.
