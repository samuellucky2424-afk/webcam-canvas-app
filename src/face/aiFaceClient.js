/**
 * LivePortrait WebSocket client.
 *
 * Uploads downscaled driver frames (the user's webcam crop around the head)
 * as JPEG blobs to the Python face-service over a binary WebSocket, and
 * receives back AI-animated head frames as JPEGs which are decoded into a
 * reusable 256×256 canvas. That canvas is the `outputCanvas` consumed by the
 * sprite head renderer in `aiHeadRenderer.js`.
 *
 * Design constraints honoured:
 *
 *  - One in flight. A new webcam frame is sent only after the previous
 *    LivePortrait response returns. Every other webcam frame is dropped.
 *  - Pre-allocated buffers. The driver scratch canvas (192×192) and the
 *    output canvas (256×256) are created once. No `new` allocations in the
 *    hot path apart from the unavoidable `Blob` and `ImageBitmap` returned
 *    by browser APIs (both are released).
 *  - Status fan-out. `connecting | open | closed` notifications
 *    let the composer flip back to the local default head atomically when
 *    the GPU side dies.
 *  - Frame pairing. Each upload is tagged with a monotonic `frameId`; on
 *    reply the buffer is pinned to the most recently sent frameId, which —
 *    given the server is latest-wins — is the best approximation of which
 *    snapshot the response corresponds to.
 *
 * No retry storm: reconnection backs off through 1s, 2s, then 5s.
 */

const DRIVER_SIZE = 192;
const OUTPUT_SIZE = 256;
const CONNECT_TIMEOUT_MS = 3000;
const RECONNECT_DELAYS_MS = [1000, 2000, 5000];
const STATUSES = ["closed", "connecting", "open"];
const MAX_BUFFERED_BYTES = 150_000;
const MAX_SEND_FPS = 15;

export function createAiFaceClient({
  url = "ws://127.0.0.1:8765/ws",
  targetFps = 12,
  jpegQuality = 0.6,
  reconnect = true,
  connectTimeoutMs = CONNECT_TIMEOUT_MS,
  reconnectDelaysMs = RECONNECT_DELAYS_MS,
  staleAfterMs = 600,
  slowPreviewLatencyMs = 2000,
  slowPreviewIntervalMs = 120000,
  syncBuffer = null
} = {}) {
  // ---------------- state ----------------
  let ws = null;
  let status = "closed";
  let inFlight = false;
  let awaitingResponse = false;
  let lastSendT = 0;
  let lastRecvT = 0;
  let lastSentFrameId = -1;
  let lastReceivedFrameId = -1;
  let sentFrames = 0;
  let rawReceivedFrames = 0;
  let receivedFrames = 0;
  let sentFps = 0;
  let receivedFps = 0;
  let lastRoundTripMs = null;
  let serverInferenceMs = null;
  let serverInferenceFps = null;
  let lastError = "";
  let skippedFrames = 0;
  let skippedAwaitingResponse = 0;
  let skippedThrottle = 0;
  let skippedEncoding = 0;
  let skippedSocket = 0;
  let skippedBuffered = 0;
  let sentSince = 0;
  let receivedSince = 0;
  let lastRateT = performance.now();
  let encoding = false;
  let reconnectAttempt = 0;
  let reconnectTimer = null;
  let connectTimer = null;
  let manualDisconnect = false;
  const listeners = new Set();

  const cappedTargetFps = Math.max(1, Math.min(targetFps, MAX_SEND_FPS));
  const minIntervalMs = 1000 / cappedTargetFps;
  const replyTimeoutMs = Math.max(3000, Math.min(staleAfterMs || 3000, slowPreviewLatencyMs || 3000));
  const backoffDelays = Array.isArray(reconnectDelaysMs) && reconnectDelaysMs.length
    ? reconnectDelaysMs
    : RECONNECT_DELAYS_MS;

  // ---------------- canvases (preallocated) ----------------
  const driverCanvas = document.createElement("canvas");
  driverCanvas.width = DRIVER_SIZE;
  driverCanvas.height = DRIVER_SIZE;
  const driverCtx = driverCanvas.getContext("2d", { alpha: false, desynchronized: true });

  const outputCanvas = document.createElement("canvas");
  outputCanvas.width = OUTPUT_SIZE;
  outputCanvas.height = OUTPUT_SIZE;
  const outputCtx = outputCanvas.getContext("2d", { alpha: true });

  // ---------------- helpers ----------------
  function setStatus(next) {
    if (!STATUSES.includes(next)) return;
    if (status === next) return;
    status = next;
    for (const cb of listeners) {
      try { cb(status); } catch (_) { /* listener guard */ }
    }
  }

  function isReady() {
    if (status !== "open") return false;
    if (!lastRecvT) return false;
    // Treat the stream as not-ready if the last AI frame is stale —
    // the composer will fall back to the local head until traffic resumes.
    if (staleAfterMs > 0 && performance.now() - lastRecvT > staleAfterMs) return false;
    return true;
  }

  function isRealtime(maxLatencyMs = 700) {
    if (!isReady()) return false;
    if (lastRoundTripMs == null) return false;
    return lastRoundTripMs <= maxLatencyMs;
  }

  function clearConnectTimer() {
    if (!connectTimer) return;
    clearTimeout(connectTimer);
    connectTimer = null;
  }

  function clearReconnectTimer() {
    if (!reconnectTimer) return;
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  function scheduleReconnect() {
    if (!reconnect || manualDisconnect || reconnectTimer) return;
    const index = Math.min(reconnectAttempt, backoffDelays.length - 1);
    const delay = backoffDelays[index];
    reconnectAttempt = Math.min(reconnectAttempt + 1, backoffDelays.length - 1);
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, delay);
  }

  function closeSocket(socket) {
    try {
      socket.close();
    } catch (error) {
      lastError = error?.message || String(error);
      console.error("error", error);
    }
  }

  function onOpen(socket) {
    if (socket !== ws) return;
    clearConnectTimer();
    reconnectAttempt = 0;
    lastError = "";
    console.info("open", { url });
    setStatus("open");
  }

  function onClose(ev, socket) {
    if (socket !== ws) return;
    clearConnectTimer();
    console.info("close", {
      url,
      code: ev?.code,
      reason: ev?.reason || "",
      wasClean: ev?.wasClean
    });
    ws = null;
    inFlight = false;
    awaitingResponse = false;
    encoding = false;
    if (syncBuffer) syncBuffer.clearPin();
    setStatus("closed");
    scheduleReconnect();
  }

  function onError(ev, socket) {
    if (socket !== ws) return;
    lastError = ev?.message || "WebSocket error";
    console.error("error", ev);
    inFlight = false;
    awaitingResponse = false;
    if (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN) {
      closeSocket(socket);
    }
  }

  async function onMessage(ev) {
    lastRecvT = performance.now();
    if (lastSendT) {
      lastRoundTripMs = lastRecvT - lastSendT;
    }
    awaitingResponse = false;
    updateInFlight();
    const data = ev.data;
    if (!data) return;
    rawReceivedFrames += 1;
    const byteLength =
      typeof data.size === "number"
        ? data.size
        : typeof data.byteLength === "number"
          ? data.byteLength
          : null;
    console.info("message", { url, bytes: byteLength });
    let bitmap = null;
    try {
      const blob = data instanceof Blob ? data : new Blob([data], { type: "image/jpeg" });
      bitmap = await createImageBitmap(blob);
      outputCtx.drawImage(bitmap, 0, 0, OUTPUT_SIZE, OUTPUT_SIZE);
      receivedFrames += 1;
      receivedSince += 1;
      lastError = "";
      lastReceivedFrameId = lastSentFrameId;
      if (syncBuffer && lastSentFrameId > 0) {
        if (lastRoundTripMs != null && lastRoundTripMs <= 2000) {
          syncBuffer.pinFrameId(lastSentFrameId);
        } else {
          syncBuffer.clearPin();
        }
      }
    } catch (error) {
      lastError = error?.message || String(error);
      console.error("error", error);
      /* decode error — keep last good frame */
    } finally {
      if (bitmap && typeof bitmap.close === "function") bitmap.close();
    }
    tickRates();
  }

  function tickRates() {
    const now = performance.now();
    if (now - lastRateT < 1000) return;
    const elapsed = now - lastRateT;
    sentFps = (sentSince * 1000) / elapsed;
    receivedFps = (receivedSince * 1000) / elapsed;
    sentSince = 0;
    receivedSince = 0;
    lastRateT = now;
    console.info("aiFace rates", {
      sentFps: Number(sentFps.toFixed(1)),
      receivedFps: Number(receivedFps.toFixed(1)),
      bufferedAmount: ws ? ws.bufferedAmount : null,
      activeInFlightCount: awaitingResponse ? 1 : 0,
      skippedFrames
    });
  }

  function connect() {
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;
    clearReconnectTimer();
    clearConnectTimer();
    manualDisconnect = false;
    setStatus("connecting");
    console.info("connecting", { url });
    try {
      const socket = new WebSocket(url);
      ws = socket;
      socket.binaryType = "blob";
      connectTimer = setTimeout(() => {
        if (socket !== ws || socket.readyState !== WebSocket.CONNECTING) return;
        connectTimer = null;
        console.error("error", new Error(`WebSocket connection timed out after ${connectTimeoutMs} ms`));
        console.info("close", {
          url,
          code: "timeout",
          reason: "connect timeout",
          wasClean: false
        });
        ws = null;
        inFlight = false;
        awaitingResponse = false;
        if (syncBuffer) syncBuffer.clearPin();
        setStatus("closed");
        closeSocket(socket);
        scheduleReconnect();
      }, connectTimeoutMs);
      socket.addEventListener("open", () => onOpen(socket));
      socket.addEventListener("close", (ev) => onClose(ev, socket));
      socket.addEventListener("error", (ev) => onError(ev, socket));
      socket.addEventListener("message", onMessage);
    } catch (error) {
      ws = null;
      clearConnectTimer();
      console.error("error", error);
      inFlight = false;
      awaitingResponse = false;
      setStatus("closed");
      scheduleReconnect();
    }
  }

  function disconnect() {
    manualDisconnect = true;
    clearReconnectTimer();
    clearConnectTimer();
    if (ws) {
      closeSocket(ws);
    }
    if (syncBuffer) syncBuffer.clearPin();
    awaitingResponse = false;
    encoding = false;
    updateInFlight();
    setStatus("closed");
  }

  function getLastReceivedFrameId() {
    return lastReceivedFrameId;
  }

  function updateInFlight() {
    inFlight = encoding || awaitingResponse;
  }

  function skipFrame(reason) {
    skippedFrames += 1;
    if (reason === "awaitingResponse") skippedAwaitingResponse += 1;
    else if (reason === "throttle") skippedThrottle += 1;
    else if (reason === "encoding") skippedEncoding += 1;
    else if (reason === "buffered") skippedBuffered += 1;
    else skippedSocket += 1;
    tickRates();
  }

  function handleReplyTimeout(now) {
    if (!awaitingResponse || !lastSendT || now - lastSendT < replyTimeoutMs) return false;
    lastError = `LivePortrait reply timed out after ${Math.round(now - lastSendT)} ms`;
    console.error("error", new Error(lastError));
    const socket = ws;
    if (socket && (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN)) {
      closeSocket(socket);
    }
    return true;
  }

  async function sendCurrentFrame(source, state, socket) {
    encoding = true;
    updateInFlight();
    const frameId = state?.frameId ?? -1;

    try {
      drawDriverCrop(source, state);
      if (driverCanvas.toBlob) {
        const blob = await new Promise((resolve) => {
          driverCanvas.toBlob(resolve, "image/jpeg", jpegQuality);
        });
        if (!blob) return;
        if (socket !== ws || socket.readyState !== WebSocket.OPEN) return;
        if (awaitingResponse) {
          skipFrame("awaitingResponse");
          return;
        }
        if (socket.bufferedAmount > MAX_BUFFERED_BYTES) {
          skipFrame("buffered");
          return;
        }
        socket.send(blob);
      } else {
        const dataUrl = driverCanvas.toDataURL("image/jpeg", jpegQuality);
        const bin = atob(dataUrl.split(",")[1]);
        const buf = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i += 1) buf[i] = bin.charCodeAt(i);
        if (socket !== ws || socket.readyState !== WebSocket.OPEN) return;
        if (awaitingResponse) {
          skipFrame("awaitingResponse");
          return;
        }
        if (socket.bufferedAmount > MAX_BUFFERED_BYTES) {
          skipFrame("buffered");
          return;
        }
        socket.send(buf.buffer);
      }

      lastSendT = performance.now();
      lastSentFrameId = frameId;
      awaitingResponse = true;
      updateInFlight();
      sentFrames += 1;
      sentSince += 1;
    } catch (error) {
      lastError = error?.message || String(error);
      console.error("error", error);
    } finally {
      encoding = false;
      updateInFlight();
      tickRates();
    }
  }

  /**
   * Push a driver frame. `state` is the unified state object — its
   * `frameId` is paired with the response. `source` is anything
   * `drawImage` accepts (HTMLVideoElement, HTMLCanvasElement, etc.).
   *
   * Crops a square region around the detected head if the unified state
   * has one, otherwise centres on the full frame.
   */
  function sendDriver(source, state) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      skipFrame("socket");
      return;
    }
    const now = performance.now();
    if (awaitingResponse) {
      handleReplyTimeout(now);
      skipFrame("awaitingResponse");
      return;
    }
    if (encoding) {
      skipFrame("encoding");
      return;
    }
    if (lastSendT && now - lastSendT < minIntervalMs) {
      skipFrame("throttle");
      return;
    }
    if (ws.bufferedAmount > MAX_BUFFERED_BYTES) {
      skipFrame("buffered");
      return;
    }
    tickRates();
    void sendCurrentFrame(source, state, ws);
  }

  function drawDriverCrop(source, state) {
    const sw = source?.videoWidth ?? source?.width ?? 0;
    const sh = source?.videoHeight ?? source?.height ?? 0;
    if (!sw || !sh) return;

    let cx = sw * 0.5;
    let cy = sh * 0.4;
    let crop = Math.min(sw, sh) * 0.55;

    // If we know where the face is in source (mirrored video) coords, crop
    // there. The `state.skeleton` has joints in source space (normalized
    // 0..1) — derive a head box from nose + ears when present.
    const sk = state?.skeleton?.joints;
    if (sk && sk.nose) {
      cx = sk.nose.x * sw;
      cy = sk.nose.y * sh;
      // Heuristic head radius from eye spacing if available.
      const lEye = sk.leftEye, rEye = sk.rightEye;
      if (lEye && rEye) {
        const eyeDx = (lEye.x - rEye.x) * sw;
        const eyeDy = (lEye.y - rEye.y) * sh;
        const eyeWidth = Math.hypot(eyeDx, eyeDy);
        crop = Math.max(eyeWidth * 4.5, 96);
      } else {
        crop = Math.min(sw, sh) * 0.45;
      }
    }

    const half = crop * 0.5;
    let x = cx - half;
    let y = cy - half;
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x + crop > sw) x = sw - crop;
    if (y + crop > sh) y = sh - crop;

    driverCtx.fillStyle = "#000";
    driverCtx.fillRect(0, 0, DRIVER_SIZE, DRIVER_SIZE);
    driverCtx.drawImage(source, x, y, crop, crop, 0, 0, DRIVER_SIZE, DRIVER_SIZE);
  }

  function onStatusChange(cb) {
    listeners.add(cb);
    try { cb(status); } catch (_) { /* listener guard */ }
    return () => listeners.delete(cb);
  }

  function setServerMetrics({ inferenceMs = null, inferenceFps = null } = {}) {
    if (Number.isFinite(inferenceMs)) serverInferenceMs = inferenceMs;
    if (Number.isFinite(inferenceFps)) serverInferenceFps = inferenceFps;
  }

  function getStatus() { return status; }
  function getStats() {
    const now = performance.now();
    return {
      status,
      url,
      sentFrames,
      rawReceivedFrames,
      receivedFrames,
      inFlight,
      awaitingResponse,
      waitingForReply: awaitingResponse,
      activeInFlightCount: awaitingResponse ? 1 : 0,
      pendingFrames: awaitingResponse ? 1 : 0,
      skippedFrames,
      skippedAwaitingResponse,
      skippedThrottle,
      skippedEncoding,
      skippedSocket,
      skippedBuffered,
      sentFps,
      receivedFps,
      lastSentFrameId,
      lastReceivedFrameId,
      lastRoundTripMs,
      websocketRttMs: lastRoundTripMs,
      serverInferenceMs,
      serverInferenceFps,
      inferenceTimeMs: serverInferenceMs,
      driverSize: DRIVER_SIZE,
      jpegQuality,
      targetFps: cappedTargetFps,
      lastError,
      slowPreview: lastRoundTripMs != null && lastRoundTripMs > slowPreviewLatencyMs,
      bufferedAmount: ws ? ws.bufferedAmount : null,
      msSinceSend: lastSendT ? now - lastSendT : null,
      msSinceReceive: lastRecvT ? now - lastRecvT : null
    };
  }
  function getOutputCanvas() { return outputCanvas; }
  function getOutputSize() { return OUTPUT_SIZE; }

  return {
    connect,
    disconnect,
    sendDriver,
    isReady,
    isRealtime,
    getStatus,
    getStats,
    setServerMetrics,
    onStatusChange,
    getLastReceivedFrameId,
    getOutputCanvas,
    getOutputSize
  };
}
