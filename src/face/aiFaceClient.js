/**
 * LivePortrait WebSocket client.
 *
 * Uploads downscaled driver frames (the user's webcam crop around the head)
 * as JPEG blobs to the Python face-service over a binary WebSocket, and
 * receives back AI-animated head frames as JPEGs which are decoded into a
 * reusable low-resolution canvas. That canvas is the `outputCanvas` consumed by the
 * sprite head renderer in `aiHeadRenderer.js`.
 *
 * Design constraints honoured:
 *
 *  - One in flight. A new webcam frame is sent only after the previous
 *    LivePortrait response returns. Every other webcam frame is dropped.
 *  - Pre-allocated buffers. The driver scratch canvas and the output canvas
 *    are created once and resized only when realtime mode changes. No `new` allocations in the
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

const DEFAULT_DRIVER_SIZE = 160;
const ULTRA_DRIVER_SIZE = 128;
const DEFAULT_OUTPUT_SIZE = 160;
const ULTRA_OUTPUT_SIZE = 128;
const DEFAULT_JPEG_QUALITY = 0.45;
const ULTRA_JPEG_QUALITY = 0.4;
const MAX_JPEG_QUALITY = 0.5;
const DEFAULT_TEXTURE_BLEND_MS = 90;
const CONNECT_TIMEOUT_MS = 3000;
const RECONNECT_DELAYS_MS = [1000, 2000, 5000];
const STATUSES = ["closed", "connecting", "open"];
const MAX_BUFFERED_BYTES = 80_000;
const MAX_SEND_FPS = 15;

export function createAiFaceClient({
  url = "ws://127.0.0.1:8765/ws",
  targetFps = 12,
  jpegQuality = DEFAULT_JPEG_QUALITY,
  driverSize = DEFAULT_DRIVER_SIZE,
  outputSize = DEFAULT_OUTPUT_SIZE,
  ultraRealtime = false,
  ultraDriverSize = ULTRA_DRIVER_SIZE,
  ultraOutputSize = ULTRA_OUTPUT_SIZE,
  ultraJpegQuality = ULTRA_JPEG_QUALITY,
  textureBlendMs = DEFAULT_TEXTURE_BLEND_MS,
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
  let uploadKBps = 0;
  let downloadKBps = 0;
  let uploadedBytes = 0;
  let downloadedBytes = 0;
  let uploadedBytesSince = 0;
  let downloadedBytesSince = 0;
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
  let ultraMode = !!ultraRealtime;
  let activeDriverSize = 0;
  let activeOutputSize = 0;
  let activeJpegQuality = 0;
  let activeUrl = url;
  let textureBlendStartT = 0;
  let textureBlendActive = false;
  let textureFrameVersion = 0;
  const listeners = new Set();

  const cappedTargetFps = Math.max(1, Math.min(targetFps, MAX_SEND_FPS));
  const minIntervalMs = 1000 / cappedTargetFps;
  const replyTimeoutMs = Math.max(3000, Math.min(staleAfterMs || 3000, slowPreviewLatencyMs || 3000));
  const backoffDelays = Array.isArray(reconnectDelaysMs) && reconnectDelaysMs.length
    ? reconnectDelaysMs
    : RECONNECT_DELAYS_MS;

  function clampNumber(value, min, max, fallback) {
    const n = Number(value);
    if (!Number.isFinite(n)) return fallback;
    return Math.max(min, Math.min(max, n));
  }

  function clampSize(value, fallback) {
    return Math.round(clampNumber(value, 96, 224, fallback));
  }

  function refreshModeValues() {
    activeDriverSize = ultraMode
      ? clampSize(ultraDriverSize, ULTRA_DRIVER_SIZE)
      : clampSize(driverSize, DEFAULT_DRIVER_SIZE);
    activeOutputSize = ultraMode
      ? clampSize(ultraOutputSize, ULTRA_OUTPUT_SIZE)
      : clampSize(outputSize, DEFAULT_OUTPUT_SIZE);
    activeJpegQuality = ultraMode
      ? clampNumber(ultraJpegQuality, 0.25, MAX_JPEG_QUALITY, ULTRA_JPEG_QUALITY)
      : clampNumber(jpegQuality, 0.25, MAX_JPEG_QUALITY, DEFAULT_JPEG_QUALITY);
  }

  refreshModeValues();

  // ---------------- canvases (preallocated) ----------------
  const driverCanvas = document.createElement("canvas");
  driverCanvas.width = activeDriverSize;
  driverCanvas.height = activeDriverSize;
  const driverCtx = driverCanvas.getContext("2d", { alpha: false, desynchronized: true });

  const outputCanvas = document.createElement("canvas");
  outputCanvas.width = activeOutputSize;
  outputCanvas.height = activeOutputSize;
  const outputCtx = outputCanvas.getContext("2d", { alpha: true });

  const currentTextureCanvas = document.createElement("canvas");
  currentTextureCanvas.width = activeOutputSize;
  currentTextureCanvas.height = activeOutputSize;
  const currentTextureCtx = currentTextureCanvas.getContext("2d", { alpha: true });

  const previousTextureCanvas = document.createElement("canvas");
  previousTextureCanvas.width = activeOutputSize;
  previousTextureCanvas.height = activeOutputSize;
  const previousTextureCtx = previousTextureCanvas.getContext("2d", { alpha: true });

  // ---------------- helpers ----------------
  function configureCanvasState() {
    if (driverCtx) {
      driverCtx.imageSmoothingEnabled = true;
      driverCtx.imageSmoothingQuality = "low";
    }
    if (outputCtx) {
      outputCtx.imageSmoothingEnabled = true;
      outputCtx.imageSmoothingQuality = "low";
    }
    if (currentTextureCtx) {
      currentTextureCtx.imageSmoothingEnabled = true;
      currentTextureCtx.imageSmoothingQuality = "low";
    }
    if (previousTextureCtx) {
      previousTextureCtx.imageSmoothingEnabled = true;
      previousTextureCtx.imageSmoothingQuality = "low";
    }
  }

  configureCanvasState();

  function resizeCanvas(canvas, width, height) {
    if (canvas.width === width && canvas.height === height) return false;
    canvas.width = width;
    canvas.height = height;
    return true;
  }

  function resizePayloadCanvases() {
    resizeCanvas(driverCanvas, activeDriverSize, activeDriverSize);
    const outputChanged = resizeCanvas(outputCanvas, activeOutputSize, activeOutputSize);
    resizeCanvas(currentTextureCanvas, activeOutputSize, activeOutputSize);
    resizeCanvas(previousTextureCanvas, activeOutputSize, activeOutputSize);
    if (outputChanged) {
      textureBlendActive = false;
      textureFrameVersion = 0;
    }
    configureCanvasState();
  }

  function easeBlend(t) {
    const x = Math.max(0, Math.min(1, t));
    return x * x * (3 - 2 * x);
  }

  function updateTextureBlend(now = performance.now()) {
    if (!textureBlendActive) return;
    const duration = Math.max(0, textureBlendMs);
    const t = duration > 0 ? (now - textureBlendStartT) / duration : 1;
    if (t >= 1) {
      outputCtx.globalAlpha = 1;
      outputCtx.clearRect(0, 0, activeOutputSize, activeOutputSize);
      outputCtx.drawImage(currentTextureCanvas, 0, 0, activeOutputSize, activeOutputSize);
      textureBlendActive = false;
      return;
    }
    const alpha = easeBlend(t);
    outputCtx.globalAlpha = 1;
    outputCtx.clearRect(0, 0, activeOutputSize, activeOutputSize);
    outputCtx.drawImage(previousTextureCanvas, 0, 0, activeOutputSize, activeOutputSize);
    outputCtx.globalAlpha = alpha;
    outputCtx.drawImage(currentTextureCanvas, 0, 0, activeOutputSize, activeOutputSize);
    outputCtx.globalAlpha = 1;
  }

  function buildWsUrl() {
    try {
      const next = new URL(url, window.location.href);
      next.searchParams.set("driver", String(activeDriverSize));
      next.searchParams.set("out", String(activeOutputSize));
      next.searchParams.set("q", String(Math.round(activeJpegQuality * 100)));
      next.searchParams.set("mode", ultraMode ? "ultra" : "realtime");
      return next.toString();
    } catch (_) {
      return url;
    }
  }

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
    console.info("open", { url: activeUrl, mode: ultraMode ? "ultra" : "realtime" });
    setStatus("open");
  }

  function onClose(ev, socket) {
    if (socket !== ws) return;
    clearConnectTimer();
    console.info("close", {
      url: activeUrl,
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
    if (byteLength != null) {
      downloadedBytes += byteLength;
      downloadedBytesSince += byteLength;
    }
    let bitmap = null;
    try {
      const hadTexture = receivedFrames > 0 || textureFrameVersion > 0;
      if (hadTexture) {
        updateTextureBlend();
        previousTextureCtx.clearRect(0, 0, activeOutputSize, activeOutputSize);
        previousTextureCtx.drawImage(outputCanvas, 0, 0, activeOutputSize, activeOutputSize);
      }
      const blob = data instanceof Blob ? data : new Blob([data], { type: "image/jpeg" });
      try {
        bitmap = await createImageBitmap(blob, {
          resizeWidth: activeOutputSize,
          resizeHeight: activeOutputSize,
          resizeQuality: "low"
        });
      } catch (_) {
        bitmap = await createImageBitmap(blob);
      }
      currentTextureCtx.clearRect(0, 0, activeOutputSize, activeOutputSize);
      currentTextureCtx.drawImage(bitmap, 0, 0, activeOutputSize, activeOutputSize);
      textureFrameVersion += 1;
      if (hadTexture && textureBlendMs > 0) {
        textureBlendStartT = performance.now();
        textureBlendActive = true;
        updateTextureBlend(textureBlendStartT);
      } else {
        textureBlendActive = false;
        outputCtx.clearRect(0, 0, activeOutputSize, activeOutputSize);
        outputCtx.drawImage(currentTextureCanvas, 0, 0, activeOutputSize, activeOutputSize);
      }
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
    uploadKBps = (uploadedBytesSince / 1024) * (1000 / elapsed);
    downloadKBps = (downloadedBytesSince / 1024) * (1000 / elapsed);
    sentSince = 0;
    receivedSince = 0;
    uploadedBytesSince = 0;
    downloadedBytesSince = 0;
    lastRateT = now;
    console.info("aiFace rates", {
      sentFps: Number(sentFps.toFixed(1)),
      receivedFps: Number(receivedFps.toFixed(1)),
      uploadKBps: Number(uploadKBps.toFixed(1)),
      downloadKBps: Number(downloadKBps.toFixed(1)),
      avgUploadBytes: sentFrames ? Math.round(uploadedBytes / sentFrames) : 0,
      avgDownloadBytes: rawReceivedFrames ? Math.round(downloadedBytes / rawReceivedFrames) : 0,
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
    activeUrl = buildWsUrl();
    console.info("connecting", {
      url: activeUrl,
      driverSize: activeDriverSize,
      outputSize: activeOutputSize,
      jpegQuality: activeJpegQuality,
      mode: ultraMode ? "ultra" : "realtime"
    });
    try {
      const socket = new WebSocket(activeUrl);
      ws = socket;
      socket.binaryType = "blob";
      connectTimer = setTimeout(() => {
        if (socket !== ws || socket.readyState !== WebSocket.CONNECTING) return;
        connectTimer = null;
        console.error("error", new Error(`WebSocket connection timed out after ${connectTimeoutMs} ms`));
        console.info("close", {
          url: activeUrl,
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
          driverCanvas.toBlob(resolve, "image/jpeg", activeJpegQuality);
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
        uploadedBytes += blob.size;
        uploadedBytesSince += blob.size;
        socket.send(blob);
      } else {
        const dataUrl = driverCanvas.toDataURL("image/jpeg", activeJpegQuality);
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
        uploadedBytes += buf.byteLength;
        uploadedBytesSince += buf.byteLength;
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
    driverCtx.fillRect(0, 0, activeDriverSize, activeDriverSize);
    driverCtx.drawImage(source, x, y, crop, crop, 0, 0, activeDriverSize, activeDriverSize);
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

  function setUltraRealtimeMode(on) {
    const next = !!on;
    if (next === ultraMode) return;
    ultraMode = next;
    refreshModeValues();
    resizePayloadCanvases();
    awaitingResponse = false;
    encoding = false;
    textureBlendActive = false;
    lastRecvT = 0;
    lastSentFrameId = -1;
    lastReceivedFrameId = -1;
    if (syncBuffer) syncBuffer.clearPin();
    updateInFlight();

    const socket = ws;
    if (socket && (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN)) {
      closeSocket(socket);
      clearReconnectTimer();
      reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connect();
      }, 0);
    }
  }

  function getUltraRealtimeMode() {
    return ultraMode;
  }

  function getStatus() { return status; }
  function getStats() {
    const now = performance.now();
    return {
      status,
      url: activeUrl || buildWsUrl(),
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
      uploadKBps,
      downloadKBps,
      uploadedBytes,
      downloadedBytes,
      avgUploadFrameBytes: sentFrames ? uploadedBytes / sentFrames : 0,
      avgDownloadFrameBytes: rawReceivedFrames ? downloadedBytes / rawReceivedFrames : 0,
      driverSize: activeDriverSize,
      outputSize: activeOutputSize,
      jpegQuality: activeJpegQuality,
      targetFps: cappedTargetFps,
      ultraRealtime: ultraMode,
      textureBlendActive,
      textureFrameVersion,
      lastError,
      slowPreview: lastRoundTripMs != null && lastRoundTripMs > slowPreviewLatencyMs,
      bufferedAmount: ws ? ws.bufferedAmount : null,
      msSinceSend: lastSendT ? now - lastSendT : null,
      msSinceReceive: lastRecvT ? now - lastRecvT : null
    };
  }
  function getOutputCanvas() {
    updateTextureBlend();
    return outputCanvas;
  }
  function getOutputSize() { return activeOutputSize; }

  return {
    connect,
    disconnect,
    sendDriver,
    isReady,
    isRealtime,
    getStatus,
    getStats,
    setServerMetrics,
    setUltraRealtimeMode,
    getUltraRealtimeMode,
    onStatusChange,
    getLastReceivedFrameId,
    getOutputCanvas,
    getOutputSize
  };
}
