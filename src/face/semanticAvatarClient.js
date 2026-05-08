import { createAvatarMotionPacket } from "../state/semanticParams.js";

const DEFAULT_RECONNECT_DELAYS_MS = [1000, 2500, 5000];
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const MAX_BUFFERED_BYTES = 12000;

function clamp(value, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return min;
  return Math.min(Math.max(n, min), max);
}

function bytesOf(text) {
  return new TextEncoder().encode(text).length;
}

function wsUrlFromBase(baseUrl, path) {
  const url = new URL(path, baseUrl);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

export function createSemanticAvatarClient({
  baseUrl = "http://127.0.0.1:8765",
  uploadPath = "/avatar/upload",
  wsPath = "/ws/semantic",
  maxPacketsPerSecond = 24,
  reconnectDelaysMs = DEFAULT_RECONNECT_DELAYS_MS,
  maxBufferedBytes = MAX_BUFFERED_BYTES,
  onStatusChange,
  onUploadStatus
} = {}) {
  let ws = null;
  let status = "closed";
  let uploadStatus = "empty";
  let avatarId = "";
  let reconnectTimer = null;
  let reconnectAttempt = 0;
  let manualClose = false;
  let lastSendT = 0;
  let lastRecvT = 0;
  let pendingPingT = 0;
  let lastPingT = 0;
  let lastRateT = performance.now();
  let sentSince = 0;
  let recvSince = 0;
  let upBytesSince = 0;
  let downBytesSince = 0;
  let sentPackets = 0;
  let receivedFrames = 0;
  let droppedPackets = 0;
  let sentPps = 0;
  let receivedFps = 0;
  let uploadKBps = 0;
  let downloadKBps = 0;
  let rttMs = null;
  let lastPacketBytes = 0;
  let lastPacket = "";
  let lastError = "";
  let serverMetrics = {};

  const minIntervalMs = 1000 / clamp(maxPacketsPerSecond, 1, 30);
  const backoffDelays =
    Array.isArray(reconnectDelaysMs) && reconnectDelaysMs.length
      ? reconnectDelaysMs
      : DEFAULT_RECONNECT_DELAYS_MS;

  const outputCanvas = document.createElement("canvas");
  outputCanvas.width = 256;
  outputCanvas.height = 256;
  const outputCtx = outputCanvas.getContext("2d", { alpha: true });

  function setStatus(next) {
    if (status === next) return;
    status = next;
    onStatusChange?.(status);
  }

  function setUploadStatus(next) {
    if (uploadStatus === next) return;
    uploadStatus = next;
    onUploadStatus?.(uploadStatus);
  }

  function tickRates(now = performance.now()) {
    if (now - lastRateT < 1000) return;
    const elapsed = now - lastRateT;
    sentPps = (sentSince * 1000) / elapsed;
    receivedFps = (recvSince * 1000) / elapsed;
    uploadKBps = (upBytesSince / 1024) * (1000 / elapsed);
    downloadKBps = (downBytesSince / 1024) * (1000 / elapsed);
    sentSince = 0;
    recvSince = 0;
    upBytesSince = 0;
    downBytesSince = 0;
    lastRateT = now;
  }

  function clearReconnectTimer() {
    if (!reconnectTimer) return;
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  function scheduleReconnect() {
    if (manualClose || reconnectTimer || !avatarId) return;
    const index = Math.min(reconnectAttempt, backoffDelays.length - 1);
    const delay = backoffDelays[index];
    reconnectAttempt = Math.min(reconnectAttempt + 1, backoffDelays.length - 1);
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, delay);
  }

  async function uploadAvatar(file) {
    if (!file) throw new Error("No avatar image selected.");
    if (!/^image\/(jpeg|png|webp)$/.test(file.type)) {
      throw new Error("Use a JPG, PNG, or WebP portrait.");
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      throw new Error("Image must be 10 MB or smaller.");
    }

    setUploadStatus("uploading");
    const form = new FormData();
    form.append("image", file);
    const response = await fetch(new URL(uploadPath, baseUrl), {
      method: "POST",
      body: form
    });
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      setUploadStatus("error");
      throw new Error(text || `Upload failed (${response.status})`);
    }
    const data = await response.json();
    avatarId = String(data.avatar_id || data.avatarId || "default");
    setUploadStatus("ready");
    return data;
  }

  function buildWsUrl() {
    const url = new URL(wsUrlFromBase(baseUrl, wsPath));
    if (avatarId) url.searchParams.set("avatar_id", avatarId);
    return url.toString();
  }

  function connect() {
    if (!avatarId) {
      setStatus("no-avatar");
      return;
    }
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;
    clearReconnectTimer();
    manualClose = false;
    lastError = "";
    setStatus("connecting");

    try {
      const socket = new WebSocket(buildWsUrl());
      ws = socket;
      socket.binaryType = "blob";
      socket.addEventListener("open", () => {
        if (socket !== ws) return;
        reconnectAttempt = 0;
        setStatus("open");
      });
      socket.addEventListener("message", (ev) => {
        void handleMessage(ev.data);
      });
      socket.addEventListener("close", () => {
        if (socket !== ws) return;
        ws = null;
        pendingPingT = 0;
        setStatus(avatarId ? "closed" : "no-avatar");
        scheduleReconnect();
      });
      socket.addEventListener("error", () => {
        if (socket !== ws) return;
        lastError = "WebSocket error";
        try {
          socket.close();
        } catch {
          // Socket is already unusable.
        }
      });
    } catch (error) {
      ws = null;
      lastError = error?.message || String(error);
      setStatus("closed");
      scheduleReconnect();
    }
  }

  function disconnect() {
    manualClose = true;
    clearReconnectTimer();
    pendingPingT = 0;
    if (ws) {
      try {
        ws.close();
      } catch {
        // Ignore shutdown errors.
      }
    }
    ws = null;
    setStatus(avatarId ? "closed" : "no-avatar");
  }

  function sendRaw(text) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    const bytes = bytesOf(text);
    if (ws.bufferedAmount + bytes > maxBufferedBytes) return false;
    ws.send(text);
    upBytesSince += bytes;
    return true;
  }

  function maybePing(now) {
    if (pendingPingT || now - lastPingT < 2000) return;
    pendingPingT = now;
    lastPingT = now;
    sendRaw(JSON.stringify({ type: "ping", t: Math.round(now) }));
  }

  function sendSemantic(params) {
    const now = performance.now();
    tickRates(now);
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      droppedPackets += 1;
      return false;
    }
    if (lastSendT && now - lastSendT < minIntervalMs) return false;
    if (ws.bufferedAmount > maxBufferedBytes) {
      droppedPackets += 1;
      return false;
    }

    const packet = createAvatarMotionPacket(params);
    const text = JSON.stringify(packet);
    const bytes = bytesOf(text);
    if (ws.bufferedAmount + bytes > maxBufferedBytes) {
      droppedPackets += 1;
      return false;
    }

    try {
      ws.send(text);
      lastPacket = text;
      lastPacketBytes = bytes;
      lastSendT = now;
      sentPackets += 1;
      sentSince += 1;
      upBytesSince += bytes;
      maybePing(now);
      return true;
    } catch (error) {
      droppedPackets += 1;
      lastError = error?.message || String(error);
      return false;
    }
  }

  async function handleMessage(data) {
    if (typeof data === "string") {
      try {
        const message = JSON.parse(data);
        if (message?.type === "pong" && pendingPingT) {
          rttMs = performance.now() - pendingPingT;
          pendingPingT = 0;
        }
        if (message?.type === "metrics") {
          serverMetrics = { ...serverMetrics, ...message };
          if (Number.isFinite(message.rtt_ms)) rttMs = message.rtt_ms;
          if (Number.isFinite(message.last_ws_latency_ms)) rttMs = message.last_ws_latency_ms;
        }
        if (message?.error) lastError = String(message.error);
      } catch {
        // Ignore non-JSON status strings.
      }
      return;
    }

    const bytes = typeof data?.size === "number" ? data.size : data?.byteLength ?? 0;
    downBytesSince += bytes;
    lastRecvT = performance.now();

    let bitmap = null;
    try {
      const blob = data instanceof Blob ? data : new Blob([data], { type: "image/jpeg" });
      bitmap = await createImageBitmap(blob, { resizeQuality: "low" });
      if (outputCanvas.width !== bitmap.width || outputCanvas.height !== bitmap.height) {
        outputCanvas.width = bitmap.width;
        outputCanvas.height = bitmap.height;
      }
      outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
      outputCtx.drawImage(bitmap, 0, 0, outputCanvas.width, outputCanvas.height);
      receivedFrames += 1;
      recvSince += 1;
      lastError = "";
    } catch (error) {
      lastError = error?.message || String(error);
    } finally {
      if (bitmap?.close) bitmap.close();
      tickRates();
    }
  }

  function getStats() {
    tickRates();
    return {
      status,
      uploadStatus,
      avatarId,
      sentPackets,
      receivedFrames,
      sentPps,
      receivedFps,
      uploadKBps,
      downloadKBps,
      droppedPackets,
      lastPacketBytes,
      lastPacket,
      rttMs,
      lastError,
      serverMetrics,
      msSinceReceive: lastRecvT ? performance.now() - lastRecvT : null,
      bufferedAmount: ws?.bufferedAmount ?? 0
    };
  }

  return {
    uploadAvatar,
    connect,
    disconnect,
    sendSemantic,
    getOutputCanvas: () => outputCanvas,
    getStats,
    getStatus: () => status,
    hasAvatar: () => Boolean(avatarId),
    isReady: () => receivedFrames > 0
  };
}
