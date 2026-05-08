import { createSemanticPacket } from "../state/semanticParams.js";

const DEFAULT_RECONNECT_DELAYS_MS = [1000, 2500, 5000];
const MAX_BUFFERED_BYTES = 16_000;

function byteLength(text) {
  return new TextEncoder().encode(text).length;
}

function clamp(value, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return min;
  return Math.min(Math.max(n, min), max);
}

export function createSemanticSocket({
  enabled = false,
  url = "",
  maxPacketsPerSecond = 24,
  reconnectDelaysMs = DEFAULT_RECONNECT_DELAYS_MS,
  maxBufferedBytes = MAX_BUFFERED_BYTES,
  compact = true,
  onStatusChange
} = {}) {
  let ws = null;
  let status = enabled && url ? "closed" : "disabled";
  let reconnectTimer = null;
  let reconnectAttempt = 0;
  let manualClose = false;
  let lastSendT = 0;
  let lastRateT = performance.now();
  let sentSinceRate = 0;
  let bytesSinceRate = 0;
  let packetsPerSecond = 0;
  let kbps = 0;
  let sentPackets = 0;
  let droppedPackets = 0;
  let lastPacketBytes = 0;
  let lastSerializedPacket = "";
  let rttMs = null;
  let lastPingT = 0;
  let pendingPingT = 0;

  const minIntervalMs = 1000 / clamp(maxPacketsPerSecond, 1, 30);
  const backoffDelays =
    Array.isArray(reconnectDelaysMs) && reconnectDelaysMs.length
      ? reconnectDelaysMs
      : DEFAULT_RECONNECT_DELAYS_MS;

  function setStatus(next) {
    if (status === next) return;
    status = next;
    onStatusChange?.(status);
  }

  function tickRates(now = performance.now()) {
    if (now - lastRateT < 1000) return;
    const elapsed = now - lastRateT;
    packetsPerSecond = (sentSinceRate * 1000) / elapsed;
    kbps = (bytesSinceRate / 1024) * (1000 / elapsed);
    sentSinceRate = 0;
    bytesSinceRate = 0;
    lastRateT = now;
  }

  function clearReconnectTimer() {
    if (!reconnectTimer) return;
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }

  function scheduleReconnect() {
    if (manualClose || !enabled || !url || reconnectTimer) return;
    const index = Math.min(reconnectAttempt, backoffDelays.length - 1);
    const delay = backoffDelays[index];
    reconnectAttempt = Math.min(reconnectAttempt + 1, backoffDelays.length - 1);
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, delay);
  }

  function sendRaw(text) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return false;
    if (ws.bufferedAmount + byteLength(text) > maxBufferedBytes) return false;
    ws.send(text);
    return true;
  }

  function maybePing(now) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (pendingPingT || now - lastPingT < 2000) return;
    pendingPingT = now;
    lastPingT = now;
    sendRaw(JSON.stringify({ type: "ping", t: Math.round(now) }));
  }

  function handleTextMessage(text) {
    try {
      const data = JSON.parse(text);
      if (data?.type === "ping") {
        sendRaw(JSON.stringify({ type: "pong", t: data.t ?? Math.round(performance.now()) }));
        return;
      }
      if (data?.type === "pong" && pendingPingT) {
        rttMs = performance.now() - pendingPingT;
        pendingPingT = 0;
      }
    } catch {
      // Semantic receivers may send non-JSON logs; ignore them.
    }
  }

  function connect() {
    if (!enabled || !url) {
      setStatus("disabled");
      return;
    }
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;

    clearReconnectTimer();
    manualClose = false;
    setStatus("connecting");

    try {
      const socket = new WebSocket(url);
      ws = socket;
      socket.addEventListener("open", () => {
        if (socket !== ws) return;
        reconnectAttempt = 0;
        setStatus("open");
      });
      socket.addEventListener("message", (ev) => {
        if (typeof ev.data === "string") handleTextMessage(ev.data);
      });
      socket.addEventListener("close", () => {
        if (socket !== ws) return;
        ws = null;
        pendingPingT = 0;
        setStatus(enabled && url ? "closed" : "disabled");
        scheduleReconnect();
      });
      socket.addEventListener("error", () => {
        if (socket !== ws) return;
        try {
          socket.close();
        } catch {
          // Socket is already unusable.
        }
      });
    } catch {
      ws = null;
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
    setStatus(enabled && url ? "closed" : "disabled");
  }

  function send(params) {
    const now = performance.now();
    const packet = createSemanticPacket(params, { compact });
    if (!packet) return false;

    const serialized = JSON.stringify(packet);
    lastSerializedPacket = serialized;
    lastPacketBytes = byteLength(serialized);
    tickRates(now);

    if (!enabled || !url) return false;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      droppedPackets += 1;
      return false;
    }
    if (lastSendT && now - lastSendT < minIntervalMs) return false;
    if (ws.bufferedAmount + lastPacketBytes > maxBufferedBytes) {
      droppedPackets += 1;
      return false;
    }

    try {
      ws.send(serialized);
      sentPackets += 1;
      sentSinceRate += 1;
      bytesSinceRate += lastPacketBytes;
      lastSendT = now;
      maybePing(now);
      return true;
    } catch {
      droppedPackets += 1;
      return false;
    }
  }

  function getStats() {
    tickRates();
    return {
      status,
      url,
      enabled: enabled && Boolean(url),
      rttMs,
      sentPackets,
      droppedPackets,
      packetsPerSecond,
      kbps,
      lastPacketBytes,
      lastSerializedPacket,
      bufferedAmount: ws?.bufferedAmount ?? 0
    };
  }

  return {
    connect,
    disconnect,
    send,
    getStats,
    getStatus: () => status
  };
}
