import { startCamera, stopCamera } from "./tracking/camera.js";
import {
  AI_FACE_CONFIG,
  AVATAR_CONFIG,
  CAMERA_CONSTRAINTS,
  PUPPET_CONFIG,
  POSE_RENDER_CONFIG,
  POSE_TRACKER_CONFIG,
  RENDER_CONFIG,
  STREAM_CONFIG,
  THREE_CONFIG
} from "./config.js";
import { createPoseTracker } from "./tracking/poseTracker.js";
import { createHandTracker } from "./tracking/handTracker.js";
import { createFaceTracker } from "./tracking/faceTracker.js";
import { createRenderer } from "./render/renderer.js";
import { createStateBuilder } from "./state/stateBuilder.js";
import { createSyncBuffer } from "./state/syncBuffer.js";
import { createAiFaceClient } from "./face/aiFaceClient.js";
import { createAiHeadRenderer } from "./face/aiHeadRenderer.js";
import { resolveHeadRenderer } from "./face/head.js";
import { createWebRtcPublisher } from "./stream/webrtc.js";
import { createThreePreview } from "./three/threePreview.js";

const video = document.querySelector("#camera");
const canvas = document.querySelector("#preview");
const threeCanvas = document.querySelector("#three-canvas");
const status = document.querySelector("#status");
const aiWsStateLabel = document.querySelector("#ai-ws-state");
const fpsLabel = document.querySelector("#fps");
const scaleLabel = document.querySelector("#scale");
const resolutionLabel = document.querySelector("#resolution");
const trackingStateLabel = document.querySelector("#tracking-state");

let stream;
let renderer;
let threePreview;
let poseTracker;
let handTracker;
let faceTracker;
let aiFaceClient;
let webrtcPublisher;
let faceIdleFps = null;
let faceActiveFps = null;
let activeCanvas = canvas;
let puppetImage = null;

function setStatus(message, tone) {
  status.textContent = message;
  status.dataset.tone = tone;
}

function setAiWsState(state) {
  if (!aiWsStateLabel) return;
  const normalized = state === "ready" ? "open" : state;
  aiWsStateLabel.textContent = `AI WS: ${normalized}`;
  aiWsStateLabel.dataset.state = normalized;
  aiWsStateLabel.hidden = false;
}

function getAiFaceStats() {
  return aiFaceClient?.getStats?.() ?? null;
}

function isAiRealtime() {
  return aiFaceClient?.isRealtime?.(AI_FACE_CONFIG.maxRealtimeLatencyMs) === true;
}

function setMetrics({ fps = 0, scale = 1, width = activeCanvas?.width, height = activeCanvas?.height } = {}) {
  fpsLabel.textContent = `${fps.toFixed(1)} FPS`;
  scaleLabel.textContent = `${Math.round(scale * 100)}% scale`;
  resolutionLabel.textContent = `${width} x ${height}`;
}

function setTrackingState(overlay, hands) {
  if (!trackingStateLabel) return;
  const poseState = overlay?.landmarks?.length ? "on" : "lost";
  trackingStateLabel.textContent = `pose: ${poseState} / hands: ${hands?.length ?? 0}`;
}

function attachSyncHud(syncBuffer, isVisible) {
  const root = document.querySelector("#sync-hud");
  if (!root) return;
  const elMode = root.querySelector("#sync-mode");
  const elDelay = root.querySelector("#sync-delay");
  const elLat = root.querySelector("#sync-latency");
  const elOff = root.querySelector("#sync-offset");
  const elInt = root.querySelector("#sync-interp");
  setInterval(() => {
    const visible = !!isVisible?.();
    root.hidden = !visible;
    if (!visible) return;
    const s = syncBuffer.getStats();
    elMode.textContent = `mode: ${s.lastMatchMode}`;
    elDelay.textContent = `delay: ${s.currentDelayMs.toFixed(1)} ms (target ${s.targetDelayMs.toFixed(0)})`;
    elLat.textContent = `lat: ${s.measuredLatencyMs.toFixed(1)} ms`;
    elOff.textContent = `offset: ${s.syncOffsetMs.toFixed(1)} ms`;
    elInt.textContent = `interp: t=${s.interpolationT.toFixed(2)}, gap=${s.bracketGapMs.toFixed(0)} ms`;
  }, 100);
}

function attachAiBridge(client, { sourceImageUrl = null } = {}) {
  const root = document.querySelector("#ai-bridge");
  if (!root || !client?.getStats) return;

  const preview = root.querySelector("#ai-bridge-preview");
  const source = root.querySelector("#ai-bridge-source");
  const state = root.querySelector("#ai-bridge-state");
  const sent = root.querySelector("#ai-bridge-sent");
  const received = root.querySelector("#ai-bridge-received");
  const bitrate = root.querySelector("#ai-bridge-bitrate");
  const latency = root.querySelector("#ai-bridge-latency");
  const last = root.querySelector("#ai-bridge-last");
  const ultraToggle = root.querySelector("#ai-ultra-mode");
  const previewCtx = preview?.getContext("2d");
  let lastDrawnReceived = -1;
  let sourcePreview = null;
  let sourcePreviewDrawn = false;
  let nextHealthCheck = 0;
  let healthInFlight = false;
  let serverInferenceMs = null;
  let serverInferenceFps = null;

  root.hidden = false;
  if (ultraToggle && client.getUltraRealtimeMode && client.setUltraRealtimeMode) {
    ultraToggle.checked = client.getUltraRealtimeMode();
    ultraToggle.addEventListener("change", () => {
      client.setUltraRealtimeMode(ultraToggle.checked);
    });
  }

  if (sourceImageUrl && previewCtx) {
    loadImage(sourceImageUrl)
      .then((img) => {
        sourcePreview = img;
        sourcePreviewDrawn = false;
      })
      .catch(() => {
        sourcePreview = null;
      });
  }

  function makeHealthUrl(wsUrl) {
    try {
      const url = new URL(wsUrl);
      url.protocol = url.protocol === "wss:" ? "https:" : "http:";
      url.pathname = "/healthz";
      url.search = "";
      url.hash = "";
      return url.toString();
    } catch (_) {
      return null;
    }
  }

  function basename(path) {
    if (typeof path !== "string" || !path) return "-";
    return path.split(/[\\/]/).pop() || path;
  }

  function refreshHealth(stats) {
    if (!source || healthInFlight) return;
    const now = performance.now();
    if (now < nextHealthCheck) return;
    const url = makeHealthUrl(stats.url);
    if (!url) return;
    nextHealthCheck = now + 3000;
    healthInFlight = true;
    fetch(url, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`health ${r.status}`))))
      .then((data) => {
        serverInferenceMs = Number(data.last_inference_ms ?? NaN);
        serverInferenceFps = Number(data.inference_fps ?? NaN);
        client.setServerMetrics?.({
          inferenceMs: serverInferenceMs,
          inferenceFps: serverInferenceFps
        });
        source.textContent = `source: ${basename(data.source_path)} / ${data.device ?? "-"}`;
      })
      .catch(() => {
        source.textContent = "source: health unavailable";
      })
      .finally(() => {
        healthInFlight = false;
      });
  }

  setInterval(() => {
    const stats = client.getStats();
    refreshHealth(stats);
    state.textContent = `state: ${stats.status}${stats.awaitingResponse ? " / waiting" : ""} / in-flight: ${stats.activeInFlightCount}`;
    sent.textContent = `sent: ${stats.sentFrames} (${stats.sentFps.toFixed(1)}/s) / skipped: ${stats.skippedFrames}`;
    received.textContent = `received: ${stats.receivedFrames}/${stats.rawReceivedFrames} (${stats.receivedFps.toFixed(1)}/s)`;
    if (bitrate) {
      bitrate.textContent = `net: up ${stats.uploadKBps.toFixed(1)} KB/s / down ${stats.downloadKBps.toFixed(1)} KB/s / avg ${Math.round(stats.avgUploadFrameBytes)}B:${Math.round(stats.avgDownloadFrameBytes)}B`;
    }
    if (latency) {
      const rtt = stats.websocketRttMs == null ? "-" : `${stats.websocketRttMs.toFixed(0)} ms`;
      const infer = Number.isFinite(serverInferenceMs) ? `${serverInferenceMs.toFixed(0)} ms` : "-";
      const inferFps = Number.isFinite(serverInferenceFps) ? ` @ ${serverInferenceFps.toFixed(1)}/s` : "";
      latency.textContent = `rtt: ${rtt} / infer: ${infer}${inferFps} / ${stats.driverSize}->${stats.outputSize} q${Math.round(stats.jpegQuality * 100)}`;
    }
    last.textContent = `last: ${
      stats.msSinceReceive == null ? "-" : `${(stats.msSinceReceive / 1000).toFixed(1)}s ago`
    }`;
    if (stats.lastError) {
      last.textContent = `error: ${stats.lastError}`;
    } else if (stats.slowPreview) {
      last.textContent = `slow CPU preview: ${(stats.lastRoundTripMs / 1000).toFixed(1)}s`;
    }

    if (previewCtx) {
      if (stats.receivedFrames > 0 && stats.receivedFrames !== lastDrawnReceived) {
        if (preview.width !== stats.outputSize || preview.height !== stats.outputSize) {
          preview.width = stats.outputSize;
          preview.height = stats.outputSize;
        }
        previewCtx.drawImage(client.getOutputCanvas(), 0, 0, preview.width, preview.height);
        lastDrawnReceived = stats.receivedFrames;
        sourcePreviewDrawn = false;
      } else if (!stats.receivedFrames && sourcePreview && !sourcePreviewDrawn) {
        previewCtx.clearRect(0, 0, preview.width, preview.height);
        const fit = fitContain(
          sourcePreview.naturalWidth || sourcePreview.width || 1,
          sourcePreview.naturalHeight || sourcePreview.height || 1,
          preview.width,
          preview.height
        );
        previewCtx.drawImage(sourcePreview, fit.x, fit.y, fit.width, fit.height);
        sourcePreviewDrawn = true;
      }
    }
  }, 250);
}

function formatError(error) {
  if (!error || typeof error !== "object") {
    return "Unable to start the camera.";
  }

  if (error instanceof Error && error.message === "Unable to initialize MediaPipe pose tracking.") {
    return "Unable to load pose tracking assets.";
  }

  switch (error.name) {
    case "NotAllowedError":
      return "Camera is blocked. Allow camera for 127.0.0.1 in the address bar, then reload.";
    case "NotFoundError":
      return "No webcam was found.";
    case "NotReadableError":
      return "The webcam is already in use by another app.";
    default:
      return "Unable to start the camera.";
  }
}

function cleanup() {
  webrtcPublisher?.stop();
  aiFaceClient?.disconnect();
  renderer?.stop();
  threePreview?.dispose();
  faceTracker?.close();
  handTracker?.close();
  poseTracker?.close();
  stopCamera(stream);
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.decoding = "async";
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(e);
    img.src = src;
  });
}

function fitContain(srcW, srcH, dstW, dstH) {
  const scale = Math.min(dstW / srcW, dstH / srcH);
  const width = srcW * scale;
  const height = srcH * scale;
  return {
    x: (dstW - width) * 0.5,
    y: (dstH - height) * 0.5,
    width,
    height
  };
}

function drawFallbackImage(image) {
  if (!image || !canvas) return;
  const context = canvas.getContext("2d");
  if (!context) return;

  const srcWidth = image.naturalWidth || image.width || 1;
  const srcHeight = image.naturalHeight || image.height || 1;
  const fit = fitContain(srcWidth, srcHeight, canvas.width, canvas.height);

  canvas.hidden = false;
  threeCanvas.hidden = true;
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, fit.x, fit.y, fit.width, fit.height);
}

async function init() {
  if (!window.isSecureContext) {
    setStatus("Open this app on localhost or HTTPS to use getUserMedia.", "error");
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("getUserMedia is not available in this browser.", "error");
    return;
  }

  const puppetEnabled = PUPPET_CONFIG.enabled;
  const aiEnabled = AI_FACE_CONFIG.enabled && Boolean(navigator.mediaDevices?.getUserMedia);

  if (aiWsStateLabel) {
    aiWsStateLabel.hidden = !aiEnabled;
    if (aiEnabled) setAiWsState("closed");
  }

  const aiBridge = document.querySelector("#ai-bridge");
  if (aiBridge) aiBridge.hidden = true;

  if (puppetEnabled) {
    try {
      puppetImage = await loadImage(PUPPET_CONFIG.imageUrl);
      drawFallbackImage(puppetImage);
      setMetrics({ fps: 0, scale: 1, width: canvas.width, height: canvas.height });
    } catch (e) {
      console.error(e);
      setStatus("Unable to load puppet image.", "error");
      return;
    }
  }

  setStatus("Requesting camera access...", "working");

  try {
    stream = await startCamera(video, CAMERA_CONSTRAINTS);

    if (THREE_CONFIG.enabled && !puppetEnabled) {
      activeCanvas = threeCanvas ?? canvas;
      if (activeCanvas !== canvas) {
        canvas.hidden = true;
        activeCanvas.hidden = false;
      }
      if (aiWsStateLabel) {
        aiWsStateLabel.hidden = true;
      }
      threePreview = await createThreePreview({
        canvas: activeCanvas,
        mode: THREE_CONFIG.mode,
        modelUrl: THREE_CONFIG.modelUrl,
        imageUrl: THREE_CONFIG.imageUrl,
        onStats: setMetrics,
        onStatus: setStatus
      });
      threePreview.start();
      setMetrics({ scale: 1 });
      return;
    }

    setStatus("Loading pose tracker...", "working");

    // Optional: load a static image as the avatar head sprite. This
    // mutates AVATAR_CONFIG.head before the renderer is constructed so
    // both the direct (no-AI) path and the AI-fallback path pick it up.
    if (AVATAR_CONFIG.testHeadImage) {
      try {
        const img = await loadImage(AVATAR_CONFIG.testHeadImage);
        AVATAR_CONFIG.head = {
          type: "image",
          source: img,
          clipOval: true
        };
      } catch (imgError) {
        console.warn("Test head image failed to load:", imgError);
      }
    }

    poseTracker = await createPoseTracker({
      video,
      ...POSE_TRACKER_CONFIG
    });

    // Hand tracking lives in its own module so it can be tuned (or replaced)
    // independently of pose. Run it slightly slower than pose tracking to
    // protect the 15 FPS floor on weaker machines.
    if (POSE_TRACKER_CONFIG.hand) {
      try {
        handTracker = await createHandTracker({
          video,
          wasmPath: POSE_TRACKER_CONFIG.wasmPath,
          modelAssetPath: POSE_TRACKER_CONFIG.hand.modelAssetPath,
          numHands: POSE_TRACKER_CONFIG.hand.numHands,
          minHandDetectionConfidence: POSE_TRACKER_CONFIG.hand.minHandDetectionConfidence,
          minHandPresenceConfidence: POSE_TRACKER_CONFIG.hand.minHandPresenceConfidence,
          minTrackingConfidence: POSE_TRACKER_CONFIG.hand.minTrackingConfidence,
          targetFps: POSE_TRACKER_CONFIG.hand.targetFps ?? POSE_TRACKER_CONFIG.targetFps
        });
      } catch (handError) {
        // Hand tracking is non-essential; warn but keep pose running.
        console.warn("Hand tracker unavailable:", handError);
      }
    }

    // Face mesh tracking — also modular and independent. Throttled lower
    // than hands because the 468-point mesh is the heaviest of the three.
    if (POSE_TRACKER_CONFIG.face) {
      try {
        faceActiveFps = POSE_TRACKER_CONFIG.face.targetFps ?? POSE_TRACKER_CONFIG.targetFps;
        faceIdleFps = Math.max(2, Math.round(faceActiveFps * 0.5));
        faceTracker = await createFaceTracker({
          video,
          wasmPath: POSE_TRACKER_CONFIG.wasmPath,
          modelAssetPath: POSE_TRACKER_CONFIG.face.modelAssetPath,
          numFaces: POSE_TRACKER_CONFIG.face.numFaces,
          minFaceDetectionConfidence: POSE_TRACKER_CONFIG.face.minFaceDetectionConfidence,
          minFacePresenceConfidence: POSE_TRACKER_CONFIG.face.minFacePresenceConfidence,
          minTrackingConfidence: POSE_TRACKER_CONFIG.face.minTrackingConfidence,
          targetFps: faceActiveFps,
          minTargetFps: faceIdleFps,
          maxTargetFps: faceActiveFps
        });
        faceTracker.setTargetFps?.(aiEnabled ? faceActiveFps : faceIdleFps);
      } catch (faceError) {
        console.warn("Face tracker unavailable:", faceError);
      }
    }

    // ------------------------------------------------------------------
    // Unified state pipeline (stage: composition + sync).
    // The state builder produces a single per-frame object that downstream
    // stages (AI face client, sync buffer, future stream encoder) all read
    // from. The sync buffer keeps recent overlays for delay-matching once
    // AI face frames start coming back from the GPU service.
    // ------------------------------------------------------------------
    const stateBuilder = createStateBuilder();
    const syncBuffer = createSyncBuffer({
      delayMs: aiEnabled ? AI_FACE_CONFIG.bufferDelayMs : 0,
      minDelayMs: AI_FACE_CONFIG.bufferMinDelayMs,
      maxDelayMs: AI_FACE_CONFIG.bufferMaxDelayMs,
      latencyAlpha: AI_FACE_CONFIG.bufferLatencyAlpha,
      driftCorrectionPerSec: AI_FACE_CONFIG.bufferDriftCorrectionPerSec
    });
    // Expose sync stats for live debug:
    //   window.__sync.getStats()
    window.__sync = syncBuffer;

    // ------------------------------------------------------------------
    // AI face integration. Disabled by default — flip
    // AI_FACE_CONFIG.enabled in config.js after starting the Python
    // face-service. When the WS opens we render the AI portrait
    // into the head slot; on close / stale frames we automatically fall
    // back to the local default head with no flicker.
    // ------------------------------------------------------------------
    let avatarHeadRenderer;
    if (aiEnabled) {
      aiFaceClient = createAiFaceClient({
        url: AI_FACE_CONFIG.url,
        connectTimeoutMs: AI_FACE_CONFIG.connectTimeoutMs,
        reconnectDelaysMs: AI_FACE_CONFIG.reconnectDelaysMs,
        targetFps: AI_FACE_CONFIG.targetFps,
        jpegQuality: AI_FACE_CONFIG.jpegQuality,
        driverSize: AI_FACE_CONFIG.driverSize,
        outputSize: AI_FACE_CONFIG.outputSize,
        ultraRealtime: AI_FACE_CONFIG.ultraRealtime,
        ultraDriverSize: AI_FACE_CONFIG.ultraDriverSize,
        ultraOutputSize: AI_FACE_CONFIG.ultraOutputSize,
        ultraJpegQuality: AI_FACE_CONFIG.ultraJpegQuality,
        staleAfterMs: AI_FACE_CONFIG.staleAfterMs,
        slowPreviewLatencyMs: AI_FACE_CONFIG.slowPreviewLatencyMs,
        slowPreviewIntervalMs: AI_FACE_CONFIG.slowPreviewIntervalMs,
        syncBuffer
      });
      const fallbackHead = resolveHeadRenderer(AVATAR_CONFIG.head, {
        palette: AVATAR_CONFIG.palette,
        style: AVATAR_CONFIG.style
      });
      avatarHeadRenderer = createAiHeadRenderer({
        client: aiFaceClient,
        fallback: fallbackHead,
        edgeFeatherPx: AI_FACE_CONFIG.edgeFeatherPx,
        edgeFeatherFrac: AI_FACE_CONFIG.edgeFeatherFrac,
        scaleBoost: AI_FACE_CONFIG.scaleBoost,
        ovalAspectY: AI_FACE_CONFIG.ovalAspectY,
        colorCorrect: AI_FACE_CONFIG.colorCorrect,
        brightness: AI_FACE_CONFIG.brightness,
        contrast: AI_FACE_CONFIG.contrast,
        saturation: AI_FACE_CONFIG.saturation,
        toneTint: AI_FACE_CONFIG.toneTint,
        toneStrength: AI_FACE_CONFIG.toneStrength,
        chinShadowStrength: AI_FACE_CONFIG.chinShadowStrength,
        rimShadowStrength: AI_FACE_CONFIG.rimShadowStrength,
        colorSmoothingAlpha: AI_FACE_CONFIG.colorSmoothingAlpha,
        smoothingAlpha: AI_FACE_CONFIG.smoothingAlpha,
        offsetX: AI_FACE_CONFIG.offsetX,
        offsetY: AI_FACE_CONFIG.offsetY,
        yawWeight: AI_FACE_CONFIG.yawWeight,
        pitchWeight: AI_FACE_CONFIG.pitchWeight,
        debug: AI_FACE_CONFIG.debug
      });
      // Expose runtime knobs for live calibration from DevTools:
      //   window.__aiHead.setCalibration({ offsetX: 4, offsetY: -2 })
      //   window.__aiHead.setDebug(true)
      window.__aiHead = avatarHeadRenderer;
      aiFaceClient.onStatusChange((s) => {
        setAiWsState(s);
        if (faceTracker && faceActiveFps && faceIdleFps) {
          faceTracker.setTargetFps?.(s === "open" || s === "ready" ? faceActiveFps : faceIdleFps);
        }
      });
      attachAiBridge(aiFaceClient, { sourceImageUrl: PUPPET_CONFIG.imageUrl });
      // Sync HUD — visible whenever the AI head is in debug mode (or when
      // AI_FACE_CONFIG.debug is true). Updated at ~10 Hz on a setInterval
      // so it doesn't add work to the render loop.
      attachSyncHud(syncBuffer, () => avatarHeadRenderer?.getDebug?.() === true);
      aiFaceClient.connect();
    }

    renderer = createRenderer({
      video,
      canvas,
      ...RENDER_CONFIG,
      beforeRender(now) {
        // Drive all trackers from the same animation tick. Each tracker
        // throttles itself internally so they coexist without blowing the
        // FPS budget.
        poseTracker?.processFrame(now);
        handTracker?.processFrame(now);
        faceTracker?.processFrame(now);
        if (handTracker) {
          poseTracker?.setHands(handTracker.getLatestHands());
        }
        if (faceTracker) {
          poseTracker?.setFaces(faceTracker.getLatestFaces());
        }
        // Capture the unified state once the trackers have fed each other,
        // then push to the sync buffer so the renderer can pull a delayed
        // snapshot when AI face frames are paired against it.
        const overlay = poseTracker?.getLatestOverlay();
        setTrackingState(overlay, handTracker?.getLatestHands?.() ?? []);
        if (overlay) {
          const state = stateBuilder.build(overlay, overlay?.headPose ?? null);
          syncBuffer.push(state);
          if (aiFaceClient) {
            aiFaceClient.sendDriver(video, state);
          }
        }
      },
      getFrame() {
        // Only sync body motion to AI frames when LivePortrait is genuinely
        // realtime. CPU inference can be tens of seconds late, so the body
        // must keep following fresh MediaPipe landmarks in that mode.
        if (aiFaceClient && isAiRealtime()) {
          const synced = syncBuffer.getRenderState();
          if (synced?.overlay) {
            return {
              overlay: synced.overlay,
              frameId: synced.frameId,
              timestamp: synced.timestamp
            };
          }
        }
        const overlay = poseTracker?.getLatestOverlay();
        const last = stateBuilder.getLast?.();
        return {
          overlay,
          frameId: last?.frameId ?? null,
          timestamp: last?.timestamp ?? null
        };
      },
      poseConfig: POSE_RENDER_CONFIG,
      avatarConfig: {
        ...AVATAR_CONFIG,
        enabled: AVATAR_CONFIG.enabled !== false && !puppetEnabled,
        puppet: puppetEnabled
          ? {
              ...PUPPET_CONFIG,
              source: puppetImage,
              getHeadSource: aiFaceClient
                ? () => (isAiRealtime() ? aiFaceClient.getOutputCanvas() : null)
                : null
            }
          : null,
        ...(avatarHeadRenderer ? { headRenderer: avatarHeadRenderer } : {})
      },
      onStats({ fps, scale, width, height }) {
        setMetrics({ fps, scale, width, height });

        if (fps < RENDER_CONFIG.minimumFps) {
          poseTracker?.relievePressure();
          faceTracker?.relievePressure?.();
          setStatus("Reducing render and tracking load to hold 15+ FPS.", "working");
          return;
        }

        if (fps > POSE_TRACKER_CONFIG.recoveryFps) {
          poseTracker?.restoreRate();
          faceTracker?.restoreRate?.();
        }

        const aiStats = getAiFaceStats();
        if (aiStats?.lastRoundTripMs > AI_FACE_CONFIG.maxRealtimeLatencyMs) {
          setStatus("Live pose preview; LivePortrait CPU preview is slow.", "ready");
        } else {
          setStatus("Live pose preview", "ready");
        }
      }
    });

    renderer.start();
    setMetrics({ scale: RENDER_CONFIG.initialScale });
    setStatus("Live pose preview", "ready");

    // ------------------------------------------------------------------
    // Outbound WebRTC stream of the rendered avatar canvas. Disabled by
    // default — set STREAM_CONFIG.enabled=true and provide a WHIP-style
    // signaling endpoint to start publishing.
    // ------------------------------------------------------------------
    if (STREAM_CONFIG.enabled && STREAM_CONFIG.endpoint) {
      try {
        webrtcPublisher = createWebRtcPublisher({
          endpoint: STREAM_CONFIG.endpoint,
          targetFps: STREAM_CONFIG.targetFps,
          bitrateKbps: STREAM_CONFIG.bitrateKbps,
          preferCodec: STREAM_CONFIG.preferCodec
        });
        webrtcPublisher.onStatusChange((s) => {
          if (s === "live") setStatus("Streaming live", "ready");
          else if (s === "failed") setStatus("Stream failed", "error");
        });
        await webrtcPublisher.start(canvas);
      } catch (streamError) {
        console.warn("WebRTC publisher failed to start:", streamError);
      }
    }
  } catch (error) {
    console.error(error);
    cleanup();
    if (puppetEnabled && puppetImage) {
      drawFallbackImage(puppetImage);
      setMetrics({ fps: 0, scale: 1, width: canvas.width, height: canvas.height });
      if (error?.name === "NotAllowedError") {
        setStatus("Camera is blocked. Allow camera for 127.0.0.1 in the address bar, then reload.", "error");
      } else {
        setStatus(`${formatError(error)} Showing the paper-doll source image only.`, "error");
      }
      return;
    }
    setStatus(formatError(error), "error");
  }
}

window.addEventListener("beforeunload", cleanup);

document.addEventListener("visibilitychange", () => {
  if (!renderer && !threePreview) {
    return;
  }

  if (document.hidden) {
    renderer?.stop();
    threePreview?.stop();
    return;
  }

  poseTracker?.reset();
  handTracker?.reset();
  faceTracker?.reset();
  renderer?.start();
  threePreview?.start();
});

init();
