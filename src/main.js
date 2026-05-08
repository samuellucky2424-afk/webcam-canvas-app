import { startCamera, stopCamera } from "./tracking/camera.js";
import {
  AVATAR_CONFIG,
  CAMERA_CONSTRAINTS,
  LIVEPORTRAIT_CONFIG,
  POSE_RENDER_CONFIG,
  POSE_TRACKER_CONFIG,
  PUPPET_CONFIG,
  RENDER_CONFIG,
  SEMANTIC_STREAM_CONFIG,
  STREAM_CONFIG,
  THREE_CONFIG
} from "./config.js";
import { createPoseTracker } from "./tracking/poseTracker.js";
import { createHandTracker } from "./tracking/handTracker.js";
import { createFaceTracker } from "./tracking/faceTracker.js";
import { createRenderer } from "./render/renderer.js";
import { createStateBuilder } from "./state/stateBuilder.js";
import { createHeadPoseEstimator } from "./state/headPose.js";
import { createSemanticParamExtractor } from "./state/semanticParams.js";
import { createSemanticAvatarClient } from "./face/semanticAvatarClient.js";
import { createVirtualCamera } from "./stream/virtualCamera.js";
import { createThreePreview } from "./three/threePreview.js";

const video = document.querySelector("#camera");
const canvas = document.querySelector("#preview");
const threeCanvas = document.querySelector("#three-canvas");
const status = document.querySelector("#status");
const semanticWsStateLabel = document.querySelector("#semantic-ws-state");
const fpsLabel = document.querySelector("#fps");
const scaleLabel = document.querySelector("#scale");
const resolutionLabel = document.querySelector("#resolution");
const trackingStateLabel = document.querySelector("#tracking-state");
const landmarkToggle = document.querySelector("#landmark-toggle");
const uploadDropzone = document.querySelector("#avatar-dropzone");
const portraitUpload = document.querySelector("#portrait-upload");
const portraitPickButton = document.querySelector("#portrait-pick");
const livePortraitBaseUrlInput = document.querySelector("#liveportrait-base-url");
const livePortraitConnectButton = document.querySelector("#liveportrait-connect");
const livePortraitDisconnectButton = document.querySelector("#liveportrait-disconnect");
const livePortraitPreview = document.querySelector("#liveportrait-preview");
const livePortraitStateLabel = document.querySelector("#liveportrait-state");
const livePortraitRateLabel = document.querySelector("#liveportrait-rate");
const livePortraitNetLabel = document.querySelector("#liveportrait-net");
const livePortraitLatencyLabel = document.querySelector("#liveportrait-latency");
const livePortraitSourceLabel = document.querySelector("#liveportrait-source");
const avatarLoadedLabel = document.querySelector("#avatar-loaded");
const pipelineStages = {
  webcam: document.querySelector("#stage-webcam"),
  landmarks: document.querySelector("#stage-landmarks"),
  motion: document.querySelector("#stage-motion"),
  renderer: document.querySelector("#stage-renderer"),
  virtualCamera: document.querySelector("#stage-virtual-camera")
};

let stream;
let renderer;
let threePreview;
let poseTracker;
let handTracker;
let faceTracker;
let avatarClient;
let virtualCamera;
let activeCanvas = canvas;
let puppetImage = null;
let uploadedPortraitImage = null;
let lastSemantic = null;
let lastUnifiedState = null;
let runtimeAvatarConfig = AVATAR_CONFIG;
let usingRemoteAvatarHead = false;

function setStage(stage, state, label) {
  const el = pipelineStages[stage];
  if (!el) return;
  el.dataset.state = state;
  if (label) el.textContent = label;
}

function setStatus(message, tone) {
  status.textContent = message;
  status.dataset.tone = tone;
}

function setSemanticWsState(state) {
  if (!semanticWsStateLabel) return;
  semanticWsStateLabel.textContent = `Semantic WS: ${state}`;
  semanticWsStateLabel.dataset.state = state;
}

function setLivePortraitState(state) {
  if (!livePortraitStateLabel) return;
  livePortraitStateLabel.textContent = `state: ${state}`;
  livePortraitStateLabel.dataset.state = state;
  setSemanticWsState(state === "open" ? "open" : state === "connecting" ? "connecting" : state);
}

function setUploadStatus(uploadStatus) {
  if (!avatarLoadedLabel) return;
  const label =
    uploadStatus === "ready" ? "avatar: loaded" :
    uploadStatus === "uploading" ? "avatar: uploading" :
    uploadStatus === "error" ? "avatar: error" :
    "avatar: none";
  avatarLoadedLabel.textContent = label;
  avatarLoadedLabel.dataset.state = uploadStatus === "ready" ? "open" : uploadStatus === "error" ? "closed" : "disabled";
}

function setMetrics({ fps = 0, scale = 1, width = activeCanvas?.width, height = activeCanvas?.height } = {}) {
  fpsLabel.textContent = `${fps.toFixed(1)} FPS`;
  scaleLabel.textContent = `${Math.round(scale * 100)}% scale`;
  resolutionLabel.textContent = `${width} x ${height}`;
}

function setTrackingState(overlay, hands) {
  if (!trackingStateLabel) return;
  const poseState = overlay?.landmarks?.length ? "on" : "lost";
  const faceState = overlay?.faces?.length ? "on" : "lost";
  trackingStateLabel.textContent = `pose: ${poseState} / hands: ${hands?.length ?? 0} / face: ${faceState}`;
}

function drawImagePreview(source) {
  const context = livePortraitPreview?.getContext("2d");
  if (!context || !source) return;
  context.clearRect(0, 0, livePortraitPreview.width, livePortraitPreview.height);
  const width = source.naturalWidth || source.videoWidth || source.width || 1;
  const height = source.naturalHeight || source.videoHeight || source.height || 1;
  const fit = fitContain(width, height, livePortraitPreview.width, livePortraitPreview.height);
  context.drawImage(source, fit.x, fit.y, fit.width, fit.height);
}

function syncRemoteAvatarHead() {
  if (!renderer) return;
  if (avatarClient?.isReady?.()) {
    if (!usingRemoteAvatarHead) {
      renderer.setHeadSource(avatarClient.getOutputCanvas());
      usingRemoteAvatarHead = true;
    }
    return;
  }

  if (usingRemoteAvatarHead || uploadedPortraitImage) {
    renderer.setHeadSource(uploadedPortraitImage);
    usingRemoteAvatarHead = false;
  }
}

function createAvatarClient(baseUrl = livePortraitBaseUrlInput?.value || LIVEPORTRAIT_CONFIG.baseUrl) {
  avatarClient?.disconnect?.();
  avatarClient = createSemanticAvatarClient({
    baseUrl,
    uploadPath: LIVEPORTRAIT_CONFIG.uploadPath,
    wsPath: LIVEPORTRAIT_CONFIG.wsPath,
    maxPacketsPerSecond: LIVEPORTRAIT_CONFIG.maxPacketsPerSecond,
    reconnectDelaysMs: LIVEPORTRAIT_CONFIG.reconnectDelaysMs,
    maxBufferedBytes: LIVEPORTRAIT_CONFIG.maxBufferedBytes,
    onStatusChange: setLivePortraitState,
    onUploadStatus: setUploadStatus
  });
  setLivePortraitState(avatarClient.getStatus());
  return avatarClient;
}

async function previewFile(file) {
  const objectUrl = URL.createObjectURL(file);
  try {
    uploadedPortraitImage = await loadImage(objectUrl);
    drawImagePreview(uploadedPortraitImage);
    renderer?.setHeadSource(uploadedPortraitImage);
    usingRemoteAvatarHead = false;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

async function uploadAvatarFile(file) {
  if (!file) return;
  try {
    setUploadStatus("uploading");
    if (livePortraitSourceLabel) livePortraitSourceLabel.textContent = `source: ${file.name}`;
    await previewFile(file);
    const client = avatarClient ?? createAvatarClient();
    await client.uploadAvatar(file);
    if (LIVEPORTRAIT_CONFIG.connectAfterUpload) client.connect();
  } catch (error) {
    console.error(error);
    setUploadStatus("error");
    if (livePortraitSourceLabel) livePortraitSourceLabel.textContent = `source: ${error?.message || "upload failed"}`;
  }
}

function attachAvatarUploadPanel() {
  if (!LIVEPORTRAIT_CONFIG.enabled) return;
  if (livePortraitBaseUrlInput) livePortraitBaseUrlInput.value = LIVEPORTRAIT_CONFIG.baseUrl;
  createAvatarClient();
  setUploadStatus("empty");

  portraitPickButton?.addEventListener("click", () => portraitUpload?.click());
  portraitUpload?.addEventListener("change", () => {
    void uploadAvatarFile(portraitUpload.files?.[0]);
  });

  uploadDropzone?.addEventListener("dragenter", (event) => {
    event.preventDefault();
    uploadDropzone.dataset.dragging = "true";
  });
  uploadDropzone?.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadDropzone.dataset.dragging = "true";
  });
  uploadDropzone?.addEventListener("dragleave", () => {
    uploadDropzone.dataset.dragging = "false";
  });
  uploadDropzone?.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadDropzone.dataset.dragging = "false";
    void uploadAvatarFile(event.dataTransfer?.files?.[0]);
  });

  livePortraitConnectButton?.addEventListener("click", () => {
    createAvatarClient(livePortraitBaseUrlInput?.value || LIVEPORTRAIT_CONFIG.baseUrl).connect();
  });

  livePortraitDisconnectButton?.addEventListener("click", () => {
    avatarClient?.disconnect();
    usingRemoteAvatarHead = false;
    if (uploadedPortraitImage) renderer?.setHeadSource(uploadedPortraitImage);
  });

  setInterval(() => {
    if (!avatarClient) return;
    const stats = avatarClient.getStats();
    if (livePortraitRateLabel) {
      livePortraitRateLabel.textContent = `packets: ${stats.sentPps.toFixed(1)}/s / frames: ${stats.receivedFps.toFixed(1)}/s`;
    }
    if (livePortraitNetLabel) {
      livePortraitNetLabel.textContent = `net: ${stats.uploadKBps.toFixed(2)} / ${stats.downloadKBps.toFixed(1)} KB/s`;
    }
    if (livePortraitLatencyLabel) {
      livePortraitLatencyLabel.textContent = `rtt: ${stats.rttMs == null ? "-" : `${stats.rttMs.toFixed(0)} ms`} / dropped: ${stats.droppedPackets}`;
    }
    syncRemoteAvatarHead();
    if (stats.receivedFrames > 0) drawImagePreview(avatarClient.getOutputCanvas());
  }, 250);
}

function attachSemanticHud(getSemantic) {
  const root = document.querySelector("#semantic-hud");
  if (!root) return;
  const packet = root.querySelector("#semantic-packet-size");
  const pps = root.querySelector("#semantic-pps");
  const bandwidth = root.querySelector("#semantic-bandwidth");
  const rtt = root.querySelector("#semantic-rtt");
  const dropped = root.querySelector("#semantic-dropped");
  const interp = root.querySelector("#semantic-interp");
  const confidence = root.querySelector("#semantic-confidence");
  const sample = root.querySelector("#semantic-sample");

  root.hidden = false;
  setInterval(() => {
    const stats = avatarClient?.getStats?.() ?? {};
    const semantic = getSemantic?.();
    const latencyMs = semantic?.timestamp ? Math.max(0, performance.now() - semantic.timestamp) : null;

    packet.textContent = `packet: ${stats.lastPacketBytes ?? 0} B`;
    pps.textContent = `pps: ${(stats.sentPps ?? 0).toFixed(1)}`;
    bandwidth.textContent = `up: ${(stats.uploadKBps ?? 0).toFixed(2)} KB/s`;
    rtt.textContent = `rtt: ${stats.rttMs == null ? "-" : `${stats.rttMs.toFixed(0)} ms`}`;
    dropped.textContent = `dropped: ${stats.droppedPackets ?? 0}`;
    interp.textContent = `render lag: ${latencyMs == null ? "-" : `${latencyMs.toFixed(1)} ms`}`;
    confidence.textContent = `confidence: ${semantic ? semantic.confidence.toFixed(2) : "-"}`;
    if (sample) sample.textContent = stats.lastPacket || "{}";
  }, 250);
}

function formatError(error) {
  if (!error || typeof error !== "object") return "Unable to start the camera.";
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
  avatarClient?.disconnect();
  virtualCamera?.stop();
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

function setLandmarkMode(on) {
  AVATAR_CONFIG.showOverlay = !!on;
  AVATAR_CONFIG.showHandDebug = !!on;
  AVATAR_CONFIG.showFaceDebug = !!on;
  runtimeAvatarConfig.showOverlay = !!on;
  runtimeAvatarConfig.showHandDebug = !!on;
  runtimeAvatarConfig.showFaceDebug = !!on;
  if (landmarkToggle) landmarkToggle.checked = !!on;
}

async function initTrackers() {
  poseTracker = await createPoseTracker({
    video,
    ...POSE_TRACKER_CONFIG
  });

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
      console.warn("Hand tracker unavailable:", handError);
    }
  }

  if (POSE_TRACKER_CONFIG.face) {
    try {
      faceTracker = await createFaceTracker({
        video,
        wasmPath: POSE_TRACKER_CONFIG.wasmPath,
        modelAssetPath: POSE_TRACKER_CONFIG.face.modelAssetPath,
        numFaces: POSE_TRACKER_CONFIG.face.numFaces,
        minFaceDetectionConfidence: POSE_TRACKER_CONFIG.face.minFaceDetectionConfidence,
        minFacePresenceConfidence: POSE_TRACKER_CONFIG.face.minFacePresenceConfidence,
        minTrackingConfidence: POSE_TRACKER_CONFIG.face.minTrackingConfidence,
        targetFps: POSE_TRACKER_CONFIG.face.targetFps,
        minTargetFps: POSE_TRACKER_CONFIG.minTargetFps,
        maxTargetFps: POSE_TRACKER_CONFIG.maxTargetFps
      });
    } catch (faceError) {
      console.warn("Face tracker unavailable:", faceError);
    }
  }
}

async function init() {
  if (!window.isSecureContext) {
    setStage("webcam", "error");
    setStatus("Open this app on localhost or HTTPS to use getUserMedia.", "error");
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    setStage("webcam", "error");
    setStatus("getUserMedia is not available in this browser.", "error");
    return;
  }

  setLandmarkMode(false);
  landmarkToggle?.addEventListener("change", () => setLandmarkMode(landmarkToggle.checked));
  attachAvatarUploadPanel();
  attachSemanticHud(() => lastSemantic);
  setSemanticWsState("no-avatar");
  window.__semantic = {
    getLatest: () => lastSemantic,
    getState: () => lastUnifiedState,
    getStats: () => avatarClient?.getStats?.()
  };

  if (PUPPET_CONFIG.enabled) {
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
  setStage("webcam", "working");

  try {
    stream = await startCamera(video, CAMERA_CONSTRAINTS);
    setStage("webcam", "ready");

    if (THREE_CONFIG.enabled && !PUPPET_CONFIG.enabled) {
      activeCanvas = threeCanvas ?? canvas;
      if (activeCanvas !== canvas) {
        canvas.hidden = true;
        activeCanvas.hidden = false;
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
      virtualCamera = createVirtualCamera({
        canvas: activeCanvas,
        targetFps: STREAM_CONFIG.targetFps,
        onStatusChange: (s) => setStage("virtualCamera", s === "ready" ? "ready" : s === "unsupported" ? "error" : "working")
      });
      virtualCamera.start();
      setStage("landmarks", "ready", "Landmarks bypassed");
      setStage("motion", "ready", "Semantics bypassed");
      setStage("renderer", "ready");
      setMetrics({ scale: 1 });
      return;
    }

    setStatus("Loading local MediaPipe trackers...", "working");
    setStage("landmarks", "working");

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

    await initTrackers();
    setStage("landmarks", "ready");
    setStage("motion", "working", "Semantics");

    const stateBuilder = createStateBuilder();
    const sharedHeadPoseEstimator = createHeadPoseEstimator({
      smoothing: AVATAR_CONFIG.headPoseSmoothing
    });
    const semanticExtractor = createSemanticParamExtractor({
      smoothingAlpha: SEMANTIC_STREAM_CONFIG.smoothingAlpha,
      deadzone: SEMANTIC_STREAM_CONFIG.deadzone
    });

    runtimeAvatarConfig = {
      ...AVATAR_CONFIG,
      enabled: AVATAR_CONFIG.enabled !== false && !PUPPET_CONFIG.enabled,
      puppet: PUPPET_CONFIG.enabled
        ? {
            ...PUPPET_CONFIG,
            source: puppetImage
          }
        : null
    };

    renderer = createRenderer({
      video,
      canvas,
      ...RENDER_CONFIG,
      beforeRender(now) {
        poseTracker?.processFrame(now);
        handTracker?.processFrame(now);
        faceTracker?.processFrame(now);

        if (handTracker) poseTracker?.setHands(handTracker.getLatestHands());
        if (faceTracker) poseTracker?.setFaces(faceTracker.getLatestFaces());

        const overlay = poseTracker?.getLatestOverlay();
        setTrackingState(overlay, handTracker?.getLatestHands?.() ?? []);

        if (overlay) {
          const headPose = sharedHeadPoseEstimator.update({
            face: overlay.faces?.[0]?.landmarks,
            poseLandmarks: overlay.landmarks
          });
          const state = stateBuilder.build(overlay, headPose);
          const semantic = semanticExtractor.update(state);
          state.semantic = semantic;
          lastUnifiedState = state;
          lastSemantic = semantic;
          avatarClient?.sendSemantic(semantic);
          syncRemoteAvatarHead();

          if (state.skeleton?.joints?.neck) setStage("motion", "ready", "Semantics");
        }
      },
      getFrame() {
        const overlay = poseTracker?.getLatestOverlay();
        const last = stateBuilder.getLast?.();
        return {
          overlay,
          frameId: last?.frameId ?? null,
          timestamp: last?.timestamp ?? null,
          semantic: last?.semantic ?? null
        };
      },
      poseConfig: POSE_RENDER_CONFIG,
      avatarConfig: runtimeAvatarConfig,
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

        setStatus("Semantic LivePortrait avatar preview", "ready");
      }
    });

    renderer.start();
    virtualCamera = createVirtualCamera({
      canvas,
      targetFps: STREAM_CONFIG.targetFps,
      onStatusChange: (s) => setStage("virtualCamera", s === "ready" ? "ready" : s === "unsupported" ? "error" : "working")
    });
    virtualCamera.start();
    setStage("renderer", "ready");
    setMetrics({ scale: RENDER_CONFIG.initialScale });
    setStatus("Semantic LivePortrait avatar preview", "ready");
  } catch (error) {
    console.error(error);
    cleanup();
    if (PUPPET_CONFIG.enabled && puppetImage) {
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
    setStage("landmarks", "error");
  }
}

window.addEventListener("beforeunload", cleanup);

document.addEventListener("visibilitychange", () => {
  if (!renderer && !threePreview) return;

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
