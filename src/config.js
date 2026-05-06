const lowEndCpu = typeof navigator.hardwareConcurrency === "number" && navigator.hardwareConcurrency <= 4;

const poseModelPath =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const handModelPath =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const faceModelPath =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

export const CAMERA_CONSTRAINTS = {
  audio: false,
  video: {
    facingMode: "user",
    width: { ideal: 640 },
    height: { ideal: 360 },
    frameRate: { ideal: 30, max: 30 }
  }
};

export const RENDER_CONFIG = {
  targetFps: 24,
  renderFps: 60,
  minimumFps: 15,
  initialScale: lowEndCpu ? 0.75 : 1,
  minScale: 0.35,
  maxScale: 1,
  downscaleStep: 0.15,
  upscaleStep: 0.05,
  upscaleThresholdFps: 20,
  maxCanvasWidth: lowEndCpu ? 720 : 1280
};

export const POSE_TRACKER_CONFIG = {
  downshiftStep: 2,
  maxPoses: 1,
  maxTargetFps: lowEndCpu ? 24 : 30,
  minTargetFps: 10,
  modelAssetPath: poseModelPath,
  recoveryFps: 24,
  targetFps: lowEndCpu ? 18 : 30,
  upshiftStep: 1,
  wasmPath: "/node_modules/@mediapipe/tasks-vision/wasm",
  minPoseDetectionConfidence: 0.5,
  minPosePresenceConfidence: 0.5,
  minTrackingConfidence: 0.5,
  hand: {
    modelAssetPath: handModelPath,
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    // Run the hand tracker slightly slower than pose to keep the FPS floor
    // safe on lower-end machines. Pose runs at `targetFps` above.
    targetFps: lowEndCpu ? 12 : 15
  },
  face: null
};

export const POSE_RENDER_CONFIG = {
  connectionColor: "rgba(64, 255, 204, 0.8)",
  connectionWidth: lowEndCpu ? 2 : 3,
  keypointColor: "rgba(255, 255, 255, 0.95)",
  keypointRadius: lowEndCpu ? 3.5 : 4.5,
  minVisibility: 0.45,
  hand: {
    connectionColor: "rgba(255, 196, 64, 0.85)",
    connectionWidth: lowEndCpu ? 1.5 : 2,
    keypointColor: "rgba(255, 255, 255, 0.95)",
    keypointRadius: lowEndCpu ? 2.5 : 3,
    minVisibility: 0
  },
  face: {
    connectionColor: "rgba(170, 230, 255, 0.65)",
    connectionWidth: 1,
    keypointColor: "rgba(170, 230, 255, 0.55)",
    keypointRadius: 0.9,
    minVisibility: 0,
    drawAllPoints: false
  }
};

export const AVATAR_CONFIG = {
  enabled: true,
  showOverlay: false, // when avatar is on, hide the raw landmark overlay for a cleaner look
  // Hand-tracking debug overlay — draws the 21 raw hand landmarks + bone
  // connections on top of the avatar. Toggle on to verify hand tracking.
  showHandDebug: true,
  // Face-mesh debug overlay — draws the 468 face-mesh points + connections
  // on top of the avatar. Toggle on to verify face tracking.
  showFaceDebug: false,
  scaleSmoothing: 0.18,
  headPoseSmoothing: 0.25,
  // Head source descriptor. Swap this at runtime to plug in an AI face.
  //
  //   { type: "default" }                        — drawn humanoid head (current)
  //   { type: "image", source: HTMLImageElement, scale, aspect, verticalOffset, clipOval, ignoreHeadPose }
  //   { type: "video", source: HTMLVideoElement, ... }
  //   { type: "canvas", source: HTMLCanvasElement | OffscreenCanvas, ... }
  //
  // The avatar maintains body alignment automatically: the head sprite is
  // centered on the rig's neck-to-head vector, scaled by the body-derived
  // head radius, and rotated to follow spine + detected head roll.
  head: { type: "default" },
  // Optional: when set, main.js loads this image at startup and pushes it
  // into the avatar's head slot via setHeadSource. Useful for testing the
  // sprite head pipeline without running the full AI face service.
  testHeadImage: null,
  style: {
    outlineWidth: lowEndCpu ? 2 : 2.5,
    accentWidth: lowEndCpu ? 1.2 : 1.6,
    minBoneConfidence: 0.15
  },
  palette: {
    // AI-style cool suit with cyan accents.
    bodyBase: "#2a3550",
    bodyHighlight: "#566285",
    bodyShadow: "#161c2c",
    accent: "#5ce6ff",
    accentGlow: "rgba(92, 230, 255, 0.55)",
    skinBase: "#e8c19a",
    skinHighlight: "#f5d8b6",
    skinShadow: "#a87a55",
    hairBase: "#1b1d24",
    hairHighlight: "#3a3d48",
    eyeWhite: "#f8f3e7",
    iris: "#3070ff",
    pupil: "#0c0f1c",
    mouth: "#7a3245",
    brow: "#15171f",
    outline: "rgba(8, 10, 18, 0.9)"
  }
};

// AI face integration (LivePortrait Python sidecar). When `enabled` is true,
// the avatar's head slot is swapped to the AI head renderer once the WS
// becomes ready, and reverts to the local default head atomically on
// disconnect / stale-frame timeout.
export const AI_FACE_CONFIG = {
  enabled: true,
  url: "wss://osts5obpvv4kv5-8765.proxy.runpod.net/ws",
  connectTimeoutMs: 3000,
  reconnectDelaysMs: [1000, 2000, 5000],
  targetFps: lowEndCpu ? 1.5 : 2,
  jpegQuality: 0.45,
  driverSize: 160,
  outputSize: 160,
  ultraRealtime: false,
  ultraDriverSize: 128,
  ultraOutputSize: 128,
  ultraJpegQuality: 0.4,
  latencyCompensation: true,
  predictionMs: 90,
  maxPredictionMs: 140,
  motionSmoothingAlpha: 0.55,
  velocitySmoothingAlpha: 0.35,
  textureBlendMs: 90,
  adaptiveSend: true,
  posePositionThreshold: 0.018,
  poseRotationThreshold: 0.07,
  poseExpressionThreshold: 0.12,
  maxPoseSilenceMs: 2200,
  compactPose: true,
  compactPoseFps: 12,
  adaptiveQuality: true,
  highRttMs: 500,
  stableRttMs: 320,
  qualityDownshiftFrames: 2,
  qualityRestoreFrames: 8,
  staleAfterMs: 45000,
  maxRealtimeLatencyMs: 700,
  slowPreviewLatencyMs: 2000,
  slowPreviewIntervalMs: 120000,
  // Sync buffer keeps the body lagged so it lines up with the AI face,
  // which is delayed by network + GPU inference (~80–120 ms is typical).
  // The buffer measures real round-trip latency from frame-id pin events
  // and continuously eases `currentDelayMs` toward it (clamped to the
  // [bufferMinDelayMs, bufferMaxDelayMs] band) for drift correction.
  bufferDelayMs: 100,
  bufferMinDelayMs: 80,
  bufferMaxDelayMs: 120,
  bufferLatencyAlpha: 0.2,
  bufferDriftCorrectionPerSec: 50,
  edgeFeatherPx: 5,
  edgeFeatherFrac: 0.10,
  scaleBoost: 1.18,
  ovalAspectY: 1.18,
  // Color matching — applied to the AI portrait so it blends with the
  // avatar palette. Smoothed across frames for anti-flicker.
  colorCorrect: true,
  brightness: 1.00,
  contrast: 1.05,
  saturation: 0.90,
  // Default tone tint matches AVATAR_CONFIG.palette.skinBase (#e8c19a).
  toneTint: "rgba(232, 193, 154, 1.0)",
  toneStrength: 0.18,
  chinShadowStrength: 0.22,
  rimShadowStrength: 0.18,
  colorSmoothingAlpha: 0.08,
  // Alignment & stability — see src/face/aiHeadRenderer.js for the full
  // pipeline. `smoothingAlpha` is the per-frame blend (cur*alpha +
  // prev*(1-alpha)) applied to anchor position, scale, rotation, yaw
  // and pitch. `offsetX/Y` are head-local calibration nudges in pixels.
  smoothingAlpha: 0.15,
  offsetX: 0,
  offsetY: 0,
  yawWeight: 0.6,
  pitchWeight: 0.6,
  // When true, draws anchor point + bounding box + alignment cross over
  // the AI face for visual debugging of drift/scale jitter. Can also be
  // toggled at runtime via `window.__aiHead.setDebug(true)`.
  debug: false
};

// Outbound WebRTC streaming of the rendered avatar canvas. Disabled by
// default — set `enabled: true` and provide a WHIP-style signaling
// `endpoint` to start publishing.
export const STREAM_CONFIG = {
  enabled: false,
  endpoint: "",
  targetFps: 30,
  bitrateKbps: 2500,
  preferCodec: "VP9"
};

export const THREE_CONFIG = {
  enabled: false,
  mode: "image",
  modelUrl: "/assets/model.glb",
  imageUrl: "/assets/person.jpg"
};

export const PUPPET_CONFIG = {
  enabled: false,
  imageUrl: "/assets/person.jpg",
  smoothing: 0.2,
  maxStepFrac: 0.25
};
