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
    frameRate: { ideal: 30, max: 60 }
  }
};

export const RENDER_CONFIG = {
  targetFps: 30,
  renderFps: 60,
  minimumFps: 15,
  initialScale: lowEndCpu ? 0.75 : 1,
  minScale: 0.35,
  maxScale: 1,
  downscaleStep: 0.15,
  upscaleStep: 0.05,
  upscaleThresholdFps: 24,
  maxCanvasWidth: lowEndCpu ? 720 : 1280
};

export const POSE_TRACKER_CONFIG = {
  downshiftStep: 2,
  maxPoses: 1,
  maxTargetFps: lowEndCpu ? 30 : 60,
  minTargetFps: 10,
  modelAssetPath: poseModelPath,
  recoveryFps: 28,
  targetFps: lowEndCpu ? 24 : 30,
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
    targetFps: lowEndCpu ? 18 : 30
  },
  face: {
    modelAssetPath: faceModelPath,
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    targetFps: lowEndCpu ? 18 : 30
  }
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
  showOverlay: false,
  showHandDebug: false,
  showFaceDebug: false,
  scaleSmoothing: 0.18,
  headPoseSmoothing: 0.25,
  head: { type: "default" },
  testHeadImage: null,
  style: {
    outlineWidth: lowEndCpu ? 2 : 2.5,
    accentWidth: lowEndCpu ? 1.2 : 1.6,
    minBoneConfidence: 0.15
  },
  palette: {
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

export const SEMANTIC_STREAM_CONFIG = {
  enabled: false,
  url: "",
  maxPacketsPerSecond: 24,
  compact: true,
  reconnectDelaysMs: [1000, 2500, 5000],
  maxBufferedBytes: 16000,
  smoothingAlpha: lowEndCpu ? 0.42 : 0.38,
  deadzone: {
    angleDeg: 0.35,
    unit: 0.012,
    point: 0.002
  }
};

export const CLOUD_ENHANCEMENT_CONFIG = {
  enabled: false,
  maxFps: 1
};

export const LIVEPORTRAIT_CONFIG = {
  enabled: true,
  baseUrl: "https://7rv82kmoo4v93t-8765.proxy.runpod.net",
  uploadPath: "/avatar/upload",
  wsPath: "/ws/semantic",
  maxPacketsPerSecond: 24,
  connectAfterUpload: true,
  reconnectDelaysMs: [1000, 2500, 5000],
  maxBufferedBytes: 12000
};

export const STREAM_CONFIG = {
  enabled: false,
  endpoint: "",
  targetFps: 30
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
