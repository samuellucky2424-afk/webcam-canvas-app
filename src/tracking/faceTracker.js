/**
 * Face tracking module.
 *
 * Wraps a MediaPipe `FaceLandmarker` so face mesh detection lives separately
 * from pose / hand tracking. Produces up to 468 landmarks per face (1 face by
 * default) at RUNNING_MODE "VIDEO". Exponential smoothing is applied frame to
 * frame to reduce jitter.
 *
 * Designed to run alongside pose + hand tracking without blocking them: the
 * caller drives `processFrame(timestamp)` from the same animation loop, and a
 * frame is throttled internally so calls faster than `targetFps` early-return.
 *
 * Pure tracking — no rendering.
 */

import { FaceLandmarker, FilesetResolver } from "../../node_modules/@mediapipe/tasks-vision/vision_bundle.mjs";

const SMOOTHING_PREV_WEIGHT = 0.7;
const SMOOTHING_CURR_WEIGHT = 0.3;

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function smoothLandmarks(previous, current) {
  if (!current?.length) return current ?? [];
  if (!previous?.length || previous.length !== current.length) {
    return current.map((p) => ({ ...p }));
  }
  const out = new Array(current.length);
  for (let i = 0; i < current.length; i += 1) {
    const a = previous[i];
    const b = current[i];
    out[i] = {
      ...b,
      x: a.x * SMOOTHING_PREV_WEIGHT + b.x * SMOOTHING_CURR_WEIGHT,
      y: a.y * SMOOTHING_PREV_WEIGHT + b.y * SMOOTHING_CURR_WEIGHT,
      z:
        typeof b.z === "number" && typeof a.z === "number"
          ? a.z * SMOOTHING_PREV_WEIGHT + b.z * SMOOTHING_CURR_WEIGHT
          : b.z
    };
  }
  return out;
}

function toConnectionPairs(connections) {
  return connections.map(({ start, end }) => [start, end]);
}

async function createLandmarker(vision, options) {
  try {
    return await FaceLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: "GPU" }
    });
  } catch {
    return await FaceLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: "CPU" }
    });
  }
}

/**
 * Creates a face tracker.
 *
 * @param {Object} opts
 * @param {HTMLVideoElement} opts.video                  - Frame source.
 * @param {string} opts.wasmPath                         - MediaPipe WASM dir.
 * @param {string} opts.modelAssetPath                   - Face mesh model URL.
 * @param {number} [opts.numFaces=1]
 * @param {number} [opts.minFaceDetectionConfidence=0.5]
 * @param {number} [opts.minFacePresenceConfidence=0.5]
 * @param {number} [opts.minTrackingConfidence=0.5]
 * @param {number} [opts.targetFps=12]                   - Throttle ceiling.
 *
 * @returns {{
 *   processFrame: (timestamp:number) => void,
 *   getLatestFaces: () => Array<{landmarks:Array, connections:Array<[number,number]>}>,
 *   getConnections: () => Array<[number,number]>,
 *   reset: () => void,
 *   close: () => void
 * }}
 */
export async function createFaceTracker({
  video,
  wasmPath,
  modelAssetPath,
  numFaces = 1,
  minFaceDetectionConfidence = 0.5,
  minFacePresenceConfidence = 0.5,
  minTrackingConfidence = 0.5,
  targetFps = 12,
  minTargetFps = 2,
  maxTargetFps = 24,
  downshiftStep = 1,
  upshiftStep = 1
}) {
  const vision = await FilesetResolver.forVisionTasks(wasmPath);
  const landmarker = await createLandmarker(vision, {
    numFaces,
    minFaceDetectionConfidence,
    minFacePresenceConfidence,
    minTrackingConfidence,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
    runningMode: "VIDEO",
    baseOptions: { modelAssetPath }
  });

  // Compose every named connection group into a single pair list for debug
  // rendering. The pose-tracker overlay used the same set.
  const connections = [
    ...toConnectionPairs(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL),
    ...toConnectionPairs(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE),
    ...toConnectionPairs(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE),
    ...toConnectionPairs(FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW),
    ...toConnectionPairs(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW),
    ...toConnectionPairs(FaceLandmarker.FACE_LANDMARKS_LIPS),
    ...toConnectionPairs(FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS),
    ...toConnectionPairs(FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS)
  ];

  const initialTargetFps = clamp(targetFps, minTargetFps, maxTargetFps);
  let currentTargetFps = initialTargetFps;
  let minimumInterval = 1000 / Math.max(initialTargetFps, 1);

  let lastTimestamp = -Infinity;
  let lastVideoTime = -1;
  let closed = false;
  let latestFaces = [];
  let previousByIndex = [];
  let latestUpdateTimestamp = -Infinity;

  function setTargetFps(nextTargetFps) {
    currentTargetFps = clamp(nextTargetFps, minTargetFps, maxTargetFps);
    minimumInterval = 1000 / Math.max(currentTargetFps, 1);
  }

  function processFrame(timestamp) {
    if (closed || !video.videoWidth || !video.videoHeight) return;
    if (video.currentTime === lastVideoTime) return;
    if (timestamp - lastTimestamp < minimumInterval) return;

    lastTimestamp = timestamp;
    lastVideoTime = video.currentTime;

    const result = landmarker.detectForVideo(video, timestamp);
    const raw = result?.faceLandmarks ?? [];

    const nextHistory = [];
    latestFaces = raw.map((landmarks, index) => {
      const previous = previousByIndex[index];
      const smoothed = smoothLandmarks(previous, landmarks);
      nextHistory[index] = smoothed;
      return { connections, landmarks: smoothed };
    });
    previousByIndex = nextHistory;
    latestUpdateTimestamp = timestamp;
  }

  return {
    processFrame,
    getLatestFaces() {
      return latestFaces;
    },
    getLatestTimestamp() {
      return latestUpdateTimestamp;
    },
    setTargetFps,
    relievePressure() {
      setTargetFps(currentTargetFps - downshiftStep);
    },
    restoreRate() {
      setTargetFps(currentTargetFps + upshiftStep);
    },
    getConnections() {
      return connections;
    },
    reset() {
      lastTimestamp = -Infinity;
      lastVideoTime = -1;
      setTargetFps(initialTargetFps);
      latestFaces = [];
      previousByIndex = [];
      latestUpdateTimestamp = -Infinity;
    },
    close() {
      if (closed) return;
      closed = true;
      latestFaces = [];
      landmarker.close();
    }
  };
}
