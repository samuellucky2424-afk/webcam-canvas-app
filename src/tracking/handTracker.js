/**
 * Hand tracking module.
 *
 * Wraps a MediaPipe `HandLandmarker` so hand detection lives separately from
 * pose / face tracking. The tracker is reused across frames (RUNNING_MODE
 * "VIDEO"), produces 21 landmarks per hand for up to two hands, and
 * exponentially smooths each hand keyed by handedness so that swapping
 * between left/right doesn't cross-blend.
 *
 * Designed to run alongside pose tracking without blocking it: callers drive
 * `processFrame(timestamp)` from the same animation loop, and a frame is
 * throttled internally so calls faster than `targetFps` early-return.
 *
 * Pure tracking — no rendering.
 */

import { FilesetResolver, HandLandmarker } from "../../node_modules/@mediapipe/tasks-vision/vision_bundle.mjs";

const SMOOTHING_PREV_WEIGHT = 0.5;
const SMOOTHING_CURR_WEIGHT = 0.5;

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
    return await HandLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: "GPU" }
    });
  } catch {
    return await HandLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: "CPU" }
    });
  }
}

/**
 * Creates a hand tracker.
 *
 * @param {Object} opts
 * @param {HTMLVideoElement} opts.video                 - Frame source.
 * @param {string} opts.wasmPath                        - MediaPipe WASM dir.
 * @param {string} opts.modelAssetPath                  - Hand model URL.
 * @param {number} [opts.numHands=2]
 * @param {number} [opts.minHandDetectionConfidence=0.5]
 * @param {number} [opts.minHandPresenceConfidence=0.5]
 * @param {number} [opts.minTrackingConfidence=0.5]
 * @param {number} [opts.targetFps=10]                  - Throttle ceiling.
 *
 * @returns {{
 *   processFrame: (timestamp:number) => void,
 *   getLatestHands: () => Array<{landmarks:Array, connections:Array<[number,number]>, handedness:string}>,
 *   getConnections: () => Array<[number,number]>,
 *   reset: () => void,
 *   close: () => void
 * }}
 */
export async function createHandTracker({
  video,
  wasmPath,
  modelAssetPath,
  numHands = 2,
  minHandDetectionConfidence = 0.5,
  minHandPresenceConfidence = 0.5,
  minTrackingConfidence = 0.5,
  targetFps = 10
}) {
  const vision = await FilesetResolver.forVisionTasks(wasmPath);
  const landmarker = await createLandmarker(vision, {
    numHands,
    minHandDetectionConfidence,
    minHandPresenceConfidence,
    minTrackingConfidence,
    runningMode: "VIDEO",
    baseOptions: { modelAssetPath }
  });

  const connections = toConnectionPairs(HandLandmarker.HAND_CONNECTIONS);
  const minimumInterval = 1000 / Math.max(targetFps, 1);

  let lastTimestamp = -Infinity;
  let lastVideoTime = -1;
  let closed = false;
  let latestHands = [];
  let previousByLabel = new Map();

  function processFrame(timestamp) {
    if (closed || !video.videoWidth || !video.videoHeight) return;
    if (video.currentTime === lastVideoTime) return;
    if (timestamp - lastTimestamp < minimumInterval) return;

    lastTimestamp = timestamp;
    lastVideoTime = video.currentTime;

    const result = landmarker.detectForVideo(video, timestamp);
    const raw = result?.landmarks ?? [];
    const handedness = result?.handedness ?? result?.handednesses ?? [];

    const nextHistory = new Map();
    latestHands = raw.map((landmarks, index) => {
      const label = handedness[index]?.[0]?.categoryName ?? `hand-${index}`;
      const previous = previousByLabel.get(label);
      const smoothed = smoothLandmarks(previous, landmarks);
      nextHistory.set(label, smoothed);
      return { connections, landmarks: smoothed, handedness: label };
    });
    previousByLabel = nextHistory;
  }

  return {
    processFrame,
    getLatestHands() {
      return latestHands;
    },
    getConnections() {
      return connections;
    },
    reset() {
      lastTimestamp = -Infinity;
      lastVideoTime = -1;
      latestHands = [];
      previousByLabel = new Map();
    },
    close() {
      if (closed) return;
      closed = true;
      latestHands = [];
      landmarker.close();
    }
  };
}

// Suppress unused-import noise for `clamp` if tree-shaking is ever added.
export const __noop = clamp;
