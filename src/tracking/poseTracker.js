/**
 * Pose tracking module.
 *
 * Wraps MediaPipe `PoseLandmarker`. Hand and face tracking each live in their
 * own modules — see `./handTracker.js` and `./faceTracker.js`. Their results
 * are injected into the exposed overlay via `setHands(hands)` / `setFaces(faces)`
 * so consumers (renderer, avatar) keep treating the overlay as a single bag
 * of pose + hands + face.
 *
 * Pure tracking — no rendering.
 */

import { FilesetResolver, PoseLandmarker } from "../../node_modules/@mediapipe/tasks-vision/vision_bundle.mjs";
import { buildSkeletonRig } from "../body/skeleton.js";

const SMOOTHING_PREV_WEIGHT = 0.55;
const SMOOTHING_CURR_WEIGHT = 0.45;

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
          : b.z,
      visibility:
        typeof b.visibility === "number" && typeof a.visibility === "number"
          ? a.visibility * SMOOTHING_PREV_WEIGHT + b.visibility * SMOOTHING_CURR_WEIGHT
          : b.visibility
    };
  }
  return out;
}

function toConnectionPairs(connections) {
  return connections.map(({ start, end }) => [start, end]);
}

function createEmptyOverlay(connections) {
  return {
    connections,
    landmarks: [],
    hands: [],
    faces: [],
    skeleton: { joints: {}, bones: {}, bonesList: [] },
    timestamp: 0
  };
}

async function createPoseLandmarker(vision, modelAssetPath, options) {
  try {
    return await PoseLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { delegate: "GPU", modelAssetPath }
    });
  } catch {
    try {
      return await PoseLandmarker.createFromOptions(vision, {
        ...options,
        baseOptions: { delegate: "CPU", modelAssetPath }
      });
    } catch {
      throw new Error("Unable to initialize MediaPipe pose tracking.");
    }
  }
}

export async function createPoseTracker({
  video,
  wasmPath,
  modelAssetPath,
  targetFps,
  minTargetFps,
  maxTargetFps,
  downshiftStep,
  upshiftStep,
  maxPoses,
  minPoseDetectionConfidence,
  minPosePresenceConfidence,
  minTrackingConfidence
}) {
  const vision = await FilesetResolver.forVisionTasks(wasmPath);
  const poseLandmarker = await createPoseLandmarker(vision, modelAssetPath, {
    minPoseDetectionConfidence,
    minPosePresenceConfidence,
    minTrackingConfidence,
    numPoses: maxPoses,
    outputSegmentationMasks: false,
    runningMode: "VIDEO"
  });

  const connections = toConnectionPairs(PoseLandmarker.POSE_CONNECTIONS);

  const initialTargetFps = clamp(targetFps, minTargetFps, maxTargetFps);
  let lastTimestamp = -Infinity;
  let lastVideoTime = -1;
  let currentTargetFps = initialTargetFps;
  let minimumInterval = 1000 / Math.max(initialTargetFps, 1);
  let closed = false;
  let latestOverlay = createEmptyOverlay(connections);
  let injectedHands = [];
  let injectedFaces = [];
  let previousPoseLandmarks = null;

  function setTargetFps(nextTargetFps) {
    currentTargetFps = clamp(nextTargetFps, minTargetFps, maxTargetFps);
    minimumInterval = 1000 / Math.max(currentTargetFps, 1);
  }

  function rebuildOverlay(timestamp, smoothedPoseLandmarks) {
    latestOverlay = {
      connections,
      landmarks: smoothedPoseLandmarks,
      hands: injectedHands,
      faces: injectedFaces,
      skeleton: buildSkeletonRig(smoothedPoseLandmarks),
      timestamp
    };
  }

  function processFrame(timestamp) {
    if (closed || !video.videoWidth || !video.videoHeight) return;
    if (video.currentTime === lastVideoTime) return;
    if (timestamp - lastTimestamp < minimumInterval) return;

    lastTimestamp = timestamp;
    lastVideoTime = video.currentTime;

    const poseResult = poseLandmarker.detectForVideo(video, timestamp);

    const rawPoseLandmarks = poseResult.landmarks?.[0] ?? [];
    const smoothedPoseLandmarks = smoothLandmarks(previousPoseLandmarks, rawPoseLandmarks);
    previousPoseLandmarks = smoothedPoseLandmarks;

    rebuildOverlay(timestamp, smoothedPoseLandmarks);
  }

  return {
    getLatestOverlay() {
      return latestOverlay;
    },
    processFrame,
    /**
     * Hand tracking lives in a separate module; the renderer / avatar still
     * read hands from the overlay, so hand results from `handTracker.js` are
     * injected here once per frame.
     */
    setHands(hands) {
      injectedHands = Array.isArray(hands) ? hands : [];
      latestOverlay = { ...latestOverlay, hands: injectedHands };
    },
    /**
     * Face tracking lives in a separate module; faces from `faceTracker.js`
     * are injected here once per frame.
     */
    setFaces(faces) {
      injectedFaces = Array.isArray(faces) ? faces : [];
      latestOverlay = { ...latestOverlay, faces: injectedFaces };
    },
    relievePressure() {
      setTargetFps(currentTargetFps - downshiftStep);
    },
    restoreRate() {
      setTargetFps(currentTargetFps + upshiftStep);
    },
    reset() {
      lastTimestamp = -Infinity;
      lastVideoTime = -1;
      setTargetFps(initialTargetFps);
      latestOverlay = createEmptyOverlay(connections);
      injectedHands = [];
      injectedFaces = [];
      previousPoseLandmarks = null;
    },
    close() {
      if (closed) return;
      closed = true;
      latestOverlay = createEmptyOverlay(connections);
      poseLandmarker.close();
    }
  };
}
