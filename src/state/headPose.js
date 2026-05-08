/**
 * Head and face semantic estimation.
 *
 * Prefers MediaPipe FaceLandmarker output, then falls back to the pose model's
 * nose/eye/ear landmarks. Values are clamped to human-safe ranges and smoothed
 * for avatar animation and semantic packet streaming.
 */

// Pose-model head landmarks.
const POSE_NOSE = 0;
const POSE_LEFT_EYE = 2;
const POSE_RIGHT_EYE = 5;
const POSE_LEFT_EAR = 7;
const POSE_RIGHT_EAR = 8;

// Face-mesh landmarks.
const FACE_LEFT_EYE_OUTER = 33;
const FACE_RIGHT_EYE_OUTER = 263;
const FACE_NOSE_TIP = 1;
const FACE_FOREHEAD = 10;
const FACE_CHIN = 152;

const FACE_LEFT_EYE_TOP = 159;
const FACE_LEFT_EYE_BOTTOM = 145;
const FACE_LEFT_EYE_INNER = 133;
const FACE_RIGHT_EYE_TOP = 386;
const FACE_RIGHT_EYE_BOTTOM = 374;
const FACE_RIGHT_EYE_INNER = 362;

const FACE_LEFT_BROW = [70, 105, 107];
const FACE_RIGHT_BROW = [300, 334, 336];
const FACE_LEFT_IRIS = [468, 469, 470, 471, 472];
const FACE_RIGHT_IRIS = [473, 474, 475, 476, 477];

const FACE_MOUTH_UPPER = 13;
const FACE_MOUTH_LOWER = 14;
const FACE_MOUTH_LEFT = 61;
const FACE_MOUTH_RIGHT = 291;

const MAX_ROLL = (30 * Math.PI) / 180;
const MAX_YAW = (35 * Math.PI) / 180;
const MAX_PITCH = (25 * Math.PI) / 180;

const DEFAULT_SMOOTHING = 0.25;

function clamp(value, limit) {
  if (value > limit) return limit;
  if (value < -limit) return -limit;
  return value;
}

function clampRange(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.min(Math.max(value, min), max);
}

function clamp01(value) {
  return clampRange(value, 0, 1);
}

function distance(a, b) {
  if (!a || !b) return 0;
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function visibilityOf(landmark) {
  return typeof landmark?.visibility === "number" ? landmark.visibility : 1;
}

function averagePoint(landmarks, indices) {
  let x = 0;
  let y = 0;
  let z = 0;
  let count = 0;
  for (const index of indices) {
    const p = landmarks?.[index];
    if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
    x += p.x;
    y += p.y;
    z += Number.isFinite(p.z) ? p.z : 0;
    count += 1;
  }
  if (!count) return null;
  return { x: x / count, y: y / count, z: z / count };
}

function eyeOpenness(top, bottom, inner, outer) {
  if (!top || !bottom || !inner || !outer) return 1;
  const h = distance(top, bottom);
  const w = Math.max(distance(inner, outer), 1e-3);
  const ratio = h / w;
  const closedThreshold = 0.16;
  const openThreshold = 0.28;
  if (ratio <= closedThreshold) return 0;
  if (ratio >= openThreshold) return 1;
  return clamp01((ratio - closedThreshold) / (openThreshold - closedThreshold));
}

function poseFromFace(faceLandmarks) {
  if (!faceLandmarks || faceLandmarks.length < 264) return null;

  const leftEye = faceLandmarks[FACE_LEFT_EYE_OUTER];
  const rightEye = faceLandmarks[FACE_RIGHT_EYE_OUTER];
  const noseTip = faceLandmarks[FACE_NOSE_TIP];
  const forehead = faceLandmarks[FACE_FOREHEAD];
  const chin = faceLandmarks[FACE_CHIN];

  if (!leftEye || !rightEye || !noseTip || !forehead || !chin) return null;

  const eyeDx = rightEye.x - leftEye.x;
  const eyeDy = rightEye.y - leftEye.y;
  const roll = Math.atan2(eyeDy, eyeDx);

  const eyeMid = midpoint(leftEye, rightEye);
  const eyeWidth = Math.max(distance(leftEye, rightEye), 1e-3);
  const faceHeight = Math.max(distance(forehead, chin), 1e-3);

  const yawRatio = (noseTip.x - eyeMid.x) / (eyeWidth * 0.5);
  const yaw = Math.atan(yawRatio * 0.6);

  const pitchRatio = (noseTip.y - eyeMid.y) / eyeWidth - 0.55;
  const pitch = Math.atan(pitchRatio);

  const leftOpen = eyeOpenness(
    faceLandmarks[FACE_LEFT_EYE_TOP],
    faceLandmarks[FACE_LEFT_EYE_BOTTOM],
    faceLandmarks[FACE_LEFT_EYE_INNER],
    faceLandmarks[FACE_LEFT_EYE_OUTER]
  );
  const rightOpen = eyeOpenness(
    faceLandmarks[FACE_RIGHT_EYE_TOP],
    faceLandmarks[FACE_RIGHT_EYE_BOTTOM],
    faceLandmarks[FACE_RIGHT_EYE_INNER],
    faceLandmarks[FACE_RIGHT_EYE_OUTER]
  );

  let mouthOpen = 0;
  let jawOpen = 0;
  let smile = 0;
  const mu = faceLandmarks[FACE_MOUTH_UPPER];
  const ml = faceLandmarks[FACE_MOUTH_LOWER];
  const mleft = faceLandmarks[FACE_MOUTH_LEFT];
  const mright = faceLandmarks[FACE_MOUTH_RIGHT];
  if (mu && ml && mleft && mright) {
    const gap = distance(mu, ml);
    const width = Math.max(distance(mleft, mright), 1e-3);
    mouthOpen = clamp01((gap / width - 0.04) / 0.5);
    jawOpen = clamp01((gap / faceHeight - 0.018) / 0.11);
    smile = clamp01((width / eyeWidth - 0.45) / 0.2);
  }

  const leftBrow = averagePoint(faceLandmarks, FACE_LEFT_BROW);
  const rightBrow = averagePoint(faceLandmarks, FACE_RIGHT_BROW);
  const brow = leftBrow && rightBrow ? midpoint(leftBrow, rightBrow) : leftBrow ?? rightBrow;
  const browGap = brow ? (eyeMid.y - brow.y) / faceHeight : 0.14;
  const browRaise = clamp01((browGap - 0.13) / 0.09);

  let eyeDirectionX = clampRange(yaw / MAX_YAW, -1, 1);
  let eyeDirectionY = clampRange(pitch / MAX_PITCH, -1, 1);
  const leftIris = averagePoint(faceLandmarks, FACE_LEFT_IRIS);
  const rightIris = averagePoint(faceLandmarks, FACE_RIGHT_IRIS);
  if (leftIris && rightIris) {
    const leftEyeCenter = midpoint(leftEye, faceLandmarks[FACE_LEFT_EYE_INNER]);
    const rightEyeCenter = midpoint(rightEye, faceLandmarks[FACE_RIGHT_EYE_INNER]);
    const leftEyeWidth = Math.max(distance(leftEye, faceLandmarks[FACE_LEFT_EYE_INNER]), 1e-3);
    const rightEyeWidth = Math.max(distance(rightEye, faceLandmarks[FACE_RIGHT_EYE_INNER]), 1e-3);
    const leftEyeHeight = Math.max(distance(faceLandmarks[FACE_LEFT_EYE_TOP], faceLandmarks[FACE_LEFT_EYE_BOTTOM]), 1e-3);
    const rightEyeHeight = Math.max(distance(faceLandmarks[FACE_RIGHT_EYE_TOP], faceLandmarks[FACE_RIGHT_EYE_BOTTOM]), 1e-3);
    const lx = (leftIris.x - leftEyeCenter.x) / leftEyeWidth;
    const rx = (rightIris.x - rightEyeCenter.x) / rightEyeWidth;
    const ly = (leftIris.y - leftEyeCenter.y) / leftEyeHeight;
    const ry = (rightIris.y - rightEyeCenter.y) / rightEyeHeight;
    eyeDirectionX = clampRange(((lx + rx) / 2) * 4.5, -1, 1);
    eyeDirectionY = clampRange(((ly + ry) / 2) * 2.4, -1, 1);
  }

  return {
    roll,
    yaw,
    pitch,
    eyeOpen: (leftOpen + rightOpen) / 2,
    leftEyeOpen: leftOpen,
    rightEyeOpen: rightOpen,
    mouthOpen,
    jawOpen,
    smile,
    browRaise,
    eyeDirectionX,
    eyeDirectionY,
    confidence: 1
  };
}

function poseFromPose(poseLandmarks) {
  if (!poseLandmarks?.length) return null;

  const nose = poseLandmarks[POSE_NOSE];
  const leftEye = poseLandmarks[POSE_LEFT_EYE];
  const rightEye = poseLandmarks[POSE_RIGHT_EYE];
  const leftEar = poseLandmarks[POSE_LEFT_EAR];
  const rightEar = poseLandmarks[POSE_RIGHT_EAR];

  if (!nose || !leftEye || !rightEye) return null;

  const confidence = Math.min(visibilityOf(nose), visibilityOf(leftEye), visibilityOf(rightEye));
  if (confidence < 0.4) return null;

  const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);
  const eyeMid = midpoint(leftEye, rightEye);
  const eyeWidth = Math.max(distance(leftEye, rightEye), 1e-3);

  let yaw = 0;
  if (leftEar && rightEar && visibilityOf(leftEar) > 0.3 && visibilityOf(rightEar) > 0.3) {
    const earMid = midpoint(leftEar, rightEar);
    const earWidth = Math.max(distance(leftEar, rightEar), 1e-3);
    yaw = Math.atan(((nose.x - earMid.x) / (earWidth * 0.5)) * 0.5);
  } else {
    yaw = Math.atan(((nose.x - eyeMid.x) / (eyeWidth * 0.5)) * 0.6);
  }

  const pitch = Math.atan(((nose.y - eyeMid.y) / (eyeWidth * 1.5)) * 1.2 - 0.3);

  return {
    roll,
    yaw,
    pitch,
    eyeOpen: 1,
    leftEyeOpen: 1,
    rightEyeOpen: 1,
    mouthOpen: 0,
    jawOpen: 0,
    smile: 0,
    browRaise: 0,
    eyeDirectionX: clampRange(yaw / MAX_YAW, -1, 1),
    eyeDirectionY: clampRange(pitch / MAX_PITCH, -1, 1),
    confidence
  };
}

export function createHeadPoseEstimator({ smoothing = DEFAULT_SMOOTHING } = {}) {
  let state = null;
  let framesSinceSource = 0;
  const holdFrames = 30;

  function reset() {
    state = null;
    framesSinceSource = 0;
  }

  function relaxCurrent(shortHold) {
    if (!state) return state;
    const orientationHold = shortHold ? 1 : 0.9;
    const confidenceDecay = shortHold ? 0.9 : 0.7;
    state = {
      roll: state.roll * orientationHold,
      yaw: state.yaw * orientationHold,
      pitch: state.pitch * orientationHold,
      eyeOpen: state.eyeOpen * 0.5 + 0.5,
      leftEyeOpen: state.leftEyeOpen * 0.5 + 0.5,
      rightEyeOpen: state.rightEyeOpen * 0.5 + 0.5,
      mouthOpen: state.mouthOpen * 0.7,
      jawOpen: state.jawOpen * 0.7,
      smile: state.smile * 0.7,
      browRaise: state.browRaise * 0.8,
      eyeDirectionX: state.eyeDirectionX * 0.8,
      eyeDirectionY: state.eyeDirectionY * 0.8,
      confidence: state.confidence * confidenceDecay
    };
    return state;
  }

  function update({ face, poseLandmarks } = {}) {
    const raw = poseFromFace(face) ?? poseFromPose(poseLandmarks);

    if (!raw) {
      framesSinceSource += 1;
      return relaxCurrent(framesSinceSource <= holdFrames);
    }

    framesSinceSource = 0;
    const target = {
      roll: clamp(raw.roll, MAX_ROLL),
      yaw: clamp(raw.yaw, MAX_YAW),
      pitch: clamp(raw.pitch, MAX_PITCH),
      eyeOpen: clamp01(raw.eyeOpen ?? 1),
      leftEyeOpen: clamp01(raw.leftEyeOpen ?? raw.eyeOpen ?? 1),
      rightEyeOpen: clamp01(raw.rightEyeOpen ?? raw.eyeOpen ?? 1),
      mouthOpen: clamp01(raw.mouthOpen ?? 0),
      jawOpen: clamp01(raw.jawOpen ?? raw.mouthOpen ?? 0),
      smile: clamp01(raw.smile ?? 0),
      browRaise: clamp01(raw.browRaise ?? 0),
      eyeDirectionX: clampRange(raw.eyeDirectionX ?? 0, -1, 1),
      eyeDirectionY: clampRange(raw.eyeDirectionY ?? 0, -1, 1),
      confidence: clamp01(raw.confidence ?? 0)
    };

    if (!state) {
      state = target;
      return state;
    }

    const a = clampRange(smoothing, 0.02, 0.95);
    const eyeCloseA = 0.85;
    const eyeOpenA = 0.35;
    const eyeA = target.eyeOpen < state.eyeOpen ? eyeCloseA : eyeOpenA;
    const leftEyeA = target.leftEyeOpen < state.leftEyeOpen ? eyeCloseA : eyeOpenA;
    const rightEyeA = target.rightEyeOpen < state.rightEyeOpen ? eyeCloseA : eyeOpenA;
    const mouthA = 0.45;
    const expressionA = 0.35;

    state = {
      roll: state.roll * (1 - a) + target.roll * a,
      yaw: state.yaw * (1 - a) + target.yaw * a,
      pitch: state.pitch * (1 - a) + target.pitch * a,
      eyeOpen: state.eyeOpen * (1 - eyeA) + target.eyeOpen * eyeA,
      leftEyeOpen: state.leftEyeOpen * (1 - leftEyeA) + target.leftEyeOpen * leftEyeA,
      rightEyeOpen: state.rightEyeOpen * (1 - rightEyeA) + target.rightEyeOpen * rightEyeA,
      mouthOpen: state.mouthOpen * (1 - mouthA) + target.mouthOpen * mouthA,
      jawOpen: state.jawOpen * (1 - mouthA) + target.jawOpen * mouthA,
      smile: state.smile * (1 - mouthA) + target.smile * mouthA,
      browRaise: state.browRaise * (1 - expressionA) + target.browRaise * expressionA,
      eyeDirectionX: state.eyeDirectionX * (1 - expressionA) + target.eyeDirectionX * expressionA,
      eyeDirectionY: state.eyeDirectionY * (1 - expressionA) + target.eyeDirectionY * expressionA,
      confidence: state.confidence * (1 - a) + target.confidence * a
    };
    return state;
  }

  return { update, reset };
}

export const HEAD_POSE_LIMITS = {
  maxRoll: MAX_ROLL,
  maxYaw: MAX_YAW,
  maxPitch: MAX_PITCH
};
