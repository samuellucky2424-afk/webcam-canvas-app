/**
 * Head pose estimation.
 *
 * Derives `{ roll, yaw, pitch, eyeOpen, mouthOpen, confidence }` from the
 * most reliable source available:
 *   1. The 478-point face mesh (preferred).
 *   2. The pose model's eye / ear / nose landmarks (fallback — no eyelid
 *      or lip data).
 *
 * Output is clamped to natural human ranges and exponentially smoothed.
 */

// Pose-model head landmarks.
const POSE_NOSE = 0;
const POSE_LEFT_EYE = 2;
const POSE_RIGHT_EYE = 5;
const POSE_LEFT_EAR = 7;
const POSE_RIGHT_EAR = 8;

// Face-mesh landmarks (478-point model).
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

function distance(a, b) {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function visibilityOf(landmark) {
  return typeof landmark?.visibility === "number" ? landmark.visibility : 1;
}

function eyeOpenness(top, bottom, inner, outer) {
  if (!top || !bottom || !inner || !outer) return 1;
  const h = distance(top, bottom);
  const w = Math.max(distance(inner, outer), 1e-3);
  const ratio = h / w; // EAR-like: ~0.30 fully open, ~0.10 closed
  // Hysteresis thresholds — below `closed` snap shut, above `open` count
  // as fully open, in between remap linearly. The gap between the two
  // thresholds prevents the blink classifier from flickering when the
  // ratio dithers on the boundary.
  const closedThreshold = 0.16;
  const openThreshold = 0.28;
  if (ratio <= closedThreshold) return 0;
  if (ratio >= openThreshold) return 1;
  return (ratio - closedThreshold) / (openThreshold - closedThreshold);
}

function poseFromFace(faceLandmarks) {
  if (!faceLandmarks || faceLandmarks.length < 264) {
    return null;
  }

  const leftEye = faceLandmarks[FACE_LEFT_EYE_OUTER];
  const rightEye = faceLandmarks[FACE_RIGHT_EYE_OUTER];
  const noseTip = faceLandmarks[FACE_NOSE_TIP];
  const forehead = faceLandmarks[FACE_FOREHEAD];
  const chin = faceLandmarks[FACE_CHIN];

  if (!leftEye || !rightEye || !noseTip || !forehead || !chin) {
    return null;
  }

  const eyeDx = rightEye.x - leftEye.x;
  const eyeDy = rightEye.y - leftEye.y;
  // Roll: angle of the inter-ocular line. atan2 handles the sign so a
  // clockwise tilt from the camera's view comes out positive.
  const roll = Math.atan2(eyeDy, eyeDx);

  const eyeMid = midpoint(leftEye, rightEye);
  const eyeWidth = Math.max(distance(leftEye, rightEye), 1e-3);

  // Yaw: nose horizontal offset from the eye midpoint, scaled by half the
  // eye width. Looking straight ahead → 0; turning right (camera left) → +.
  const yawRatio = (noseTip.x - eyeMid.x) / (eyeWidth * 0.5);
  const yaw = Math.atan(yawRatio * 0.6);

  // Pitch: nose vertical offset from the eye midpoint, normalized by eye
  // width (which is roughly invariant to pitch — face height shrinks as
  // the head tilts forward, so it's a worse divisor). The 0.55 baseline
  // is the typical ratio of (nose - eye-mid)/eye-width when looking
  // straight ahead, so subtracting it centers neutral at 0.
  const pitchRatio = (noseTip.y - eyeMid.y) / eyeWidth - 0.55;
  const pitch = Math.atan(pitchRatio * 1.0);

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
  const eyeOpen = (leftOpen + rightOpen) / 2;

  let mouthOpen = 0;
  let smile = 0;
  const mu = faceLandmarks[FACE_MOUTH_UPPER];
  const ml = faceLandmarks[FACE_MOUTH_LOWER];
  const mleft = faceLandmarks[FACE_MOUTH_LEFT];
  const mright = faceLandmarks[FACE_MOUTH_RIGHT];
  if (mu && ml && mleft && mright) {
    const gap = distance(mu, ml);
    const width = Math.max(distance(mleft, mright), 1e-3);
    const ratio = gap / width;
    mouthOpen = Math.min(Math.max((ratio - 0.04) / 0.5, 0), 1);

    // Smile: mouth width relative to inter-ocular width. Neutral mouth is
    // ~0.45 × eye-width; a wide smile reaches ~0.65+. Normalize that
    // window to [0, 1] and clamp.
    const widthRatio = width / eyeWidth;
    smile = Math.min(Math.max((widthRatio - 0.45) / 0.20, 0), 1);
  }

  return { roll, yaw, pitch, eyeOpen, mouthOpen, smile, confidence: 1 };
}

function poseFromPose(poseLandmarks) {
  if (!poseLandmarks?.length) {
    return null;
  }

  const nose = poseLandmarks[POSE_NOSE];
  const leftEye = poseLandmarks[POSE_LEFT_EYE];
  const rightEye = poseLandmarks[POSE_RIGHT_EYE];
  const leftEar = poseLandmarks[POSE_LEFT_EAR];
  const rightEar = poseLandmarks[POSE_RIGHT_EAR];

  if (!nose || !leftEye || !rightEye) {
    return null;
  }

  const confidence = Math.min(
    visibilityOf(nose),
    visibilityOf(leftEye),
    visibilityOf(rightEye)
  );

  if (confidence < 0.4) {
    return null;
  }

  const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);
  const eyeMid = midpoint(leftEye, rightEye);
  const eyeWidth = Math.max(distance(leftEye, rightEye), 1e-3);

  let yaw = 0;
  if (leftEar && rightEar && visibilityOf(leftEar) > 0.3 && visibilityOf(rightEar) > 0.3) {
    const earMid = midpoint(leftEar, rightEar);
    const earWidth = Math.max(distance(leftEar, rightEar), 1e-3);
    const yawRatio = (nose.x - earMid.x) / (earWidth * 0.5);
    yaw = Math.atan(yawRatio * 0.5);
  } else {
    const yawRatio = (nose.x - eyeMid.x) / (eyeWidth * 0.5);
    yaw = Math.atan(yawRatio * 0.6);
  }

  const pitchRatio = (nose.y - eyeMid.y) / (eyeWidth * 1.5);
  const pitch = Math.atan(pitchRatio * 1.2 - 0.3);

  return { roll, yaw, pitch, eyeOpen: 1, mouthOpen: 0, smile: 0, confidence };
}

export function createHeadPoseEstimator({ smoothing = DEFAULT_SMOOTHING } = {}) {
  let state = null;
  // Frames since the face/pose source last produced a usable reading. Used
  // to decide whether to hold the last known orientation (short dropouts —
  // e.g. a single missed face-mesh frame) or slowly relax toward neutral
  // (long absence — user has left frame).
  let framesSinceSource = 0;
  // Hold the last orientation untouched for this many frames before we
  // start easing toward rest. ~30 frames at 24 FPS ≈ 1.25 s — long enough
  // to ride out brief detection gaps without the head twitching.
  const HOLD_FRAMES = 30;

  function reset() {
    state = null;
    framesSinceSource = 0;
  }

  function update({ face, poseLandmarks } = {}) {
    const raw = poseFromFace(face) ?? poseFromPose(poseLandmarks);

    if (!raw) {
      framesSinceSource += 1;
      if (state) {
        if (framesSinceSource <= HOLD_FRAMES) {
          // Fallback to last known position: keep orientation, only let
          // transient signals (eyes/mouth) decay so a stuck blink doesn't
          // freeze on the avatar's face.
          state = {
            ...state,
            eyeOpen: state.eyeOpen * 0.5 + 0.5,
            mouthOpen: state.mouthOpen * 0.7,
            smile: state.smile * 0.7,
            confidence: state.confidence * 0.9
          };
        } else {
          // Long absence: ease orientation back toward rest so the avatar
          // doesn't get stuck looking sideways forever.
          state = {
            roll: state.roll * 0.9,
            yaw: state.yaw * 0.9,
            pitch: state.pitch * 0.9,
            eyeOpen: state.eyeOpen * 0.5 + 0.5,
            mouthOpen: state.mouthOpen * 0.7,
            smile: state.smile * 0.7,
            confidence: state.confidence * 0.7
          };
        }
      }
      return state;
    }

    framesSinceSource = 0;
    const target = {
      roll: clamp(raw.roll, MAX_ROLL),
      yaw: clamp(raw.yaw, MAX_YAW),
      pitch: clamp(raw.pitch, MAX_PITCH),
      eyeOpen: raw.eyeOpen ?? 1,
      mouthOpen: raw.mouthOpen ?? 0,
      smile: raw.smile ?? 0,
      confidence: raw.confidence
    };

    if (!state) {
      state = target;
      return state;
    }

    const a = smoothing;
    // Asymmetric eye smoothing: closing fast (snap shut on a blink)
    // and opening slower (eyelid lifts smoothly back). Mouth uses a
    // single rate because lip motion isn't perceived as a discrete event.
    const eyeCloseA = 0.85;
    const eyeOpenA = 0.35;
    const eyeA = target.eyeOpen < state.eyeOpen ? eyeCloseA : eyeOpenA;
    const mouthA = 0.45;
    state = {
      roll: state.roll * (1 - a) + target.roll * a,
      yaw: state.yaw * (1 - a) + target.yaw * a,
      pitch: state.pitch * (1 - a) + target.pitch * a,
      eyeOpen: state.eyeOpen * (1 - eyeA) + target.eyeOpen * eyeA,
      mouthOpen: state.mouthOpen * (1 - mouthA) + target.mouthOpen * mouthA,
      smile: state.smile * (1 - mouthA) + target.smile * mouthA,
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
