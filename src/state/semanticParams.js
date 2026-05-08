import { HEAD_POSE_LIMITS } from "./headPose.js";

const RAD_TO_DEG = 180 / Math.PI;
const POSE = {
  nose: 0,
  leftShoulder: 11,
  rightShoulder: 12,
  leftHip: 23,
  rightHip: 24
};

const DEFAULTS = {
  smoothingAlpha: 0.38,
  deadzone: {
    angleDeg: 0.35,
    unit: 0.012,
    point: 0.002
  },
  maxStepPerSecond: {
    angleDeg: 360,
    unit: 8,
    point: 6
  }
};

function clamp(value, min, max) {
  const n = Number(value);
  if (!Number.isFinite(n)) return min;
  return Math.min(Math.max(n, min), max);
}

function clamp01(value) {
  return clamp(value, 0, 1);
}

function round(value, digits = 3) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  const factor = 10 ** digits;
  return Math.round(n * factor) / factor;
}

function angleDelta(from, to) {
  let d = to - from;
  while (d > Math.PI) d -= Math.PI * 2;
  while (d < -Math.PI) d += Math.PI * 2;
  return d;
}

function deg(rad) {
  return rad * RAD_TO_DEG;
}

function confidenceOf(point) {
  return clamp01(point?.visibility ?? 1);
}

function pointFromLandmark(point) {
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) return null;
  return [clamp01(point.x), clamp01(point.y)];
}

function smoothScalar(prev, target, dt, alpha, deadzone, maxStep) {
  if (!Number.isFinite(target)) return Number.isFinite(prev) ? prev : 0;
  if (!Number.isFinite(prev)) return target;
  if (Math.abs(target - prev) < deadzone) return prev;
  const stepped = prev + clamp(target - prev, -maxStep * dt, maxStep * dt);
  return prev * (1 - alpha) + stepped * alpha;
}

function smoothPoint(prev, target, dt, alpha, deadzone, maxStep) {
  if (!target) return Array.isArray(prev) ? prev : null;
  const px = Array.isArray(prev) ? prev[0] : Number.NaN;
  const py = Array.isArray(prev) ? prev[1] : Number.NaN;
  return [
    smoothScalar(px, target[0], dt, alpha, deadzone, maxStep),
    smoothScalar(py, target[1], dt, alpha, deadzone, maxStep)
  ];
}

function bodyConfidence(landmarks) {
  const keys = [POSE.nose, POSE.leftShoulder, POSE.rightShoulder, POSE.leftHip, POSE.rightHip];
  let total = 0;
  let count = 0;
  for (const index of keys) {
    const point = landmarks?.[index];
    if (!point) continue;
    total += confidenceOf(point);
    count += 1;
  }
  return count ? total / count : 0;
}

function bodyAngles(skeleton) {
  const spine = skeleton?.bones?.spine;
  const head = skeleton?.bones?.head;
  const torsoAngle = spine ? deg(angleDelta(Math.PI / 2, spine.angle)) : 0;
  const neckRotation = spine && head ? deg(angleDelta(spine.angle + Math.PI, head.angle)) : 0;
  return {
    torsoAngle: clamp(torsoAngle, -45, 45),
    neckRotation: clamp(neckRotation, -45, 45)
  };
}

function makeRawParams(state) {
  const hp = state?.headPose ?? {};
  const landmarks = state?.overlay?.landmarks ?? [];
  const headPosition = pointFromLandmark(landmarks[POSE.nose]);
  const leftShoulder = pointFromLandmark(landmarks[POSE.leftShoulder]);
  const rightShoulder = pointFromLandmark(landmarks[POSE.rightShoulder]);
  const shoulderCenter = leftShoulder && rightShoulder
    ? [(leftShoulder[0] + rightShoulder[0]) * 0.5, (leftShoulder[1] + rightShoulder[1]) * 0.5]
    : leftShoulder ?? rightShoulder;
  const { torsoAngle, neckRotation } = bodyAngles(state?.skeleton);
  const confidence = Math.max(clamp01(hp.confidence ?? 0), bodyConfidence(landmarks));

  return {
    yaw: clamp(deg(hp.yaw ?? 0), -deg(HEAD_POSE_LIMITS.maxYaw), deg(HEAD_POSE_LIMITS.maxYaw)),
    pitch: clamp(deg(hp.pitch ?? 0), -deg(HEAD_POSE_LIMITS.maxPitch), deg(HEAD_POSE_LIMITS.maxPitch)),
    roll: clamp(deg(hp.roll ?? 0), -deg(HEAD_POSE_LIMITS.maxRoll), deg(HEAD_POSE_LIMITS.maxRoll)),
    blinkLeft: clamp01(1 - (hp.leftEyeOpen ?? hp.eyeOpen ?? 1)),
    blinkRight: clamp01(1 - (hp.rightEyeOpen ?? hp.eyeOpen ?? 1)),
    mouthOpen: clamp01(hp.mouthOpen ?? 0),
    jawOpen: clamp01(hp.jawOpen ?? hp.mouthOpen ?? 0),
    smile: clamp01(hp.smile ?? 0),
    browRaise: clamp01(hp.browRaise ?? 0),
    eyeDirectionX: clamp(hp.eyeDirectionX ?? 0, -1, 1),
    eyeDirectionY: clamp(hp.eyeDirectionY ?? 0, -1, 1),
    headPosition,
    shoulderLeft: leftShoulder,
    shoulderRight: rightShoulder,
    shoulderCenter,
    torsoAngle,
    neckRotation,
    confidence
  };
}

export function createSemanticParamExtractor(options = {}) {
  const config = {
    ...DEFAULTS,
    ...options,
    deadzone: { ...DEFAULTS.deadzone, ...(options.deadzone ?? {}) },
    maxStepPerSecond: { ...DEFAULTS.maxStepPerSecond, ...(options.maxStepPerSecond ?? {}) }
  };

  let last = null;
  let lastTimestamp = 0;

  function smooth(raw, timestamp) {
    const dt = lastTimestamp ? clamp((timestamp - lastTimestamp) / 1000, 1 / 120, 0.12) : 1 / 30;
    const alpha = clamp(config.smoothingAlpha, 0.01, 1);
    const dz = config.deadzone;
    const max = config.maxStepPerSecond;

    const next = {
      yaw: smoothScalar(last?.yaw, raw.yaw, dt, alpha, dz.angleDeg, max.angleDeg),
      pitch: smoothScalar(last?.pitch, raw.pitch, dt, alpha, dz.angleDeg, max.angleDeg),
      roll: smoothScalar(last?.roll, raw.roll, dt, alpha, dz.angleDeg, max.angleDeg),
      blinkLeft: smoothScalar(last?.blinkLeft, raw.blinkLeft, dt, alpha, dz.unit, max.unit),
      blinkRight: smoothScalar(last?.blinkRight, raw.blinkRight, dt, alpha, dz.unit, max.unit),
      mouthOpen: smoothScalar(last?.mouthOpen, raw.mouthOpen, dt, alpha, dz.unit, max.unit),
      jawOpen: smoothScalar(last?.jawOpen, raw.jawOpen, dt, alpha, dz.unit, max.unit),
      smile: smoothScalar(last?.smile, raw.smile, dt, alpha, dz.unit, max.unit),
      browRaise: smoothScalar(last?.browRaise, raw.browRaise, dt, alpha, dz.unit, max.unit),
      eyeDirectionX: smoothScalar(last?.eyeDirectionX, raw.eyeDirectionX, dt, alpha, dz.unit, max.unit),
      eyeDirectionY: smoothScalar(last?.eyeDirectionY, raw.eyeDirectionY, dt, alpha, dz.unit, max.unit),
      headPosition: smoothPoint(last?.headPosition, raw.headPosition, dt, alpha, dz.point, max.point),
      shoulderLeft: smoothPoint(last?.shoulderLeft, raw.shoulderLeft, dt, alpha, dz.point, max.point),
      shoulderRight: smoothPoint(last?.shoulderRight, raw.shoulderRight, dt, alpha, dz.point, max.point),
      shoulderCenter: smoothPoint(last?.shoulderCenter, raw.shoulderCenter, dt, alpha, dz.point, max.point),
      torsoAngle: smoothScalar(last?.torsoAngle, raw.torsoAngle, dt, alpha, dz.angleDeg, max.angleDeg),
      neckRotation: smoothScalar(last?.neckRotation, raw.neckRotation, dt, alpha, dz.angleDeg, max.angleDeg),
      confidence: smoothScalar(last?.confidence, raw.confidence, dt, alpha, dz.unit, max.unit)
    };

    last = next;
    lastTimestamp = timestamp;
    return next;
  }

  function update(state) {
    const timestamp = state?.timestamp ?? performance.now();
    const params = smooth(makeRawParams(state), timestamp);
    return {
      ...params,
      timestamp,
      wallTime: Date.now(),
      hasFace: state?.hasFace === true
    };
  }

  function reset() {
    last = null;
    lastTimestamp = 0;
  }

  return { update, reset };
}

export function createSemanticPacket(params, { compact = true } = {}) {
  if (!params) return null;
  const left = params.shoulderLeft;
  const right = params.shoulderRight;

  if (compact) {
    return {
      t: Math.round(params.wallTime ?? Date.now()),
      y: round(params.yaw, 1),
      p: round(params.pitch, 1),
      r: round(params.roll, 1),
      bl: round(params.blinkLeft, 3),
      br: round(params.blinkRight, 3),
      mo: round(params.mouthOpen, 3),
      sm: round(params.smile, 3),
      jaw: round(params.jawOpen, 3),
      brow: round(params.browRaise, 3),
      ex: round(params.eyeDirectionX, 3),
      ey: round(params.eyeDirectionY, 3),
      sl: left ? [round(left[0], 3), round(left[1], 3)] : null,
      sr: right ? [round(right[0], 3), round(right[1], 3)] : null,
      ta: round(params.torsoAngle, 1),
      nr: round(params.neckRotation, 1),
      c: round(params.confidence, 3)
    };
  }

  return {
    t: Math.round(params.wallTime ?? Date.now()),
    yaw: round(params.yaw, 1),
    pitch: round(params.pitch, 1),
    roll: round(params.roll, 1),
    blinkLeft: round(params.blinkLeft, 3),
    blinkRight: round(params.blinkRight, 3),
    mouthOpen: round(params.mouthOpen, 3),
    smile: round(params.smile, 3),
    jawOpen: round(params.jawOpen, 3),
    browRaise: round(params.browRaise, 3),
    eyeDirection: {
      x: round(params.eyeDirectionX, 3),
      y: round(params.eyeDirectionY, 3)
    },
    shoulders: {
      left: left ? [round(left[0], 3), round(left[1], 3)] : null,
      right: right ? [round(right[0], 3), round(right[1], 3)] : null
    },
    torsoAngle: round(params.torsoAngle, 1),
    neckRotation: round(params.neckRotation, 1),
    poseConfidence: round(params.confidence, 3)
  };
}

export function createAvatarMotionPacket(params) {
  const head = params?.headPosition;
  const shoulder = params?.shoulderCenter;
  return {
    t: Math.round(params?.wallTime ?? Date.now()),
    yaw: round(params?.yaw ?? 0, 1),
    pitch: round(params?.pitch ?? 0, 1),
    roll: round(params?.roll ?? 0, 1),
    blinkLeft: round(params?.blinkLeft ?? 0, 3),
    blinkRight: round(params?.blinkRight ?? 0, 3),
    mouthOpen: round(params?.mouthOpen ?? 0, 3),
    browRaise: round(params?.browRaise ?? 0, 3),
    smile: round(params?.smile ?? 0, 3),
    pupilX: round(params?.eyeDirectionX ?? 0, 3),
    pupilY: round(params?.eyeDirectionY ?? 0, 3),
    headX: round(head?.[0] ?? 0.5, 3),
    headY: round(head?.[1] ?? 0.35, 3),
    shoulderX: round(shoulder?.[0] ?? 0.5, 3),
    shoulderY: round(shoulder?.[1] ?? 0.6, 3),
    confidence: round(params?.confidence ?? 0, 3)
  };
}
