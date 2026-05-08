/**
 * Avatar renderer.
 *
 * Consumes a skeleton rig (see `./skeleton.js`) and draws a stylized humanoid
 * avatar onto a 2D canvas context. The body is composed of filled polygons
 * (torso trapezoid with chest/ab paneling), tapered limb segments, rounded
 * hands and feet, a neck cylinder, and a detailed head with hair, eyes,
 * brows and mouth.
 *
 * Design goals:
 *   - **Anchor at hips.** All limbs are chained outward from the pelvis.
 *   - **Consistent proportions.** Bone lengths are fixed ratios of a smoothed
 *     reference scale (the spine length).
 *   - **Smooth animation.** The rig is already smoothed in the tracker; the
 *     reference scale is additionally low-pass filtered here.
 *   - **Humanoid look without textures.** All visual richness is from layered
 *     shapes, gradients, and a small accent palette — no images required.
 *
 * Pure rendering — no MediaPipe imports.
 */

import { resolveHeadRenderer } from "../face/head.js";

const DEFAULT_PROPORTIONS = {
  head: 0.32,
  shoulderHalfWidth: 0.42,
  hipHalfWidth: 0.24,
  upperArm: 0.55,
  forearm: 0.48,
  thigh: 0.70,
  shin: 0.65,
  headRadius: 0.20,
  neckHalfWidth: 0.09,
  upperArmThickness: 0.13,
  forearmThickness: 0.10,
  thighThickness: 0.16,
  shinThickness: 0.12,
  handRadius: 0.09,
  footLength: 0.18,
  footHeight: 0.07
};

const DEFAULT_PALETTE = {
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
  outline: "rgba(8, 10, 18, 0.85)"
};

const DEFAULT_STYLE = {
  outlineWidth: 2.5,
  accentWidth: 1.6,
  minBoneConfidence: 0.15
};

const DEFAULT_SCALE_SMOOTHING = 0.18;

// --- Geometry helpers -------------------------------------------------------

function polar(origin, angle, length) {
  return {
    x: origin.x + Math.cos(angle) * length,
    y: origin.y + Math.sin(angle) * length
  };
}

function offsetPerp(point, axisAngle, distance) {
  return {
    x: point.x + Math.cos(axisAngle + Math.PI / 2) * distance,
    y: point.y + Math.sin(axisAngle + Math.PI / 2) * distance
  };
}

function directionAngle(from, to, fallback) {
  if (!from || !to) return fallback;
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  if (dx === 0 && dy === 0) return fallback;
  return Math.atan2(dy, dx);
}

function lerpPoint(a, b, t) {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

// --- Drawing primitives -----------------------------------------------------

function strokePath(context, points, color, width, closed = false) {
  if (!points.length) return;
  context.strokeStyle = color;
  context.lineWidth = width;
  context.beginPath();
  context.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i += 1) {
    context.lineTo(points[i].x, points[i].y);
  }
  if (closed) context.closePath();
  context.stroke();
}

function fillPath(context, points, fillStyle) {
  if (!points.length) return;
  context.fillStyle = fillStyle;
  context.beginPath();
  context.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i += 1) {
    context.lineTo(points[i].x, points[i].y);
  }
  context.closePath();
  context.fill();
}

/**
 * Tapered limb segment as a filled trapezoid with outline + accent stripe.
 */
function drawTaperedSegment(context, start, end, startThickness, endThickness, palette, style) {
  if (!start || !end) return;

  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.hypot(dx, dy);
  if (length < 1) return;

  const angle = Math.atan2(dy, dx);
  const halfStart = startThickness / 2;
  const halfEnd = endThickness / 2;

  const s1 = offsetPerp(start, angle, -halfStart);
  const s2 = offsetPerp(start, angle, halfStart);
  const e2 = offsetPerp(end, angle, halfEnd);
  const e1 = offsetPerp(end, angle, -halfEnd);

  const grad = context.createLinearGradient(s1.x, s1.y, s2.x, s2.y);
  grad.addColorStop(0, palette.bodyHighlight);
  grad.addColorStop(0.5, palette.bodyBase);
  grad.addColorStop(1, palette.bodyShadow);

  fillPath(context, [s1, s2, e2, e1], grad);
  strokePath(context, [s1, s2, e2, e1], palette.outline, style.outlineWidth, true);

  // Joint cap at proximal end so successive segments visually merge.
  context.fillStyle = palette.bodyBase;
  context.beginPath();
  context.arc(start.x, start.y, halfStart, 0, Math.PI * 2);
  context.fill();
  context.strokeStyle = palette.outline;
  context.lineWidth = style.outlineWidth;
  context.beginPath();
  context.arc(start.x, start.y, halfStart, 0, Math.PI * 2);
  context.stroke();
}

function drawHand(context, wrist, axisAngle, radius, palette, style) {
  if (!wrist) return;
  const center = polar(wrist, axisAngle, radius * 0.6);

  context.save();
  context.translate(center.x, center.y);
  context.rotate(axisAngle);

  const grad = context.createRadialGradient(-radius * 0.3, -radius * 0.3, radius * 0.1, 0, 0, radius);
  grad.addColorStop(0, palette.skinHighlight);
  grad.addColorStop(1, palette.skinShadow);

  context.fillStyle = grad;
  context.beginPath();
  context.ellipse(0, 0, radius, radius * 0.85, 0, 0, Math.PI * 2);
  context.fill();
  context.strokeStyle = palette.outline;
  context.lineWidth = style.outlineWidth;
  context.stroke();

  context.restore();
}

// Hand landmark indices (MediaPipe Hand Landmarker, 21 points).
// 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky.
const FINGER_CHAINS = [
  [0, 1, 2, 3, 4],
  [0, 5, 6, 7, 8],
  [0, 9, 10, 11, 12],
  [0, 13, 14, 15, 16],
  [0, 17, 18, 19, 20]
];
const PALM_POLYGON = [0, 1, 5, 9, 13, 17];

/**
 * Draws a stylized hand using MediaPipe's 21 hand landmarks (already projected
 * to canvas space). Renders a palm silhouette and five tapered finger chains
 * with an outline pass beneath a skin-coloured top pass.
 *
 * `anchor` (optional): if provided, the hand is translated so landmark[0]
 * (the hand wrist) lands exactly on `anchor`, keeping the fingers attached
 * to the rig wrist even when the two trackers disagree slightly.
 *
 * `wristAngle` (optional, radians): if provided, the hand is rotated around
 * the anchor so the landmark[0] → landmark[9] (wrist → middle-finger MCP)
 * axis lines up with this angle. Used to align the hand with the rig's
 * forearm direction so it doesn't look detached when the pose-tracker wrist
 * and hand-tracker wrist disagree on roll.
 */
function drawDetailedHand(context, landmarks, scale, palette, style, anchor = null, wristAngle = null) {
  if (!landmarks || landmarks.length < 21) return;

  let pts = landmarks;

  // Rotate around landmark[0] so the hand's natural axis matches the rig
  // forearm. We blend the source hand orientation with the target wrist
  // orientation so the rotation feels guided rather than locked — the
  // fingers still curl naturally, but the whole hand swings with the arm.
  if (wristAngle != null && landmarks.length > 9) {
    const w = landmarks[0];
    const m = landmarks[9];
    const sourceAngle = Math.atan2(m.y - w.y, m.x - w.x);
    // Wrap delta into [-PI, PI] to avoid 360° flips.
    let delta = wristAngle - sourceAngle;
    while (delta > Math.PI) delta -= Math.PI * 2;
    while (delta < -Math.PI) delta += Math.PI * 2;
    // 75% align with rig wrist, 25% keep hand-tracker rotation. Tune for
    // taste — full lock (1.0) feels too rigid, no lock (0) lets the hand
    // drift away visually.
    const blend = 0.75;
    const applied = delta * blend;
    const cos = Math.cos(applied);
    const sin = Math.sin(applied);
    pts = landmarks.map((p) => {
      const dx = p.x - w.x;
      const dy = p.y - w.y;
      return { x: w.x + dx * cos - dy * sin, y: w.y + dx * sin + dy * cos };
    });
  }

  if (anchor) {
    const dx = anchor.x - pts[0].x;
    const dy = anchor.y - pts[0].y;
    pts = pts.map((p) => ({ x: p.x + dx, y: p.y + dy }));
  }

  const fingerThickness = Math.max(scale * 0.045, 4);
  const tipThickness = Math.max(scale * 0.028, 3);
  const outlineExtra = Math.max(style.outlineWidth * 1.1, 1.5);

  context.save();
  context.lineCap = "round";
  context.lineJoin = "round";

  // Palm silhouette behind the fingers.
  const palmPts = PALM_POLYGON.map((i) => pts[i]).filter(Boolean);
  if (palmPts.length >= 3) {
    const wrist = pts[0];
    const middleMcp = pts[9] ?? wrist;
    const grad = context.createLinearGradient(wrist.x, wrist.y, middleMcp.x, middleMcp.y);
    grad.addColorStop(0, palette.skinShadow);
    grad.addColorStop(1, palette.skinHighlight);
    fillPath(context, palmPts, grad);
    strokePath(context, palmPts, palette.outline, style.outlineWidth, true);
  }

  // Outline pass for fingers.
  context.strokeStyle = palette.outline;
  for (const chain of FINGER_CHAINS) {
    for (let i = 0; i < chain.length - 1; i += 1) {
      const a = pts[chain[i]];
      const b = pts[chain[i + 1]];
      if (!a || !b) continue;
      const t = i / (chain.length - 2);
      const seg = fingerThickness * (1 - t) + tipThickness * t;
      context.lineWidth = seg + outlineExtra;
      context.beginPath();
      context.moveTo(a.x, a.y);
      context.lineTo(b.x, b.y);
      context.stroke();
    }
  }

  // Skin pass for fingers.
  context.strokeStyle = palette.skinBase;
  for (const chain of FINGER_CHAINS) {
    for (let i = 0; i < chain.length - 1; i += 1) {
      const a = pts[chain[i]];
      const b = pts[chain[i + 1]];
      if (!a || !b) continue;
      const t = i / (chain.length - 2);
      context.lineWidth = fingerThickness * (1 - t) + tipThickness * t;
      context.beginPath();
      context.moveTo(a.x, a.y);
      context.lineTo(b.x, b.y);
      context.stroke();
    }
  }

  // Highlight at each fingertip.
  context.fillStyle = palette.skinHighlight;
  for (const chain of FINGER_CHAINS) {
    const tip = pts[chain[chain.length - 1]];
    if (!tip) continue;
    context.beginPath();
    context.arc(tip.x, tip.y, tipThickness * 0.55, 0, Math.PI * 2);
    context.fill();
  }

  context.restore();
}

/**
 * Picks the detected hand (from the supplied list) closest to the given
 * wrist position and removes it from the list so it can't be reused for the
 * other arm. Returns null if no hand is within range.
 */
function takeClosestHand(handsList, wrist, maxDistance) {
  if (!handsList?.length || !wrist) return null;
  let bestIndex = -1;
  let bestDist = Infinity;
  for (let i = 0; i < handsList.length; i += 1) {
    const candidate = handsList[i]?.landmarks?.[0];
    if (!candidate) continue;
    const dx = candidate.x - wrist.x;
    const dy = candidate.y - wrist.y;
    const d = Math.hypot(dx, dy);
    if (d < bestDist) {
      bestDist = d;
      bestIndex = i;
    }
  }
  if (bestIndex === -1 || bestDist > maxDistance) return null;
  return handsList.splice(bestIndex, 1)[0];
}

function drawFoot(context, ankle, legAxisAngle, length, height, palette, style) {
  if (!ankle) return;

  // Foot points "forward" — perpendicular to the leg axis (canvas down = +y
  // when the leg points down, so the foot extends along legAxisAngle + PI/2).
  const forwardAngle = legAxisAngle + Math.PI / 2;
  const heel = polar(ankle, forwardAngle - Math.PI, length * 0.25);
  const toe = polar(ankle, forwardAngle, length * 0.75);

  context.save();
  context.translate((heel.x + toe.x) / 2, (heel.y + toe.y) / 2);
  context.rotate(forwardAngle);

  const grad = context.createLinearGradient(0, -height / 2, 0, height / 2);
  grad.addColorStop(0, palette.bodyHighlight);
  grad.addColorStop(1, palette.bodyShadow);

  context.fillStyle = grad;
  context.beginPath();
  context.ellipse(0, 0, length / 2, height / 2, 0, 0, Math.PI * 2);
  context.fill();
  context.strokeStyle = palette.outline;
  context.lineWidth = style.outlineWidth;
  context.stroke();

  context.restore();
}

/**
 * Torso as a filled silhouette (shoulders → narrowed waist → hips), with
 * chest / ab dividers and a glowing accent line down the sternum.
 */
function drawTorso(context, leftShoulder, rightShoulder, leftHip, rightHip, neck, pelvis, palette, style) {
  const leftWaist = lerpPoint(leftShoulder, leftHip, 0.55);
  const rightWaist = lerpPoint(rightShoulder, rightHip, 0.55);
  const spineMid = lerpPoint(neck, pelvis, 0.55);
  const waistPullL = lerpPoint(leftWaist, spineMid, 0.18);
  const waistPullR = lerpPoint(rightWaist, spineMid, 0.18);

  const torsoPath = [
    rightShoulder,
    waistPullR,
    rightHip,
    leftHip,
    waistPullL,
    leftShoulder
  ];

  const grad = context.createLinearGradient(neck.x, neck.y, pelvis.x, pelvis.y);
  grad.addColorStop(0, palette.bodyHighlight);
  grad.addColorStop(0.4, palette.bodyBase);
  grad.addColorStop(1, palette.bodyShadow);

  fillPath(context, torsoPath, grad);
  strokePath(context, torsoPath, palette.outline, style.outlineWidth, true);

  // Chest line.
  const sternum = lerpPoint(neck, pelvis, 0.28);
  const beltLine = lerpPoint(neck, pelvis, 0.62);
  context.strokeStyle = palette.bodyShadow;
  context.lineWidth = Math.max(style.outlineWidth * 0.8, 1);
  context.beginPath();
  context.moveTo(leftShoulder.x, leftShoulder.y);
  context.quadraticCurveTo(sternum.x, sternum.y, rightShoulder.x, rightShoulder.y);
  context.stroke();

  // Belt / abdominal divider.
  context.beginPath();
  context.moveTo(waistPullL.x, waistPullL.y);
  context.quadraticCurveTo(beltLine.x, beltLine.y, waistPullR.x, waistPullR.y);
  context.stroke();

  // Glowing accent down the center.
  context.shadowColor = palette.accentGlow;
  context.shadowBlur = 8;
  strokePath(context, [sternum, beltLine], palette.accent, style.accentWidth + 0.5);
  context.shadowBlur = 0;
}

/**
 * Head: skin oval + hair cap + eyes (whites/iris/pupil/highlight) + brows +
 * nose + mouth. Reacts to head pose: roll rotates, yaw squashes silhouette
 * and shifts gaze, pitch shifts features vertically.
 *
 * NOTE: head drawing now lives in `./head.js`. The avatar invokes whichever
 * head renderer was passed in (default, image, video, or canvas source) so
 * optional local texture sources can replace the head
 * without touching this module.
 */

function drawNeck(context, neck, axisAngle, headCenter, halfWidth, palette, style) {
  const a1 = offsetPerp(neck, axisAngle, -halfWidth);
  const a2 = offsetPerp(neck, axisAngle, halfWidth);
  const b1 = offsetPerp(headCenter, axisAngle, -halfWidth * 0.85);
  const b2 = offsetPerp(headCenter, axisAngle, halfWidth * 0.85);

  const grad = context.createLinearGradient(a1.x, a1.y, a2.x, a2.y);
  grad.addColorStop(0, palette.skinHighlight);
  grad.addColorStop(0.5, palette.skinBase);
  grad.addColorStop(1, palette.skinShadow);

  fillPath(context, [a1, a2, b2, b1], grad);
  strokePath(context, [a1, a2, b2, b1], palette.outline, style.outlineWidth * 0.8, true);
}

// --- Public API -------------------------------------------------------------

export function createAvatarRenderer({
  proportions = {},
  style = {},
  palette = {},
  scaleSmoothing = DEFAULT_SCALE_SMOOTHING,
  headRenderer = null,
  head = null
} = {}) {
  const P = { ...DEFAULT_PROPORTIONS, ...proportions };
  const S_STYLE = { ...DEFAULT_STYLE, ...style };
  const PALETTE = { ...DEFAULT_PALETTE, ...palette, ...(style.palette ?? {}) };
  // Resolve the head renderer once. Callers can either pass a fully-formed
  // `headRenderer` (anything with a `draw(context, frame)` method) or a
  // descriptor `head` such as `{ type: "video", source: videoEl }`.
  const activeHeadRenderer =
    headRenderer ?? resolveHeadRenderer(head, { palette: PALETTE, style: S_STYLE });
  let smoothedScale = 0;
  // Per-side hand presence in [0,1]. Drives a cross-fade between the
  // detailed 21-landmark hand and the fallback oval so hands don't pop in
  // and out when the hand tracker briefly loses them.
  let leftHandPresence = 0;
  let rightHandPresence = 0;
  // Last-seen landmarks per side. Re-used for one extra frame when the
  // tracker drops the hand momentarily, so the visible hand doesn't snap
  // back to the wrist before the fade-out completes.
  let lastLeftHand = null;
  let lastRightHand = null;

  function reset() {
    smoothedScale = 0;
    leftHandPresence = 0;
    rightHandPresence = 0;
    lastLeftHand = null;
    lastRightHand = null;
    activeHeadRenderer.reset?.();
  }

  function setHeadRenderer(next) {
    if (next?.draw) {
      activeHeadRenderer.dispose?.();
      // Replace by mutating the closed-over reference target. We expose a
      // setter rather than recreating the avatar so wiring stays simple.
      Object.assign(activeHeadRenderer, next);
    }
  }

  function setHeadSource(source) {
    activeHeadRenderer.setSource?.(source);
  }

  function draw(context, rig, options = {}) {
    if (!context || !rig) return;

    const { bones, joints } = rig;
    if (!joints?.neck || !bones?.spine) return;

    const neckVisibility = joints.neck.visibility ?? 1;
    if (neckVisibility < S_STYLE.minBoneConfidence) return;

    const measured = bones.spine.length;
    const measurementConfidence = bones.spine.confidence ?? 1;
    if (measured > 0) {
      if (smoothedScale <= 0) {
        smoothedScale = measured;
      } else if (measurementConfidence >= S_STYLE.minBoneConfidence) {
        smoothedScale = smoothedScale * (1 - scaleSmoothing) + measured * scaleSmoothing;
      }
    }
    if (!(smoothedScale > 0)) return;
    const scale = smoothedScale;

    // Forward kinematics from pelvis (synthesized if hips off-screen).
    const spineUpAngle = bones.spine.angle + Math.PI;
    const spineDownAngle = bones.spine.angle;
    const pelvis =
      joints.pelvis && (joints.pelvis.visibility ?? 1) >= S_STYLE.minBoneConfidence
        ? joints.pelvis
        : polar(joints.neck, spineDownAngle, scale);
    const neck = polar(pelvis, spineUpAngle, scale);

    const headAngle = bones.head ? bones.head.angle : spineUpAngle;
    const headRadius = scale * P.headRadius;
    const headCenter = polar(neck, headAngle, headRadius * 1.05);

    const spinePerpLeft = spineUpAngle - Math.PI / 2;
    const spinePerpRight = spineUpAngle + Math.PI / 2;
    const leftShoulderDir = directionAngle(joints.neck, joints.leftShoulder, spinePerpLeft);
    const rightShoulderDir = directionAngle(joints.neck, joints.rightShoulder, spinePerpRight);
    const leftHipDir = directionAngle(pelvis, joints.leftHip, spinePerpLeft);
    const rightHipDir = directionAngle(pelvis, joints.rightHip, spinePerpRight);

    const leftShoulder = polar(neck, leftShoulderDir, scale * P.shoulderHalfWidth);
    const rightShoulder = polar(neck, rightShoulderDir, scale * P.shoulderHalfWidth);
    const leftHip = polar(pelvis, leftHipDir, scale * P.hipHalfWidth);
    const rightHip = polar(pelvis, rightHipDir, scale * P.hipHalfWidth);

    const minConf = S_STYLE.minBoneConfidence;

    const leftElbow =
      bones.leftUpperArm && bones.leftUpperArm.confidence >= minConf
        ? polar(leftShoulder, bones.leftUpperArm.angle, scale * P.upperArm)
        : null;
    const leftWrist =
      leftElbow && bones.leftForearm && bones.leftForearm.confidence >= minConf
        ? polar(leftElbow, bones.leftForearm.angle, scale * P.forearm)
        : null;
    const rightElbow =
      bones.rightUpperArm && bones.rightUpperArm.confidence >= minConf
        ? polar(rightShoulder, bones.rightUpperArm.angle, scale * P.upperArm)
        : null;
    const rightWrist =
      rightElbow && bones.rightForearm && bones.rightForearm.confidence >= minConf
        ? polar(rightElbow, bones.rightForearm.angle, scale * P.forearm)
        : null;

    const leftKnee =
      bones.leftThigh && bones.leftThigh.confidence >= minConf
        ? polar(leftHip, bones.leftThigh.angle, scale * P.thigh)
        : null;
    const leftAnkle =
      leftKnee && bones.leftShin && bones.leftShin.confidence >= minConf
        ? polar(leftKnee, bones.leftShin.angle, scale * P.shin)
        : null;
    const rightKnee =
      bones.rightThigh && bones.rightThigh.confidence >= minConf
        ? polar(rightHip, bones.rightThigh.angle, scale * P.thigh)
        : null;
    const rightAnkle =
      rightKnee && bones.rightShin && bones.rightShin.confidence >= minConf
        ? polar(rightKnee, bones.rightShin.angle, scale * P.shin)
        : null;

    // --- Render order: legs → torso → arms → neck → head ---------------
    context.save();
    context.lineCap = "round";
    context.lineJoin = "round";

    // Legs (drawn behind the torso).
    if (leftKnee) {
      drawTaperedSegment(
        context,
        leftHip,
        leftKnee,
        scale * P.thighThickness,
        scale * P.thighThickness * 0.78,
        PALETTE,
        S_STYLE
      );
    }
    if (leftAnkle) {
      drawTaperedSegment(
        context,
        leftKnee,
        leftAnkle,
        scale * P.shinThickness,
        scale * P.shinThickness * 0.7,
        PALETTE,
        S_STYLE
      );
      drawFoot(context, leftAnkle, bones.leftShin.angle, scale * P.footLength, scale * P.footHeight, PALETTE, S_STYLE);
    }
    if (rightKnee) {
      drawTaperedSegment(
        context,
        rightHip,
        rightKnee,
        scale * P.thighThickness,
        scale * P.thighThickness * 0.78,
        PALETTE,
        S_STYLE
      );
    }
    if (rightAnkle) {
      drawTaperedSegment(
        context,
        rightKnee,
        rightAnkle,
        scale * P.shinThickness,
        scale * P.shinThickness * 0.7,
        PALETTE,
        S_STYLE
      );
      drawFoot(context, rightAnkle, bones.rightShin.angle, scale * P.footLength, scale * P.footHeight, PALETTE, S_STYLE);
    }

    drawTorso(context, leftShoulder, rightShoulder, leftHip, rightHip, neck, pelvis, PALETTE, S_STYLE);

    // Match detected MediaPipe hands to each rig wrist by proximity so the
    // avatar's hands gain real fingers when the hand tracker has them.
    const handsAvailable = Array.isArray(options.hands) ? [...options.hands] : [];
    const handMatchRadius = scale * 0.6;
    const leftHandMatch = leftWrist ? takeClosestHand(handsAvailable, leftWrist, handMatchRadius) : null;
    const rightHandMatch = rightWrist ? takeClosestHand(handsAvailable, rightWrist, handMatchRadius) : null;

    // --- Smooth presence + carry-over of last-seen hands ---------------
    // Asymmetric easing: snap in fast (~120 ms) so a returning hand looks
    // immediate, fade out slowly (~250 ms) so a single dropped frame from
    // the hand tracker doesn't make the fingers flicker.
    const ATTACK = 0.35;
    const RELEASE = 0.18;
    if (leftHandMatch) {
      lastLeftHand = leftHandMatch;
      leftHandPresence = leftHandPresence + (1 - leftHandPresence) * ATTACK;
    } else {
      leftHandPresence = leftHandPresence * (1 - RELEASE);
      if (leftHandPresence < 0.02) {
        leftHandPresence = 0;
        lastLeftHand = null;
      }
    }
    if (rightHandMatch) {
      lastRightHand = rightHandMatch;
      rightHandPresence = rightHandPresence + (1 - rightHandPresence) * ATTACK;
    } else {
      rightHandPresence = rightHandPresence * (1 - RELEASE);
      if (rightHandPresence < 0.02) {
        rightHandPresence = 0;
        lastRightHand = null;
      }
    }

    // Arms (drawn over the torso so shoulders look attached).
    if (leftElbow) {
      drawTaperedSegment(
        context,
        leftShoulder,
        leftElbow,
        scale * P.upperArmThickness,
        scale * P.upperArmThickness * 0.85,
        PALETTE,
        S_STYLE
      );
    }
    if (leftWrist) {
      drawTaperedSegment(
        context,
        leftElbow,
        leftWrist,
        scale * P.forearmThickness,
        scale * P.forearmThickness * 0.78,
        PALETTE,
        S_STYLE
      );
      // Cross-fade fallback oval (under) with detailed hand (over). The oval
      // gives the wrist a stable silhouette while the detailed hand fades
      // in/out, so transitions feel smooth instead of pop-in.
      const ovalAlpha = 1 - leftHandPresence;
      if (ovalAlpha > 0.01) {
        context.save();
        context.globalAlpha = ovalAlpha;
        drawHand(context, leftWrist, bones.leftForearm.angle, scale * P.handRadius, PALETTE, S_STYLE);
        context.restore();
      }
      if (lastLeftHand && leftHandPresence > 0.01) {
        context.save();
        context.globalAlpha = leftHandPresence;
        drawDetailedHand(
          context,
          lastLeftHand.landmarks,
          scale,
          PALETTE,
          S_STYLE,
          leftWrist,
          bones.leftForearm.angle
        );
        context.restore();
      }
    }
    if (rightElbow) {
      drawTaperedSegment(
        context,
        rightShoulder,
        rightElbow,
        scale * P.upperArmThickness,
        scale * P.upperArmThickness * 0.85,
        PALETTE,
        S_STYLE
      );
    }
    if (rightWrist) {
      drawTaperedSegment(
        context,
        rightElbow,
        rightWrist,
        scale * P.forearmThickness,
        scale * P.forearmThickness * 0.78,
        PALETTE,
        S_STYLE
      );
      const ovalAlpha = 1 - rightHandPresence;
      if (ovalAlpha > 0.01) {
        context.save();
        context.globalAlpha = ovalAlpha;
        drawHand(context, rightWrist, bones.rightForearm.angle, scale * P.handRadius, PALETTE, S_STYLE);
        context.restore();
      }
      if (lastRightHand && rightHandPresence > 0.01) {
        context.save();
        context.globalAlpha = rightHandPresence;
        drawDetailedHand(
          context,
          lastRightHand.landmarks,
          scale,
          PALETTE,
          S_STYLE,
          rightWrist,
          bones.rightForearm.angle
        );
        context.restore();
      }
    }

    drawNeck(context, neck, spineUpAngle, headCenter, scale * P.neckHalfWidth, PALETTE, S_STYLE);
    activeHeadRenderer.draw(context, {
      center: headCenter,
      radius: headRadius,
      headAngle,
      headPose: options.headPose ?? null,
      neck,
      leftShoulder,
      rightShoulder,
      faceLandmarks: options.faceLandmarks ?? null,
      frameId: options.frameId ?? null,
      timestamp: options.timestamp ?? null
    });

    context.restore();
  }

  return { draw, reset, setHeadRenderer, setHeadSource };
}
