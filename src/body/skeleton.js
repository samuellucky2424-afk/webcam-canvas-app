/**
 * Skeleton rig builder.
 *
 * Converts a flat array of pose landmarks (canvas-space coordinates) into a
 * structured rig of bones. Each bone exposes:
 *   - `name` — stable identifier (e.g. `"spine"`, `"leftUpperArm"`)
 *   - `start` / `end` — references to landmark objects (with `x`, `y`)
 *   - `length` — Euclidean distance in canvas pixels
 *   - `angle` — orientation in radians, computed via `Math.atan2(dy, dx)`
 *               (0 = pointing along +x, increases clockwise on a canvas
 *               because the y axis grows downward)
 *   - `angleDeg` — same angle in degrees for convenience
 *   - `confidence` — min(visibility) of the two endpoints
 *
 * Pure computation only — no drawing, no DOM. Render layers can consume the
 * returned rig however they like.
 *
 * Pose landmark indices follow the MediaPipe Pose Landmark model.
 */

const POSE_INDEX = {
  nose: 0,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  rightKnee: 26,
  leftAnkle: 27,
  rightAnkle: 28
};

/**
 * Bone definitions: which landmarks form each bone of the rig.
 * Keep the list small and skeletal — head, spine, arms, legs.
 */
const BONE_DEFINITIONS = [
  { name: "head", start: "neck", end: "nose" },
  { name: "spine", start: "neck", end: "pelvis" },

  { name: "leftUpperArm", start: "leftShoulder", end: "leftElbow" },
  { name: "leftForearm", start: "leftElbow", end: "leftWrist" },
  { name: "rightUpperArm", start: "rightShoulder", end: "rightElbow" },
  { name: "rightForearm", start: "rightElbow", end: "rightWrist" },

  { name: "leftThigh", start: "leftHip", end: "leftKnee" },
  { name: "leftShin", start: "leftKnee", end: "leftAnkle" },
  { name: "rightThigh", start: "rightHip", end: "rightKnee" },
  { name: "rightShin", start: "rightKnee", end: "rightAnkle" }
];

function midpoint(a, b) {
  if (!a || !b) {
    return null;
  }

  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: typeof a.z === "number" && typeof b.z === "number" ? (a.z + b.z) / 2 : undefined,
    visibility:
      typeof a.visibility === "number" && typeof b.visibility === "number"
        ? Math.min(a.visibility, b.visibility)
        : undefined
  };
}

/**
 * Builds a name → landmark map that includes the raw MediaPipe joints plus
 * two derived joints used as bone roots:
 *   - `neck`   = midpoint(leftShoulder, rightShoulder)
 *   - `pelvis` = midpoint(leftHip, rightHip)
 */
function buildJointMap(landmarks) {
  const joints = {};

  for (const [name, index] of Object.entries(POSE_INDEX)) {
    joints[name] = landmarks[index] ?? null;
  }

  joints.neck = midpoint(joints.leftShoulder, joints.rightShoulder);
  joints.pelvis = midpoint(joints.leftHip, joints.rightHip);

  return joints;
}

function computeBone(name, start, end) {
  if (!start || !end) {
    return null;
  }

  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.hypot(dx, dy);
  const angle = Math.atan2(dy, dx);

  const startVisibility = typeof start.visibility === "number" ? start.visibility : 1;
  const endVisibility = typeof end.visibility === "number" ? end.visibility : 1;

  return {
    name,
    start,
    end,
    length,
    angle,
    angleDeg: (angle * 180) / Math.PI,
    confidence: Math.min(startVisibility, endVisibility)
  };
}

/**
 * Builds a skeleton rig from an array of pose landmarks.
 *
 * Returns:
 * {
 *   joints: { nose, neck, leftShoulder, ..., pelvis, ... },
 *   bones: { head, spine, leftUpperArm, leftForearm, rightUpperArm, ... },
 *   bonesList: [...same bones in definition order]
 * }
 *
 * Bones whose endpoints are missing are omitted from the maps.
 *
 * This function is pure and safe to call every frame.
 */
export function buildSkeletonRig(landmarks) {
  if (!landmarks?.length) {
    return { joints: {}, bones: {}, bonesList: [] };
  }

  const joints = buildJointMap(landmarks);
  const bones = {};
  const bonesList = [];

  for (const definition of BONE_DEFINITIONS) {
    const bone = computeBone(definition.name, joints[definition.start], joints[definition.end]);

    if (!bone) {
      continue;
    }

    bones[definition.name] = bone;
    bonesList.push(bone);
  }

  return { joints, bones, bonesList };
}

export { BONE_DEFINITIONS, POSE_INDEX };
