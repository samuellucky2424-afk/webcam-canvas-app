/**
 * Unified per-frame state builder.
 *
 * Composes the outputs of the three trackers (pose, hands, face) and the
 * head-pose estimator into a single immutable `state` object that downstream
 * stages (body rig, AI face client, composer, stream encoder) all read from.
 *
 * Tracking is NEVER duplicated here — this module reads what the trackers
 * have already produced. It does no MediaPipe work of its own.
 *
 * Output shape per frame:
 *
 *   {
 *     timestamp: number,             // performance.now() at build time
 *     frameId:   number,              // monotonic id (driver→server pairing)
 *     skeleton:  {...},               // body joints (canvas-space if mapped)
 *     hands:     [{landmarks, ...}],  // up to 2 hands
 *     headPose:  { roll, yaw, pitch, eyeOpen, mouthOpen, smile, confidence },
 *     faceSignals: {
 *       blink:            number,     // 1 - eyeOpen, clamped 0..1
 *       mouthOpen:        number,     // 0..1
 *       expressionValues: { smile, yaw, pitch, roll }
 *     },
 *     hasFace: boolean
 *   }
 */

let nextFrameId = 1;

export function createStateBuilder() {
  let lastState = null;

  /**
   * Build a unified state object from the current overlay (already produced
   * by `poseTracker` + `setHands` + `setFaces`) and the head-pose estimator.
   *
   * @param {object} overlay   - poseTracker.getLatestOverlay()
   * @param {object|null} headPose - headPoseEstimator.update(...) result
   * @returns {object} unified state
   */
  function build(overlay, headPose) {
    const now = performance.now();
    const eyeOpen = headPose?.eyeOpen ?? 1;
    const mouthOpen = headPose?.mouthOpen ?? 0;
    const smile = headPose?.smile ?? 0;
    const hasFace = Array.isArray(overlay?.faces) && overlay.faces.length > 0;

    lastState = {
      timestamp: now,
      frameId: nextFrameId++,
      overlay: overlay ?? null,
      skeleton: overlay?.skeleton ?? null,
      hands: overlay?.hands ?? [],
      headPose: headPose
        ? {
            roll: headPose.roll ?? 0,
            yaw: headPose.yaw ?? 0,
            pitch: headPose.pitch ?? 0,
            eyeOpen,
            mouthOpen,
            smile,
            confidence: headPose.confidence ?? 0
          }
        : null,
      faceSignals: {
        blink: Math.min(Math.max(1 - eyeOpen, 0), 1),
        mouthOpen,
        expressionValues: {
          smile,
          yaw: headPose?.yaw ?? 0,
          pitch: headPose?.pitch ?? 0,
          roll: headPose?.roll ?? 0
        }
      },
      hasFace
    };
    return lastState;
  }

  function getLast() {
    return lastState;
  }

  function reset() {
    lastState = null;
  }

  return { build, getLast, reset };
}
