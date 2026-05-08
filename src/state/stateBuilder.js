/**
 * Unified per-frame state builder.
 *
 * Composes the outputs of the three trackers (pose, hands, face) and the
 * head-pose estimator into a single per-frame `state` object that downstream
 * stages (body rig, semantic encoder, local renderer) all read from.
 *
 * Tracking is NEVER duplicated here — this module reads what the trackers
 * have already produced. It does no MediaPipe work of its own.
 *
 * Output shape per frame:
 *
 *   {
 *     timestamp: number,             // performance.now() at build time
 *     frameId:   number,              // monotonic id for semantic packets
 *     skeleton:  {...},               // body joints (canvas-space if mapped)
 *     hands:     [{landmarks, ...}],  // up to 2 hands
 *     headPose:  {
 *       roll, yaw, pitch,
 *       eyeOpen, leftEyeOpen, rightEyeOpen,
 *       mouthOpen, jawOpen, smile, browRaise,
 *       eyeDirectionX, eyeDirectionY, confidence
 *     },
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
    const leftEyeOpen = headPose?.leftEyeOpen ?? eyeOpen;
    const rightEyeOpen = headPose?.rightEyeOpen ?? eyeOpen;
    const mouthOpen = headPose?.mouthOpen ?? 0;
    const jawOpen = headPose?.jawOpen ?? mouthOpen;
    const smile = headPose?.smile ?? 0;
    const browRaise = headPose?.browRaise ?? 0;
    const eyeDirectionX = headPose?.eyeDirectionX ?? 0;
    const eyeDirectionY = headPose?.eyeDirectionY ?? 0;
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
            leftEyeOpen,
            rightEyeOpen,
            mouthOpen,
            jawOpen,
            smile,
            browRaise,
            eyeDirectionX,
            eyeDirectionY,
            confidence: headPose.confidence ?? 0
          }
        : null,
      faceSignals: {
        blink: Math.min(Math.max(1 - eyeOpen, 0), 1),
        blinkLeft: Math.min(Math.max(1 - leftEyeOpen, 0), 1),
        blinkRight: Math.min(Math.max(1 - rightEyeOpen, 0), 1),
        mouthOpen,
        jawOpen,
        browRaise,
        eyeDirection: { x: eyeDirectionX, y: eyeDirectionY },
        expressionValues: {
          smile,
          yaw: headPose?.yaw ?? 0,
          pitch: headPose?.pitch ?? 0,
          roll: headPose?.roll ?? 0,
          browRaise,
          eyeDirectionX,
          eyeDirectionY
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
