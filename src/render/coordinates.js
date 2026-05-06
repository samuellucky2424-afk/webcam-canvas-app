/**
 * Coordinate transformation utilities.
 *
 * MediaPipe returns landmarks in normalized [0, 1] image space, where (0, 0) is
 * the top-left of the source video frame and (1, 1) is the bottom-right. This
 * module converts those normalized coordinates into canvas-pixel coordinates
 * while correctly handling aspect-ratio mismatches between the source video
 * and the destination canvas (letterboxing / pillarboxing).
 *
 * All transformation logic lives here so trackers stay producer-only and the
 * renderer stays consumer-only.
 */

/**
 * Computes how a source frame of `videoWidth` x `videoHeight` should map onto
 * a canvas of `canvasWidth` x `canvasHeight` while preserving aspect ratio.
 *
 * Returns the offset (in canvas pixels) and the per-axis scale factor needed
 * to convert a normalized [0, 1] coordinate into canvas pixels.
 *
 * If the canvas matches the video aspect ratio (the common case in this app),
 * `offsetX` and `offsetY` will be 0 and `scaleX === scaleY === canvasWidth /
 * videoWidth`.
 */
export function computeCanvasMapping({ canvasWidth, canvasHeight, videoWidth, videoHeight }) {
  const safeVideoWidth = Math.max(videoWidth || 0, 1);
  const safeVideoHeight = Math.max(videoHeight || 0, 1);
  const safeCanvasWidth = Math.max(canvasWidth || 0, 1);
  const safeCanvasHeight = Math.max(canvasHeight || 0, 1);

  const videoAspect = safeVideoWidth / safeVideoHeight;
  const canvasAspect = safeCanvasWidth / safeCanvasHeight;

  let drawWidth = safeCanvasWidth;
  let drawHeight = safeCanvasHeight;

  if (canvasAspect > videoAspect) {
    // Canvas wider than video → pillarbox (bars on the sides).
    drawWidth = safeCanvasHeight * videoAspect;
  } else if (canvasAspect < videoAspect) {
    // Canvas taller than video → letterbox (bars on top/bottom).
    drawHeight = safeCanvasWidth / videoAspect;
  }

  const offsetX = (safeCanvasWidth - drawWidth) / 2;
  const offsetY = (safeCanvasHeight - drawHeight) / 2;

  return {
    offsetX,
    offsetY,
    scaleX: drawWidth,
    scaleY: drawHeight,
    drawWidth,
    drawHeight
  };
}

/**
 * Converts a single normalized landmark `{ x, y, z?, visibility? }` to canvas
 * pixel coordinates, preserving non-positional fields.
 */
export function projectLandmark(landmark, mapping) {
  if (!landmark) {
    return landmark;
  }

  return {
    ...landmark,
    x: mapping.offsetX + landmark.x * mapping.scaleX,
    y: mapping.offsetY + landmark.y * mapping.scaleY,
    // Z is normalized to roughly the image width; rescale alongside x for
    // visual depth consistency without changing semantics.
    z: typeof landmark.z === "number" ? landmark.z * mapping.scaleX : landmark.z,
    normalizedX: landmark.x,
    normalizedY: landmark.y
  };
}

/**
 * Projects every landmark in an array into canvas pixel coordinates.
 */
export function projectLandmarks(landmarks, mapping) {
  if (!landmarks?.length) {
    return [];
  }

  const projected = new Array(landmarks.length);

  for (let index = 0; index < landmarks.length; index += 1) {
    projected[index] = projectLandmark(landmarks[index], mapping);
  }

  return projected;
}

/**
 * Projects an entire tracker overlay (pose body landmarks + hand landmarks +
 * face landmarks) into canvas pixel coordinates.
 *
 * The shape of the returned object matches the input overlay so the renderer
 * can keep iterating it the same way; only the coordinate space changes.
 */
export function projectOverlay(overlay, mapping) {
  if (!overlay) {
    return overlay;
  }

  return {
    ...overlay,
    landmarks: projectLandmarks(overlay.landmarks, mapping),
    hands: (overlay.hands ?? []).map((hand) => ({
      ...hand,
      landmarks: projectLandmarks(hand.landmarks, mapping)
    })),
    faces: (overlay.faces ?? []).map((face) => ({
      ...face,
      landmarks: projectLandmarks(face.landmarks, mapping)
    })),
    mapping
  };
}
