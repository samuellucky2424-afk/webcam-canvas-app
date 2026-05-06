import { computeCanvasMapping, projectOverlay } from "./coordinates.js";
import { buildSkeletonRig } from "../body/skeleton.js";
import { createAvatarRenderer } from "../body/avatar.js";
import { createPuppetRenderer } from "../body/puppet.js";
import { createHeadPoseEstimator } from "../state/headPose.js";

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function isVisibleLandmark(landmark, minVisibility) {
  return Boolean(landmark) && (landmark.visibility ?? 1) >= minVisibility;
}

const POSE_FACE_INDEX_MAX = 10;
const POSE_LEFT_WRIST = 15;
const POSE_RIGHT_WRIST = 16;

function distance(a, b) {
  if (!a || !b) return Infinity;
  return Math.hypot((a.x ?? 0) - (b.x ?? 0), (a.y ?? 0) - (b.y ?? 0));
}

function handWristCandidates(projected) {
  return (projected?.hands ?? [])
    .map((hand) => hand?.landmarks?.[0])
    .filter((wrist) => wrist && Number.isFinite(wrist.x) && Number.isFinite(wrist.y));
}

function assistPoseWristsFromHands(projected) {
  const landmarks = projected?.landmarks;
  const wrists = handWristCandidates(projected);
  if (!landmarks?.length || !wrists.length) return projected;

  const targets = [
    { index: POSE_LEFT_WRIST, point: landmarks[POSE_LEFT_WRIST] },
    { index: POSE_RIGHT_WRIST, point: landmarks[POSE_RIGHT_WRIST] }
  ].filter((target) => target.point);
  if (!targets.length) return projected;

  const used = new Set();
  for (const target of targets) {
    let bestIndex = -1;
    let bestDistance = Infinity;
    for (let i = 0; i < wrists.length; i += 1) {
      if (used.has(i)) continue;
      const d = distance(target.point, wrists[i]);
      if (d < bestDistance) {
        bestDistance = d;
        bestIndex = i;
      }
    }
    if (bestIndex < 0) continue;
    used.add(bestIndex);
    const handWrist = wrists[bestIndex];
    landmarks[target.index] = {
      ...target.point,
      x: target.point.x * 0.2 + handWrist.x * 0.8,
      y: target.point.y * 0.2 + handWrist.y * 0.8,
      z: typeof handWrist.z === "number" ? handWrist.z : target.point.z,
      visibility: Math.max(target.point.visibility ?? 0, handWrist.visibility ?? 1)
    };
  }

  return projected;
}

function drawLandmarkSet(context, landmarks, connections, style, options = {}) {
  if (!landmarks?.length || !style) {
    return;
  }

  const skipFace = options.skipPoseFace === true;
  const drawDots = options.drawDots !== false;

  context.save();
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = style.connectionColor;
  context.fillStyle = style.keypointColor;
  context.lineWidth = style.connectionWidth;

  for (const [startIndex, endIndex] of connections ?? []) {
    if (skipFace && (startIndex <= POSE_FACE_INDEX_MAX || endIndex <= POSE_FACE_INDEX_MAX)) {
      continue;
    }

    const start = landmarks[startIndex];
    const end = landmarks[endIndex];

    if (!isVisibleLandmark(start, style.minVisibility) || !isVisibleLandmark(end, style.minVisibility)) {
      continue;
    }

    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
  }

  if (drawDots) {
    for (let index = 0; index < landmarks.length; index += 1) {
      if (skipFace && index <= POSE_FACE_INDEX_MAX) {
        continue;
      }

      const landmark = landmarks[index];

      if (!isVisibleLandmark(landmark, style.minVisibility)) {
        continue;
      }

      context.beginPath();
      context.arc(landmark.x, landmark.y, style.keypointRadius, 0, Math.PI * 2);
      context.fill();
    }
  }

  context.restore();
}

function drawPoseOverlay(context, overlay, poseConfig) {
  if (!overlay || !poseConfig) {
    return;
  }

  const hasFaceMesh = Boolean(poseConfig.face && overlay.faces?.length);

  drawLandmarkSet(context, overlay.landmarks, overlay.connections, poseConfig, {
    skipPoseFace: hasFaceMesh
  });

  if (hasFaceMesh) {
    for (const face of overlay.faces) {
      drawLandmarkSet(context, face.landmarks, face.connections, poseConfig.face, {
        drawDots: poseConfig.face.drawAllPoints !== false
      });
    }
  }

  if (poseConfig.hand && overlay.hands?.length) {
    for (const hand of overlay.hands) {
      drawLandmarkSet(context, hand.landmarks, hand.connections, poseConfig.hand);
    }
  }
}

export function createRenderer({
  video,
  canvas,
  targetFps,
  minimumFps,
  initialScale,
  minScale,
  maxScale,
  downscaleStep,
  upscaleStep,
  upscaleThresholdFps,
  maxCanvasWidth,
  beforeRender,
  getOverlay,
  getFrame,
  poseConfig,
  avatarConfig,
  onStats
}) {
  const context = canvas.getContext("2d", {
    alpha: true,
    desynchronized: true
  });

  if (!context) {
    throw new Error("Canvas 2D context is not available.");
  }

  context.imageSmoothingEnabled = false;

  const avatarEnabled = avatarConfig?.enabled !== false && Boolean(avatarConfig);
  const overlayEnabled = !avatarEnabled || avatarConfig?.showOverlay !== false;
  const handDebugEnabled = avatarConfig?.showHandDebug === true;
  const faceDebugEnabled = avatarConfig?.showFaceDebug === true;
  const puppetConfig = avatarConfig?.puppet ?? null;
  const puppetEnabled = puppetConfig?.enabled === true && puppetConfig?.source;
  const avatarRenderer = avatarEnabled
    ? createAvatarRenderer({
        proportions: avatarConfig?.proportions,
        style: avatarConfig?.style,
        palette: avatarConfig?.palette,
        scaleSmoothing: avatarConfig?.scaleSmoothing,
        head: avatarConfig?.head,
        headRenderer: avatarConfig?.headRenderer
      })
    : null;
  const puppetRenderer = puppetEnabled
    ? createPuppetRenderer({
        image: puppetConfig.source,
        imageWidth: puppetConfig.source.width || puppetConfig.source.naturalWidth || 1,
        imageHeight: puppetConfig.source.height || puppetConfig.source.naturalHeight || 1,
        getHeadSource: puppetConfig.getHeadSource,
        smoothing: puppetConfig.smoothing,
        maxStepFrac: puppetConfig.maxStepFrac
      })
    : null;
  const headPoseEstimator = avatarEnabled
    ? createHeadPoseEstimator({ smoothing: avatarConfig?.headPoseSmoothing })
    : null;

  let animationFrameId = 0;
  let lastRenderTime = 0;
  let sampleStartTime = 0;
  let frameCount = 0;
  let renderScale = initialScale;

  function getCanvasScale() {
    return clamp(Math.min(renderScale, maxCanvasWidth / video.videoWidth), minScale, maxScale);
  }

  function resizeCanvas() {
    const scale = getCanvasScale();
    const nextWidth = Math.max(1, Math.round(video.videoWidth * scale));
    const nextHeight = Math.max(1, Math.round(video.videoHeight * scale));

    if (canvas.width !== nextWidth || canvas.height !== nextHeight) {
      canvas.width = nextWidth;
      canvas.height = nextHeight;
    }
  }

  function adjustScale(fps) {
    if (fps < minimumFps && renderScale > minScale) {
      renderScale = clamp(Number((renderScale - downscaleStep).toFixed(2)), minScale, maxScale);
      resizeCanvas();
      return;
    }

    if (fps > upscaleThresholdFps && renderScale < maxScale) {
      renderScale = clamp(Number((renderScale + upscaleStep).toFixed(2)), minScale, maxScale);
      resizeCanvas();
    }
  }

  function emitStats(now) {
    frameCount += 1;

    if (!sampleStartTime) {
      sampleStartTime = now;
      return;
    }

    const elapsed = now - sampleStartTime;

    if (elapsed < 1000) {
      return;
    }

    const fps = (frameCount * 1000) / elapsed;
    frameCount = 0;
    sampleStartTime = now;

    adjustScale(fps);

    onStats?.({
      fps,
      height: canvas.height,
      scale: renderScale,
      width: canvas.width
    });
  }

  function renderFrame(now) {
    animationFrameId = requestAnimationFrame(renderFrame);

    if (!video.videoWidth || !video.videoHeight) {
      return;
    }

    if (!lastRenderTime) {
      lastRenderTime = now;
    }

    const frameInterval = 1000 / targetFps;
    const elapsedSinceRender = now - lastRenderTime;

    if (elapsedSinceRender < frameInterval) {
      return;
    }

    lastRenderTime = now - (elapsedSinceRender % frameInterval);
    beforeRender?.(now);
    resizeCanvas();
    // Avatar-only stage: clear so the CSS background shows through. The live
    // webcam is shown separately as a picture-in-picture element.
    context.clearRect(0, 0, canvas.width, canvas.height);

    const frame = getFrame?.() ?? getOverlay?.();
    const rawOverlay = frame?.overlay ?? frame;
    const frameId = frame?.frameId ?? null;
    const stateTimestamp = frame?.timestamp ?? null;

    if (puppetRenderer) {
      const projected = rawOverlay
        ? projectOverlay(rawOverlay, computeCanvasMapping({
            canvasWidth: canvas.width,
            canvasHeight: canvas.height,
            videoWidth: video.videoWidth,
            videoHeight: video.videoHeight
          }))
        : null;
      if (projected) assistPoseWristsFromHands(projected);
      puppetRenderer.draw(context, projected);
      emitStats(now);
      return;
    }

    if (rawOverlay) {
      const mapping = computeCanvasMapping({
        canvasWidth: canvas.width,
        canvasHeight: canvas.height,
        videoWidth: video.videoWidth,
        videoHeight: video.videoHeight
      });
      const projected = assistPoseWristsFromHands(projectOverlay(rawOverlay, mapping));

      if (overlayEnabled) {
        drawPoseOverlay(context, projected, poseConfig);
      } else {
        // Independent debug overlays per tracker so each can be verified
        // in isolation while the avatar still renders normally.
        if (handDebugEnabled && poseConfig?.hand && projected.hands?.length) {
          for (const hand of projected.hands) {
            drawLandmarkSet(context, hand.landmarks, hand.connections, poseConfig.hand);
          }
        }
        if (faceDebugEnabled && poseConfig?.face && projected.faces?.length) {
          for (const face of projected.faces) {
            drawLandmarkSet(context, face.landmarks, face.connections, poseConfig.face, {
              drawAllPoints: poseConfig.face.drawAllPoints
            });
          }
        }
      }

      if (avatarRenderer) {
        // Rebuild the rig in canvas space so the avatar can use canvas-pixel
        // angles and lengths directly.
        const canvasRig = buildSkeletonRig(projected.landmarks);
        const headPose = headPoseEstimator?.update({
          face: projected.faces?.[0]?.landmarks,
          poseLandmarks: projected.landmarks
        });
        avatarRenderer.draw(context, canvasRig, {
          headPose,
          hands: projected.hands,
          faceLandmarks: projected.faces?.[0]?.landmarks ?? null,
          frameId,
          timestamp: stateTimestamp ?? rawOverlay.timestamp ?? null
        });
      }
    }

    emitStats(now);
  }

  return {
    start() {
      if (animationFrameId) {
        return;
      }

      lastRenderTime = 0;
      sampleStartTime = 0;
      frameCount = 0;
      animationFrameId = requestAnimationFrame(renderFrame);
    },
    stop() {
      if (!animationFrameId) {
        return;
      }

      cancelAnimationFrame(animationFrameId);
      animationFrameId = 0;
      avatarRenderer?.reset();
      puppetRenderer?.resetCalibration?.();
      headPoseEstimator?.reset();
      context.clearRect(0, 0, canvas.width, canvas.height);
    }
  };
}
