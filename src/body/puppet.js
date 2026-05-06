const LM = {
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

const SOURCE = {
  head: { x: 0.39, y: 0.02, w: 0.21, h: 0.16 },
  torso: { x: 0.29, y: 0.14, w: 0.42, h: 0.30 },
  pelvis: { x: 0.33, y: 0.40, w: 0.34, h: 0.10 },
  leftUpperArm: { x: 0.20, y: 0.16, w: 0.13, h: 0.20 },
  leftLowerArm: { x: 0.17, y: 0.34, w: 0.12, h: 0.21 },
  rightUpperArm: { x: 0.67, y: 0.16, w: 0.13, h: 0.20 },
  rightLowerArm: { x: 0.71, y: 0.34, w: 0.12, h: 0.21 },
  leftUpperLeg: { x: 0.38, y: 0.45, w: 0.13, h: 0.24 },
  leftLowerLeg: { x: 0.39, y: 0.67, w: 0.11, h: 0.26 },
  rightUpperLeg: { x: 0.50, y: 0.45, w: 0.13, h: 0.24 },
  rightLowerLeg: { x: 0.50, y: 0.67, w: 0.11, h: 0.26 }
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function midpoint(a, b) {
  return { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 };
}

function distance(a, b) {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function fitContain(srcW, srcH, dstW, dstH) {
  const scale = Math.min(dstW / srcW, dstH / srcH);
  const width = srcW * scale;
  const height = srcH * scale;
  return {
    x: (dstW - width) * 0.5,
    y: (dstH - height) * 0.5,
    width,
    height
  };
}

function getPoint(landmarks, index) {
  const lm = landmarks?.[index];
  if (!lm || !Number.isFinite(lm.x) || !Number.isFinite(lm.y)) return null;
  return { x: lm.x, y: lm.y };
}

function getRectPx(region, imageWidth, imageHeight) {
  return {
    x: region.x * imageWidth,
    y: region.y * imageHeight,
    w: region.w * imageWidth,
    h: region.h * imageHeight
  };
}

function drawStaticImage(ctx, image, imageWidth, imageHeight) {
  const fit = fitContain(imageWidth, imageHeight, ctx.canvas.width, ctx.canvas.height);
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(image, fit.x, fit.y, fit.width, fit.height);
}

function drawSegment(ctx, image, srcRect, start, end, width) {
  if (!start || !end) return;
  const length = distance(start, end);
  if (!(length > 1)) return;
  const angle = Math.atan2(end.y - start.y, end.x - start.x) + Math.PI / 2;
  ctx.save();
  ctx.translate(start.x, start.y);
  ctx.rotate(angle);
  ctx.drawImage(
    image,
    srcRect.x,
    srcRect.y,
    srcRect.w,
    srcRect.h,
    -width * 0.5,
    0,
    width,
    length
  );
  ctx.restore();
}

function drawCenteredPart(ctx, image, srcRect, center, angle, width, height, anchorY = 0.5) {
  if (!center) return;
  ctx.save();
  ctx.translate(center.x, center.y);
  ctx.rotate(angle);
  ctx.drawImage(
    image,
    srcRect.x,
    srcRect.y,
    srcRect.w,
    srcRect.h,
    -width * 0.5,
    -height * anchorY,
    width,
    height
  );
  ctx.restore();
}

export function createPuppetRenderer({
  image,
  imageWidth,
  imageHeight,
  getHeadSource = null,
  smoothing = 0.2,
  maxStepFrac = 0.25
} = {}) {
  if (!image) throw new Error("createPuppetRenderer: image is required");

  let prevLandmarks = null;

  function resolveHeadSource() {
    if (typeof getHeadSource !== "function") return null;
    const source = getHeadSource();
    if (!source || !source.width || !source.height) return null;
    return source;
  }

  function smoothDest(projectedLandmarks, scale) {
    if (!projectedLandmarks?.length) return null;
    const maxStep = Math.max(2, scale * maxStepFrac);

    if (!prevLandmarks || prevLandmarks.length !== projectedLandmarks.length) {
      prevLandmarks = projectedLandmarks.map((lm) => ({
        x: lm.x,
        y: lm.y
      }));
      return prevLandmarks;
    }

    for (let i = 0; i < projectedLandmarks.length; i += 1) {
      const cur = projectedLandmarks[i];
      const prev = prevLandmarks[i];
      if (!cur || !prev) continue;
      const dx = cur.x - prev.x;
      const dy = cur.y - prev.y;
      const len = Math.hypot(dx, dy);
      let tx = cur.x;
      let ty = cur.y;
      if (len > maxStep && len > 1e-6) {
        const k = maxStep / len;
        tx = prev.x + dx * k;
        ty = prev.y + dy * k;
      }
      prev.x = prev.x * (1 - smoothing) + tx * smoothing;
      prev.y = prev.y * (1 - smoothing) + ty * smoothing;
    }

    return prevLandmarks;
  }

  function draw(ctx, overlay) {
    const landmarks = overlay?.landmarks;
    if (!landmarks?.length) {
      drawStaticImage(ctx, image, imageWidth, imageHeight);
      return;
    }

    const lS = getPoint(landmarks, LM.leftShoulder);
    const rS = getPoint(landmarks, LM.rightShoulder);
    const scale = lS && rS ? distance(lS, rS) : 120;
    const smoothedLandmarks = smoothDest(landmarks, scale) ?? landmarks.map((lm) => ({ x: lm.x, y: lm.y }));

    const leftShoulder = getPoint(smoothedLandmarks, LM.leftShoulder);
    const rightShoulder = getPoint(smoothedLandmarks, LM.rightShoulder);
    const leftHip = getPoint(smoothedLandmarks, LM.leftHip);
    const rightHip = getPoint(smoothedLandmarks, LM.rightHip);
    const leftElbow = getPoint(smoothedLandmarks, LM.leftElbow);
    const rightElbow = getPoint(smoothedLandmarks, LM.rightElbow);
    const leftWrist = getPoint(smoothedLandmarks, LM.leftWrist);
    const rightWrist = getPoint(smoothedLandmarks, LM.rightWrist);
    const leftKnee = getPoint(smoothedLandmarks, LM.leftKnee);
    const rightKnee = getPoint(smoothedLandmarks, LM.rightKnee);
    const leftAnkle = getPoint(smoothedLandmarks, LM.leftAnkle);
    const rightAnkle = getPoint(smoothedLandmarks, LM.rightAnkle);
    const nose = getPoint(smoothedLandmarks, LM.nose);

    if (!leftShoulder || !rightShoulder || !leftHip || !rightHip) {
      drawStaticImage(ctx, image, imageWidth, imageHeight);
      return;
    }

    const shoulderMid = midpoint(leftShoulder, rightShoulder);
    const hipMid = midpoint(leftHip, rightHip);
    const shoulderSpan = distance(leftShoulder, rightShoulder);
    const hipSpan = distance(leftHip, rightHip);
    const torsoHeight = distance(shoulderMid, hipMid) * 1.15;
    const torsoWidth = Math.max(shoulderSpan * 1.45, hipSpan * 1.25);
    const torsoAngle = Math.atan2(hipMid.y - shoulderMid.y, hipMid.x - shoulderMid.x) + Math.PI / 2;
    const pelvisCenter = midpoint(hipMid, { x: hipMid.x, y: lerp(shoulderMid.y, hipMid.y, 1.15) });
    const headCenter = nose
      ? { x: lerp(shoulderMid.x, nose.x, 0.75), y: nose.y + shoulderSpan * 0.14 }
      : { x: shoulderMid.x, y: shoulderMid.y - shoulderSpan * 0.72 };
    const headSize = shoulderSpan * 0.9;

    const src = {};
    for (const [key, region] of Object.entries(SOURCE)) {
      src[key] = getRectPx(region, imageWidth, imageHeight);
    }

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    drawSegment(ctx, image, src.leftUpperLeg, leftHip, leftKnee, shoulderSpan * 0.42);
    drawSegment(ctx, image, src.leftLowerLeg, leftKnee, leftAnkle, shoulderSpan * 0.36);
    drawSegment(ctx, image, src.rightUpperLeg, rightHip, rightKnee, shoulderSpan * 0.42);
    drawSegment(ctx, image, src.rightLowerLeg, rightKnee, rightAnkle, shoulderSpan * 0.36);

    drawSegment(ctx, image, src.leftUpperArm, leftShoulder, leftElbow, shoulderSpan * 0.28);
    drawSegment(ctx, image, src.leftLowerArm, leftElbow, leftWrist, shoulderSpan * 0.22);
    drawSegment(ctx, image, src.rightUpperArm, rightShoulder, rightElbow, shoulderSpan * 0.28);
    drawSegment(ctx, image, src.rightLowerArm, rightElbow, rightWrist, shoulderSpan * 0.22);

    drawCenteredPart(ctx, image, src.torso, midpoint(shoulderMid, hipMid), torsoAngle, torsoWidth, torsoHeight, 0.46);
    drawCenteredPart(ctx, image, src.pelvis, pelvisCenter, torsoAngle, hipSpan * 1.5, shoulderSpan * 0.42, 0.5);
    const liveHead = resolveHeadSource();
    if (liveHead) {
      drawCenteredPart(
        ctx,
        liveHead,
        { x: 0, y: 0, w: liveHead.width, h: liveHead.height },
        headCenter,
        0,
        headSize,
        headSize * 1.15,
        0.52
      );
    } else {
      drawCenteredPart(ctx, image, src.head, headCenter, 0, headSize, headSize * 1.15, 0.52);
    }

    ctx.restore();
  }

  function resetCalibration() {
    prevLandmarks = null;
  }

  return { draw, resetCalibration };
}
