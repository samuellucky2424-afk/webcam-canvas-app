/**
 * AI head renderer — bridge between the LivePortrait WebSocket client and
 * the avatar's pluggable head slot.
 *
 * Implements the HeadRenderer interface: `{ draw(ctx, frame), reset, dispose }`
 * where the avatar passes:
 *   frame = {
 *     center:    { x, y },              // head anchor in canvas space
 *     radius:    number,                // body-derived head radius
 *     headAngle: number,                // neck→head bone angle (incl. roll)
 *     headPose:  { roll, yaw, pitch, eyeOpen, mouthOpen, smile, confidence }
 *   }
 *
 * Pipeline stages:
 *
 *   1. Local pose drives anchor / scale / rotation immediately. Cloud AI
 *      frames are treated as texture refreshes, not motion timing.
 *      Velocity extrapolation predicts the next local head transform.
 *   2. Color match — `ctx.filter` applies brightness / contrast /
 *      saturation to the AI portrait so it blends with the avatar's
 *      palette. The strengths are smoothed across frames so subtle
 *      differences in JPEG decode don't strobe.
 *   3. Tone tint — a low-opacity `soft-light` layer biases the AI face
 *      toward the surrounding skin/body tone.
 *   4. Elliptical feathered mask — a true ellipse (respecting
 *      `ovalAspectY`) with a smoothstep-style two-stop gradient gives a
 *      soft 3–6 px alpha falloff that hides the JPEG seam.
 *   5. Chin shadow — a bottom-half radial gradient inside the mask adds
 *      grounding so the face does not look like a sticker.
 *   6. Rim shadow — a 1.5 px dark ellipse around the silhouette grounds
 *      the head against neck/torso lighting.
 *
 *   Debug overlays are unchanged: cyan bounding box + alignment cross
 *   + raw/smoothed anchor dots.
 */

const TWO_PI = Math.PI * 2;

export function createAiHeadRenderer({
  client,
  fallback,
  // ---- mask & edge ----
  // Feather radius (px) for the elliptical mask. 3–6 looks best.
  edgeFeatherPx = 5,
  // Width of the gradient ring as a fraction of the radius. Higher =
  // softer edge. Effective falloff = max(edgeFeatherPx, radius*edgeFeatherFrac).
  edgeFeatherFrac = 0.10,
  // Aspect ratio of the head oval (height/width). Matches typical face
  // portraits; the mask is a true ellipse, not a circle scaled later.
  ovalAspectY = 1.18,
  // Slight upscale so the AI face fills the body's head slot fully.
  scaleBoost = 1.18,
  // ---- color matching ----
  // Master toggle — disable to isolate alignment issues from color work.
  colorCorrect = true,
  brightness = 1.0,
  contrast = 1.05,
  saturation = 0.9,
  // Soft-light tint pulled toward the avatar skin tone. Strength 0..1.
  toneTint = "rgba(232, 193, 154, 1.0)", // matches AVATAR_CONFIG.palette.skinBase
  toneStrength = 0.18,
  // Bottom-half chin shadow opacity 0..1.
  chinShadowStrength = 0.22,
  // Rim shadow around the silhouette (alpha 0..1).
  rimShadowStrength = 0.18,
  // ---- alignment / pose ----
  smoothingAlpha = 0.15,
  offsetX = 0,
  offsetY = 0,
  yawWeight = 0.6,
  pitchWeight = 0.6,
  minYawScale = 0.78,
  minPitchScale = 0.82,
  latencyCompensation = true,
  predictionMs = 90,
  maxPredictionMs = 140,
  motionSmoothingAlpha = 0.55,
  velocitySmoothingAlpha = 0.35,
  // Anti-flicker: how strongly per-frame color adjustments are smoothed
  // toward their target values (0 = frozen at first frame, 1 = no
  // smoothing). 0.08 keeps subtle JPEG flicker off the face without
  // making the system feel laggy.
  colorSmoothingAlpha = 0.08,
  debug = false
} = {}) {
  if (!client) throw new Error("createAiHeadRenderer: client is required");
  if (!fallback) throw new Error("createAiHeadRenderer: fallback is required");

  const FACE_LEFT_EYE_OUTER = 33;
  const FACE_RIGHT_EYE_OUTER = 263;
  const FACE_NOSE_TIP = 1;

  const scratch = document.createElement("canvas");
  let scratchSize = 0;
  let scratchFrameKey = -1;
  let scratchColorKey = "";

  // Smoothed transform state.
  const s = {
    cx: null, cy: null, radius: null, angle: null,
    yaw: 0, pitch: 0, valid: false
  };
  const motion = {
    lastT: 0,
    last: null,
    vx: 0,
    vy: 0,
    vr: 0,
    va: 0,
    vyaw: 0,
    vpitch: 0
  };
  const spring = {
    lastT: 0,
    vx: 0,
    vy: 0,
    vr: 0,
    va: 0,
    vyaw: 0,
    vpitch: 0
  };
  // Smoothed color state — primed at the configured targets so the very
  // first frame already looks right.
  const c = {
    brightness, contrast, saturation,
    toneStrength, chinShadowStrength, rimShadowStrength
  };

  const calibration = { offsetX, offsetY };
  let debugOn = debug;

  function ensureScratch(size) {
    if (scratchSize === size) return;
    scratch.width = size;
    scratch.height = size;
    scratchSize = size;
    scratchFrameKey = -1;
  }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  function angleDelta(from, to) {
    let d = to - from;
    while (d > Math.PI) d -= TWO_PI;
    while (d < -Math.PI) d += TWO_PI;
    return d;
  }

  function clampStep(prev, next, maxStep) {
    if (prev === null || !Number.isFinite(prev)) return next;
    const d = next - prev;
    if (d > maxStep) return prev + maxStep;
    if (d < -maxStep) return prev - maxStep;
    return next;
  }

  function clampAngleStep(prev, next, maxStep) {
    if (prev === null || !Number.isFinite(prev)) return next;
    let d = next - prev;
    while (d > Math.PI) d -= TWO_PI;
    while (d < -Math.PI) d += TWO_PI;
    if (d > maxStep) d = maxStep;
    if (d < -maxStep) d = -maxStep;
    return prev + d;
  }

  function lerp(prev, cur, alpha) {
    if (prev === null || !Number.isFinite(prev)) return cur;
    return prev * (1 - alpha) + cur * alpha;
  }
  function lerpAngle(prev, cur, alpha) {
    if (prev === null || !Number.isFinite(prev)) return cur;
    return prev + angleDelta(prev, cur) * alpha;
  }

  function midpoint(a, b) {
    return { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 };
  }

  function distance(a, b) {
    return Math.hypot(b.x - a.x, b.y - a.y);
  }

  function pickFaceAnchor(faceLandmarks) {
    if (!faceLandmarks?.length) return null;
    const l = faceLandmarks[FACE_LEFT_EYE_OUTER];
    const r = faceLandmarks[FACE_RIGHT_EYE_OUTER];
    const n = faceLandmarks[FACE_NOSE_TIP];
    if (!l || !r || !n) return null;
    const eyeC = midpoint(l, r);
    return {
      anchor: { x: eyeC.x * 0.65 + n.x * 0.35, y: eyeC.y * 0.65 + n.y * 0.35 },
      eyeDist: distance(l, r)
    };
  }

  function clampPointStep(prev, next, maxStep) {
    if (!prev) return next;
    const dx = next.x - prev.x;
    const dy = next.y - prev.y;
    const len = Math.hypot(dx, dy);
    if (len <= maxStep || len < 1e-6) return next;
    const k = maxStep / len;
    return { x: prev.x + dx * k, y: prev.y + dy * k };
  }

  function predictTarget(target, now) {
    if (!latencyCompensation) return target;

    if (motion.last && motion.lastT) {
      const dt = clamp((now - motion.lastT) / 1000, 1 / 120, 0.15);
      const a = clamp(velocitySmoothingAlpha, 0.05, 0.9);
      motion.vx = lerp(motion.vx, (target.cx - motion.last.cx) / dt, a);
      motion.vy = lerp(motion.vy, (target.cy - motion.last.cy) / dt, a);
      motion.vr = lerp(motion.vr, (target.radius - motion.last.radius) / dt, a);
      motion.va = lerp(motion.va, angleDelta(motion.last.angle, target.angle) / dt, a);
      motion.vyaw = lerp(motion.vyaw, (target.yaw - motion.last.yaw) / dt, a);
      motion.vpitch = lerp(motion.vpitch, (target.pitch - motion.last.pitch) / dt, a);
    }

    motion.last = { ...target };
    motion.lastT = now;

    const lead = clamp(predictionMs, 0, maxPredictionMs) / 1000;
    const maxShift = Math.max(2, target.radius * 0.35);
    const dx = clamp(motion.vx * lead, -maxShift, maxShift);
    const dy = clamp(motion.vy * lead, -maxShift, maxShift);
    return {
      cx: target.cx + dx,
      cy: target.cy + dy,
      radius: clamp(target.radius + motion.vr * lead, target.radius * 0.75, target.radius * 1.25),
      angle: target.angle + clamp(motion.va * lead, -0.35, 0.35),
      yaw: clamp(target.yaw + motion.vyaw * lead, -1.25, 1.25),
      pitch: clamp(target.pitch + motion.vpitch * lead, -1.25, 1.25)
    };
  }

  function springScalar(current, target, velocityKey, dt, stiffness, damping) {
    const velocity = (spring[velocityKey] + (target - current) * stiffness * dt) * Math.exp(-damping * dt);
    spring[velocityKey] = velocity;
    return current + velocity * dt;
  }

  function springAngle(current, target, velocityKey, dt, stiffness, damping) {
    const delta = angleDelta(current, target);
    const velocity = (spring[velocityKey] + delta * stiffness * dt) * Math.exp(-damping * dt);
    spring[velocityKey] = velocity;
    return current + velocity * dt;
  }

  function smoothFrame(frame) {
    const motionA = clamp(latencyCompensation ? motionSmoothingAlpha : smoothingAlpha, 0.02, 0.85);
    const posA = clamp(latencyCompensation ? motionA : smoothingAlpha + 0.05, 0.02, 0.85);
    const scaleA = clamp(latencyCompensation ? motionA * 0.78 : smoothingAlpha - 0.05, 0.02, 0.65);
    const rotA = clamp(latencyCompensation ? motionA * 0.92 : smoothingAlpha, 0.02, 0.8);

    const faceInfo = pickFaceAnchor(frame.faceLandmarks);
    const confidence = clamp(frame.headPose?.confidence ?? 0, 0, 1);

    const baseAnchor = frame.neck
      ? { x: frame.neck.x * 0.15 + frame.center.x * 0.85, y: frame.neck.y * 0.15 + frame.center.y * 0.85 }
      : frame.center;

    let currentAnchor = baseAnchor;
    if (faceInfo?.anchor) {
      const refineW = 0.35 * confidence;
      currentAnchor = {
        x: baseAnchor.x * (1 - refineW) + faceInfo.anchor.x * refineW,
        y: baseAnchor.y * (1 - refineW) + faceInfo.anchor.y * refineW
      };
    }

    let maxPosStep = Math.max(2, (s.radius ?? frame.radius ?? 24) * 0.25);
    let clampedAnchor = clampPointStep(
      s.cx == null || s.cy == null ? null : { x: s.cx, y: s.cy },
      currentAnchor,
      maxPosStep
    );

    let yaw = clamp(frame.headPose?.yaw ?? 0, -1.2, 1.2);
    let pitch = clamp(frame.headPose?.pitch ?? 0, -1.2, 1.2);
    const roll = clamp(frame.headPose?.roll ?? 0, -1.2, 1.2);

    let targetRadius = frame.radius;
    if (frame.leftShoulder && frame.rightShoulder) {
      const shoulderDist = distance(frame.leftShoulder, frame.rightShoulder);
      const radiusFromShoulders = shoulderDist * 0.24;
      if (Number.isFinite(radiusFromShoulders) && radiusFromShoulders > 1) {
        targetRadius = targetRadius * 0.8 + radiusFromShoulders * 0.2;
      }
    }
    if (faceInfo?.eyeDist) {
      const radiusFromEyes = faceInfo.eyeDist * 1.35;
      if (Number.isFinite(radiusFromEyes) && radiusFromEyes > 1) {
        const w = 0.35 * confidence;
        targetRadius = targetRadius * (1 - w) + radiusFromEyes * w;
      }
    }

    const targetAngle = (frame.headAngle ?? 0) + roll;
    const predicted = predictTarget({
      cx: currentAnchor.x,
      cy: currentAnchor.y,
      radius: targetRadius,
      angle: targetAngle,
      yaw,
      pitch
    }, performance.now());

    currentAnchor = { x: predicted.cx, y: predicted.cy };
    targetRadius = predicted.radius;
    yaw = predicted.yaw;
    pitch = predicted.pitch;
    maxPosStep = Math.max(2, (s.radius ?? targetRadius ?? 24) * (latencyCompensation ? 0.55 : 0.25));
    clampedAnchor = clampPointStep(
      s.cx == null || s.cy == null ? null : { x: s.cx, y: s.cy },
      currentAnchor,
      maxPosStep
    );

    const maxRadiusStep = Math.max(1, (s.radius ?? targetRadius ?? 24) * (latencyCompensation ? 0.35 : 0.18));
    const radiusStep = clampStep(s.radius, targetRadius, maxRadiusStep);

    const maxAngleStep = latencyCompensation ? 0.65 : 0.35;
    const angleStep = clampAngleStep(s.angle, predicted.angle, maxAngleStep);

    const maxAxisStep = latencyCompensation ? 0.7 : 0.35;
    const yawStep = clampStep(s.yaw, yaw, maxAxisStep);
    const pitchStep = clampStep(s.pitch, pitch, maxAxisStep);

    if (latencyCompensation && s.valid) {
      const now = performance.now();
      const dt = clamp((now - (spring.lastT || now)) / 1000, 1 / 120, 1 / 20);
      spring.lastT = now;
      s.cx = springScalar(s.cx, clampedAnchor.x, "vx", dt, 120, 17);
      s.cy = springScalar(s.cy, clampedAnchor.y, "vy", dt, 120, 17);
      s.radius = springScalar(s.radius, radiusStep, "vr", dt, 95, 16);
      s.angle = springAngle(s.angle, angleStep, "va", dt, 115, 17);
      s.yaw = springScalar(s.yaw, yawStep, "vyaw", dt, 105, 16);
      s.pitch = springScalar(s.pitch, pitchStep, "vpitch", dt, 105, 16);
    } else {
      spring.lastT = performance.now();
      s.cx = lerp(s.cx, clampedAnchor.x, posA);
      s.cy = lerp(s.cy, clampedAnchor.y, posA);
      s.radius = lerp(s.radius, radiusStep, scaleA);
      s.angle = lerpAngle(s.angle, angleStep, rotA);
      s.yaw = lerp(s.yaw, yawStep, rotA);
      s.pitch = lerp(s.pitch, pitchStep, rotA);
    }
    s.valid = true;

    // Smooth the color adjustments toward their configured targets so
    // that runtime tweaks (window.__aiHead.setColor(...)) ease in.
    const ca = colorSmoothingAlpha;
    c.brightness = lerp(c.brightness, brightness, ca);
    c.contrast = lerp(c.contrast, contrast, ca);
    c.saturation = lerp(c.saturation, saturation, ca);
    c.toneStrength = lerp(c.toneStrength, toneStrength, ca);
    c.chinShadowStrength = lerp(c.chinShadowStrength, chinShadowStrength, ca);
    c.rimShadowStrength = lerp(c.rimShadowStrength, rimShadowStrength, ca);
  }

  function buildScratch(drawSize, frameKey) {
    const colorKey = [
      c.brightness.toFixed(3),
      c.contrast.toFixed(3),
      c.saturation.toFixed(3),
      c.toneStrength.toFixed(3),
      c.chinShadowStrength.toFixed(3)
    ].join("|");
    if (scratchFrameKey === frameKey && scratchColorKey === colorKey) return;
    scratchFrameKey = frameKey;
    scratchColorKey = colorKey;

    const sctx = scratch.getContext("2d");
    const W = drawSize;
    const H = Math.round(drawSize * ovalAspectY);
    if (scratch.height !== H) scratch.height = H;

    const cx = W * 0.5;
    const cy = H * 0.5;
    const rx = (W * 0.5) - 1;
    const ry = (H * 0.5) - 1;

    sctx.save();
    sctx.clearRect(0, 0, W, H);

    // -- 1. Source AI portrait drawn into the elliptical area. The AI
    //    canvas is square (256×256); we scale it to fill the ellipse's
    //    bounding box. ctx.filter handles brightness/contrast/saturation
    //    in one shot.
    if (colorCorrect && hasFilterSupport()) {
      sctx.filter = `brightness(${c.brightness.toFixed(3)}) contrast(${c.contrast.toFixed(3)}) saturate(${c.saturation.toFixed(3)})`;
    }
    const ai = client.getOutputCanvas();
    sctx.drawImage(ai, 0, 0, W, H);
    sctx.filter = "none";

    // -- 2. Tone tint pulled toward the avatar's skin tone using
    //    soft-light. This shifts hue/value gently without flattening
    //    contrast.
    if (colorCorrect && c.toneStrength > 0.001) {
      sctx.globalCompositeOperation = "soft-light";
      sctx.fillStyle = applyAlpha(toneTint, c.toneStrength);
      sctx.fillRect(0, 0, W, H);
    }

    // -- 3. Chin / lower-face shadow inside the ellipse. A vertical
    //    radial gradient rooted near the chin darkens the lower third
    //    so the face has weight against the neck.
    if (c.chinShadowStrength > 0.001) {
      sctx.globalCompositeOperation = "multiply";
      const chinY = cy + ry * 0.55;
      const grad = sctx.createRadialGradient(cx, chinY, ry * 0.10, cx, chinY, ry * 0.95);
      grad.addColorStop(0, `rgba(40, 30, 25, ${(c.chinShadowStrength * 0.85).toFixed(3)})`);
      grad.addColorStop(0.6, `rgba(50, 40, 35, ${(c.chinShadowStrength * 0.35).toFixed(3)})`);
      grad.addColorStop(1, "rgba(80, 70, 60, 0)");
      sctx.fillStyle = grad;
      sctx.fillRect(0, 0, W, H);
    }

    // -- 4. Elliptical feathered mask. We compose the mask with
    //    `destination-in` so the AI face only survives inside the
    //    ellipse. The two-stop gradient — opaque at innerR, transparent
    //    at outerR — produces a smooth 3–6 px alpha falloff that hides
    //    the AI portrait's hard JPEG edge.
    sctx.globalCompositeOperation = "destination-in";
    const featherPx = Math.max(edgeFeatherPx, Math.min(rx, ry) * edgeFeatherFrac);
    const innerScale = Math.max(0, 1 - featherPx / Math.min(rx, ry));

    // Scale the gradient to be circular, then squash it via setTransform
    // so it follows the ellipse exactly.
    sctx.save();
    sctx.setTransform(rx, 0, 0, ry, cx, cy);
    const mask = sctx.createRadialGradient(0, 0, innerScale, 0, 0, 1);
    mask.addColorStop(0, "rgba(0,0,0,1)");
    mask.addColorStop(Math.min(0.999, innerScale + (1 - innerScale) * 0.5),
                      "rgba(0,0,0,0.85)");
    mask.addColorStop(1, "rgba(0,0,0,0)");
    sctx.fillStyle = mask;
    sctx.fillRect(-1.05, -1.05, 2.1, 2.1);
    sctx.restore();

    sctx.restore();
  }

  function drawDebug(ctx, halfW, halfH) {
    ctx.save();
    ctx.lineWidth = 1.25;
    ctx.strokeStyle = "rgba(92, 230, 255, 0.9)";
    ctx.strokeRect(-halfW, -halfH, halfW * 2, halfH * 2);
    ctx.beginPath();
    ctx.moveTo(-8, 0); ctx.lineTo(8, 0);
    ctx.moveTo(0, -8); ctx.lineTo(0, 8);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
    ctx.stroke();
    ctx.restore();
  }

  function drawDebugWorld(ctx, frame) {
    ctx.save();
    ctx.fillStyle = "rgba(255, 60, 90, 0.95)";
    ctx.beginPath();
    ctx.arc(frame.center.x, frame.center.y, 3.5, 0, TWO_PI);
    ctx.fill();
    ctx.fillStyle = "rgba(120, 255, 140, 0.95)";
    ctx.beginPath();
    ctx.arc(s.cx, s.cy, 3.0, 0, TWO_PI);
    ctx.fill();
    ctx.restore();
  }

  function drawWarpedTexture(ctx, image, halfW, halfH) {
    const width = halfW * 2;
    const height = halfH * 2;
    const yawN = clamp(s.yaw * yawWeight, -1, 1);
    const pitchN = clamp(s.pitch * pitchWeight, -1, 1);
    const xScale = clamp(Math.cos(s.yaw * yawWeight), minYawScale, 1);
    const yScale = clamp(Math.cos(s.pitch * pitchWeight), minPitchScale, 1);
    const skewX = Math.sin(yawN) * 0.12;
    const skewY = -Math.sin(pitchN) * 0.08;
    const rows = latencyCompensation ? 18 : 1;

    ctx.save();
    ctx.scale(xScale, yScale);
    ctx.transform(1, skewY, skewX, 1, 0, 0);

    if (rows <= 1) {
      ctx.drawImage(image, -halfW, -halfH, width, height);
      ctx.restore();
      return;
    }

    const srcH = image.height || height;
    const srcW = image.width || width;
    for (let i = 0; i < rows; i += 1) {
      const v0 = i / rows;
      const v1 = (i + 1) / rows;
      const mid = (v0 + v1) * 0.5;
      const y = -halfH + v0 * height;
      const rowH = (v1 - v0) * height + 1;
      const perspectiveScale = 1 + pitchN * (mid - 0.5) * 0.22;
      const rowW = width * perspectiveScale;
      const rowX = -rowW * 0.5 + yawN * (mid - 0.5) * halfW * 0.16;
      ctx.drawImage(
        image,
        0,
        v0 * srcH,
        srcW,
        Math.max(1, (v1 - v0) * srcH),
        rowX,
        y,
        rowW,
        rowH
      );
    }
    ctx.restore();
  }

  function draw(ctx, frame) {
    if (!client.isReady()) {
      s.valid = false;
      s.cx = s.cy = s.radius = s.angle = null;
      fallback.draw(ctx, frame);
      return;
    }

    const receivedFrameId = client.getLastReceivedFrameId?.();
    const ai = client.getOutputCanvas();
    if (!ai || !ai.width) {
      fallback.draw(ctx, frame);
      return;
    }

    smoothFrame(frame);

    const drawSize = Math.max(8, Math.round(s.radius * 2 * scaleBoost));
    ensureScratch(drawSize);
    const clientStats = client.getStats?.();
    const outputFrameKey = clientStats?.textureBlendActive
      ? `${clientStats.textureFrameVersion}:${Math.floor(performance.now() / 16)}`
      : clientStats?.textureFrameVersion ?? receivedFrameId ?? 0;
    buildScratch(drawSize, outputFrameKey);

    const slideX = Math.sin(s.yaw * yawWeight) * s.radius * 0.12;
    const slideY = Math.sin(s.pitch * pitchWeight) * s.radius * 0.10;

    ctx.save();
    ctx.translate(s.cx, s.cy);
    ctx.rotate(s.angle);
    ctx.translate(calibration.offsetX + slideX, calibration.offsetY + slideY);

    const halfW = drawSize * 0.5;
    const halfH = (drawSize * ovalAspectY) * 0.5;
    drawWarpedTexture(ctx, scratch, halfW, halfH);

    // Rim shadow grounds the AI portrait against the body silhouette.
    if (c.rimShadowStrength > 0.001) {
      ctx.globalCompositeOperation = "source-over";
      ctx.beginPath();
      ctx.ellipse(0, 0, halfW * 0.99, halfH * 0.99, 0, 0, TWO_PI);
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = `rgba(0,0,0,${c.rimShadowStrength.toFixed(3)})`;
      ctx.stroke();
    }

    if (debugOn) drawDebug(ctx, halfW, halfH);
    ctx.restore();
    if (debugOn) drawDebugWorld(ctx, frame);

    if (frame.headPose) frame.headPose.__consumedBy = "ai";
  }

  function reset() {
    s.valid = false;
    s.cx = s.cy = s.radius = s.angle = null;
    s.yaw = 0; s.pitch = 0;
    motion.lastT = 0;
    motion.last = null;
    motion.vx = motion.vy = motion.vr = motion.va = motion.vyaw = motion.vpitch = 0;
    spring.lastT = 0;
    spring.vx = spring.vy = spring.vr = spring.va = spring.vyaw = spring.vpitch = 0;
    scratchFrameKey = -1;
    if (typeof fallback.reset === "function") fallback.reset();
  }

  function dispose() {
    if (typeof fallback.dispose === "function") fallback.dispose();
  }

  function setCalibration({ offsetX: ox, offsetY: oy } = {}) {
    if (Number.isFinite(ox)) calibration.offsetX = ox;
    if (Number.isFinite(oy)) calibration.offsetY = oy;
  }
  function getCalibration() { return { ...calibration }; }

  // Live color knobs — assign new TARGET values; the smoother eases
  // toward them so the change is anti-flicker.
  function setColor(updates = {}) {
    if (Number.isFinite(updates.brightness)) brightness = updates.brightness;
    if (Number.isFinite(updates.contrast)) contrast = updates.contrast;
    if (Number.isFinite(updates.saturation)) saturation = updates.saturation;
    if (Number.isFinite(updates.toneStrength)) toneStrength = updates.toneStrength;
    if (Number.isFinite(updates.chinShadowStrength)) chinShadowStrength = updates.chinShadowStrength;
    if (Number.isFinite(updates.rimShadowStrength)) rimShadowStrength = updates.rimShadowStrength;
    if (typeof updates.toneTint === "string") toneTint = updates.toneTint;
    scratchFrameKey = -1;
  }

  function setDebug(on) { debugOn = !!on; }
  function getDebug() { return debugOn; }

  return {
    draw, reset, dispose,
    setCalibration, getCalibration,
    setColor,
    setDebug, getDebug
  };
}

// ----------------------------------------------------------------
// Module-level helpers
// ----------------------------------------------------------------

let _filterSupport = null;
function hasFilterSupport() {
  if (_filterSupport !== null) return _filterSupport;
  try {
    const c = document.createElement("canvas").getContext("2d");
    _filterSupport = "filter" in c;
  } catch (_) {
    _filterSupport = false;
  }
  return _filterSupport;
}

// Apply a multiplier to the alpha channel of an `rgba(...)` or `rgb(...)`
// or `#hex` color string. Used so toneStrength can ease without
// re-parsing colors per frame.
function applyAlpha(color, alpha) {
  if (typeof color !== "string") return `rgba(0,0,0,${alpha})`;
  const a = Math.max(0, Math.min(1, alpha));
  const m = color.match(/^rgba?\(([^)]+)\)$/i);
  if (m) {
    const parts = m[1].split(",").map((s) => s.trim());
    const [r, g, b] = parts;
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }
  if (color.startsWith("#")) {
    let hex = color.slice(1);
    if (hex.length === 3) hex = hex.split("").map((ch) => ch + ch).join("");
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }
  return color;
}
