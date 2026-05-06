/**
 * Head rendering module.
 *
 * The avatar's head is drawn through a pluggable "head renderer" so it can be
 * swapped at runtime — for example, replaced by an external video stream
 * (live face) or a static image (AI-generated portrait) without touching the
 * body rendering pipeline.
 *
 * All head renderers share the same interface:
 *
 *   {
 *     draw(context, frame): void,
 *     // Optional lifecycle hooks; the avatar will call them if present.
 *     reset?(): void,
 *     dispose?(): void
 *   }
 *
 * Where `frame` is the per-frame placement computed by the body rig:
 *
 *   {
 *     center:   { x, y },        // canvas-space center of the head
 *     radius:   number,           // body-derived head radius (pixels)
 *     headAngle: number,          // rig direction (neck → head top), radians
 *     headPose: { roll, yaw, pitch, confidence } | null
 *   }
 *
 * `headAngle + PI/2` rotates a sprite so its local "up" follows the spine.
 * `headPose.roll` adds head-only rotation on top. Yaw / pitch are in radians
 * and intended for stylistic effects (gaze direction, parallax, etc.) when
 * the renderer cares; static-image renderers can ignore them.
 *
 * Pure rendering — no MediaPipe imports.
 */

// ---------------------------------------------------------------------------
// Default (drawn) humanoid head — same look as before, isolated here so it
// can be replaced wholesale.
// ---------------------------------------------------------------------------

export function createDefaultHeadRenderer({ palette, style }) {
  function draw(context, frame) {
    const { center, radius, headAngle, headPose } = frame;
    const baseRotation = headAngle + Math.PI / 2;
    const roll = headPose?.roll ?? 0;
    const yaw = headPose?.yaw ?? 0;
    const pitch = headPose?.pitch ?? 0;
    const eyeOpen = headPose?.eyeOpen ?? 1;
    const mouthOpen = headPose?.mouthOpen ?? 0;
    const smile = headPose?.smile ?? 0;

    context.save();
    context.translate(center.x, center.y);
    context.rotate(baseRotation + roll);

    const yawSquash = Math.max(Math.cos(yaw), 0.55);
    const yawShift = Math.sin(yaw) * radius * 0.18;
    const pitchShift = Math.sin(pitch) * radius * 0.45;

    const halfW = radius * yawSquash;
    const halfH = radius;

    // Skin face.
    const faceGrad = context.createRadialGradient(
      -halfW * 0.3,
      -halfH * 0.3,
      radius * 0.1,
      0,
      0,
      radius * 1.1
    );
    faceGrad.addColorStop(0, palette.skinHighlight);
    faceGrad.addColorStop(0.6, palette.skinBase);
    faceGrad.addColorStop(1, palette.skinShadow);

    context.fillStyle = faceGrad;
    context.beginPath();
    context.ellipse(0, 0, halfW, halfH, 0, 0, Math.PI * 2);
    context.fill();
    context.lineWidth = style.outlineWidth;
    context.strokeStyle = palette.outline;
    context.stroke();

    // Hair cap (clipped to head silhouette).
    context.save();
    context.beginPath();
    context.ellipse(0, 0, halfW * 1.02, halfH * 1.02, 0, 0, Math.PI * 2);
    context.clip();

    const hairGrad = context.createLinearGradient(0, -halfH, 0, 0);
    hairGrad.addColorStop(0, palette.hairHighlight);
    hairGrad.addColorStop(1, palette.hairBase);
    context.fillStyle = hairGrad;
    context.beginPath();
    context.moveTo(-halfW * 1.05, -halfH * 1.1);
    context.lineTo(-halfW * 1.05, -halfH * 0.05);
    context.bezierCurveTo(
      -halfW * 0.6, -halfH * 0.45,
      -halfW * 0.2, -halfH * 0.55,
      yawShift * 0.5, -halfH * 0.4
    );
    context.bezierCurveTo(
      halfW * 0.3, -halfH * 0.55,
      halfW * 0.7, -halfH * 0.45,
      halfW * 1.05, -halfH * 0.05
    );
    context.lineTo(halfW * 1.05, -halfH * 1.1);
    context.closePath();
    context.fill();
    context.restore();

    // Eyes & brows.
    const eyeY = -halfH * 0.18 + pitchShift * 0.6;
    const eyeOffsetX = halfW * 0.42;
    const eyeRadiusX = halfW * 0.16;
    const eyeRadiusY = halfH * 0.10;

    const browY = eyeY - halfH * 0.18;
    context.strokeStyle = palette.brow;
    context.lineWidth = Math.max(radius * 0.045, 1.5);
    context.lineCap = "round";
    context.beginPath();
    context.moveTo(-eyeOffsetX - eyeRadiusX * 0.9 + yawShift, browY + halfH * 0.02);
    context.quadraticCurveTo(
      -eyeOffsetX + yawShift, browY - halfH * 0.04,
      -eyeOffsetX + eyeRadiusX * 0.9 + yawShift, browY
    );
    context.stroke();
    context.beginPath();
    context.moveTo(eyeOffsetX - eyeRadiusX * 0.9 + yawShift, browY);
    context.quadraticCurveTo(
      eyeOffsetX + yawShift, browY - halfH * 0.04,
      eyeOffsetX + eyeRadiusX * 0.9 + yawShift, browY + halfH * 0.02
    );
    context.stroke();

    function drawEye(cx) {
      // Sclera (with closed-eye fallback when eyelids are shut).
      if (eyeOpen <= 0.08) {
        // Fully closed — draw a curved lash line.
        context.strokeStyle = palette.outline;
        context.lineWidth = Math.max(style.outlineWidth * 0.8, 1.4);
        context.beginPath();
        context.moveTo(cx + yawShift - eyeRadiusX, eyeY);
        context.quadraticCurveTo(
          cx + yawShift, eyeY + eyeRadiusY * 0.4,
          cx + yawShift + eyeRadiusX, eyeY
        );
        context.stroke();
        return;
      }

      // Eye opening height tracks blink amount (clamped to a minimum
      // squint so the iris stays readable above ~0.08).
      const openY = eyeRadiusY * Math.max(eyeOpen, 0.15);

      context.save();
      context.beginPath();
      context.ellipse(cx + yawShift, eyeY, eyeRadiusX, openY, 0, 0, Math.PI * 2);
      context.clip();

      context.fillStyle = palette.eyeWhite;
      context.fillRect(
        cx + yawShift - eyeRadiusX,
        eyeY - eyeRadiusY,
        eyeRadiusX * 2,
        eyeRadiusY * 2
      );

      const irisR = eyeRadiusY * 0.85;
      const gazeX = cx + yawShift + Math.sin(yaw) * eyeRadiusX * 0.4;
      const gazeY = eyeY + Math.sin(pitch) * eyeRadiusY * 0.4;
      context.fillStyle = palette.iris;
      context.beginPath();
      context.arc(gazeX, gazeY, irisR, 0, Math.PI * 2);
      context.fill();

      context.fillStyle = palette.pupil;
      context.beginPath();
      context.arc(gazeX, gazeY, irisR * 0.45, 0, Math.PI * 2);
      context.fill();

      context.fillStyle = palette.eyeWhite;
      context.beginPath();
      context.arc(gazeX - irisR * 0.35, gazeY - irisR * 0.35, irisR * 0.22, 0, Math.PI * 2);
      context.fill();

      context.restore();

      // Eye outline shaped to the current opening so the eyelid visibly
      // closes during a blink.
      context.strokeStyle = palette.outline;
      context.lineWidth = Math.max(style.outlineWidth * 0.6, 1);
      context.beginPath();
      context.ellipse(cx + yawShift, eyeY, eyeRadiusX, openY, 0, 0, Math.PI * 2);
      context.stroke();
    }
    drawEye(-eyeOffsetX);
    drawEye(eyeOffsetX);

    // Nose.
    const noseTopY = eyeY + halfH * 0.05;
    const noseBottomY = halfH * 0.18 + pitchShift * 0.4;
    context.strokeStyle = palette.skinShadow;
    context.lineWidth = Math.max(radius * 0.04, 1);
    context.beginPath();
    context.moveTo(yawShift * 1.6, noseTopY);
    context.quadraticCurveTo(
      yawShift * 1.6 - halfW * 0.04,
      (noseTopY + noseBottomY) / 2,
      yawShift * 1.6,
      noseBottomY
    );
    context.stroke();
    context.fillStyle = palette.skinShadow;
    context.beginPath();
    context.arc(yawShift * 1.6 - halfW * 0.04, noseBottomY, radius * 0.018, 0, Math.PI * 2);
    context.arc(yawShift * 1.6 + halfW * 0.04, noseBottomY, radius * 0.018, 0, Math.PI * 2);
    context.fill();

    // Mouth — closes/opens with detected jaw drop. When sufficiently open,
    // shows teeth (a thin white band) and a darker mouth interior behind.
    const mouthY = halfH * 0.42 + pitchShift * 0.3;
    const mouthW = halfW * 0.45;
    const mouthH = mouthOpen * halfH * 0.25;

    if (mouthOpen > 0.12) {
      context.save();
      context.translate(yawShift, mouthY);
      context.fillStyle = "#2a0e16";
      context.beginPath();
      context.ellipse(0, mouthH * 0.1, mouthW, mouthH, 0, 0, Math.PI * 2);
      context.fill();
      context.strokeStyle = palette.outline;
      context.lineWidth = Math.max(style.outlineWidth * 0.7, 1);
      context.stroke();

      context.save();
      context.beginPath();
      context.ellipse(0, mouthH * 0.1, mouthW, mouthH, 0, 0, Math.PI * 2);
      context.clip();
      context.fillStyle = "#f3ead4";
      context.fillRect(-mouthW, -mouthH * 0.55, mouthW * 2, Math.max(mouthH * 0.45, 2));
      context.strokeStyle = "rgba(120, 90, 70, 0.35)";
      context.lineWidth = 1;
      const teeth = 6;
      for (let i = 1; i < teeth; i += 1) {
        const tx = -mouthW + (i / teeth) * mouthW * 2;
        context.beginPath();
        context.moveTo(tx, -mouthH * 0.55);
        context.lineTo(tx, -mouthH * 0.1);
        context.stroke();
      }
      context.restore();

      context.strokeStyle = palette.mouth;
      context.lineWidth = Math.max(radius * 0.04, 1.5);
      context.beginPath();
      context.ellipse(0, mouthH * 0.1, mouthW, mouthH, 0, 0, Math.PI * 2);
      context.stroke();

      context.restore();
    } else {
      // Closed-mouth shape morphs from a slight frown (rest) through a
      // neutral line into a smile. The control point's y is biased up
      // (negative, since canvas y grows downward) by `smile`.
      context.strokeStyle = palette.mouth;
      context.lineWidth = Math.max(radius * 0.055, 1.5);
      context.lineCap = "round";
      // Smile widens the mouth a touch, lifts the corners, and bows the
      // line upward (positive smile -> control y goes "up").
      const smileW = mouthW * (1 + smile * 0.18);
      const cornerLift = -smile * halfH * 0.06;
      const cy = mouthY + halfH * 0.05 - smile * halfH * 0.12;
      context.beginPath();
      context.moveTo(-smileW + yawShift, mouthY + cornerLift);
      context.quadraticCurveTo(yawShift, cy, smileW + yawShift, mouthY + cornerLift);
      context.stroke();

      // Subtle cheek lift dots when smiling hard, to read as a grin even
      // when the lips don't part.
      if (smile > 0.5) {
        context.fillStyle = palette.skinShadow;
        const cheekR = Math.max(radius * 0.018, 1) * smile;
        context.beginPath();
        context.arc(-smileW * 1.05 + yawShift, mouthY + cornerLift - halfH * 0.03, cheekR, 0, Math.PI * 2);
        context.arc(smileW * 1.05 + yawShift, mouthY + cornerLift - halfH * 0.03, cheekR, 0, Math.PI * 2);
        context.fill();
      }
    }

    context.restore();
  }

  return { draw };
}

// ---------------------------------------------------------------------------
// Sprite-based head renderer — accepts any drawable source (HTMLImageElement,
// HTMLCanvasElement, HTMLVideoElement, ImageBitmap, OffscreenCanvas).
//
// This is the integration point for AI-generated face frames:
//   - A static portrait → pass an HTMLImageElement.
//   - A live face video → pass an HTMLVideoElement (e.g. a hidden <video>
//     attached to a MediaStream from getUserMedia, an AI face filter, or a
//     remote stream).
//   - An off-screen canvas where another module paints AI face frames →
//     pass that canvas. The avatar will resample it each frame.
// ---------------------------------------------------------------------------

const DEFAULT_SPRITE_OPTIONS = {
  // Multiplier on the body-derived head radius. AI face crops are usually
  // tighter than the avatar's geometric head, so 1.0 is a good default.
  scale: 1.0,
  // Aspect ratio (width / height). 1 = square crop. Override for non-square
  // sources (e.g. 0.78 for portrait).
  aspect: 1.0,
  // Vertical offset along the head's local "up" axis, in radius units. Use
  // this to nudge the AI face up/down to match the avatar's expected
  // forehead/chin alignment.
  verticalOffset: 0,
  // Clip the sprite to an oval matching the head silhouette. Set false to
  // show the full rectangular source.
  clipOval: true,
  // If true, ignore detected head pose and draw upright in the head's local
  // frame. Useful when the source already encodes its own rotation.
  ignoreHeadPose: false
};

export function createSpriteHeadRenderer({ source, ...options } = {}) {
  let currentSource = source ?? null;
  const opts = { ...DEFAULT_SPRITE_OPTIONS, ...options };

  function isReady(src) {
    if (!src) return false;
    if (src instanceof HTMLVideoElement) {
      return src.readyState >= 2 && src.videoWidth > 0 && src.videoHeight > 0;
    }
    if (src instanceof HTMLImageElement) {
      return src.complete && src.naturalWidth > 0;
    }
    // HTMLCanvasElement, OffscreenCanvas, ImageBitmap — assume drawable when
    // a non-zero size is reported.
    const w = src.width ?? src.videoWidth ?? 0;
    const h = src.height ?? src.videoHeight ?? 0;
    return w > 0 && h > 0;
  }

  function setSource(next) {
    currentSource = next ?? null;
  }

  function draw(context, frame) {
    if (!isReady(currentSource)) return;

    const { center, radius, headAngle, headPose } = frame;
    const baseRotation = headAngle + Math.PI / 2;
    const roll = opts.ignoreHeadPose ? 0 : headPose?.roll ?? 0;

    const drawRadius = radius * opts.scale;
    const halfH = drawRadius;
    const halfW = drawRadius * opts.aspect;

    context.save();
    context.translate(center.x, center.y);
    context.rotate(baseRotation + roll);
    context.translate(0, opts.verticalOffset * radius);

    if (opts.clipOval) {
      context.beginPath();
      context.ellipse(0, 0, halfW, halfH, 0, 0, Math.PI * 2);
      context.clip();
    }

    try {
      context.drawImage(currentSource, -halfW, -halfH, halfW * 2, halfH * 2);
    } catch {
      // Source became invalid mid-frame (e.g. video tore down). Silently
      // skip — next frame will retry the readiness check.
    }

    context.restore();
  }

  return { draw, setSource };
}

// ---------------------------------------------------------------------------
// Factory dispatcher used by the avatar to pick a renderer from config.
// ---------------------------------------------------------------------------

/**
 * Resolves a head renderer from a config descriptor.
 *
 * Descriptor shapes:
 *   { type: "default" }                 → drawn humanoid head (uses palette/style)
 *   { type: "image", source: HTMLImageElement, ...spriteOptions }
 *   { type: "video", source: HTMLVideoElement, ...spriteOptions }
 *   { type: "canvas", source: HTMLCanvasElement | OffscreenCanvas, ...spriteOptions }
 *
 * If `descriptor` is null/undefined or `type` is unknown, falls back to the
 * default head renderer.
 *
 * `palette` and `style` are forwarded to the default renderer; sprite-based
 * renderers ignore them.
 */
export function resolveHeadRenderer(descriptor, { palette, style }) {
  const type = descriptor?.type ?? "default";

  if (type === "image" || type === "video" || type === "canvas") {
    return createSpriteHeadRenderer({
      source: descriptor.source,
      scale: descriptor.scale,
      aspect: descriptor.aspect,
      verticalOffset: descriptor.verticalOffset,
      clipOval: descriptor.clipOval,
      ignoreHeadPose: descriptor.ignoreHeadPose
    });
  }

  return createDefaultHeadRenderer({ palette, style });
}
