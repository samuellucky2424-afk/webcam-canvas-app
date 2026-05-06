/**
 * Sync buffer.
 *
 * The AI face pipeline (driver upload → GPU inference → JPEG download →
 * decode) introduces ~80–120 ms of latency relative to the local body
 * skeleton. If the body renders from "now" while the face shows the user's
 * expression from "now − 100 ms", lips, head turns, and shoulders desync
 * visibly.
 *
 * This buffer holds recent state snapshots in a circular ring keyed by
 * `frameId` and `timestamp`. The renderer asks for a body state delayed
 * by `currentDelayMs`, and the buffer:
 *
 *   1. Frame-id matches first when the AI client has pinned a frame id —
 *      that frame's timestamp is the ground truth for AI lag and is fed
 *      into the latency estimator.
 *   2. Time-matches otherwise: locates the two bracketing snapshots and
 *      LINEARLY INTERPOLATES landmarks (pose, hands, faces) and head pose
 *      so the body moves continuously rather than snapping between
 *      tracker samples.
 *   3. Adapts continuously: `currentDelayMs` eases toward the smoothed
 *      `measuredLatencyMs`, clamped to [minDelayMs, maxDelayMs] (default
 *      80–120 ms). The easing rate (`driftCorrectionPerSec`) is small so
 *      drift correction is invisible.
 *
 * Hot-path allocations are avoided: landmark arrays in the interpolated
 * overlay are reused across frames and only grown when the source shape
 * changes.
 */

const DEFAULT_CAPACITY = 90; // ~3 s at 30 FPS

export function createSyncBuffer({
  capacity = DEFAULT_CAPACITY,
  delayMs = 100,
  minDelayMs = 80,
  maxDelayMs = 120,
  // Smoothing for the measured AI latency (per pin event).
  latencyAlpha = 0.2,
  // Maximum delay correction in ms per second of wall-clock time.
  // Small value → drift correction is imperceptible.
  driftCorrectionPerSec = 50
} = {}) {
  const ring = new Array(capacity).fill(null).map(() => ({
    frameId: -1, t: 0, state: null
  }));
  let head = 0;     // next write position
  let count = 0;
  let pinnedFrameId = -1;
  let pinnedAt = 0;

  // Latency / delay state
  let measuredLatencyMs = delayMs;
  let currentDelayMs = clamp(delayMs, minDelayMs, maxDelayMs);
  let targetDelayMs = currentDelayMs;
  let lastDriftTickT = 0;

  // Debug stats
  const stats = {
    currentDelayMs,
    targetDelayMs,
    measuredLatencyMs,
    lastMatchMode: "none", // "pinned" | "interpolated" | "newest" | "oldest" | "none"
    syncOffsetMs: 0,        // (target time − chosen sample time)
    interpolationT: 0,      // 0..1 alpha used in last interpolation
    bracketGapMs: 0,        // time gap between bracketing samples
    pushedCount: 0
  };

  // Reusable interpolation scratch (one overlay object, mutated per call).
  const scratch = makeScratchOverlay();

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
  function lerp(a, b, t) { return a + (b - a) * t; }

  function push(state) {
    if (!state) return;
    const slot = ring[head];
    slot.frameId = state.frameId;
    slot.t = state.timestamp;
    slot.state = state;
    head = (head + 1) % capacity;
    if (count < capacity) count += 1;
    stats.pushedCount += 1;
  }

  /**
   * Called by the AI face client when a server reply lands. The pinned
   * frameId's timestamp is the ground truth for the round-trip latency,
   * which feeds the smoothed estimator and (eventually) the chosen
   * playback delay.
   */
  function pinFrameId(frameId) {
    pinnedFrameId = frameId;
    pinnedAt = performance.now();
    const slot = findSlotByFrameId(frameId);
    if (slot) {
      const observed = pinnedAt - slot.t;
      // Reject obviously wrong samples (e.g. very stale or negative).
      if (observed > 5 && observed < 2000) {
        measuredLatencyMs = lerp(measuredLatencyMs, observed, latencyAlpha);
        targetDelayMs = clamp(measuredLatencyMs, minDelayMs, maxDelayMs);
      }
    }
  }

  function clearPin() {
    pinnedFrameId = -1;
    pinnedAt = 0;
    targetDelayMs = clamp(delayMs, minDelayMs, maxDelayMs);
  }

  function findSlotByFrameId(frameId) {
    for (let i = 0; i < count; i += 1) {
      const slot = ring[i];
      if (slot.frameId === frameId) return slot;
    }
    return null;
  }

  function getNewestSlot() {
    if (count === 0) return null;
    return ring[(head - 1 + capacity) % capacity];
  }

  function getOldestSlot() {
    if (count === 0) return null;
    if (count < capacity) return ring[0];
    return ring[head]; // oldest is at head when full
  }

  /**
   * Find slots A (older or equal) and B (newer or equal) bracketing the
   * target time. Returns { a, b, t } where t is the interpolation alpha
   * along [a.t .. b.t]. If only one side is available, returns the same
   * slot for both with t=0.
   */
  function bracket(targetT) {
    if (count === 0) return null;
    let a = null, b = null;
    for (let i = 0; i < count; i += 1) {
      const slot = ring[i];
      if (!slot.state) continue;
      if (slot.t <= targetT) {
        if (!a || slot.t > a.t) a = slot;
      }
      if (slot.t >= targetT) {
        if (!b || slot.t < b.t) b = slot;
      }
    }
    if (!a && !b) return null;
    if (!a) return { a: b, b, t: 0 };
    if (!b) return { a, b: a, t: 0 };
    if (a === b) return { a, b, t: 0 };
    const span = b.t - a.t;
    const t = span > 0 ? clamp((targetT - a.t) / span, 0, 1) : 0;
    return { a, b, t, span };
  }

  // ---------- drift correction ----------
  function tickDrift(now) {
    if (!lastDriftTickT) { lastDriftTickT = now; return; }
    const dt = (now - lastDriftTickT) / 1000;
    lastDriftTickT = now;
    if (dt <= 0) return;
    const maxStep = driftCorrectionPerSec * dt;
    const diff = targetDelayMs - currentDelayMs;
    if (Math.abs(diff) <= maxStep) currentDelayMs = targetDelayMs;
    else currentDelayMs += Math.sign(diff) * maxStep;
  }

  // ---------- main accessor ----------
  /**
   * Return the body state to render this frame.
   *
   * Strategy:
   *   - If a frameId is pinned, prefer the snapshot with that exact id
   *     (acknowledging it may be slightly off the smoothed delay).
   *   - Otherwise, interpolate between bracketing snapshots at
   *     `now − currentDelayMs`.
   */
  function getRenderState() {
    const now = performance.now();
    tickDrift(now);

    if (count === 0) {
      stats.lastMatchMode = "none";
      return null;
    }

    // Pinned-frame fast path.
    if (pinnedFrameId > 0) {
      const slot = findSlotByFrameId(pinnedFrameId);
      if (slot) {
        stats.lastMatchMode = "pinned";
        stats.syncOffsetMs = (now - currentDelayMs) - slot.t;
        stats.interpolationT = 0;
        stats.bracketGapMs = 0;
        return slot.state;
      }
    }

    // Time-based bracket + interpolation.
    const target = now - currentDelayMs;
    const br = bracket(target);
    if (!br) {
      const newest = getNewestSlot();
      stats.lastMatchMode = "newest";
      stats.syncOffsetMs = newest ? target - newest.t : 0;
      stats.interpolationT = 0;
      stats.bracketGapMs = 0;
      return newest?.state ?? null;
    }

    if (br.a === br.b || !(br.span > 0)) {
      stats.lastMatchMode = br.a === getOldestSlot() ? "oldest" : "newest";
      stats.syncOffsetMs = target - br.a.t;
      stats.interpolationT = 0;
      stats.bracketGapMs = 0;
      return br.a.state;
    }

    stats.lastMatchMode = "interpolated";
    stats.syncOffsetMs = target - lerp(br.a.t, br.b.t, br.t);
    stats.interpolationT = br.t;
    stats.bracketGapMs = br.span;

    return interpolateState(scratch, br.a.state, br.b.state, br.t);
  }

  function getNewest() {
    return getNewestSlot()?.state ?? null;
  }

  function getStats() {
    stats.currentDelayMs = currentDelayMs;
    stats.targetDelayMs = targetDelayMs;
    stats.measuredLatencyMs = measuredLatencyMs;
    return stats;
  }

  function reset() {
    for (const slot of ring) {
      slot.frameId = -1;
      slot.t = 0;
      slot.state = null;
    }
    head = 0;
    count = 0;
    pinnedFrameId = -1;
    pinnedAt = 0;
    measuredLatencyMs = delayMs;
    currentDelayMs = clamp(delayMs, minDelayMs, maxDelayMs);
    targetDelayMs = currentDelayMs;
    lastDriftTickT = 0;
    stats.lastMatchMode = "none";
    stats.syncOffsetMs = 0;
    stats.interpolationT = 0;
    stats.bracketGapMs = 0;
  }

  return {
    push,
    pinFrameId,
    clearPin,
    getRenderState,
    getNewest,
    getStats,
    reset
  };
}

// ============================================================
// Interpolation helpers — kept module-level so they don't close
// over the per-instance ring and stay JIT-friendly.
// ============================================================

function makeScratchOverlay() {
  return {
    landmarks: [],
    hands: [],
    faces: [],
    skeleton: null,
    headPose: { roll: 0, yaw: 0, pitch: 0, eyeOpen: 1, mouthOpen: 0, smile: 0, confidence: 0 },
    __interpolated: true
  };
}

function ensureArr(arr, len, factory) {
  while (arr.length < len) arr.push(factory());
  arr.length = len;
}

function lerpN(a, b, t) { return a + (b - a) * t; }

function interpLandmark(out, a, b, t) {
  out.x = lerpN(a.x ?? 0, b.x ?? 0, t);
  out.y = lerpN(a.y ?? 0, b.y ?? 0, t);
  out.z = lerpN(a.z ?? 0, b.z ?? 0, t);
  out.visibility = lerpN(a.visibility ?? 0, b.visibility ?? 0, t);
  return out;
}

function interpLandmarkArray(outArr, a, b, t) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
    // Shape mismatch — copy the newer side verbatim.
    const src = b ?? a ?? [];
    ensureArr(outArr, src.length, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
    for (let i = 0; i < src.length; i += 1) {
      const s = src[i];
      const o = outArr[i];
      o.x = s.x ?? 0; o.y = s.y ?? 0; o.z = s.z ?? 0; o.visibility = s.visibility ?? 0;
    }
    return;
  }
  ensureArr(outArr, a.length, () => ({ x: 0, y: 0, z: 0, visibility: 0 }));
  for (let i = 0; i < a.length; i += 1) {
    interpLandmark(outArr[i], a[i], b[i], t);
  }
}

function interpHands(outArr, ha, hb, t) {
  // Use the newer side's structure (handedness, connections) and
  // interpolate landmarks when shapes match exactly.
  if (!Array.isArray(hb)) { outArr.length = 0; return; }
  ensureArr(outArr, hb.length, () => ({ landmarks: [], connections: null, handedness: null }));
  for (let i = 0; i < hb.length; i += 1) {
    const slot = outArr[i];
    const newer = hb[i];
    const older = ha?.[i];
    slot.handedness = newer.handedness ?? null;
    slot.connections = newer.connections ?? null;
    if (older && Array.isArray(older.landmarks) && older.landmarks.length === newer.landmarks.length) {
      slot.landmarks = slot.landmarks || [];
      interpLandmarkArray(slot.landmarks, older.landmarks, newer.landmarks, t);
    } else {
      slot.landmarks = newer.landmarks;
    }
  }
}

function interpFaces(outArr, fa, fb, t) {
  if (!Array.isArray(fb)) { outArr.length = 0; return; }
  ensureArr(outArr, fb.length, () => ({ landmarks: [], connections: null }));
  for (let i = 0; i < fb.length; i += 1) {
    const slot = outArr[i];
    const newer = fb[i];
    const older = fa?.[i];
    slot.connections = newer.connections ?? null;
    if (older && Array.isArray(older.landmarks) && older.landmarks.length === newer.landmarks.length) {
      slot.landmarks = slot.landmarks || [];
      interpLandmarkArray(slot.landmarks, older.landmarks, newer.landmarks, t);
    } else {
      slot.landmarks = newer.landmarks;
    }
  }
}

function interpHeadPose(out, a, b, t) {
  if (!a || !b) {
    const src = b || a;
    if (!src) return null;
    out.roll = src.roll ?? 0;
    out.yaw = src.yaw ?? 0;
    out.pitch = src.pitch ?? 0;
    out.eyeOpen = src.eyeOpen ?? 1;
    out.mouthOpen = src.mouthOpen ?? 0;
    out.smile = src.smile ?? 0;
    out.confidence = src.confidence ?? 0;
    return out;
  }
  out.roll = lerpN(a.roll ?? 0, b.roll ?? 0, t);
  out.yaw = lerpN(a.yaw ?? 0, b.yaw ?? 0, t);
  out.pitch = lerpN(a.pitch ?? 0, b.pitch ?? 0, t);
  out.eyeOpen = lerpN(a.eyeOpen ?? 1, b.eyeOpen ?? 1, t);
  out.mouthOpen = lerpN(a.mouthOpen ?? 0, b.mouthOpen ?? 0, t);
  out.smile = lerpN(a.smile ?? 0, b.smile ?? 0, t);
  out.confidence = lerpN(a.confidence ?? 0, b.confidence ?? 0, t);
  return out;
}

function interpolateState(scratch, sa, sb, t) {
  // Build an interpolated `state` object whose `overlay` shape is what
  // the renderer's `getOverlay()` consumer expects (landmarks + hands +
  // faces in source space).
  const oa = sa?.overlay ?? null;
  const ob = sb?.overlay ?? null;
  if (!oa && !ob) return sb || sa;

  // Pose landmarks — primary signal for the body rig.
  interpLandmarkArray(scratch.landmarks,
    oa?.landmarks ?? [], ob?.landmarks ?? [], t);

  interpHands(scratch.hands, oa?.hands, ob?.hands, t);
  interpFaces(scratch.faces, oa?.faces, ob?.faces, t);

  // Skeleton is rebuilt from landmarks downstream by the renderer, so we
  // do not need to interpolate it explicitly. Pass through the newer.
  scratch.skeleton = ob?.skeleton ?? oa?.skeleton ?? null;

  interpHeadPose(scratch.headPose, sa?.headPose, sb?.headPose, t);

  // Compose the returned state — mirror the unified state shape so
  // downstream consumers (renderer, AI client) see the same fields.
  return {
    timestamp: lerpN(sa.timestamp ?? 0, sb.timestamp ?? 0, t),
    frameId: sb.frameId ?? sa.frameId ?? -1,
    overlay: scratch,
    skeleton: scratch.skeleton,
    hands: scratch.hands,
    headPose: scratch.headPose,
    faceSignals: sb.faceSignals ?? sa.faceSignals ?? null,
    hasFace: (sb.hasFace ?? false) || (sa.hasFace ?? false),
    __interpolated: true
  };
}
