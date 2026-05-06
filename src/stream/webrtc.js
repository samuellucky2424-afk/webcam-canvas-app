/**
 * WebRTC publisher — streams the rendered avatar canvas to a WHIP-style
 * signaling endpoint over a single RTCPeerConnection.
 *
 * Server side is intentionally not opinionated here: any endpoint that
 * accepts an SDP offer (Content-Type application/sdp) and returns an SDP
 * answer will work. This matches the WHIP draft used by mediasoup,
 * Galène, Janus, LiveKit ingest, aiortc, etc.
 *
 * Low-latency pipeline:
 *  - `canvas.captureStream(targetFps)` → MediaStreamTrack at native colour.
 *  - `addTransceiver('video', {direction:'sendonly'})` so we don't
 *    negotiate a recvonly channel by accident.
 *  - SDP munging swaps `useinbandfec=1` and bumps start bitrate via
 *    `b=AS:` to encourage the encoder to ramp fast.
 *  - `RTCRtpSender.setParameters` enables `degradationPreference:
 *    "maintain-framerate"` so the encoder drops resolution before frames.
 *
 * Server-side NVENC / hardware encode happens in the receiver — the
 * browser side just emits VP8/VP9/H264 according to the SDP answer.
 */

const DEFAULT_BITRATE_KBPS = 2500;

export function createWebRtcPublisher({
  endpoint,
  targetFps = 30,
  bitrateKbps = DEFAULT_BITRATE_KBPS,
  preferCodec = "VP9"
} = {}) {
  if (!endpoint) throw new Error("createWebRtcPublisher: endpoint is required");

  let pc = null;
  let stream = null;
  let track = null;
  let status = "idle"; // idle | connecting | live | failed | closed
  const listeners = new Set();

  function setStatus(s) {
    if (s === status) return;
    status = s;
    for (const cb of listeners) {
      try { cb(status); } catch (_) { /* ignore */ }
    }
  }

  function preferCodecInSdp(sdp, codec) {
    const lines = sdp.split("\r\n");
    const mLineIdx = lines.findIndex((l) => l.startsWith("m=video"));
    if (mLineIdx < 0) return sdp;

    const codecRe = new RegExp(`a=rtpmap:(\\d+) ${codec}/`, "i");
    const wanted = [];
    for (const l of lines) {
      const m = l.match(codecRe);
      if (m) wanted.push(m[1]);
    }
    if (!wanted.length) return sdp;

    const mParts = lines[mLineIdx].split(" ");
    const header = mParts.slice(0, 3);
    const rest = mParts.slice(3);
    const reordered = [...wanted, ...rest.filter((p) => !wanted.includes(p))];
    lines[mLineIdx] = [...header, ...reordered].join(" ");
    return lines.join("\r\n");
  }

  function setBitrateInSdp(sdp, kbps) {
    const lines = sdp.split("\r\n");
    const out = [];
    for (const l of lines) {
      out.push(l);
      if (l.startsWith("m=video")) {
        out.push("b=AS:" + kbps);
        out.push("b=TIAS:" + (kbps * 1000));
      }
    }
    return out.join("\r\n");
  }

  async function start(canvas) {
    if (pc) return;
    if (!canvas || typeof canvas.captureStream !== "function") {
      throw new Error("createWebRtcPublisher.start: canvas with captureStream() required");
    }
    setStatus("connecting");

    stream = canvas.captureStream(targetFps);
    [track] = stream.getVideoTracks();
    if (track && "contentHint" in track) track.contentHint = "motion";

    pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      bundlePolicy: "max-bundle"
    });

    const transceiver = pc.addTransceiver(track, { direction: "sendonly", streams: [stream] });

    // Bias encoder toward smooth motion over crisp resolution.
    try {
      const sender = transceiver.sender;
      const params = sender.getParameters();
      if (!params.encodings) params.encodings = [{}];
      params.encodings[0].maxBitrate = bitrateKbps * 1000;
      params.encodings[0].maxFramerate = targetFps;
      params.degradationPreference = "maintain-framerate";
      await sender.setParameters(params);
    } catch (_) { /* not all browsers expose setParameters */ }

    pc.addEventListener("connectionstatechange", () => {
      if (!pc) return;
      if (pc.connectionState === "connected") setStatus("live");
      else if (pc.connectionState === "failed") setStatus("failed");
      else if (pc.connectionState === "closed") setStatus("closed");
    });

    const offer = await pc.createOffer({ offerToReceiveVideo: false, offerToReceiveAudio: false });
    let munged = offer.sdp;
    munged = preferCodecInSdp(munged, preferCodec);
    munged = setBitrateInSdp(munged, bitrateKbps);
    await pc.setLocalDescription({ type: offer.type, sdp: munged });

    const resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/sdp" },
      body: pc.localDescription.sdp
    });
    if (!resp.ok) {
      setStatus("failed");
      stop();
      throw new Error("WebRTC signaling failed: " + resp.status);
    }
    const answerSdp = await resp.text();
    await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });
  }

  function stop() {
    if (track) {
      try { track.stop(); } catch (_) { /* ignore */ }
      track = null;
    }
    if (stream) {
      for (const t of stream.getTracks()) try { t.stop(); } catch (_) { /* ignore */ }
      stream = null;
    }
    if (pc) {
      try { pc.close(); } catch (_) { /* ignore */ }
      pc = null;
    }
    if (status !== "failed") setStatus("closed");
  }

  function onStatusChange(cb) {
    listeners.add(cb);
    return () => listeners.delete(cb);
  }

  return {
    start,
    stop,
    onStatusChange,
    getStatus: () => status,
    getTrack: () => track
  };
}
