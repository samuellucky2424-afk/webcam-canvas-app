export function createVirtualCamera({
  canvas,
  targetFps = 30,
  onStatusChange
} = {}) {
  let stream = null;
  let track = null;
  let status = "idle";

  function setStatus(next) {
    if (status === next) return;
    status = next;
    onStatusChange?.(status);
  }

  function start() {
    if (stream) return stream;
    if (!canvas || typeof canvas.captureStream !== "function") {
      setStatus("unsupported");
      return null;
    }

    stream = canvas.captureStream(targetFps);
    [track] = stream.getVideoTracks();
    if (track && "contentHint" in track) {
      track.contentHint = "motion";
    }
    window.__virtualCameraStream = stream;
    setStatus(track ? "ready" : "unsupported");
    return stream;
  }

  function stop() {
    if (track) {
      try { track.stop(); } catch (_) { /* ignore */ }
      track = null;
    }
    if (stream) {
      for (const mediaTrack of stream.getTracks()) {
        try { mediaTrack.stop(); } catch (_) { /* ignore */ }
      }
      stream = null;
    }
    if (window.__virtualCameraStream) {
      window.__virtualCameraStream = null;
    }
    setStatus("closed");
  }

  return {
    start,
    stop,
    getStatus: () => status,
    getStream: () => stream,
    getTrack: () => track
  };
}
