function waitForMetadata(video) {
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
    return Promise.resolve();
  }

  return new Promise((resolve) => {
    video.addEventListener("loadedmetadata", resolve, { once: true });
  });
}

export async function startCamera(video, constraints) {
  const stream = await navigator.mediaDevices.getUserMedia(constraints);

  video.srcObject = stream;
  await waitForMetadata(video);
  await video.play();

  return stream;
}

export function stopCamera(stream) {
  stream?.getTracks().forEach((track) => {
    track.stop();
  });
}