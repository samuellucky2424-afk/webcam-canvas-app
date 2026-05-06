import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

function disposeMaterial(material) {
  if (!material) return;
  if (Array.isArray(material)) {
    for (const m of material) disposeMaterial(m);
    return;
  }
  for (const key of Object.keys(material)) {
    const value = material[key];
    if (value && typeof value === "object" && typeof value.dispose === "function") {
      value.dispose();
    }
  }
  material.dispose?.();
}

function disposeObject3d(object) {
  if (!object) return;
  object.traverse((child) => {
    if (child.geometry?.dispose) child.geometry.dispose();
    if (child.material) disposeMaterial(child.material);
  });
}

function frameCamera(camera, object, { coverage = 0.65, yFocus = 0.25, distanceMult = 1.15 } = {}) {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const height = Math.max(size.y, 1e-3);

  object.position.sub(center);

  const fov = (camera.fov * Math.PI) / 180;
  const targetHeight = height * coverage;
  const distance = (targetHeight / 2) / Math.tan(fov / 2);

  camera.position.set(0, height * yFocus, distance * distanceMult);
  camera.lookAt(0, height * yFocus, 0);
  camera.updateProjectionMatrix();
}

export async function createThreePreview({
  canvas,
  mode = "model",
  modelUrl,
  imageUrl,
  onStats,
  onStatus
} = {}) {
  if (!canvas) throw new Error("createThreePreview: canvas is required");

  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
    powerPreference: "high-performance"
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 0.05, 100);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x223355, 1.0);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff, 1.1);
  dir.position.set(2, 3, 2);
  scene.add(dir);

  const root = new THREE.Group();
  scene.add(root);

  const clock = new THREE.Clock();
  let mixer = null;
  let raf = 0;
  let running = false;
  let lastSampleT = 0;
  let frames = 0;

  function resize() {
    const width = Math.max(1, Math.floor(canvas.clientWidth || canvas.width));
    const height = Math.max(1, Math.floor(canvas.clientHeight || canvas.height));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  function tick(now) {
    if (!running) return;
    raf = requestAnimationFrame(tick);
    resize();
    const dt = clock.getDelta();
    mixer?.update(dt);
    renderer.render(scene, camera);

    frames += 1;
    if (!lastSampleT) lastSampleT = now;
    const elapsed = now - lastSampleT;
    if (elapsed >= 1000) {
      const fps = (frames * 1000) / elapsed;
      frames = 0;
      lastSampleT = now;
      onStats?.({
        fps,
        scale: 1,
        width: canvas.width,
        height: canvas.height
      });
    }
  }

  async function load() {
    mixer?.stopAllAction();
    mixer = null;
    disposeObject3d(root);
    root.clear();
    root.position.set(0, 0, 0);

    if (mode === "image") {
      onStatus?.("Loading 3D image...", "working");
      const loader = new THREE.TextureLoader();
      const texture = await new Promise((resolve, reject) => {
        loader.load(
          imageUrl,
          (t) => resolve(t),
          undefined,
          (err) => reject(err)
        );
      });

      texture.colorSpace = THREE.SRGBColorSpace;

      const iw = texture.image?.width ?? 1;
      const ih = texture.image?.height ?? 1;
      const aspect = iw / Math.max(ih, 1e-6);
      const height = 1.9;
      const width = height * aspect;

      const geom = new THREE.PlaneGeometry(width, height);
      const mat = new THREE.MeshBasicMaterial({ map: texture, transparent: true });
      const plane = new THREE.Mesh(geom, mat);
      plane.position.set(0, height * 0.5, 0);
      root.add(plane);
      frameCamera(camera, root, { coverage: 1.05, yFocus: 0.5, distanceMult: 1.35 });
      onStatus?.("3D image preview", "ready");
      return;
    }

    onStatus?.("Loading 3D model...", "working");
    const loader = new GLTFLoader();
    const gltf = await new Promise((resolve, reject) => {
      loader.load(
        modelUrl,
        (data) => resolve(data),
        undefined,
        (err) => reject(err)
      );
    });

    const model = gltf.scene ?? gltf.scenes?.[0];
    if (!model) {
      throw new Error("Model did not contain a scene.");
    }

    root.add(model);
    frameCamera(camera, model, { coverage: 0.65, yFocus: 0.25, distanceMult: 1.15 });

    if (Array.isArray(gltf.animations) && gltf.animations.length) {
      mixer = new THREE.AnimationMixer(model);
      const action = mixer.clipAction(gltf.animations[0]);
      action.play();
    }
    onStatus?.("3D model preview", "ready");
  }

  await load();

  function start() {
    if (running) return;
    running = true;
    clock.getDelta();
    raf = requestAnimationFrame(tick);
  }

  function stop() {
    if (!running) return;
    running = false;
    if (raf) cancelAnimationFrame(raf);
    raf = 0;
  }

  const onResize = () => resize();
  window.addEventListener("resize", onResize);
  resize();

  function dispose() {
    stop();
    window.removeEventListener("resize", onResize);
    disposeObject3d(root);
    renderer.dispose();
  }

  return { start, stop, dispose, resize };
}
