import init, { PhysicsWorld3D } from "../../src/wasm/rubble_wasm.js";
import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Three.js WebGPURenderer stores per-instance transforms in a uniform buffer,
// so keep this comfortably below the browser's 64 KiB binding limit.
const MAX_INSTANCES = 1024;
const DEMO_SEED = 0x3d5eed;

const SPHERE_COLORS = [0xff6b35, 0xf7c948, 0x4ecdc4, 0x45b7d1, 0x96ceb4];
const BOX_COLORS = [0xff6f69, 0xffcc5c, 0x88d8b0, 0xc3aed6, 0xffd166];

function createRng(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rng = createRng(DEMO_SEED);

let world: PhysicsWorld3D;

// Track per-body info for rendering
interface BodyInfo {
  type: number; // 0=sphere, 1=box, 2=capsule, 99=plane
  instanceIndex: number;
  radius?: number;
  halfExtents?: [number, number, number];
}
const bodies: BodyInfo[] = [];
let sphereCount = 0;
let boxCount = 0;
let controls: OrbitControls | null = null;

// Three.js setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
scene.fog = new THREE.Fog(0x111111, 40, 80);

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  200,
);
camera.position.set(15, 12, 20);
camera.lookAt(0, 3, 0);

let renderer: THREE.WebGLRenderer | WebGPURenderer;
let renderBackendLabel = "WebGL";
let frameInFlight = false;

async function initRenderer() {
  try {
    const webgpuRenderer = new WebGPURenderer({ antialias: true });
    await webgpuRenderer.init();
    renderer = webgpuRenderer;
    renderBackendLabel =
      "isWebGPUBackend" in renderer.backend &&
      (renderer.backend as { isWebGPUBackend?: boolean }).isWebGPUBackend
        ? "WebGPU"
        : "WebGL2";
  } catch (error) {
    console.warn("Falling back to WebGL renderer", error);
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderBackendLabel = "WebGL";
  }
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  document.body.appendChild(renderer.domElement);
}

// Lighting
const ambientLight = new THREE.AmbientLight(0x404040, 1.0);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 2.0);
dirLight.position.set(10, 20, 10);
dirLight.castShadow = true;
dirLight.shadow.camera.left = -20;
dirLight.shadow.camera.right = 20;
dirLight.shadow.camera.top = 20;
dirLight.shadow.camera.bottom = -20;
dirLight.shadow.mapSize.width = 2048;
dirLight.shadow.mapSize.height = 2048;
scene.add(dirLight);

const fillLight = new THREE.DirectionalLight(0x6688cc, 0.5);
fillLight.position.set(-5, 10, -5);
scene.add(fillLight);

// Ground plane
const groundGeo = new THREE.PlaneGeometry(100, 100);
const groundMat = new THREE.MeshStandardMaterial({
  color: 0x222222,
  roughness: 0.8,
  metalness: 0.2,
});
const groundMesh = new THREE.Mesh(groundGeo, groundMat);
groundMesh.rotation.x = -Math.PI / 2;
groundMesh.receiveShadow = true;
scene.add(groundMesh);

// Grid helper
const grid = new THREE.GridHelper(100, 100, 0x333333, 0x222222);
grid.position.y = 0.01;
scene.add(grid);

// Instanced meshes for spheres and boxes
const sphereGeo = new THREE.SphereGeometry(1, 24, 24);
const sphereMat = new THREE.MeshStandardMaterial({
  roughness: 0.4,
  metalness: 0.3,
});
const sphereInstances = new THREE.InstancedMesh(
  sphereGeo,
  sphereMat,
  MAX_INSTANCES,
);
sphereInstances.castShadow = true;
sphereInstances.receiveShadow = true;
sphereInstances.count = 0;
scene.add(sphereInstances);

// Per-instance colors
const sphereColorAttr = new Float32Array(MAX_INSTANCES * 3);
const boxColorAttr = new Float32Array(MAX_INSTANCES * 3);
sphereInstances.instanceColor = new THREE.InstancedBufferAttribute(
  sphereColorAttr,
  3,
);

const boxGeo = new THREE.BoxGeometry(1, 1, 1);
const boxMat = new THREE.MeshStandardMaterial({
  roughness: 0.5,
  metalness: 0.2,
});
const boxInstances = new THREE.InstancedMesh(boxGeo, boxMat, MAX_INSTANCES);
boxInstances.castShadow = true;
boxInstances.receiveShadow = true;
boxInstances.count = 0;
boxInstances.instanceColor = new THREE.InstancedBufferAttribute(boxColorAttr, 3);
scene.add(boxInstances);

const tempMatrix = new THREE.Matrix4();
const tempQuat = new THREE.Quaternion();
const tempPos = new THREE.Vector3();
const tempScale = new THREE.Vector3();
const tempColor = new THREE.Color();

function addSphere(x: number, y: number, z: number, radius: number, mass: number) {
  const idx = world.add_sphere(x, y, z, radius, mass);
  const renderable = sphereCount < MAX_INSTANCES;

  if (renderable) {
    const color = SPHERE_COLORS[sphereCount % SPHERE_COLORS.length];
    tempColor.set(color);
    sphereColorAttr[sphereCount * 3] = tempColor.r;
    sphereColorAttr[sphereCount * 3 + 1] = tempColor.g;
    sphereColorAttr[sphereCount * 3 + 2] = tempColor.b;
  }

  bodies.push({
    type: 0,
    instanceIndex: renderable ? sphereCount : -1,
    radius,
  });
  if (renderable) {
    sphereCount++;
    sphereInstances.count = sphereCount;
    sphereInstances.instanceColor!.needsUpdate = true;
  }
  return idx;
}

function addBox(
  x: number,
  y: number,
  z: number,
  hw: number,
  hh: number,
  hd: number,
  mass: number,
) {
  const idx = world.add_box(x, y, z, hw, hh, hd, mass);
  const renderable = boxCount < MAX_INSTANCES;

  if (renderable) {
    const color = BOX_COLORS[boxCount % BOX_COLORS.length];
    tempColor.set(color);
    boxColorAttr[boxCount * 3] = tempColor.r;
    boxColorAttr[boxCount * 3 + 1] = tempColor.g;
    boxColorAttr[boxCount * 3 + 2] = tempColor.b;
  }

  bodies.push({
    type: 1,
    instanceIndex: renderable ? boxCount : -1,
    halfExtents: [hw, hh, hd],
  });
  if (renderable) {
    boxCount++;
    boxInstances.count = boxCount;
    boxInstances.instanceColor!.needsUpdate = true;
  }
  return idx;
}

function spawnRandomBody(x: number, y: number, z: number) {
  const isSphere = rng() > 0.4;
  if (isSphere) {
    const r = 0.3 + rng() * 0.5;
    addSphere(x, y, z, r, 1.0);
  } else {
    const hw = 0.3 + rng() * 0.5;
    const hh = 0.3 + rng() * 0.5;
    const hd = 0.3 + rng() * 0.5;
    addBox(x, y, z, hw, hh, hd, 1.0);
  }
  ensureTransformBuffer();
}

function ensureTransformBuffer() {
  const handleCount = world.handle_count();
  if (cachedTransforms.length !== handleCount * 7) {
    cachedTransforms = new Float32Array(handleCount * 7);
  }
}

function syncTransformCache() {
  ensureTransformBuffer();
  if (cachedTransforms.length > 0) {
    world.copy_transforms_into(cachedTransforms);
  }
}

function updateTransforms() {
  const transforms = cachedTransforms;

  for (let i = 0; i < bodies.length; i++) {
    const b = bodies[i];
    if (b.instanceIndex < 0) {
      continue;
    }
    const off = i * 7;
    const px = transforms[off];
    const py = transforms[off + 1];
    const pz = transforms[off + 2];
    const qx = transforms[off + 3];
    const qy = transforms[off + 4];
    const qz = transforms[off + 5];
    const qw = transforms[off + 6];

    tempPos.set(px, py, pz);
    tempQuat.set(qx, qy, qz, qw);

    if (b.type === 0 && b.radius !== undefined) {
      tempScale.setScalar(b.radius);
      tempMatrix.compose(tempPos, tempQuat, tempScale);
      sphereInstances.setMatrixAt(b.instanceIndex, tempMatrix);
    } else if (b.type === 1 && b.halfExtents) {
      tempScale.set(
        b.halfExtents[0] * 2,
        b.halfExtents[1] * 2,
        b.halfExtents[2] * 2,
      );
      tempMatrix.compose(tempPos, tempQuat, tempScale);
      boxInstances.setMatrixAt(b.instanceIndex, tempMatrix);
    }
  }

  sphereInstances.instanceMatrix.needsUpdate = true;
  boxInstances.instanceMatrix.needsUpdate = true;
}

// FPS tracking
let frameCount = 0;
let lastFpsTime = performance.now();
let stepCount = 0;
let lastRenderMs = 0;
const fpsEl = document.getElementById("fps")!;
const bodiesEl = document.getElementById("bodies")!;
const timingsEl = document.getElementById("timings")!;

const TIMING_LABELS = [
  ["Upload",      "(CPU)"],
  ["Predict",     "(GPU)"],
  ["Broadphase",  "(GPU+CPU)"],
  ["Narrowphase", "(GPU)"],
  ["Contacts",    "(GPU>CPU)"],
  ["Solve",       "(GPU)"],
  ["Extract",     "(GPU)"],
] as const;

const BROADPHASE_LABELS = [
  ["Bounds",   "(CPU)"],
  ["Sort",     "(GPU)"],
  ["Build",    "(CPU+GPU)"],
  ["Traverse", "(GPU)"],
  ["Readback", "(CPU)"],
] as const;

// Cached data for test hooks (avoids borrow conflicts during async step)
let cachedTransforms = new Float32Array(0);
let cachedTimings = new Float32Array(TIMING_LABELS.length);
let cachedBroadphase = new Float32Array(BROADPHASE_LABELS.length);

function syncTimingCache() {
  world.copy_last_step_timings_into(cachedTimings);
  world.copy_last_broadphase_breakdown_into(cachedBroadphase);
}

async function renderScene() {
  if ("isWebGPURenderer" in renderer && renderer.isWebGPURenderer) {
    await renderer.renderAsync(scene, camera);
    return;
  }
  renderer.render(scene, camera);
}

function formatTimings(
  timings: Float32Array,
  broadphase: Float32Array,
  renderMs: number,
): string {
  const total = timings.reduce((a, b) => a + b, 0);
  const lines: string[] = [`Step: ${total.toFixed(2)} ms`];
  for (let i = 0; i < TIMING_LABELS.length; i++) {
    const [name, tag] = TIMING_LABELS[i];
    const ms = timings[i] ?? 0;
    const pct = total > 0 ? ((ms / total) * 100) : 0;
    lines.push(
      `  ${name.padEnd(11)} ${tag.padEnd(8)} ${ms.toFixed(2).padStart(6)} ms ${pct.toFixed(0).padStart(3)}%`
    );
    if (name === "Broadphase") {
      const bpTotal = broadphase.reduce((a, b) => a + b, 0);
      for (let j = 0; j < BROADPHASE_LABELS.length; j++) {
        const bpMs = broadphase[j] ?? 0;
        if (bpMs <= 0) {
          continue;
        }
        const [bpName, bpTag] = BROADPHASE_LABELS[j];
        const bpPct = bpTotal > 0 ? ((bpMs / bpTotal) * 100) : 0;
        lines.push(
          `    ${bpName.padEnd(9)} ${bpTag.padEnd(10)} ${bpMs.toFixed(2).padStart(6)} ms ${bpPct.toFixed(0).padStart(3)}%`
        );
      }
    }
  }
  lines.push(`Render      (${renderBackendLabel}) ${renderMs.toFixed(2).padStart(6)} ms`);
  return lines.join("\n");
}

async function loop_() {
  if (frameInFlight) {
    return;
  }
  frameInFlight = true;

  try {
  await world.step();
  stepCount++;

  // Cache data after step completes (world is no longer borrowed)
  syncTransformCache();

  const t0 = performance.now();
  controls?.update();
  updateTransforms();
  await renderScene();
  lastRenderMs = performance.now() - t0;

  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    syncTimingCache();
    fpsEl.textContent = `FPS: ${frameCount}`;
    bodiesEl.textContent = `Bodies: ${world.body_count()}`;
    timingsEl.textContent = formatTimings(cachedTimings, cachedBroadphase, lastRenderMs);
    frameCount = 0;
    lastFpsTime = now;
  }

  // Update test hooks
  if (window.__rubble_test) {
    window.__rubble_test.stepCount = stepCount;
    window.__rubble_test.bodyCount = world.body_count();
  }
  } finally {
    frameInFlight = false;
  }
}

async function main() {
  await init();

  world = await PhysicsWorld3D.create(0.0, -9.81, 0.0, 1.0 / 60.0);

  // Ground plane
  world.add_ground_plane(0.0);
  bodies.push({ type: 99, instanceIndex: -1 });

  // Spawn initial bodies. Body count is configurable via `?bodies=N` query param
  // so E2E tests (SwiftShader, ~1s/step broadphase readback) can use a smaller count.
  const bodyCountParam = new URL(window.location.href).searchParams.get("bodies");
  const initialBodies = bodyCountParam
    ? Math.max(1, Math.min(MAX_INSTANCES, parseInt(bodyCountParam, 10) || 1000))
    : 1000;
  for (let i = 0; i < initialBodies; i++) {
    const x = (rng() - 0.5) * 12;
    const y = 3 + rng() * 15;
    const z = (rng() - 0.5) * 12;
    spawnRandomBody(x, y, z);
  }

  // Init renderer
  await initRenderer();
  syncTransformCache();
  syncTimingCache();

  controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 3, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.update();

  // Resize handler
  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // Click to spawn
  renderer.domElement.addEventListener("click", (_e: MouseEvent) => {
    // Spawn above the center
    for (let i = 0; i < 5; i++) {
      const x = (rng() - 0.5) * 8;
      const y = 10 + rng() * 5;
      const z = (rng() - 0.5) * 8;
      spawnRandomBody(x, y, z);
    }
  });

  // Expose test hooks
  window.__rubble_test = {
    ready: true,
    stepCount: 0,
    bodyCount: world.body_count(),
    getTransforms: () => cachedTransforms,
    error: null,
  };

  document.getElementById("loading")!.style.display = "none";
  await renderer.setAnimationLoop(() => {
    void loop_();
  });
}

main().catch((e) => {
  window.__rubble_test = {
    ready: false,
    stepCount: 0,
    bodyCount: 0,
    getTransforms: () => new Float32Array(0),
    error: e.message || String(e),
  };
  document.getElementById("loading")!.textContent =
    `Failed to initialize: ${e.message}. WebGPU required.`;
  console.error(e);
});
