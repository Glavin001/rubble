import init, { PhysicsWorld3D } from "../../src/wasm/rubble_wasm.js";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const MAX_INSTANCES = 2048;

const SPHERE_COLORS = [0xff6b35, 0xf7c948, 0x4ecdc4, 0x45b7d1, 0x96ceb4];
const BOX_COLORS = [0xff6f69, 0xffcc5c, 0x88d8b0, 0xc3aed6, 0xffd166];

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

let renderer: THREE.WebGLRenderer;

function initRenderer() {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
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

const boxGeo = new THREE.BoxGeometry(1, 1, 1);
const boxMat = new THREE.MeshStandardMaterial({
  roughness: 0.5,
  metalness: 0.2,
});
const boxInstances = new THREE.InstancedMesh(boxGeo, boxMat, MAX_INSTANCES);
boxInstances.castShadow = true;
boxInstances.receiveShadow = true;
boxInstances.count = 0;
scene.add(boxInstances);

const tempMatrix = new THREE.Matrix4();
const tempQuat = new THREE.Quaternion();
const tempPos = new THREE.Vector3();
const tempScale = new THREE.Vector3();
const tempColor = new THREE.Color();

function addSphere(x: number, y: number, z: number, radius: number, mass: number) {
  const idx = world.add_sphere(x, y, z, radius, mass);

  const color = SPHERE_COLORS[sphereCount % SPHERE_COLORS.length];
  tempColor.set(color);
  sphereColorAttr[sphereCount * 3] = tempColor.r;
  sphereColorAttr[sphereCount * 3 + 1] = tempColor.g;
  sphereColorAttr[sphereCount * 3 + 2] = tempColor.b;

  bodies.push({
    type: 0,
    instanceIndex: sphereCount,
    radius,
  });
  sphereCount++;
  sphereInstances.count = sphereCount;
  sphereInstances.instanceColor = new THREE.InstancedBufferAttribute(
    sphereColorAttr.slice(0, sphereCount * 3),
    3,
  );
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

  const color = BOX_COLORS[boxCount % BOX_COLORS.length];
  tempColor.set(color);
  boxColorAttr[boxCount * 3] = tempColor.r;
  boxColorAttr[boxCount * 3 + 1] = tempColor.g;
  boxColorAttr[boxCount * 3 + 2] = tempColor.b;

  bodies.push({
    type: 1,
    instanceIndex: boxCount,
    halfExtents: [hw, hh, hd],
  });
  boxCount++;
  boxInstances.count = boxCount;
  boxInstances.instanceColor = new THREE.InstancedBufferAttribute(
    boxColorAttr.slice(0, boxCount * 3),
    3,
  );
  return idx;
}

function spawnRandomBody(x: number, y: number, z: number) {
  const isSphere = Math.random() > 0.4;
  if (isSphere) {
    const r = 0.3 + Math.random() * 0.5;
    addSphere(x, y, z, r, 1.0);
  } else {
    const hw = 0.3 + Math.random() * 0.5;
    const hh = 0.3 + Math.random() * 0.5;
    const hd = 0.3 + Math.random() * 0.5;
    addBox(x, y, z, hw, hh, hd, 1.0);
  }
}

function updateTransforms() {
  const transforms = new Float32Array(world.get_transforms());

  for (let i = 0; i < bodies.length; i++) {
    const b = bodies[i];
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
  ["Broadphase",  "(CPU)"],
  ["Narrowphase", "(GPU)"],
  ["Contacts",    "(GPU>CPU)"],
  ["Solve",       "(GPU)"],
  ["Extract",     "(GPU)"],
] as const;

// Cached data for test hooks (avoids borrow conflicts during async step)
let cachedTransforms = new Float32Array(0);

function formatTimings(timings: Float32Array, renderMs: number): string {
  const total = timings.reduce((a, b) => a + b, 0);
  const lines: string[] = [`Step: ${total.toFixed(2)} ms`];
  for (let i = 0; i < TIMING_LABELS.length; i++) {
    const [name, tag] = TIMING_LABELS[i];
    const ms = timings[i] ?? 0;
    const pct = total > 0 ? ((ms / total) * 100) : 0;
    lines.push(
      `  ${name.padEnd(11)} ${tag.padEnd(8)} ${ms.toFixed(2).padStart(6)} ms ${pct.toFixed(0).padStart(3)}%`
    );
  }
  lines.push(`Render      (JS)     ${renderMs.toFixed(2).padStart(6)} ms`);
  return lines.join("\n");
}

async function loop_() {
  await world.step();
  stepCount++;

  // Cache data after step completes (world is no longer borrowed)
  cachedTransforms = new Float32Array(world.get_transforms());

  const t0 = performance.now();
  updateTransforms();
  renderer.render(scene, camera);
  lastRenderMs = performance.now() - t0;

  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    fpsEl.textContent = `FPS: ${frameCount}`;
    bodiesEl.textContent = `Bodies: ${world.body_count()}`;
    timingsEl.textContent = formatTimings(
      new Float32Array(world.last_step_timings_ms()),
      lastRenderMs,
    );
    frameCount = 0;
    lastFpsTime = now;
  }

  // Update test hooks
  if (window.__rubble_test) {
    window.__rubble_test.stepCount = stepCount;
    window.__rubble_test.bodyCount = world.body_count();
  }

  requestAnimationFrame(loop_);
}

async function main() {
  await init();

  world = await PhysicsWorld3D.create(0.0, -9.81, 0.0, 1.0 / 60.0);

  // Ground plane
  world.add_ground_plane(0.0);
  bodies.push({ type: 99, instanceIndex: -1 });

  // Spawn initial bodies in a grid
  for (let i = 0; i < 100; i++) {
    const x = (Math.random() - 0.5) * 12;
    const y = 3 + Math.random() * 15;
    const z = (Math.random() - 0.5) * 12;
    spawnRandomBody(x, y, z);
  }

  // Init renderer
  initRenderer();

  const controls = new OrbitControls(camera, renderer.domElement);
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
  renderer.domElement.addEventListener("click", (e) => {
    // Spawn above the center
    for (let i = 0; i < 5; i++) {
      const x = (Math.random() - 0.5) * 8;
      const y = 10 + Math.random() * 5;
      const z = (Math.random() - 0.5) * 8;
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

  // Update orbit controls in the loop
  const originalLoop = loop_;
  function loopWithControls() {
    controls.update();
    originalLoop();
  }
  loopWithControls();
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
