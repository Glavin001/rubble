import init, { PhysicsWorld3D } from "../../src/wasm/rubble_wasm.js";
import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Three.js InstanceNode uses a UBO when mesh.count <= 1000 and vertex
// attributes when > 1000. The UBO path sends the full instanceMatrix array,
// so the InstancedMesh constructor count must stay <= 1000 for small scenes
// or > 1000 for large ones. We recreate meshes per scene with exact capacity.
const DEMO_SEED = 0x3d5eed;
const GPU_RESIDENT_STEP_BODY_THRESHOLD = 1500;

const SPHERE_COLORS = [0xff6b35, 0xf7c948, 0x4ecdc4, 0x45b7d1, 0x96ceb4];
const BOX_COLORS = [0xff6f69, 0xffcc5c, 0x88d8b0, 0xc3aed6, 0xffd166];
const CAPSULE_COLORS = [0x9ae6b4, 0xf6ad55, 0x4fd1c5, 0xb794f4, 0xfbb6ce];

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
  type: number; // 0=sphere, 1=box, 2=capsule, 99=plane/hidden
  instanceIndex: number;
  radius?: number;
  halfExtents?: [number, number, number];
  halfHeight?: number;
}
let bodies: BodyInfo[] = [];
let sphereCount = 0;
let boxCount = 0;
let capsuleCount = 0;
// Base dimensions used to build the current scene's capsule geometry.
// Per-body scale is derived from these.
let capsuleBaseHalfHeight = 0.3;
let capsuleBaseRadius = 0.2;
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
let sceneLoading = false;
let firstStepLoggedFor: string | null = null;

const forceWebGL = new URL(window.location.href).searchParams.get("webgl") === "1";

async function initRenderer() {
  try {
    if (forceWebGL) throw new Error("forced WebGL");
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

// Ground plane (decorative — scenes add their own physics ground)
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

// Instanced meshes for spheres, boxes, and capsules.
// These are recreated per scene load with the exact capacity needed.
const sphereGeo = new THREE.SphereGeometry(1, 24, 24);
const sphereMat = new THREE.MeshStandardMaterial({
  roughness: 0.4,
  metalness: 0.3,
});
const boxGeo = new THREE.BoxGeometry(1, 1, 1);
const boxMat = new THREE.MeshStandardMaterial({
  roughness: 0.5,
  metalness: 0.2,
});
const capsuleMat = new THREE.MeshStandardMaterial({
  roughness: 0.45,
  metalness: 0.25,
});

let sphereInstances: THREE.InstancedMesh;
let boxInstances: THREE.InstancedMesh;
let capsuleInstances: THREE.InstancedMesh;
let sphereColorAttr: Float32Array;
let boxColorAttr: Float32Array;
let capsuleColorAttr: Float32Array;

/**
 * (Re)create the three InstancedMesh objects with the given capacities.
 * Removes previous meshes from the scene if they exist.
 */
function allocateInstancedMeshes(sphereCap: number, boxCap: number, capsuleCap: number) {
  sphereCap = Math.max(sphereCap, 1);
  boxCap = Math.max(boxCap, 1);
  capsuleCap = Math.max(capsuleCap, 1);

  // Tear down old meshes
  if (sphereInstances) { scene.remove(sphereInstances); sphereInstances.dispose(); }
  if (boxInstances) { scene.remove(boxInstances); boxInstances.dispose(); }
  if (capsuleInstances) { scene.remove(capsuleInstances); capsuleInstances.dispose(); }

  // Color arrays
  sphereColorAttr = new Float32Array(sphereCap * 3);
  boxColorAttr = new Float32Array(boxCap * 3);
  capsuleColorAttr = new Float32Array(capsuleCap * 3);

  // Spheres
  sphereInstances = new THREE.InstancedMesh(sphereGeo, sphereMat, sphereCap);
  sphereInstances.castShadow = true;
  sphereInstances.receiveShadow = true;
  sphereInstances.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  sphereInstances.count = 0;
  sphereInstances.instanceColor = new THREE.InstancedBufferAttribute(sphereColorAttr, 3);
  scene.add(sphereInstances);

  // Boxes
  boxInstances = new THREE.InstancedMesh(boxGeo, boxMat, boxCap);
  boxInstances.castShadow = true;
  boxInstances.receiveShadow = true;
  boxInstances.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  boxInstances.count = 0;
  boxInstances.instanceColor = new THREE.InstancedBufferAttribute(boxColorAttr, 3);
  scene.add(boxInstances);

  // Capsules
  capsuleInstances = new THREE.InstancedMesh(
    new THREE.CapsuleGeometry(capsuleBaseRadius, capsuleBaseHalfHeight * 2, 4, 12),
    capsuleMat,
    capsuleCap,
  );
  capsuleInstances.castShadow = true;
  capsuleInstances.receiveShadow = true;
  capsuleInstances.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  capsuleInstances.count = 0;
  capsuleInstances.instanceColor = new THREE.InstancedBufferAttribute(capsuleColorAttr, 3);
  scene.add(capsuleInstances);
}

// Initial allocation (small — will be resized on first scene load)
allocateInstancedMeshes(64, 64, 64);

function rebuildCapsuleGeometry(halfHeight: number, radius: number, capacity: number) {
  capsuleBaseHalfHeight = halfHeight;
  capsuleBaseRadius = radius;
  scene.remove(capsuleInstances);
  capsuleInstances.geometry.dispose();
  capsuleInstances.dispose();
  capsuleColorAttr = new Float32Array(Math.max(capacity, 1) * 3);
  capsuleInstances = new THREE.InstancedMesh(
    new THREE.CapsuleGeometry(radius, halfHeight * 2, 4, 12),
    capsuleMat,
    Math.max(capacity, 1),
  );
  capsuleInstances.castShadow = true;
  capsuleInstances.receiveShadow = true;
  capsuleInstances.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  capsuleInstances.count = 0;
  capsuleInstances.instanceColor = new THREE.InstancedBufferAttribute(capsuleColorAttr, 3);
  scene.add(capsuleInstances);
}

const tempColor = new THREE.Color();

function pushColor(
  attr: Float32Array,
  palette: readonly number[],
  idx: number,
) {
  const color = palette[idx % palette.length];
  tempColor.set(color);
  attr[idx * 3] = tempColor.r;
  attr[idx * 3 + 1] = tempColor.g;
  attr[idx * 3 + 2] = tempColor.b;
}

function markInstanceColorDirty(mesh: THREE.InstancedMesh) {
  if (mesh.instanceColor) {
    mesh.instanceColor.needsUpdate = true;
  }
}

function addSphere(x: number, y: number, z: number, radius: number, mass: number) {
  const idx = world.add_sphere(x, y, z, radius, mass);
  pushColor(sphereColorAttr, SPHERE_COLORS, sphereCount);
  bodies.push({ type: 0, instanceIndex: sphereCount, radius });
  sphereCount++;
  sphereInstances.count = sphereCount;
  markInstanceColorDirty(sphereInstances);
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
  pushColor(boxColorAttr, BOX_COLORS, boxCount);
  bodies.push({ type: 1, instanceIndex: boxCount, halfExtents: [hw, hh, hd] });
  boxCount++;
  boxInstances.count = boxCount;
  markInstanceColorDirty(boxInstances);
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

async function stepAndSyncTransformCache() {
  ensureTransformBuffer();
  if (cachedTransforms.length > 0 && world.handle_count() >= GPU_RESIDENT_STEP_BODY_THRESHOLD) {
    await world.step_and_copy_transforms_into(cachedTransforms);
  } else {
    await world.step();
    syncTransformCache();
    world.copy_last_step_timings_into(cachedTimings);
    return;
  }
  world.copy_last_step_timings_into(cachedTimings);
}

function writeMatrix(
  out: Float32Array,
  base: number,
  px: number,
  py: number,
  pz: number,
  qx: number,
  qy: number,
  qz: number,
  qw: number,
  sx: number,
  sy: number,
  sz: number,
) {
  const x2 = qx + qx;
  const y2 = qy + qy;
  const z2 = qz + qz;
  const xx = qx * x2;
  const xy = qx * y2;
  const xz = qx * z2;
  const yy = qy * y2;
  const yz = qy * z2;
  const zz = qz * z2;
  const wx = qw * x2;
  const wy = qw * y2;
  const wz = qw * z2;

  out[base] = (1 - (yy + zz)) * sx;
  out[base + 1] = (xy + wz) * sx;
  out[base + 2] = (xz - wy) * sx;
  out[base + 3] = 0;

  out[base + 4] = (xy - wz) * sy;
  out[base + 5] = (1 - (xx + zz)) * sy;
  out[base + 6] = (yz + wx) * sy;
  out[base + 7] = 0;

  out[base + 8] = (xz + wy) * sz;
  out[base + 9] = (yz - wx) * sz;
  out[base + 10] = (1 - (xx + yy)) * sz;
  out[base + 11] = 0;

  out[base + 12] = px;
  out[base + 13] = py;
  out[base + 14] = pz;
  out[base + 15] = 1;
}

function updateTransforms() {
  const transforms = cachedTransforms;
  const sphereMatrices = sphereInstances.instanceMatrix.array as Float32Array;
  const boxMatrices = boxInstances.instanceMatrix.array as Float32Array;
  const capsuleMatrices = capsuleInstances.instanceMatrix.array as Float32Array;

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

    if (b.type === 0 && b.radius !== undefined) {
      const base = b.instanceIndex * 16;
      writeMatrix(sphereMatrices, base, px, py, pz, qx, qy, qz, qw, b.radius, b.radius, b.radius);
    } else if (b.type === 1 && b.halfExtents) {
      const base = b.instanceIndex * 16;
      writeMatrix(
        boxMatrices,
        base,
        px,
        py,
        pz,
        qx,
        qy,
        qz,
        qw,
        b.halfExtents[0] * 2,
        b.halfExtents[1] * 2,
        b.halfExtents[2] * 2,
      );
    } else if (b.type === 2 && b.halfHeight !== undefined && b.radius !== undefined) {
      // Scale relative to the base capsule geometry.
      const sxz = b.radius / capsuleBaseRadius;
      // CapsuleGeometry(radius, length) has total height = length + 2*radius.
      // Scaling uniformly in Y won't preserve hemispheres, so approximate via
      // radius-based scale for X/Z and length ratio for Y.
      const totalHalf = b.halfHeight + b.radius;
      const baseTotalHalf = capsuleBaseHalfHeight + capsuleBaseRadius;
      const sy = totalHalf / baseTotalHalf;
      const base = b.instanceIndex * 16;
      writeMatrix(capsuleMatrices, base, px, py, pz, qx, qy, qz, qw, sxz, sy, sxz);
    }
  }

  sphereInstances.instanceMatrix.needsUpdate = true;
  boxInstances.instanceMatrix.needsUpdate = true;
  capsuleInstances.instanceMatrix.needsUpdate = true;
}

// FPS tracking
let frameCount = 0;
let lastFpsTime = performance.now();
let stepCount = 0;
let lastRenderMs = 0;
const fpsEl = document.getElementById("fps")!;
const bodiesEl = document.getElementById("bodies")!;
const timingsEl = document.getElementById("timings")!;

function updateTimingsOverlay() {
  timingsEl.textContent = world.last_step_overlay_text(
    renderBackendLabel,
    lastRenderMs,
  );
}

// Cached data for test hooks (avoids borrow conflicts during async step)
let cachedTransforms = new Float32Array(0);
let cachedTimings = new Float32Array(7);
// [step+transform_copy, matrix_update, render, total]
let cachedFrameTimings = new Float32Array(4);

// URL parameter ?norender=1 disables rendering for stability diagnostics
const noRender = new URL(window.location.href).searchParams.get("norender") === "1";

async function renderScene() {
  if (noRender) return;
  if ("isWebGPURenderer" in renderer && renderer.isWebGPURenderer) {
    await renderer.renderAsync(scene, camera);
    return;
  }
  renderer.render(scene, camera);
}

async function runViewerFrame() {
  const frameStart = performance.now();
  const stepStart = performance.now();
  await stepAndSyncTransformCache();
  const stepAndCopyMs = performance.now() - stepStart;

  // Stability diagnostic: detect the first step where bodies go unstable.
  if (stepCount < 200) {
    const n = cachedTransforms.length / 7;
    for (let i = 1; i < n; i++) {
      const y = cachedTransforms[i * 7 + 1];
      if (y > 20 || y < -2) {
        console.warn(
          `[STABILITY] Body ${i} at y=${y.toFixed(2)} on step ${stepCount + 1}`,
        );
      }
    }
  }

  if (firstStepLoggedFor) {
    console.log(
      `[scene] "${firstStepLoggedFor}" first step: ${stepAndCopyMs.toFixed(0)}ms`,
    );
    firstStepLoggedFor = null;
    const loadingEl = document.getElementById("loading");
    if (loadingEl) loadingEl.style.display = "none";
  }
  stepCount++;

  const updateStart = performance.now();
  controls?.update();
  updateTransforms();
  const matrixUpdateMs = performance.now() - updateStart;

  const renderStart = performance.now();
  await renderScene();
  const renderMs = performance.now() - renderStart;
  lastRenderMs = renderMs;

  cachedFrameTimings[0] = stepAndCopyMs;
  cachedFrameTimings[1] = matrixUpdateMs;
  cachedFrameTimings[2] = renderMs;
  cachedFrameTimings[3] = performance.now() - frameStart;

  frameCount++;
  updateTimingsOverlay();
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    fpsEl.textContent = `FPS: ${frameCount}`;
    bodiesEl.textContent = `Bodies: ${world.body_count()}`;
    frameCount = 0;
    lastFpsTime = now;
  }

  if (window.__rubble_test) {
    window.__rubble_test.stepCount = stepCount;
    window.__rubble_test.bodyCount = world.body_count();
    window.__rubble_test.lastStepTimingsMs = cachedTimings;
    window.__rubble_test.lastFrameTimingsMs = cachedFrameTimings;
  }
}

async function loop_() {
  if (frameInFlight || sceneLoading) {
    return;
  }
  frameInFlight = true;

  try {
    await runViewerFrame();
  } catch (e) {
    // Surface step errors to console and test hooks (otherwise the animation
    // loop silently swallows them via `void loop_()` and E2E tests hang).
    console.error("world.step() failed:", e);
    if (window.__rubble_test) {
      window.__rubble_test.error = (e as Error)?.message || String(e);
    }
  } finally {
    frameInFlight = false;
  }
}

async function loadScene(name: string) {
  sceneLoading = true;
  const loadingEl = document.getElementById("loading")!;
  loadingEl.textContent = `Loading "${name}"…`;
  loadingEl.style.display = "flex";
  // Yield so the browser can paint the overlay before we start GPU work.
  await new Promise((r) => requestAnimationFrame(() => r(null)));

  const tStart = performance.now();
  // Wait for any in-flight frame on the old world to finish before we swap.
  while (frameInFlight) {
    await new Promise((r) => setTimeout(r, 4));
  }

  loadingEl.textContent = `Loading "${name}"… creating world`;
  await new Promise((r) => requestAnimationFrame(() => r(null)));

  // Throw away old world & per-body state. Rebuild from the named scene.
  const oldWorld = world as PhysicsWorld3D | undefined;
  world = await PhysicsWorld3D.create(0.0, -9.81, 0.0, 1.0 / 60.0);
  if (oldWorld) {
    try {
      oldWorld.free();
    } catch (e) {
      console.warn("failed to free old world", e);
    }
  }
  const tCreated = performance.now();
  bodies = [];
  sphereCount = 0;
  boxCount = 0;
  capsuleCount = 0;
  sphereInstances.count = 0;
  boxInstances.count = 0;
  capsuleInstances.count = 0;

  loadingEl.textContent = `Loading "${name}"… spawning bodies`;
  await new Promise((r) => requestAnimationFrame(() => r(null)));
  try {
    world.load_scene(name);
  } catch (e) {
    console.error(`load_scene("${name}") failed:`, e);
    loadingEl.textContent = `Failed to load "${name}": ${(e as Error).message ?? e}`;
    sceneLoading = false;
    throw e;
  }
  const tSpawned = performance.now();

  const shapeTypes = world.get_shape_types();
  const shapeSizes = world.get_shape_sizes();
  const shapeOffsets = world.get_shape_size_offsets();

  // First pass: count shapes per type and find capsule dimensions.
  let nSpheres = 0, nBoxes = 0, nCapsules = 0;
  let capsuleDimIdx = -1;
  for (let i = 0; i < shapeTypes.length; i++) {
    if (shapeTypes[i] === 0) nSpheres++;
    else if (shapeTypes[i] === 1) nBoxes++;
    else if (shapeTypes[i] === 2) {
      nCapsules++;
      if (capsuleDimIdx < 0) capsuleDimIdx = i;
    }
  }

  // Allocate meshes with exact capacity
  allocateInstancedMeshes(nSpheres, nBoxes, nCapsules);

  // Rebuild capsule geometry if this scene has capsules
  if (capsuleDimIdx >= 0) {
    const off = shapeOffsets[capsuleDimIdx];
    rebuildCapsuleGeometry(shapeSizes[off], shapeSizes[off + 1], nCapsules);
  }

  // Second pass: populate bodies and colors
  for (let i = 0; i < shapeTypes.length; i++) {
    const type_ = shapeTypes[i];
    const off = shapeOffsets[i];
    if (type_ === 0) {
      const r = shapeSizes[off];
      pushColor(sphereColorAttr, SPHERE_COLORS, sphereCount);
      bodies.push({ type: 0, instanceIndex: sphereCount, radius: r });
      sphereCount++;
    } else if (type_ === 1) {
      const hw = shapeSizes[off];
      const hh = shapeSizes[off + 1];
      const hd = shapeSizes[off + 2];
      pushColor(boxColorAttr, BOX_COLORS, boxCount);
      bodies.push({ type: 1, instanceIndex: boxCount, halfExtents: [hw, hh, hd] });
      boxCount++;
    } else if (type_ === 2) {
      const halfHeight = shapeSizes[off];
      const radius = shapeSizes[off + 1];
      pushColor(capsuleColorAttr, CAPSULE_COLORS, capsuleCount);
      bodies.push({ type: 2, instanceIndex: capsuleCount, halfHeight, radius });
      capsuleCount++;
    } else {
      bodies.push({ type: 99, instanceIndex: -1 });
    }
  }

  sphereInstances.count = sphereCount;
  boxInstances.count = boxCount;
  capsuleInstances.count = capsuleCount;
  markInstanceColorDirty(sphereInstances);
  markInstanceColorDirty(boxInstances);
  markInstanceColorDirty(capsuleInstances);

  syncTransformCache();
  updateTransforms();
  updateTimingsOverlay();

  const totalBodies = world.body_count();
  const tDone = performance.now();
  console.log(
    `[scene] "${name}": ${totalBodies} bodies — rendering ` +
      `${sphereCount} spheres, ${boxCount} boxes, ${capsuleCount} capsules` +
      ` — timings: create ${(tCreated - tStart).toFixed(0)}ms, spawn ${(tSpawned - tCreated).toFixed(0)}ms, mesh ${(tDone - tSpawned).toFixed(0)}ms`,
  );
  if (totalBodies > 1500) {
    console.warn(
      `[scene] "${name}" has ${totalBodies} bodies — GPU step may take hundreds of ms per frame on non-discrete GPUs.`,
    );
  }

  if (window.__rubble_test) {
    window.__rubble_test.bodyCount = totalBodies;
  }
  // Track first-step timing so we can tell if GPU step() is hanging.
  firstStepLoggedFor = name;
  // Show the setup breakdown in the overlay. We keep it visible until the
  // first GPU step completes, so on mobile/without devtools the user can see
  // whether the world was built and whether step() is actually running.
  loadingEl.innerHTML =
    `<div style="text-align:center;line-height:1.6">` +
    `<div>Loaded "${name}": ${totalBodies} bodies</div>` +
    `<div style="font-size:0.85em;color:#777">create ${(tCreated - tStart).toFixed(0)}ms · spawn ${(tSpawned - tCreated).toFixed(0)}ms · mesh ${(tDone - tSpawned).toFixed(0)}ms</div>` +
    `<div style="font-size:0.85em;color:#777">waiting for first GPU step…</div>` +
    `</div>`;
  sceneLoading = false;
}

async function main() {
  await init();

  // Create an initial world just so we can ask it which scenes exist.
  world = await PhysicsWorld3D.create(0.0, -9.81, 0.0, 1.0 / 60.0);
  const sceneNames: string[] = world.scene_names();
  const initialName: string = world.initial_scene_name();

  const sceneSelect = document.getElementById("scene-select") as HTMLSelectElement;
  for (const name of sceneNames) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    if (name === initialName) opt.selected = true;
    sceneSelect.appendChild(opt);
  }

  // `?bodies=N` overrides the scene picker and does a simple imperative spawn.
  // E2E tests on SwiftShader (CI) use this to keep per-step cost low.
  const bodyCountParam = new URL(window.location.href).searchParams.get("bodies");
  if (bodyCountParam !== null) {
    const initialBodies = Math.max(
      1,
      Math.min(50_000, parseInt(bodyCountParam, 10) || 1000),
    );
    // Pre-allocate meshes with enough capacity for the requested body count.
    allocateInstancedMeshes(initialBodies, initialBodies, initialBodies);
    world.add_ground_plane(0.0);
    bodies.push({ type: 99, instanceIndex: -1 });
    for (let i = 0; i < initialBodies; i++) {
      const x = (rng() - 0.5) * 12;
      const y = 3 + rng() * 15;
      const z = (rng() - 0.5) * 12;
      spawnRandomBody(x, y, z);
    }
    syncTransformCache();
    updateTransforms();
    updateTimingsOverlay();
    if (window.__rubble_test) {
      window.__rubble_test.bodyCount = world.body_count();
    }
    sceneSelect.disabled = true;
  } else {
    await loadScene(initialName);
  }

  sceneSelect.addEventListener("change", () => {
    void loadScene(sceneSelect.value);
  });

  // Init renderer
  await initRenderer();

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
  let loopRunning = true;
  window.__rubble_test = {
    ready: true,
    stepCount: 0,
    bodyCount: world.body_count(),
    getTransforms: () => cachedTransforms,
    lastStepTimingsMs: cachedTimings,
    lastFrameTimingsMs: cachedFrameTimings,
    error: null,
    // Benchmark-only: stop the render loop and wait for any in-flight frame.
    stopLoop: async () => {
      loopRunning = false;
      // Wait for any in-flight frame to finish
      while (frameInFlight) {
        await new Promise((r) => setTimeout(r, 10));
      }
    },
    // Benchmark-only: run one physics step without rendering.
    // Call stopLoop() first to prevent conflicts with the animation loop.
    // Returns the 7-float timings array after the step completes.
    benchStep: async () => {
      const frameStart = performance.now();
      const stepStart = performance.now();
      await stepAndSyncTransformCache();
      const stepAndCopyMs = performance.now() - stepStart;
      stepCount++;
      const updateStart = performance.now();
      controls?.update();
      updateTransforms();
      const matrixUpdateMs = performance.now() - updateStart;
      cachedFrameTimings[0] = stepAndCopyMs;
      cachedFrameTimings[1] = matrixUpdateMs;
      cachedFrameTimings[2] = 0;
      cachedFrameTimings[3] = performance.now() - frameStart;
      updateTimingsOverlay();
      window.__rubble_test!.stepCount = stepCount;
      window.__rubble_test!.bodyCount = world.body_count();
      window.__rubble_test!.lastStepTimingsMs = cachedTimings;
      window.__rubble_test!.lastFrameTimingsMs = cachedFrameTimings;
      return Array.from(cachedTimings);
    },
    loopStep: async () => {
      await loop_();
    },
  };

  document.getElementById("loading")!.style.display = "none";

  // Animation loop
  async function stepTick() {
    if (!loopRunning || frameInFlight) return;
    frameInFlight = true;
    try {
      await runViewerFrame();
    } catch (e) {
      console.error("step failed:", e);
      if (window.__rubble_test) {
        window.__rubble_test.error = (e as Error)?.message || String(e);
      }
    } finally {
      frameInFlight = false;
    }
    if (loopRunning) setTimeout(stepTick, 0);
  }
  setTimeout(() => void stepTick(), 0);
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
