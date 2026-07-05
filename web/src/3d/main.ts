import init, { PhysicsWorld3D } from "../../src/wasm/rubble_wasm.js";
import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Three.js r172's WebGPURenderer stores per-instance transforms in a 64 KiB
// uniform buffer ONLY when the instance count at shader-build time is <= 1000
// (see InstanceNode.setup). Above 1000 it falls back to instanced vertex
// attributes, which have no such limit. We force that attribute path via a
// warm-up compile (see forceInstanceAttributePath), so this cap is bounded by
// physics/render cost, not the uniform buffer. At 10240 the per-mesh instance
// data is ~640 KiB of matrices + ~120 KiB of colors, well within vertex-buffer
// territory. The device-loss issue that motivated the old 768 cap only occurs
// on the <=1000 uniform path, which the warm-up compile rules out.
const MAX_INSTANCES = 10240;
const DEMO_SEED = 0x3d5eed;

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
sphereInstances.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
scene.add(sphereInstances);

// Per-instance colors
const sphereColorAttr = new Float32Array(MAX_INSTANCES * 3);
const boxColorAttr = new Float32Array(MAX_INSTANCES * 3);
const capsuleColorAttr = new Float32Array(MAX_INSTANCES * 3);
const sphereInstanceColor = new THREE.InstancedBufferAttribute(sphereColorAttr, 3);
const boxInstanceColor = new THREE.InstancedBufferAttribute(boxColorAttr, 3);
const capsuleInstanceColor = new THREE.InstancedBufferAttribute(capsuleColorAttr, 3);

const boxGeo = new THREE.BoxGeometry(1, 1, 1);
const boxMat = new THREE.MeshStandardMaterial({
  roughness: 0.5,
  metalness: 0.2,
});
const boxInstances = new THREE.InstancedMesh(boxGeo, boxMat, MAX_INSTANCES);
boxInstances.castShadow = true;
boxInstances.receiveShadow = true;
boxInstances.count = 0;
boxInstances.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
boxInstances.instanceColor = new THREE.InstancedBufferAttribute(boxColorAttr, 3);
scene.add(boxInstances);

const capsuleMat = new THREE.MeshStandardMaterial({
  roughness: 0.45,
  metalness: 0.25,
});
let capsuleInstances: THREE.InstancedMesh = new THREE.InstancedMesh(
  new THREE.CapsuleGeometry(capsuleBaseRadius, capsuleBaseHalfHeight * 2, 4, 12),
  capsuleMat,
  MAX_INSTANCES,
);
capsuleInstances.castShadow = true;
capsuleInstances.receiveShadow = true;
capsuleInstances.count = 0;
capsuleInstances.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
scene.add(capsuleInstances);

sphereInstances.instanceColor = sphereInstanceColor;
boxInstances.instanceColor = boxInstanceColor;
capsuleInstances.instanceColor = capsuleInstanceColor;

function rebuildCapsuleGeometry(halfHeight: number, radius: number) {
  if (
    Math.abs(capsuleBaseHalfHeight - halfHeight) < 1e-6 &&
    Math.abs(capsuleBaseRadius - radius) < 1e-6
  ) {
    return;
  }

  capsuleBaseHalfHeight = halfHeight;
  capsuleBaseRadius = radius;
  scene.remove(capsuleInstances);
  capsuleInstances.geometry.dispose();
  capsuleInstances = new THREE.InstancedMesh(
    new THREE.CapsuleGeometry(
      radius,
      halfHeight * 2,
      4,
      12,
    ),
    capsuleMat,
    MAX_INSTANCES,
  );
  capsuleInstances.castShadow = true;
  capsuleInstances.receiveShadow = true;
  capsuleInstances.count = 0;
  capsuleInstances.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  capsuleInstances.instanceColor = capsuleInstanceColor;
  scene.add(capsuleInstances);
}

const tempMatrix = new THREE.Matrix4();
const tempQuat = new THREE.Quaternion();
const tempPos = new THREE.Vector3();
const tempScale = new THREE.Vector3();
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
  const renderable = sphereCount < MAX_INSTANCES;

  if (renderable) {
    pushColor(sphereColorAttr, SPHERE_COLORS, sphereCount);
  }

  bodies.push({
    type: 0,
    instanceIndex: renderable ? sphereCount : -1,
    radius,
  });
  if (renderable) {
    sphereCount++;
    sphereInstances.count = sphereCount;
    markInstanceColorDirty(sphereInstances);
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
    pushColor(boxColorAttr, BOX_COLORS, boxCount);
  }

  bodies.push({
    type: 1,
    instanceIndex: renderable ? boxCount : -1,
    halfExtents: [hw, hh, hd],
  });
  if (renderable) {
    boxCount++;
    boxInstances.count = boxCount;
    markInstanceColorDirty(boxInstances);
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
    } else if (b.type === 2 && b.halfHeight !== undefined && b.radius !== undefined) {
      // Scale relative to the base capsule geometry.
      const sxz = b.radius / capsuleBaseRadius;
      // CapsuleGeometry(radius, length) has total height = length + 2*radius.
      // Scaling uniformly in Y won't preserve hemispheres, so approximate via
      // radius-based scale for X/Z and length ratio for Y.
      const totalHalf = b.halfHeight + b.radius;
      const baseTotalHalf = capsuleBaseHalfHeight + capsuleBaseRadius;
      const sy = totalHalf / baseTotalHalf;
      tempScale.set(sxz, sy, sxz);
      tempMatrix.compose(tempPos, tempQuat, tempScale);
      capsuleInstances.setMatrixAt(b.instanceIndex, tempMatrix);
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

// Fixed-timestep accumulator: the world advances a fixed dt (1/60 s) per step,
// so stepping once per animation frame would tie simulation speed to the
// display refresh rate (2x real time on a 120 Hz panel). We instead accumulate
// real elapsed time and consume it in fixed dt slices, clamped so a stall
// (minimized tab, GC) can't trigger a catch-up spiral.
const FIXED_DT = 1 / 60;
const MAX_CATCHUP_STEPS = 4;
let simAccumulator = 0;
let lastFrameTime = performance.now();
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

async function renderScene() {
  if ("isWebGPURenderer" in renderer && renderer.isWebGPURenderer) {
    await renderer.renderAsync(scene, camera);
    return;
  }
  renderer.render(scene, camera);
}

// Three's WebGPU/WebGL node renderers pick the instancing path the first time
// a mesh's shader is built, based on `mesh.count` at that moment (<=1000 =>
// 64 KiB uniform buffer; >1000 => unbounded instanced attributes). We compile
// every instanced mesh with count = MAX_INSTANCES up front so they permanently
// take the attribute path, then restore the real counts. Zero-filled extra
// matrices are degenerate (scale 0) and invisible. Safe no-op before the
// renderer exists (the initial loadScene runs before initRenderer).
async function forceInstanceAttributePath() {
  if (!renderer) {
    return;
  }
  const meshes = [sphereInstances, boxInstances, capsuleInstances];
  const savedCounts = meshes.map((m) => m.count);
  for (const m of meshes) {
    m.count = MAX_INSTANCES;
  }
  try {
    if ("isWebGPURenderer" in renderer && renderer.isWebGPURenderer) {
      await (renderer as WebGPURenderer).compileAsync(scene, camera);
    } else {
      renderer.render(scene, camera);
    }
  } catch (e) {
    console.warn("forceInstanceAttributePath compile failed", e);
  } finally {
    for (let i = 0; i < meshes.length; i++) {
      meshes[i].count = savedCounts[i];
    }
  }
}

async function loop_() {
  if (frameInFlight || sceneLoading) {
    return;
  }
  frameInFlight = true;

  try {
  // Advance simulation by real elapsed wall-clock time in fixed dt slices, so
  // the sim runs at real time regardless of the animation-frame rate.
  const now = performance.now();
  let elapsed = (now - lastFrameTime) / 1000;
  lastFrameTime = now;
  if (elapsed > 0.25) elapsed = 0.25; // clamp after a stall (avoid catch-up spiral)
  simAccumulator += elapsed;

  let steps = 0;
  while (simAccumulator >= FIXED_DT && steps < MAX_CATCHUP_STEPS) {
    const stepStart = performance.now();
    await world.step();
    simAccumulator -= FIXED_DT;
    steps++;
    stepCount++;
    if (firstStepLoggedFor) {
      const elapsedMs = performance.now() - stepStart;
      console.log(
        `[scene] "${firstStepLoggedFor}" first step: ${elapsedMs.toFixed(0)}ms`,
      );
      firstStepLoggedFor = null;
      // Hide the overlay once the first step finishes — proves GPU is alive.
      const loadingEl = document.getElementById("loading");
      if (loadingEl) loadingEl.style.display = "none";
    }
  }
  // If we saturated the catch-up budget, drop the backlog so we never run
  // faster than real time to "make up" for a long stall.
  if (steps === MAX_CATCHUP_STEPS) {
    simAccumulator = 0;
  }

  // Cache data after the step(s) complete (world is no longer borrowed)
  syncTransformCache();

  const t0 = performance.now();
  controls?.update();
  updateTransforms();
  await renderScene();
  lastRenderMs = performance.now() - t0;

  frameCount++;
  // Step timings reflect the last `world.step()`; refresh every frame so the overlay
  // is never up to ~1s stale (FPS is still averaged once per second).
  updateTimingsOverlay();
  const nowFps = performance.now();
  if (nowFps - lastFpsTime >= 1000) {
    fpsEl.textContent = `FPS: ${frameCount}`;
    bodiesEl.textContent = `Bodies: ${world.body_count()}`;
    frameCount = 0;
    lastFpsTime = nowFps;
  }

  // Update test hooks
  if (window.__rubble_test) {
    window.__rubble_test.stepCount = stepCount;
    window.__rubble_test.bodyCount = world.body_count();
  }
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

  // First pass: find capsule dimensions so we can rebuild the capsule geometry
  // to match this scene. All capsules in a scene are assumed to share dims.
  for (let i = 0; i < shapeTypes.length; i++) {
    if (shapeTypes[i] === 2) {
      const off = shapeOffsets[i];
      rebuildCapsuleGeometry(shapeSizes[off], shapeSizes[off + 1]);
      break;
    }
  }

  for (let i = 0; i < shapeTypes.length; i++) {
    const type_ = shapeTypes[i];
    const off = shapeOffsets[i];
    if (type_ === 0) {
      const r = shapeSizes[off];
      const renderable = sphereCount < MAX_INSTANCES;
      if (renderable) pushColor(sphereColorAttr, SPHERE_COLORS, sphereCount);
      bodies.push({ type: 0, instanceIndex: renderable ? sphereCount : -1, radius: r });
      if (renderable) sphereCount++;
    } else if (type_ === 1) {
      const hw = shapeSizes[off];
      const hh = shapeSizes[off + 1];
      const hd = shapeSizes[off + 2];
      const renderable = boxCount < MAX_INSTANCES;
      if (renderable) pushColor(boxColorAttr, BOX_COLORS, boxCount);
      bodies.push({
        type: 1,
        instanceIndex: renderable ? boxCount : -1,
        halfExtents: [hw, hh, hd],
      });
      if (renderable) boxCount++;
    } else if (type_ === 2) {
      const halfHeight = shapeSizes[off];
      const radius = shapeSizes[off + 1];
      const renderable = capsuleCount < MAX_INSTANCES;
      if (renderable) pushColor(capsuleColorAttr, CAPSULE_COLORS, capsuleCount);
      bodies.push({
        type: 2,
        instanceIndex: renderable ? capsuleCount : -1,
        halfHeight,
        radius,
      });
      if (renderable) capsuleCount++;
    } else {
      // Plane or unrenderable: placeholder entry to keep indices aligned.
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

  // Lock every instanced mesh (including any capsule mesh rebuilt above) onto
  // the instanced-attribute path before the animation loop renders them at the
  // real, possibly <=1000, count. sceneLoading is still true here so the loop
  // is paused. No-op until the renderer is initialized (initial load path).
  await forceInstanceAttributePath();

  const totalBodies = world.body_count();
  const dropped =
    (bodies.filter((b) => b.type === 0).length - sphereCount) +
    (bodies.filter((b) => b.type === 1).length - boxCount) +
    (bodies.filter((b) => b.type === 2).length - capsuleCount);
  const tDone = performance.now();
  console.log(
    `[scene] "${name}": ${totalBodies} bodies — rendering ` +
      `${sphereCount} spheres, ${boxCount} boxes, ${capsuleCount} capsules` +
      (dropped > 0 ? ` (${dropped} bodies exceed MAX_INSTANCES=${MAX_INSTANCES} and are hidden)` : "") +
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
      Math.min(MAX_INSTANCES, parseInt(bodyCountParam, 10) || 1000),
    );
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

  // The initial scene was built before the renderer existed, so its meshes have
  // not been compiled yet. Force the attribute path now, before the first frame.
  await forceInstanceAttributePath();

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
