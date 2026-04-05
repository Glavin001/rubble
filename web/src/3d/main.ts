import init, { PhysicsWorld3D } from "../../src/wasm/rubble_wasm.js";
import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

// Three.js WebGPURenderer stores per-instance transforms in a uniform buffer.
// A 4x4 matrix is 64 bytes, so 1024 instances lands exactly on a 64 KiB limit;
// keep real headroom here to avoid context/device loss on Chrome SwiftShader.
const MAX_INSTANCES = 768;
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
  if (frameInFlight || sceneLoading) {
    return;
  }
  frameInFlight = true;

  try {
  const stepStart = performance.now();
  await world.step();
  if (firstStepLoggedFor) {
    const elapsed = performance.now() - stepStart;
    console.log(
      `[scene] "${firstStepLoggedFor}" first step: ${elapsed.toFixed(0)}ms`,
    );
    firstStepLoggedFor = null;
    // Hide the overlay once the first step finishes — proves GPU is alive.
    const loadingEl = document.getElementById("loading");
    if (loadingEl) loadingEl.style.display = "none";
  }
  stepCount++;

  // Cache data after step completes (world is no longer borrowed)
  syncTransformCache();

  const t0 = performance.now();
  controls?.update();
  updateTransforms();
  await renderScene();
  lastRenderMs = performance.now() - t0;

  frameCount++;
  // Step timings reflect the last `world.step()`; refresh every frame so the overlay
  // is never up to ~1s stale (FPS is still averaged once per second).
  syncTimingCache();
  timingsEl.textContent = formatTimings(cachedTimings, cachedBroadphase, lastRenderMs);
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    fpsEl.textContent = `FPS: ${frameCount}`;
    bodiesEl.textContent = `Bodies: ${world.body_count()}`;
    frameCount = 0;
    lastFpsTime = now;
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
  syncTimingCache();

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
    syncTimingCache();
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
