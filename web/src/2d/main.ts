import init, { PhysicsWorld2D } from "../../src/wasm/rubble_wasm.js";

// World dimensions in physics units. Scenes are authored for a variety of
// sizes; we pick a wide view that fits every demo (pyramid, scatter, stacks).
const WORLD_W = 60;
const WORLD_H = 40;
const WORLD_CENTER_X = 0;
const WORLD_CENTER_Y = 10;

// Colors for bodies
const COLORS = [
  "#ff6b35", "#f7c948", "#4ecdc4", "#45b7d1", "#96ceb4",
  "#ff6f69", "#ffcc5c", "#88d8b0", "#c3aed6", "#ffd166",
];
const STATIC_COLOR = "#555";

const DEMO_SEED = 0x2d5eed;

let world: PhysicsWorld2D;
let shapeTypes: Uint32Array;
let shapeSizes: Float32Array;
let shapeSizeOffsets: Uint32Array;
let bodyColors: string[] = [];

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const fpsEl = document.getElementById("fps")!;
const bodiesEl = document.getElementById("bodies")!;
const timingsEl = document.getElementById("timings")!;

function updateTimingsOverlay() {
  timingsEl.textContent = world.last_step_overlay_text("Canvas", lastRenderMs);
}

function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
window.addEventListener("resize", resize);
resize();

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

// Convert physics coords to screen coords. World is centered on
// (WORLD_CENTER_X, WORLD_CENTER_Y) and scaled to fit the viewport.
function physScale(): number {
  return Math.min(canvas.width / WORLD_W, canvas.height / WORLD_H);
}

function toScreen(px: number, py: number): [number, number] {
  const scale = physScale();
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  return [cx + (px - WORLD_CENTER_X) * scale, cy - (py - WORLD_CENTER_Y) * scale];
}

// Convert screen coords to physics coords
function toPhysics(sx: number, sy: number): [number, number] {
  const scale = physScale();
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  return [(sx - cx) / scale + WORLD_CENTER_X, WORLD_CENTER_Y - (sy - cy) / scale];
}

function randomColor(): string {
  return COLORS[Math.floor(rng() * COLORS.length)];
}

function spawnRandomBody(px: number, py: number, refreshCaches = true) {
  const isCircle = rng() > 0.4;
  if (isCircle) {
    const r = 0.2 + rng() * 0.15;
    world.add_circle(px, py, r, 1.0);
  } else {
    const hw = 0.2 + rng() * 0.15;
    const hh = 0.2 + rng() * 0.15;
    const angle = rng() * Math.PI;
    world.add_rect(px, py, hw, hh, angle, 1.0);
  }
  bodyColors.push(randomColor());
  if (refreshCaches) {
    updateShapeCache();
    ensureBodyStateBuffers();
  }
}

function updateShapeCache() {
  shapeTypes = new Uint32Array(world.get_shape_types());
  shapeSizes = new Float32Array(world.get_shape_sizes());
  shapeSizeOffsets = new Uint32Array(world.get_shape_size_offsets());
}

function ensureBodyStateBuffers() {
  const handleCount = world.handle_count();
  if (cachedPositions.length !== handleCount * 2) {
    cachedPositions = new Float32Array(handleCount * 2);
  }
  if (cachedAngles.length !== handleCount) {
    cachedAngles = new Float32Array(handleCount);
  }
}

function syncBodyStateCache() {
  ensureBodyStateBuffers();
  if (cachedPositions.length > 0) {
    world.copy_positions_into(cachedPositions);
  }
  if (cachedAngles.length > 0) {
    world.copy_angles_into(cachedAngles);
  }
}

function draw() {
  const scale = physScale();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw background
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < shapeTypes.length; i++) {
    const px = cachedPositions[i * 2];
    const py = cachedPositions[i * 2 + 1];
    const angle = cachedAngles[i];
    const type_ = shapeTypes[i];
    const sizeOff = shapeSizeOffsets[i];
    const [sx, sy] = toScreen(px, py);

    ctx.save();
    ctx.translate(sx, sy);
    ctx.rotate(-angle); // canvas Y is inverted

    if (type_ === 0) {
      // Circle
      const radius = shapeSizes[sizeOff] * scale;
      ctx.beginPath();
      ctx.arc(0, 0, radius, 0, Math.PI * 2);
      ctx.fillStyle = bodyColors[i] || "#666";
      ctx.fill();
      // Draw a line to show rotation
      ctx.strokeStyle = "rgba(0,0,0,0.3)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(radius, 0);
      ctx.stroke();
    } else if (type_ === 1) {
      // Rect
      const hw = shapeSizes[sizeOff] * scale;
      const hh = shapeSizes[sizeOff + 1] * scale;
      ctx.fillStyle = bodyColors[i] || "#666";
      ctx.fillRect(-hw, -hh, hw * 2, hh * 2);
      ctx.strokeStyle = "rgba(0,0,0,0.2)";
      ctx.lineWidth = 1;
      ctx.strokeRect(-hw, -hh, hw * 2, hh * 2);
    } else if (type_ === 3) {
      // Capsule — draw as rounded rect
      const halfH = shapeSizes[sizeOff] * scale;
      const r = shapeSizes[sizeOff + 1] * scale;
      ctx.fillStyle = bodyColors[i] || "#666";
      ctx.beginPath();
      ctx.arc(0, -halfH, r, Math.PI, 0);
      ctx.arc(0, halfH, r, 0, Math.PI);
      ctx.closePath();
      ctx.fill();
    }

    ctx.restore();
  }
}

// FPS tracking
let frameCount = 0;
let lastFpsTime = performance.now();
let stepCount = 0;
let lastRenderMs = 0;

// Cached data for test hooks (avoids borrow conflicts during async step)
let cachedPositions = new Float32Array(0);
let cachedAngles = new Float32Array(0);

async function loop_() {
  await world.step();
  stepCount++;

  // Cache data after step completes (world is no longer borrowed)
  syncBodyStateCache();

  const t0 = performance.now();
  draw();
  lastRenderMs = performance.now() - t0;

  frameCount++;
  // Step timings reflect the last `world.step()`; refresh every frame so the overlay
  // is never up to ~1s stale (FPS is still averaged once per second).
  updateTimingsOverlay();
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

  requestAnimationFrame(loop_);
}

async function loadScene(name: string) {
  const oldWorld = world as PhysicsWorld2D | undefined;
  world = await PhysicsWorld2D.create(0.0, -9.81, 1.0 / 60.0);
  if (oldWorld) {
    try {
      oldWorld.free();
    } catch (e) {
      console.warn("failed to free old 2D world", e);
    }
  }
  bodyColors = [];
  world.load_scene(name);
  console.log(`[scene 2D] "${name}": ${world.body_count()} bodies`);

  updateShapeCache();
  // Assign colors: static (mass 0) bodies get a muted shade, dynamics get
  // a palette colour. We approximate "static" as index 0 for scenes whose
  // first body is the ground; for safety, just colour all bodies from the
  // palette — static bodies still read well on the dark background.
  const handleCount = world.handle_count();
  for (let i = 0; i < handleCount; i++) {
    bodyColors.push(i === 0 ? STATIC_COLOR : randomColor());
  }
  ensureBodyStateBuffers();
  syncBodyStateCache();
  updateTimingsOverlay();

  if (window.__rubble_test) {
    window.__rubble_test.bodyCount = world.body_count();
  }
}

async function main() {
  await init();

  // Bootstrap world so we can query scene names.
  world = await PhysicsWorld2D.create(0.0, -9.81, 1.0 / 60.0);
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

  // `?bodies=N` overrides the scene picker and does an imperative walled
  // spawn. E2E tests on SwiftShader use this for a bounded, stable world.
  const bodyCountParam = new URL(window.location.href).searchParams.get("bodies");
  if (bodyCountParam !== null) {
    const initialBodies = Math.max(
      1,
      Math.min(5000, parseInt(bodyCountParam, 10) || 50),
    );
    bodyColors = [];
    const wallThickness = 1.0;
    // 3 static walls (ground, left, right) in a [0, WORLD_W] x [0, WORLD_H]
    // frame, then a grid of dynamic bodies inside.
    world.add_static_rect(WORLD_W / 2, wallThickness / 2, WORLD_W / 2, wallThickness / 2, 0.0);
    bodyColors.push(STATIC_COLOR);
    world.add_static_rect(wallThickness / 2, WORLD_H / 2, wallThickness / 2, WORLD_H / 2, 0.0);
    bodyColors.push(STATIC_COLOR);
    world.add_static_rect(WORLD_W - wallThickness / 2, WORLD_H / 2, wallThickness / 2, WORLD_H / 2, 0.0);
    bodyColors.push(STATIC_COLOR);

    const cols = Math.ceil(Math.sqrt(initialBodies * 1.6));
    const rows = Math.ceil(initialBodies / cols);
    const xStart = 4.0;
    const yStart = 6.5;
    const xSpacing = (WORLD_W - 8) / Math.max(1, cols - 1);
    const ySpacing = (WORLD_H - 15) / Math.max(1, rows - 1);
    let spawned = 0;
    for (let row = 0; row < rows && spawned < initialBodies; row++) {
      for (let col = 0; col < cols && spawned < initialBodies; col++) {
        const px = xStart + col * xSpacing;
        const py = yStart + row * ySpacing;
        spawnRandomBody(px, py, false);
        spawned++;
      }
    }
    updateShapeCache();
    ensureBodyStateBuffers();
    syncBodyStateCache();
    updateTimingsOverlay();
    sceneSelect.disabled = true;
    if (window.__rubble_test) {
      window.__rubble_test.bodyCount = world.body_count();
    }
  } else {
    await loadScene(initialName);
  }

  sceneSelect.addEventListener("change", () => {
    void loadScene(sceneSelect.value);
  });

  // Click to spawn
  canvas.addEventListener("click", (e) => {
    const [px, py] = toPhysics(e.clientX, e.clientY);
    for (let i = 0; i < 5; i++) {
      spawnRandomBody(px + (rng() - 0.5) * 2, py + (rng() - 0.5) * 2);
    }
  });

  // Expose test hooks
  window.__rubble_test = {
    ready: true,
    stepCount: 0,
    bodyCount: world.body_count(),
    getPositions: () => cachedPositions,
    getAngles: () => cachedAngles,
    error: null,
  };

  document.getElementById("loading")!.style.display = "none";
  loop_();
}

main().catch((e) => {
  window.__rubble_test = {
    ready: false,
    stepCount: 0,
    bodyCount: 0,
    getPositions: () => new Float32Array(0),
    getAngles: () => new Float32Array(0),
    error: e.message || String(e),
  };
  document.getElementById("loading")!.textContent =
    `Failed to initialize: ${e.message}. WebGPU required.`;
  console.error(e);
});
