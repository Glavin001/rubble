import init, { PhysicsWorld2D } from "../../src/wasm/rubble_wasm.js";

// World dimensions in physics units
const WORLD_W = 40;
const WORLD_H = 30;
const WALL_THICKNESS = 1;

// Colors for bodies
const COLORS = [
  "#ff6b35", "#f7c948", "#4ecdc4", "#45b7d1", "#96ceb4",
  "#ff6f69", "#ffcc5c", "#88d8b0", "#c3aed6", "#ffd166",
];

let world: PhysicsWorld2D;
let shapeTypes: Uint32Array;
let shapeSizes: Float32Array;
let shapeSizeOffsets: Uint32Array;
let bodyColors: string[] = [];

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const fpsEl = document.getElementById("fps")!;
const bodiesEl = document.getElementById("bodies")!;

function resize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}
window.addEventListener("resize", resize);
resize();

// Convert physics coords to screen coords
function toScreen(px: number, py: number): [number, number] {
  const scale = Math.min(canvas.width / WORLD_W, canvas.height / WORLD_H);
  const ox = (canvas.width - WORLD_W * scale) / 2;
  const oy = (canvas.height - WORLD_H * scale) / 2;
  return [ox + px * scale, oy + (WORLD_H - py) * scale];
}

function physScale(): number {
  return Math.min(canvas.width / WORLD_W, canvas.height / WORLD_H);
}

// Convert screen coords to physics coords
function toPhysics(sx: number, sy: number): [number, number] {
  const scale = physScale();
  const ox = (canvas.width - WORLD_W * scale) / 2;
  const oy = (canvas.height - WORLD_H * scale) / 2;
  return [(sx - ox) / scale, WORLD_H - (sy - oy) / scale];
}

function randomColor(): string {
  return COLORS[Math.floor(Math.random() * COLORS.length)];
}

function spawnRandomBody(px: number, py: number) {
  const isCircle = Math.random() > 0.4;
  if (isCircle) {
    const r = 0.3 + Math.random() * 0.5;
    world.add_circle(px, py, r, 1.0);
  } else {
    const hw = 0.3 + Math.random() * 0.5;
    const hh = 0.3 + Math.random() * 0.5;
    const angle = Math.random() * Math.PI;
    world.add_rect(px, py, hw, hh, angle, 1.0);
  }
  bodyColors.push(randomColor());
  updateShapeCache();
}

function updateShapeCache() {
  shapeTypes = new Uint32Array(world.get_shape_types());
  shapeSizes = new Float32Array(world.get_shape_sizes());
  shapeSizeOffsets = new Uint32Array(world.get_shape_size_offsets());
}

function draw() {
  const scale = physScale();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw background
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const positions = new Float32Array(world.get_positions());
  const angles = new Float32Array(world.get_angles());

  for (let i = 0; i < shapeTypes.length; i++) {
    const px = positions[i * 2];
    const py = positions[i * 2 + 1];
    const angle = angles[i];
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

function loop_() {
  world.step();
  draw();

  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    fpsEl.textContent = `FPS: ${frameCount}`;
    bodiesEl.textContent = `Bodies: ${world.body_count()}`;
    frameCount = 0;
    lastFpsTime = now;
  }

  requestAnimationFrame(loop_);
}

async function main() {
  await init();

  world = await PhysicsWorld2D.create(0.0, -9.81, 1.0 / 60.0);

  // Ground
  world.add_static_rect(WORLD_W / 2, WALL_THICKNESS / 2, WORLD_W / 2, WALL_THICKNESS / 2, 0.0);
  bodyColors.push("#333");

  // Left wall
  world.add_static_rect(
    WALL_THICKNESS / 2,
    WORLD_H / 2,
    WALL_THICKNESS / 2,
    WORLD_H / 2,
    0.0,
  );
  bodyColors.push("#333");

  // Right wall
  world.add_static_rect(
    WORLD_W - WALL_THICKNESS / 2,
    WORLD_H / 2,
    WALL_THICKNESS / 2,
    WORLD_H / 2,
    0.0,
  );
  bodyColors.push("#333");

  // Spawn initial bodies
  for (let i = 0; i < 50; i++) {
    const px = 3 + Math.random() * (WORLD_W - 6);
    const py = 10 + Math.random() * 15;
    spawnRandomBody(px, py);
  }

  updateShapeCache();

  // Click to spawn
  canvas.addEventListener("click", (e) => {
    const [px, py] = toPhysics(e.clientX, e.clientY);
    for (let i = 0; i < 5; i++) {
      spawnRandomBody(px + (Math.random() - 0.5) * 2, py + (Math.random() - 0.5) * 2);
    }
  });

  document.getElementById("loading")!.style.display = "none";
  loop_();
}

main().catch((e) => {
  document.getElementById("loading")!.textContent =
    `Failed to initialize: ${e.message}. WebGPU required.`;
  console.error(e);
});
