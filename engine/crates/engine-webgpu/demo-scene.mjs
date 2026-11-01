// demo.html's built scenes (wall, ring, gables, pyramid, pile, columns and the
// large tiers) as body descriptions, and their packing into
// `WebWorld.createBodies` records (engine-world-wasm demo.rs, BODY_RECORD
// floats per body). The scene builders are scene-builders.mjs; World makes the
// GPU records. Pure: tests/spatial_order.test.mjs checks the packing.
import { brickQuaternion, groundHalfExtent, sphereMassProperties, buildRing, buildGables } from './scene-builders.mjs';
import { mortonSceneOrder } from './spatial-order.mjs';

export const BODY_RECORD = 24;
const TYPE = { static: 0, kinematic: 1, dynamic: 2 };
const SHAPE = { sphere: 0, box: 2 };
// Principal moments of a solid box (half extents h) and a solid sphere.
const boxInertia = (m, hx, hy, hz) => [m * (hy * hy + hz * hz) / 3, m * (hx * hx + hz * hz) / 3, m * (hx * hx + hy * hy) / 3];

// Bodies in scene order: { type, p, q, v, w, mass, inertia, shape, dims, bullet,
// density }; body 0 is the static ground slab.
export function buildDemoScene(scene, cfg, { ccd }) {
  const bodies = [];
  // Static ground box slab (three-avbd / Metal): top at y = 0.5. Deep half-height
  // so soft Jacobi contacts cannot tunnel.
  const groundHalfH = (scene.startsWith('ring') || scene.startsWith('gables')) ? 0.5 : 25;
  const groundHalfXZ = groundHalfExtent(cfg.ground || 200);
  bodies.push({ type: 'static', p: [0, 0.5 - groundHalfH, 0], shape: 'box', dims: [groundHalfXZ, groundHalfH, groundHalfXZ], density: 0 });
  const half = [0.5, 0.25, 0.25];
  const addBrick = (p, angle, mass, density) => {
    bodies.push({ type: 'dynamic', p, q: brickQuaternion(angle), mass, inertia: boxInertia(mass, ...half), shape: 'box', dims: half, density });
  };
  const addBox = (p, hx, hy, hz, mass, q) => {
    bodies.push({ type: 'dynamic', p, q: q || [0, 0, 0, 1], mass, inertia: boxInertia(mass, hx, hy, hz), shape: 'box', dims: [hx, hy, hz], density: 1 });
  };
  const addBall = (p, radius, density, v, w) => {
    const { mass, invInertia } = sphereMassProperties(radius, density);
    const I = 1 / invInertia;
    bodies.push({ type: 'dynamic', p, v, w, mass, inertia: [I, I, I], shape: 'sphere', dims: [radius, 0, 0], density, bullet: ccd });
  };
  if (scene === 'ring') buildRing(addBrick, addBall, 12, 20, 2, 10);
  else if (scene === 'ring110k') buildRing(addBrick, addBall, 80, 40, 4, 10);
  else if (scene === 'gables') buildGables(addBrick, addBall, 4, 8, 20);
  else if (scene === 'gables506k') buildGables(addBrick, addBall, 16, 68, 30);
  else if (scene === 'pyramid') {
    const levels = cfg.levels, mass = 0.25;
    // three-avbd brickPyramid: 0.75 + level*0.5 (margin seats; no pre-sink).
    for (let level = 0; level < levels; level++) {
      const n = levels - level;
      for (let x = 0; x < n; x++) for (let z = 0; z < n; z++)
        addBrick([(x - (n - 1) / 2) * 1.05, 0.75 + level * 0.5, (z - (n - 1) / 2) * 0.55], 0, mass, 1);
    }
  } else if (scene === 'pile') {
    // three-avbd boxPile(40, 20); generate (z, x, y) then Morton-upload.
    let seed = 12345;
    const rand = () => { seed = (Math.imul(seed, 1103515245) + 12345) >>> 0; return seed / 2 ** 32; };
    const n = 40, h = 20;
    const specs = [];
    for (let z = 0; z < h; z++) for (let x = 0; x < n; x++) for (let y = 0; y < n; y++) {
      const hx = 0.25 + rand() * 0.35, hy = 0.25 + rand() * 0.35, hz = 0.25 + rand() * 0.35;
      const pos = [(x - n / 2) * 1.6, 2 + z * 1.6, (y - n / 2) * 1.6];
      const qx = rand() - 0.5, qy = rand() - 0.5, qz = rand() - 0.5, qw = rand() - 0.5;
      const ql = Math.hypot(qx, qy, qz, qw) || 1;
      specs.push({ pos, hx, hy, hz, q: [qx / ql, qy / ql, qz / ql, qw / ql] });
    }
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const s of specs) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], s.pos[k]); hi[k] = Math.max(hi[k], s.pos[k]); }
    const spread = (x) => {
      x &= 0x3ff;
      x = (x | (x << 16)) & 0x30000ff;
      x = (x | (x << 8)) & 0x300f00f;
      x = (x | (x << 4)) & 0x30c30c3;
      x = (x | (x << 2)) & 0x9249249;
      return x;
    };
    const morton = (p) => {
      const q = (k) => Math.max(0, Math.min(1023, Math.floor((p[k] - lo[k]) / Math.max(hi[k] - lo[k], 1e-9) * 1024)));
      return spread(q(0)) | (spread(q(1)) << 1) | (spread(q(2)) << 2);
    };
    const order = specs.map((_, i) => i).sort((ia, ib) => morton(specs[ia].pos) - morton(specs[ib].pos));
    for (const i of order) { const s = specs[i]; addBox(s.pos, s.hx, s.hy, s.hz, 8 * s.hx * s.hy * s.hz, s.q); }
  } else if (scene === 'columns') {
    const n = 100, h = 10;
    for (let x = 0; x < n; x++) for (let z = 0; z < n; z++) for (let y = 0; y < h; y++)
      addBox([(x - n / 2) * 1.5, 1 + y, (z - n / 2) * 1.5], 0.5, 0.5, 0.5, 1);
  } else {
    // Wall smash.
    const mass = 0.6 * 0.25;
    const w = 40, courses = 25;
    for (let course = 0; course < courses; course++) {
      const offset = course % 2 === 0 ? 0 : 0.5;
      for (let x = 0; x < w; x++) for (let depth = 0; depth < 2; depth++)
        addBrick([(x - w / 2) + offset, 0.75 + course * 0.5, depth * 0.5], 0, mass, 0.6);
    }
    addBall([0, 5, -25], 2, 20, [0, 4, 30], [0, 0, 0]);
  }
  return bodies;
}

// The records `createBodies` takes, in GPU slot order. With Morton ordering
// (`order=morton`) the dynamic bodies are uploaded spatially sorted; the ground
// stays at slot 0 for the renderer. Returns the records and the order maps
// (`gpuToScene[slot]`, `sceneToGpu[body]`).
export function packDemoBodies(bodies, { spatialOrder = false } = {}) {
  let order = null;
  if (spatialOrder) {
    const view = bodies.map((b) => ({ p: b.p, invMass: b.type === 'dynamic' ? 1 / b.mass : 0 }));
    order = mortonSceneOrder(view, view.map((_, body) => ({ body })));
  }
  const gpuToScene = order ? order.gpuToScene : Uint32Array.from(bodies, (_, i) => i);
  const sceneToGpu = order ? order.sceneToGpu : gpuToScene;
  const records = new Float32Array(bodies.length * BODY_RECORD);
  for (let slot = 0; slot < bodies.length; slot++) {
    const b = bodies[gpuToScene[slot]];
    const q = b.q || [0, 0, 0, 1], v = b.v || [0, 0, 0], w = b.w || [0, 0, 0];
    const inertia = b.inertia || [0, 0, 0];
    records.set([TYPE[b.type], b.bullet ? 1 : 0, ...b.p, ...q, ...v, ...w, b.mass || 0, ...inertia, SHAPE[b.shape], ...b.dims, b.density ?? 1], slot * BODY_RECORD);
  }
  return { records, gpuToScene, sceneToGpu };
}
