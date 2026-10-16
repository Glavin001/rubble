// three-avbd adapter: the pinned GpuSolver3D (WebGPU, WGSL) through Dawn, headless, as the
// repo's own `pnpm bench3d:gpu` runs it. Mirrors crates/bench-core/src/protocol.rs step for
// step (same lead-in, rounds, batches, checkpoints and output files), so its numbers line up
// with every Rust adapter's. Build-time behaviour also follows its bench (bench-cases.ts
// measureCase): bodyCapacity = bodies + 16 and `adapt` from the counters every 30 lead-in
// steps (never inside the timed window).
//
// The scene is converted back to three-avbd's Z-up frame (the inverse of scenes/export.ts):
// world vectors (x, y, z) → (x, −z, y); body-local extents/inertia (x, y, z) → (x, z, y);
// joint anchors (x, y, z) → (x, −z, y); quaternion vector parts like world vectors.
//
// Scene format v2 (tools/fracture-export): a single-hull body becomes three-avbd's own hull
// (shapes.ts `hull()` of hull.ts `convexHull(points)`, points mapped like vectors, (x, −z, y)).
// convexHull returns the hull in its own principal frame (`center`, `rotation` in the points'
// frame); the adapter maps vertices, face planes, size and radius back into the scene's body
// frame (the fragment's centre of mass and principal axes: a rigid re-expression of the same
// polyhedron — wgsl-hull.ts takes any body frame), so the body pose, mass and principal moments
// are the scene's exactly and readback needs no frame offset. Hulls over MAX_HULL_VERTICES (32)
// are simplified by convexHull (recorded). Bodies of several hulls are unsupported (exit 3:
// three-avbd has no compound shapes). The ground plane becomes a static box like three-avbd's
// own bench ground (bench-scenes.ts `ground()`: max(200, 4·extent) square, 1 m thick) whose top
// face is the plane; its readback is shifted back onto the plane.
//
// Usage (repo root, Node ≥ 22.6 with type stripping):
//   node --experimental-transform-types --no-warnings bench/adapters/three-avbd/run.ts \
//     --scene bench/scenes/out/<id>.scene.json --out <result.json> [--poses p.bin]
//     [--mode batch|sync] [--iterations N] [--set key=value]... [--max-seconds S] [--label t]
// `--set` keys: any boolean/number GpuParams3D field (reuseContacts, startAtRest, matchNearest,
// faceBias, massPenalty, ...), spatialSort=on|off.

import { createHash } from 'node:crypto';
import { execSync } from 'node:child_process';
import { readFileSync, writeFileSync, openSync, writeSync, closeSync, mkdirSync } from 'node:fs';
import { cpus, hostname, arch, platform, release, totalmem } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { adapterLabel, dawn, device, errors, skip } from './device.ts';
import { B_ANGVEL, B_POS, B_ROT, B_VEL, BODY_FLOATS, SHAPE_HULL as GPU_SHAPE_HULL } from '../../../inspiration/pinned/three-avbd/src/avbd3d/gpu/layout.ts';
import { GpuSolver3D } from '../../../inspiration/pinned/three-avbd/src/avbd3d/gpu/solver.ts';
import { Rigid } from '../../../inspiration/pinned/three-avbd/src/avbd3d/ref/body.ts';
import { Joint } from '../../../inspiration/pinned/three-avbd/src/avbd3d/ref/forces.ts';
import { Solver } from '../../../inspiration/pinned/three-avbd/src/avbd3d/ref/solver.ts';
import { convexHull, hull, MAX_HULL_VERTICES, sphere, type HullShape } from '../../../inspiration/pinned/three-avbd/src/avbd3d/shapes.ts';

const here = dirname(fileURLToPath(import.meta.url));
const pinned = resolve(here, '../../../inspiration/pinned/three-avbd');
const BODY_WORDS = 24;
const JOINT_WORDS = 12;
const SAMPLE_FLOATS = 13;

// --- Arguments (crates/bench-core/src/cli.rs) ----------------------------------------------
const argv = process.argv.slice(2);
const opts: Record<string, string> = {};
const sets: Record<string, string> = {};
let quiet = false;
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--quiet') quiet = true;
  else if (a === '--smoke') opts.smoke = 'on';
  else if (a === '--set') {
    const [k, v] = argv[++i].split('=');
    sets[k] = v;
  } else if (a.startsWith('--')) opts[a.slice(2)] = argv[++i];
  else throw new Error(`unknown argument ${a}`);
}
if (!opts.scene || !opts.out) throw new Error('--scene and --out are required');
const mode = (opts.mode ?? 'batch') as 'batch' | 'sync';
if (mode !== 'batch' && mode !== 'sync') throw new Error(`--mode ${mode}`);
if (opts.sleep === 'on') throw new Error('three-avbd has no sleeping');
const outPath = resolve(opts.out);
const posesPath = resolve(opts.poses ?? outPath.replace(/\.json$/, '') + '.poses.bin');
const maxSeconds = opts['max-seconds'] ? Number(opts['max-seconds']) : undefined;
const log = (s: string) => quiet || console.error(`[three-avbd ${meta?.id}] ${s}`);

// --- Scene (crates/bench-core/src/scene.rs) ------------------------------------------------
const scenePath = resolve(opts.scene);
const meta = JSON.parse(readFileSync(scenePath, 'utf8'));
if (meta.format !== 'rubble-bench-scene' || (meta.version !== 1 && meta.version !== 2)) throw new Error(`unsupported scene format ${meta.format} v${meta.version}`);
const binBytes = readFileSync(join(dirname(scenePath), meta.bin));
const sha = createHash('sha256').update(binBytes).digest('hex');
if (sha !== meta.binSha256) throw new Error(`scene bin sha256 ${sha} != ${meta.binSha256}; re-export`);
const words = new Uint32Array(binBytes.buffer, binBytes.byteOffset, binBytes.byteLength / 4);
const floats = new Float32Array(binBytes.buffer, binBytes.byteOffset, binBytes.byteLength / 4);
if (words[0] !== 0x43534252 || (words[1] !== 1 && words[1] !== 2)) throw new Error('scene bin: bad header');
const [nb, nj] = [words[2], words[3]];
// v2 header: + hull and vertex counts; hull table (first vertex, count) and vertices after the joints
const head = words[1] === 2 ? 6 : 4;
const [nh, nv] = words[1] === 2 ? [words[4], words[5]] : [0, 0];
const hullBase = head + nb * BODY_WORDS + nj * JOINT_WORDS;
const vertBase = hullBase + nh * 2;
if (words.length !== vertBase + nv * 3) throw new Error(`scene bin: size does not match ${nb} bodies, ${nj} joints, ${nh} hulls, ${nv} vertices`);
const SHAPE_BOX = 0, SHAPE_SPHERE = 1, SHAPE_HULLS = 2, SHAPE_PLANE = 3;
{
  let compound = 0;
  for (let i = 0; i < nb; i++) if (words[head + i * BODY_WORDS] === SHAPE_HULLS && words[head + i * BODY_WORDS + 3] > 1) compound++;
  if (compound) {
    console.error(`[three-avbd ${meta.id}] unsupported: ${compound} compound bodies (bodies of more than one convex hull; three-avbd has no compound shapes)`);
    process.exit(3);
  }
}
if (!device) {
  console.error(`no WebGPU device: ${skip}`);
  process.exit(2);
}

const zWorld = (v: ArrayLike<number>) => [v[0], -v[2], v[1]];
const zLocalAbs = (v: ArrayLike<number>) => [v[0], v[2], v[1]];
type V3 = [number, number, number];
/** Rotate v by unit quaternion q (x, y, z, w). */
const qrot = (q: ArrayLike<number>, v: ArrayLike<number>): V3 => {
  const [x, y, z, w] = [q[0], q[1], q[2], q[3]];
  const t: V3 = [2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])];
  return [v[0] + w * t[0] + (y * t[2] - z * t[1]), v[1] + w * t[1] + (z * t[0] - x * t[2]), v[2] + w * t[2] + (x * t[1] - y * t[0])];
};
const dot3 = (a: ArrayLike<number>, b: ArrayLike<number>) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * three-avbd's hull of a scene hull (body-frame vertices, already Z-up), re-expressed from
 * convexHull's principal frame (local u, placed at center + rotation·u) into the scene's body
 * frame. Also: how far the farthest input point lies outside it (convexHull's > 32-vertex
 * simplification shrinks hulls), and how far convexHull's frame sits from the scene's.
 */
function sceneHull(points: number[]): { shape: HullShape; outside: number; offset: number; angle: number; principal: V3; volume: number } {
  const h = convexHull(points);
  if (!h) throw new Error('convexHull: the points span no volume');
  const r = h.rotation;
  const c = h.center;
  const vertices = new Float32Array(h.vertices.length);
  const size: V3 = [0, 0, 0];
  let radius = 0;
  for (let i = 0; i < h.vertices.length; i += 3) {
    const u = qrot(r, h.vertices.subarray(i, i + 3));
    const v: V3 = [u[0] + c[0], u[1] + c[1], u[2] + c[2]];
    vertices.set(v, i);
    for (let j = 0; j < 3; j++) size[j] = Math.max(size[j], 2 * Math.abs(v[j]));
    radius = Math.max(radius, Math.hypot(v[0], v[1], v[2]));
  }
  const faces = h.faces.map((f) => {
    const n = qrot(r, f.normal);
    return { normal: n, d: f.d + dot3(n, c), verts: f.verts };
  });
  let outside = 0;
  for (let i = 0; i < points.length; i += 3) for (const f of faces) outside = Math.max(outside, dot3(f.normal, points.slice(i, i + 3)) - f.d);
  const angle = 2 * Math.acos(Math.min(1, Math.abs(r[3])));
  const shape: HullShape = { ...h, vertices, faces, size, radius, center: [0, 0, 0], rotation: [0, 0, 0, 1] };
  return { shape, outside, offset: Math.hypot(c[0], c[1], c[2]), angle, principal: h.moments, volume: h.volume };
}

device!.pushErrorScope('out-of-memory');
device!.pushErrorScope('validation');
const t0 = performance.now();
const ref = new Solver();
ref.clear();
ref.dt = meta.dt;
ref.gravity = meta.gravity[1];
ref.iterations = opts.iterations ? Number(opts.iterations) : meta.iterations;
ref.alpha = meta.avbd.alpha;
ref.gamma = meta.avbd.gamma;
ref.betaLin = meta.avbd.betaLin;
ref.betaAng = meta.avbd.betaAng;
const bodies: Rigid[] = [];
// Ground-plane bodies: the slab's centre sits `PLANE_DEPTH` below the plane along body +z (Z-up)
const PLANE_DEPTH = 0.5;
const planes = new Set<number>();
let groundSize = 0;
const hullStats = { bodies: 0, simplified: 0, maxOutside: 0, maxOffset: 0, maxAngle: 0, maxMomentDiff: 0, verticesIn: 0, verticesOut: 0 };
for (let i = 0; i < nb; i++) {
  const o = head + i * BODY_WORDS;
  const shapeKind = words[o];
  if (shapeKind > SHAPE_PLANE) throw new Error(`body ${i}: unknown shape ${shapeKind}`);
  const isSphereBody = shapeKind === SHAPE_SPHERE;
  const isStatic = (words[o + 1] & 1) !== 0;
  const half = floats.subarray(o + 2, o + 5);
  const mass = floats[o + 5];
  const friction = floats[o + 9];
  const pos = zWorld(floats.subarray(o + 10, o + 13));
  const q = floats.subarray(o + 13, o + 17);
  const vel = zWorld(floats.subarray(o + 17, o + 20));
  const qz = [q[0], -q[2], q[1], q[3]];
  let b: Rigid;
  if (isSphereBody) {
    b = sphere(ref, half[0], isStatic ? 0 : 1, friction, pos, vel);
  } else if (shapeKind === SHAPE_PLANE) {
    // bench-scenes.ts ground(): max(200, 4·extent) square, 1 m thick; top face on the plane
    let extent = meta.bounds?.extent ?? 0;
    if (!extent) for (let k = 0; k < nb; k++) if (k !== i) extent = Math.max(extent, Math.hypot(floats[head + k * BODY_WORDS + 10], floats[head + k * BODY_WORDS + 12]));
    groundSize = Math.max(200, 4 * extent);
    const down = qrot(qz, [0, 0, -PLANE_DEPTH]);
    b = new Rigid(ref, [groundSize, groundSize, 2 * PLANE_DEPTH], 0, friction, [pos[0] + down[0], pos[1] + down[1], pos[2] + down[2]]);
    planes.add(i);
  } else if (shapeKind === SHAPE_HULLS) {
    const first = words[o + 2];
    if (words[o + 3] !== 1 || first >= nh) throw new Error(`body ${i}: hulls ${first}+${words[o + 3]}`);
    const [v0, vc] = [words[hullBase + 2 * first], words[hullBase + 2 * first + 1]];
    const points: number[] = [];
    for (let v = v0; v < v0 + vc; v++) points.push(...zWorld(floats.subarray(vertBase + 3 * v, vertBase + 3 * v + 3)));
    const h = sceneHull(points);
    b = hull(ref, h.shape, isStatic ? 0 : 1, friction, pos, qz, vel);
    hullStats.bodies++;
    hullStats.verticesIn += vc;
    hullStats.verticesOut += h.shape.vertices.length / 3;
    if (h.outside > 1e-5) hullStats.simplified++;
    hullStats.maxOutside = Math.max(hullStats.maxOutside, h.outside);
    hullStats.maxOffset = Math.max(hullStats.maxOffset, h.offset);
    hullStats.maxAngle = Math.max(hullStats.maxAngle, h.angle);
    // three-avbd's own principal moments (scaled to the scene mass) against the scene's, sorted
    if (!isStatic) {
      const own = h.principal.map((m) => (m * mass) / h.volume).sort((x, y) => x - y);
      const scn = [...floats.subarray(o + 6, o + 9)].sort((x, y) => x - y);
      for (let j = 0; j < 3; j++) hullStats.maxMomentDiff = Math.max(hullStats.maxMomentDiff, Math.abs(own[j] - scn[j]) / scn[j]);
    }
  } else {
    b = new Rigid(ref, zLocalAbs(half).map((h) => 2 * h), isStatic ? 0 : 1, friction, pos, vel);
  }
  // Exact mass properties from the scene (three-avbd's own formulas made them)
  b.mass = isStatic ? 0 : mass;
  b.moment.set(isStatic ? [0, 0, 0] : zLocalAbs(floats.subarray(o + 6, o + 9)));
  b.positionAng.set(qz);
  b.velocityAng.set(zWorld(floats.subarray(o + 20, o + 23)));
  bodies.push(b);
}
const jointBase = head + nb * BODY_WORDS;
for (let k = 0; k < nj; k++) {
  const o = jointBase + k * JOINT_WORDS;
  if (words[o + 2] !== 0) throw new Error(`joint ${k}: kind ${words[o + 2]}`);
  if (words[o + 3] & 1) throw new Error('three-avbd cannot let joined bodies collide');
  new Joint(ref, bodies[words[o]], bodies[words[o + 1]], zWorld(floats.subarray(o + 4, o + 7)), zWorld(floats.subarray(o + 7, o + 10)));
}

const spatialSort = sets.spatialSort ?? 'on';
const gpu = new GpuSolver3D(device, ref, { bodyCapacity: nb + 16, spatialSort: spatialSort !== 'off' });
delete sets.spatialSort;
for (const [k, v] of Object.entries(sets)) {
  const params = gpu.params as unknown as Record<string, unknown>;
  if (!(k in params)) throw new Error(`--set ${k}: not a GpuParams3D field`);
  params[k] = typeof params[k] === 'boolean' ? v === 'on' || v === 'true' || v === '1' : Number(v);
}
const buildMs = performance.now() - t0;

// --- Engine ops -------------------------------------------------------------------------------
const drain = () => device.queue.onSubmittedWorkDone();
const step = () => gpu.step();
const gpuOrder = Array.from({ length: nb }, (_, i) => gpu.gpuIndex(i));
/** Bodies the GPU body buffer tags as hulls (angVel.w >= SHAPE_HULL), at the first readback. */
let gpuHullBodies = -1;
async function readState(): Promise<Float32Array> {
  const raw = await gpu.readBodies();
  const out = new Float32Array(nb * SAMPLE_FLOATS);
  const countHulls = gpuHullBodies < 0;
  if (countHulls) gpuHullBodies = 0;
  for (let i = 0; i < nb; i++) {
    const g = gpuOrder[i] * BODY_FLOATS;
    const s = i * SAMPLE_FLOATS;
    if (countHulls && raw[g + B_ANGVEL + 3] >= GPU_SHAPE_HULL) gpuHullBodies++;
    // Z-up → Y-up: (x, y, z) → (x, z, −y)
    const v3 = (at: number, to: number) => {
      out[s + to] = raw[g + at];
      out[s + to + 1] = raw[g + at + 2];
      out[s + to + 2] = -raw[g + at + 1];
    };
    v3(B_POS, 0);
    out[s + 3] = raw[g + B_ROT];
    out[s + 4] = raw[g + B_ROT + 2];
    out[s + 5] = -raw[g + B_ROT + 1];
    out[s + 6] = raw[g + B_ROT + 3];
    v3(B_VEL, 7);
    v3(B_ANGVEL, 10);
    if (planes.has(i)) {
      // ground slab centre → the plane point: + PLANE_DEPTH along body +y (Y-up readback)
      const up = qrot(out.subarray(s + 3, s + 7), [0, PLANE_DEPTH, 0]);
      for (let j = 0; j < 3; j++) out[s + j] += up[j];
    }
  }
  return out;
}
async function counters() {
  const c = await gpu.readCounters();
  return { contacts: c.contacts, manifolds: c.manifolds, pairs: c.pairs, colors: c.colors, clashes: c.clashes, overflow: c.overflow };
}

// --- Protocol (crates/bench-core/src/protocol.rs) ------------------------------------------
const p = meta.protocol;
// --smoke (crates/bench-core/src/scene.rs Protocol::smoke): every code path, almost no compute
if (opts.smoke) {
  Object.assign(p, { lead: 2, rounds: 1, warmSteps: 1, batches: 1, perBatch: 2 });
  p.steps = p.lead + p.rounds * (p.warmSteps + p.batches * p.perBatch);
  p.checkpoints = [0, p.steps];
}
const span = p.rounds * (p.warmSteps + p.batches * p.perBatch);
const windowEnd = p.lead + span;
mkdirSync(dirname(outPath), { recursive: true });
const poseFd = openSync(posesPath, 'w');
const checkpointDocs: unknown[] = [];
async function capture(at: number) {
  await drain();
  const s = await readState();
  const head = new Uint32Array([0x53504252, at, nb, SAMPLE_FLOATS]);
  writeSync(poseFd, new Uint8Array(head.buffer));
  writeSync(poseFd, new Uint8Array(s.buffer));
  checkpointDocs.push({ step: at, counters: await counters() });
}
const trace: [number, number][] = [];
const chunk = mode === 'sync' ? 1 : Math.max(1, p.perBatch);
let done = 0;
async function advance(to: number, adaptEvery30: boolean) {
  while (done < to) {
    const n = Math.min(chunk, to - done);
    const t = performance.now();
    for (let i = 0; i < n; i++) step();
    await drain();
    trace.push([done + n, (performance.now() - t) / n]);
    const before = done;
    done += n;
    // bench-cases.ts: solver.adapt(readCounters) every 30 lead-in steps
    if (adaptEvery30 && Math.floor(done / 30) > Math.floor(before / 30)) gpu.adapt(await gpu.readCounters());
  }
}
const median = (v: number[]) => [...v].sort((a, b) => a - b)[v.length >> 1];
const pct = (v: number[], q: number) => [...v].sort((a, b) => a - b)[Math.round((v.length - 1) * q)];

const started = performance.now();
await capture(0);
for (const c of p.checkpoints.filter((c: number) => c > 0 && c <= p.lead)) {
  await advance(c, true);
  await capture(c);
}
await advance(p.lead, true);
log(`lead-in ${p.lead} steps done in ${((performance.now() - started) / 1000).toFixed(1)} s; timing`);
const rounds: { samplesMs: number[]; medianMs: number }[] = [];
const stepMs: number[] = [];
for (let r = 0; r < p.rounds; r++) {
  for (let i = 0; i < p.warmSteps; i++) step();
  await drain();
  done += p.warmSteps;
  const batchMs: number[] = [];
  for (let b = 0; b < p.batches; b++) {
    if (mode === 'batch') {
      const t = performance.now();
      for (let i = 0; i < p.perBatch; i++) step();
      await drain();
      batchMs.push((performance.now() - t) / p.perBatch);
    } else {
      let sum = 0;
      for (let i = 0; i < p.perBatch; i++) {
        const t = performance.now();
        step();
        await drain();
        const ms = performance.now() - t;
        stepMs.push(ms);
        sum += ms;
      }
      batchMs.push(sum / p.perBatch);
    }
    done += p.perBatch;
  }
  rounds.push({ samplesMs: batchMs, medianMs: median(batchMs) });
}
const wallMs = Math.min(...rounds.map((r) => r.medianMs));
const allBatches = rounds.flatMap((r) => r.samplesMs);
log(`wall ${wallMs.toFixed(3)} ms/step (${mode})`);
let truncated = false;
const tailStart = performance.now();
for (const c of p.checkpoints.filter((c: number) => c >= windowEnd)) {
  if (maxSeconds !== undefined && (performance.now() - tailStart) / 1000 > maxSeconds) {
    truncated = true;
    log(`time budget ${maxSeconds} s spent at step ${done}; later checkpoints not reached`);
    break;
  }
  await advance(c, false);
  await capture(c);
}
closeSync(poseFd);
for (const scope of ['validation', 'out-of-memory']) {
  const err = await device.popErrorScope();
  if (err) errors.push(`${scope}: ${err.message}`);
}

// --- Result (same document as protocol::run) -----------------------------------------------
const git = (cmd: string) => {
  try {
    return execSync(cmd, { cwd: pinned }).toString().trim();
  } catch {
    return '';
  }
};
const dirty = git('git status --porcelain -- src').length > 0;
// Dawn 0.2 lacks GPUDevice.adapterInfo; its strings can come back as char codes
const info = (device as unknown as { adapterInfo?: GPUAdapterInfo }).adapterInfo ?? ((globalThis as { __dawn?: [unknown, GPUAdapter] }).__dawn?.[1] as unknown as { info?: GPUAdapterInfo })?.info;
const adapter = adapterLabel || [info?.vendor, info?.architecture, info?.device, info?.description].filter(Boolean).join(' · ') || 'unknown adapter';
const gp = gpu.params as unknown as Record<string, unknown>;
const config: Record<string, string> = { dawn, spatialSort, bodyCapacity: String(nb + 16) };
if (planes.size) config.groundPlane = `static box ${groundSize}×${groundSize}×${2 * PLANE_DEPTH} m (bench-scenes.ts ground() sizing), top face on the scene's plane: three-avbd has no plane shape`;
if (hullStats.bodies) {
  config.hulls =
    `three-avbd convexHull() of each hull's points (Z-up), re-expressed in the scene's body frame (exact scene pose, mass, principal moments); ` +
    `collided as ${gpu.hulls ? 'hulls' : 'BOUNDING BOXES (device cannot bind 9 storage buffers)'} (maxStorageBuffersPerShaderStage ${device.limits.maxStorageBuffersPerShaderStage}; ` +
    `${gpuHullBodies} bodies tagged SHAPE_HULL in the GPU body buffer)`;
  if (gpuHullBodies !== hullStats.bodies) log(`warning: ${hullStats.bodies} hull bodies but ${gpuHullBodies} collide as hulls on the GPU`);
  config.hullResult =
    `${hullStats.bodies} hulls: ${hullStats.verticesIn} input points -> ${hullStats.verticesOut} vertices; ${hullStats.simplified} simplified to <= ${MAX_HULL_VERTICES} vertices ` +
    `(farthest input point outside ${hullStats.maxOutside.toExponential(2)} m); convexHull's own frame vs the scene's: CoM offset <= ${hullStats.maxOffset.toExponential(2)} m, ` +
    `axes <= ${hullStats.maxAngle.toFixed(3)} rad apart, principal moments within ${(100 * hullStats.maxMomentDiff).toFixed(2)}% (unused: the scene's are set)`;
}
for (const k of ['reuseContacts', 'startAtRest', 'matchNearest', 'faceBias', 'massPenalty', 'alpha', 'gamma', 'betaLin', 'betaAng']) config[k] = String(gp[k]);
const unixCpu = () => {
  try {
    return platform() === 'darwin' ? execSync('sysctl -n machdep.cpu.brand_string').toString().trim() : cpus()[0]?.model ?? '';
  } catch {
    return cpus()[0]?.model ?? '';
  }
};
const unused = Object.keys(sets).filter((k) => !(k in gp));
if (unused.length) throw new Error(`unknown --set keys ${unused}`);
const doc = {
  format: 'rubble-bench-result',
  version: 1,
  label: opts.label ?? null,
  date: new Date().toISOString().replace(/\.\d+Z$/, 'Z'),
  machine: { os: platform() === 'darwin' ? 'macos' : platform(), arch: arch() === 'arm64' ? 'aarch64' : arch(), cpu: unixCpu(), logicalCores: cpus().length, hostname: hostname(), osVersion: release(), memoryBytes: totalmem() },
  scene: { id: meta.id, tier: meta.tier, sha256: meta.binSha256, bodies: nb, dynamic: meta.counts.dynamic, joints: nj, dt: meta.dt, referenceIterations: meta.iterations },
  engine: {
    engine: 'three-avbd',
    version: `${git('git rev-parse --short HEAD')}${dirty ? '+dirty' : ''}`,
    backend: `dawn/${adapter.split(' · ')[1] || 'webgpu'}`,
    device: adapter,
    threads: 1,
    pipelined: true,
    solver: 'AVBD',
    iterations: ref.iterations,
    iterationsMeaning: 'AVBD iterations (primal+dual) per step',
    sleeping: false,
    config,
  },
  protocol: { mode, lead: p.lead, rounds: p.rounds, warmSteps: p.warmSteps, batches: p.batches, perBatch: p.perBatch, steps: p.steps, stepsRun: done, truncated, smoke: !!opts.smoke },
  buildMs,
  timing: {
    wallMs,
    batchMeanMs: allBatches.reduce((a, b) => a + b, 0) / allBatches.length,
    batchMaxMs: Math.max(...allBatches),
    rounds,
    step: stepMs.length ? { meanMs: stepMs.reduce((a, b) => a + b, 0) / stepMs.length, p50Ms: pct(stepMs, 0.5), p95Ms: pct(stepMs, 0.95), maxMs: Math.max(...stepMs), samplesMs: stepMs } : null,
    untimedMeanMs: trace.length ? trace.reduce((a, t) => a + t[1], 0) / trace.length : null,
    trace,
  },
  checkpoints: checkpointDocs,
  poses: basename(posesPath),
  totalSeconds: (performance.now() - started) / 1000,
};
if (errors.length) (doc as Record<string, unknown>).error = `uncaptured GPU errors: ${errors.slice(0, 5).join(' | ')}`;
writeFileSync(outPath, `${JSON.stringify(doc, null, 2)}\n`);
if (errors.length) {
  console.error(`[three-avbd ${meta.id}] ${errors.length} uncaptured GPU errors, e.g. ${errors[0]}`);
  process.exit(1);
}
log(`wrote ${outPath}`);
gpu.destroy();
process.exit(0);
