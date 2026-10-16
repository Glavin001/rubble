// Export the suite (./suite.ts) as engine-neutral scene files: <id>.scene.json (metadata,
// protocol, groups, gates) + <id>.scene.bin (bodies and joints, little-endian f32/u32).
// Bodies come straight from the pinned three-avbd builders, then are turned from its Z-up
// frame into the canonical Y-up frame by a proper rotation (−90° about x):
//   (x, y, z) → (x, z, −y)
// applied to positions, velocities, angular velocities and the vector part of quaternions
// (q' = m·q·m⁻¹). Conjugating the orientation turns each body's local frame too, so local data
// follows the same map: joint anchors (x, z, −y), half extents and principal inertia (x, z, y).
// An axis-aligned box stays axis-aligned with the identity rotation (the ground is a slab
// thin in y).
//
// Usage (repo root): node --experimental-transform-types bench/scenes/export.ts [--out dir] [tier|id,...]

import { createHash } from 'node:crypto';
import { execSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Joint } from '../../inspiration/pinned/three-avbd/src/avbd3d/ref/forces.ts';
import { Solver } from '../../inspiration/pinned/three-avbd/src/avbd3d/ref/solver.ts';
import { isSphere } from '../../inspiration/pinned/three-avbd/src/avbd3d/shapes.ts';
import { PROTOCOL, SUITE, TIMED_SPAN, type GroupContext } from './suite.ts';

export const SCENE_FORMAT = 'rubble-bench-scene';
export const SCENE_VERSION = 1;
/** u32 words per body / joint record in the .bin (see README "Scene format"). */
export const BODY_WORDS = 24;
export const JOINT_WORDS = 12;
export const SHAPE_BOX = 0;
export const SHAPE_SPHERE = 1;
export const JOINT_BALL = 0;

const here = dirname(fileURLToPath(import.meta.url));
const pinned = resolve(here, '../../inspiration/pinned/three-avbd');

const args = process.argv.slice(2);
const option = (name: string) => {
  const i = args.indexOf(name);
  return i >= 0 ? args.splice(i, 2)[1] : undefined;
};
const outDir = resolve(option('--out') ?? join(here, 'out'));
const only = args[0]?.split(',');

const toYUp = (v: ArrayLike<number>): [number, number, number] => [v[0], v[2], -v[1]];

function revision(): { commit: string; dirty: string[] } {
  try {
    const commit = execSync('git rev-parse HEAD', { cwd: pinned }).toString().trim();
    const dirty = execSync('git status --porcelain -- src', { cwd: pinned })
      .toString()
      .split('\n')
      .filter(Boolean)
      .map((l) => l.trim());
    return { commit, dirty };
  } catch {
    return { commit: 'unknown', dirty: [] };
  }
}

mkdirSync(outDir, { recursive: true });
const rev = revision();
const index: unknown[] = [];

for (const spec of SUITE) {
  if (only && !only.includes(spec.tier) && !only.includes(spec.id)) continue;
  const t0 = performance.now();
  const solver = new Solver();
  spec.build(solver);
  const bodies = solver.bodies;
  const joints = solver.forces.filter((f): f is Joint => f instanceof Joint);
  const others = solver.forces.length - joints.length;
  if (others) throw new Error(`${spec.id}: ${others} non-joint forces (springs/ignore-collision) are not in scene format v1`);

  const indexOf = new Map(bodies.map((b, i) => [b, i]));
  const bin = new ArrayBuffer(16 + bodies.length * BODY_WORDS * 4 + joints.length * JOINT_WORDS * 4);
  const u32 = new Uint32Array(bin);
  const f32 = new Float32Array(bin);
  u32.set([0x43534252 /* 'RBSC' */, SCENE_VERSION, bodies.length, joints.length]);

  const ctx: GroupContext = { pos: [], isSphere: [], isStatic: [], radius: [] };
  let lo = [Infinity, Infinity, Infinity];
  let hi = [-Infinity, -Infinity, -Infinity];
  let spheres = 0;
  let statics = 0;
  bodies.forEach((b, i) => {
    const o = 4 + i * BODY_WORDS;
    const sphere = isSphere(b);
    const isStatic = b.mass <= 0;
    const pos = toYUp(b.positionLin);
    const q = b.positionAng;
    u32[o] = sphere ? SHAPE_SPHERE : SHAPE_BOX;
    u32[o + 1] = isStatic ? 1 : 0;
    // Half extents in the body frame (a sphere: its radius three times)
    f32.set(sphere ? [b.radius, b.radius, b.radius] : [b.size[0] / 2, b.size[2] / 2, b.size[1] / 2], o + 2);
    f32[o + 5] = isStatic ? 0 : b.mass;
    f32.set(isStatic ? [0, 0, 0] : [b.moment[0], b.moment[2], b.moment[1]], o + 6);
    f32[o + 9] = b.friction;
    f32.set(pos, o + 10);
    f32.set([q[0], q[2], -q[1], q[3]], o + 13);
    f32.set(toYUp(b.velocityLin), o + 17);
    f32.set(toYUp(b.velocityAng), o + 20);
    f32[o + 23] = 0; // restitution: AVBD has none, so every engine runs inelastic contacts
    ctx.pos.push(pos);
    ctx.isSphere.push(sphere);
    ctx.isStatic.push(isStatic);
    ctx.radius.push(b.radius);
    if (sphere) spheres++;
    if (isStatic) statics++;
    else {
      lo = lo.map((v, k) => Math.min(v, pos[k]));
      hi = hi.map((v, k) => Math.max(v, pos[k]));
    }
  });
  const jointBase = 4 + bodies.length * BODY_WORDS;
  joints.forEach((j, k) => {
    const o = jointBase + k * JOINT_WORDS;
    if (j.stiffnessLin !== Infinity || j.stiffnessAng !== 0 || j.fracture !== Infinity) {
      throw new Error(`${spec.id}: joint ${k} is not a hard ball joint (lin ${j.stiffnessLin}, ang ${j.stiffnessAng}, fracture ${j.fracture})`);
    }
    if (!j.bodyA) throw new Error(`${spec.id}: joint ${k} attaches to the world; scene format v1 needs two bodies`);
    u32[o] = indexOf.get(j.bodyA)!;
    u32[o + 1] = indexOf.get(j.bodyB)!;
    u32[o + 2] = JOINT_BALL;
    u32[o + 3] = 0; // flags: bit 0 = joined bodies collide (AVBD skips contacts between joined bodies)
    f32.set(toYUp(j.rA), o + 4);
    f32.set(toYUp(j.rB), o + 7);
    f32[o + 10] = Infinity; // linear stiffness: hard
    f32[o + 11] = 0; // angular stiffness: free (a ball joint)
  });

  const bytes = new Uint8Array(bin);
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const groups = spec.groups?.(ctx) ?? {};
  const checkpoints = [...new Set([0, spec.lead, spec.lead + TIMED_SPAN, spec.steps, ...spec.gates.map((g) => g.at)])].sort((a, b) => a - b);
  if (spec.steps < spec.lead + TIMED_SPAN) throw new Error(`${spec.id}: steps ${spec.steps} < lead + timed span`);
  // A readback inside the timed window would perturb the timing it measures
  const inWindow = checkpoints.filter((c) => c > spec.lead && c < spec.lead + TIMED_SPAN);
  if (inWindow.length) throw new Error(`${spec.id}: checkpoints ${inWindow} fall inside the timed window ${spec.lead}..${spec.lead + TIMED_SPAN}`);
  if (checkpoints.at(-1)! > spec.steps) throw new Error(`${spec.id}: a gate is past the last step`);
  for (const g of spec.gates) {
    if (g.group && g.group !== 'dynamic' && !groups[g.group]?.length) throw new Error(`${spec.id}: gate group ${g.group} is empty`);
  }
  const extent = Math.max(...hi.map((v, k) => v - lo[k]));
  const meta = {
    format: SCENE_FORMAT,
    version: SCENE_VERSION,
    id: spec.id,
    tier: spec.tier,
    source: {
      builder: 'three-avbd bench-scenes.ts',
      threeAvbdName: spec.threeAvbdName,
      threeAvbdCommit: rev.commit,
      threeAvbdDirty: rev.dirty,
    },
    frame: 'Y-up, right-handed; three-avbd Z-up mapped by (x,y,z)→(x,z,−y)',
    units: 'SI (m, kg, s, rad); quaternions (x, y, z, w)',
    dt: solver.dt,
    gravity: [0, solver.gravity, 0],
    iterations: spec.iterations,
    avbd: { alpha: solver.alpha, gamma: solver.gamma, betaLin: solver.betaLin, betaAng: solver.betaAng },
    frictionCombine: 'three-avbd: sqrt(a·b); engines without a geometric mean use the average (suite pairs are 0.5/0.6: 0.548 vs 0.55)',
    protocol: { lead: spec.lead, ...PROTOCOL, steps: spec.steps, checkpoints },
    counts: { bodies: bodies.length, dynamic: bodies.length - statics, static: statics, spheres, boxes: bodies.length - spheres, joints: joints.length },
    bounds: { min: lo, max: hi, extent },
    bin: `${spec.id}.scene.bin`,
    binBytes: bytes.length,
    binSha256: sha256,
    groups,
    gates: spec.gates,
  };
  writeFileSync(join(outDir, `${spec.id}.scene.bin`), bytes);
  writeFileSync(join(outDir, `${spec.id}.scene.json`), `${JSON.stringify(meta)}\n`);
  index.push({ id: spec.id, tier: spec.tier, bodies: bodies.length, joints: joints.length, sha256 });
  console.log(
    `${spec.id.padEnd(20)} ${String(bodies.length).padStart(7)} bodies ${String(joints.length).padStart(6)} joints ${String(spheres).padStart(2)} spheres ` +
      `${(bytes.length / 2 ** 20).toFixed(1).padStart(6)} MB  sha256 ${sha256.slice(0, 12)}  (${((performance.now() - t0) / 1000).toFixed(1)} s)`,
  );
}
writeFileSync(join(outDir, 'index.json'), `${JSON.stringify({ threeAvbd: rev, scenes: index }, null, 1)}\n`);
console.log(`wrote ${outDir}`);
