// Regenerates three-avbd-reference.json from the pinned three-avbd checkout.
// This is the ONLY file that reads inspiration/ (gitignored, absent in CI and
// worktrees); every test reads the checked-in fixture instead.
//
//   nvm use
//   node --experimental-transform-types engine/crates/engine-webgpu/tests/fixtures/generate-three-avbd-reference.ts
//
// INSPIRATION_DIR overrides the default ../../../../../inspiration/pinned.
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { stripTypeScriptTypes } from 'node:module';
import { runInNewContext } from 'node:vm';
import { pathToFileURL } from 'node:url';

const root = process.env.INSPIRATION_DIR
  ? pathToFileURL(process.env.INSPIRATION_DIR.replace(/\/?$/, '/'))
  : new URL('../../../../../inspiration/pinned/', import.meta.url);
const avbd = new URL('three-avbd/', root);
const { brickRing, brickGables } = await import(new URL('src/avbd3d/bench-scenes.ts', avbd).href);
const { Solver } = await import(new URL('src/avbd3d/ref/solver.ts', avbd).href);
const revision = execFileSync('git', ['-C', avbd.pathname, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();

// The pinned Morton routine itself (module-private upstream), not a copy.
const source = readFileSync(new URL('src/avbd3d/gpu/solver.ts', avbd), 'utf8');
const start = source.indexOf('function mortonOrder('), end = source.indexOf('/** Evaluated lazily', start);
if (start < 0 || end <= start) throw new Error('pinned Morton function extraction failed');
const mortonOrder = runInNewContext(stripTypeScriptTypes(source.slice(start, end)) + '\nmortonOrder');

// Demo upload order: ground stays at index 0, then the pinned Morton order.
const uploadOrder = (positionsZUp: number[][]) =>
  [0, ...mortonOrder(positionsZUp.map(positionLin => ({ positionLin }))).filter((id: number) => id !== 0)];
const sha256 = (order: number[]) => createHash('sha256').update(new Uint8Array(Uint32Array.from(order).buffer)).digest('hex');
const sampleIndices = (n: number, k = 128) => {
  const s = new Set<number>([...Array(8).keys(), ...Array.from({ length: 8 }, (_, i) => n - 1 - i)]);
  for (let i = 0; i < k; i++) s.add(Math.floor((i + 0.5) * n / k));
  return [...s].filter(i => i >= 0 && i < n).sort((a, b) => a - b);
};

// Scene bodies in three-avbd's own frame (Z-up); tests convert their Y-up builders.
const record = (b: any) => ({
  positionLin: [...b.positionLin], positionAng: [...b.positionAng], mass: b.mass,
  moment: [...b.moment], velocityLin: [...b.velocityLin], velocityAng: [...b.velocityAng],
});
const fields = ['positionLin', 'positionAng', 'moment', 'velocityLin', 'velocityAng'] as const;
const sums = (bodies: any[]) => {
  const out: Record<string, number[]> = { mass: [0] };
  for (const f of fields) out[f] = new Array(bodies[0][f].length).fill(0);
  for (const b of bodies) {
    out.mass[0] += b.mass;
    for (const f of fields) b[f].forEach((x: number, k: number) => { out[f][k] += Number.isFinite(x) ? x : 0; });
  }
  return out;
};

const scenes: Record<string, any> = {};
const morton: Record<string, any> = {};
for (const [name, build] of [
  ['ring9k', (s: any) => brickRing(s, 12, 20, 2, 10)],
  ['ring110k', (s: any) => brickRing(s)],
  ['gables506k', (s: any) => brickGables(s)],
] as const) {
  const solver = new Solver();
  build(solver);
  const bodies = solver.bodies.slice(1); // ground is supplied separately by the demo
  scenes[name] = {
    count: bodies.length,
    sums: sums(bodies),
    samples: sampleIndices(bodies.length).map(i => ({ index: i + 1, ...record(bodies[i]) })),
  };
  if (name !== 'ring9k') {
    const order = uploadOrder(solver.bodies.map((b: any) => [...b.positionLin]));
    morton[name] = { count: order.length, sha256: sha256(order), samples: sampleIndices(order.length).map(gpu => [gpu, order[gpu]]) };
  }
}

// Small hand-made edge cases (Y-up positions, as the test builds them).
const edge = [[[0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [12, 0, -8], [-12, 0, 8], [1, 0, 3], [1, 0, 3], [0, 0, -8]]];
morton.edgeCases = edge.map(ps => ({ positions: ps, order: uploadOrder(ps.map(p => [p[0], p[2], p[1]])) }));

const ground = [12, 80].map(radius => {
  const solver = new Solver();
  brickRing(solver, radius, 2, 1, 10);
  return { radius, size: [...solver.bodies[0].size], positionLin: [...solver.bodies[0].positionLin] };
});

const out = new URL('three-avbd-reference.json', import.meta.url);
writeFileSync(out, JSON.stringify({ source: 'three-avbd', revision, generator: 'generate-three-avbd-reference.ts', scenes, morton, ground }) + '\n');
console.log(`wrote ${out.pathname} from three-avbd ${revision}`);
