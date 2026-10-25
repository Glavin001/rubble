// Port the blast-stress-solver reference rigs (crush-wall, reference-building,
// reference-building-crush) into rubble as scene packs, with their authored shots.
//
//   node port-reference-packs.mjs <blast-stress-solver checkout> [git ref]
//
// The packs are copied as authored (ScenePack v2/v3; v3's `materials[].crush` and
// `nodes[].m` are kept byte-for-byte in meaning) and gain one rubble field,
// `defaults.shots`: the source demo's own shots as data, so the destruction-reactions
// harness and the Fracture Lab fire the same projectile. Re-run it to refresh from the
// source; nothing at build or test time reads the source repository.
//
// The shots follow the source demo's launcher for a single structure, one wave
// (`demos/blast-stress-demo/mini_city_main.cpp` `createProjectiles`, wave 0):
//   target   = (0, max(1.5, 0.45 * H), 0), H the top of the highest node
//   origin   = target + (-12, 1.5, -2.25)
//   velocity = normalize(target - origin) * speed * speedScale * 1.05
//   mass     = mass * massScale
//   spin     = (1, 2, 0.5) rad/s, launched after the 2 s settle the scripts use
// The scales are the ones the source's recordings and ctest gates run
// (`record_crush_videos.sh`, `CMakeLists.txt`, `docs/chunk-crushing.md`). The CLI's own
// default massScale is 1.5, but chunk-crushing.md's wall energy ("the ball's 6.3 MJ" at
// 1.4x) is the pack's 4800 kg at 35 * 1.4 * 1.05 m/s, and its building rows are named as
// multiples of the pack's mass ("2x projectile mass"), so the base shot here is massScale 1.
import { execFileSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const [repo, ref = 'origin/perf/rubble-field-60hz'] = process.argv.slice(2);
if (!repo) throw new Error('usage: node port-reference-packs.mjs <blast-stress-solver checkout> [git ref]');
const show = (p) => execFileSync('git', ['-C', repo, 'show', `${ref}:${p}`], { encoding: 'utf8', maxBuffer: 1 << 26 });
const commit = execFileSync('git', ['-C', repo, 'rev-parse', ref], { encoding: 'utf8' }).trim();

const round = (n) => Math.round(n * 1e5) / 1e5;
const vec = (x, y, z) => ({ x: round(x), y: round(y), z: round(z) });

function shot(pack, label, massScale, speedScale, note) {
  const { nodes, nodeSizes } = pack.scenario;
  const top = Math.max(...nodes.map((n, i) => n.centroid.y + (nodeSizes?.[i]?.y ?? 0) / 2));
  const p = pack.defaults.projectile;
  const target = [0, Math.max(1.5, 0.45 * top), 0];
  const origin = [target[0] - 12, target[1] + 1.5, target[2] - 2.25];
  const d = target.map((t, k) => t - origin[k]);
  const len = Math.hypot(...d);
  const speed = p.speed * speedScale * 1.05;
  return {
    label,
    massScale,
    speedScale,
    radius: p.radius,
    mass: round(p.mass * massScale),
    origin: vec(...origin),
    velocity: vec(...d.map((x) => (x / len) * speed)),
    angularVelocity: vec(1, 2, 0.5),
    launchAt: 2,
    note,
  };
}

const SCENES = {
  'crush-wall': (pack) => [
    shot(pack, '1.0x', 1, 1.0, 'ctest crush_wall_bite / crush_wall_bond_only'),
    shot(pack, '1.1x', 1, 1.1, 'record_crush_videos wall-mid: bond-only cracks internally, nothing visible'),
    shot(pack, '1.4x', 1, 1.4, 'record_crush_videos wall-punch: crush off, plug blown through, ball exits at 36.5 m/s; crush on, wall absorbs 5.2 of 6.3 MJ, ball 2.7 m/s'),
  ],
  'reference-building': (pack) => [
    shot(pack, 'ordinary', 1, 1.0, 'chunk-crushing.md ordinary impact'),
    shot(pack, '2x', 2, 1.0, 'record_crush_videos building'),
    shot(pack, '8x', 8, 1.0, 'ctest reference_building_crush_hard_impact'),
    shot(pack, '16x', 16, 1.0, 'chunk-crushing.md'),
  ],
};
SCENES['reference-building-crush'] = SCENES['reference-building'];

const index = [];
for (const [key, shots] of Object.entries(SCENES)) {
  const pack = JSON.parse(show(`blast/blast-stress-solver/assets/reference/${key}.json`));
  pack.defaults.shots = shots(pack);
  pack.defaults.source = {
    repository: 'blast-stress-solver',
    commit,
    path: `blast/blast-stress-solver/assets/reference/${key}.json`,
  };
  const out = path.join(here, `${key}.json`);
  writeFileSync(out, `${JSON.stringify(pack, null, 1)}\n`);
  console.log(`wrote ${out}: ${pack.scenario.nodes.length} nodes, ${pack.scenario.bonds.length} bonds, ${pack.defaults.shots.length} shots`);
  index.push({ key, title: pack.title, file: `${key}.json`, version: pack.version, nodes: pack.scenario.nodes.length, bonds: pack.scenario.bonds.length, shots: pack.defaults.shots.map((s) => s.label) });
}
// The Fracture Lab's Reference rigs menu (main.mjs).
writeFileSync(path.join(here, 'index.json'), `${JSON.stringify({ packs: index }, null, 1)}\n`);
