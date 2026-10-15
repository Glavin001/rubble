// vehicle.mjs must agree with the Slang layout and with the Rust packer.
// Run `cargo run -p engine-webgpu --bin wgsl_export` first (writes generated/vehicle_*.json).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { L, pack, requirements, bodyWords, colliderWords, centerOfMass, bitsf } from '../vehicle.mjs';

const slangPath = new URL('../../../slang/physics/vehicle_layout.slang', import.meta.url);
const goldenPath = new URL('../generated/vehicle_golden.json', import.meta.url);
const presetsPath = new URL('../generated/vehicle_presets.json', import.meta.url);

test('layout constants match slang/physics/vehicle_layout.slang', () => {
  const src = readFileSync(slangPath, 'utf8');
  const re = /static const uint (\w+) = (0x[0-9A-Fa-f]+|\d+)u;/g;
  let n = 0;
  for (const m of src.matchAll(re)) {
    const v = Number(m[2]);
    assert.ok(m[1] in L, `${m[1]} missing from vehicle.mjs`);
    assert.equal(L[m[1]], v, `${m[1]}: JS ${L[m[1]]} vs Slang ${v}`);
    n++;
  }
  assert.ok(n > 100, `parsed ${n} constants`);
});

// Integers must match exactly; float words may differ by f32 rounding of JS Math.
function sameWords(a, b, label) {
  assert.equal(a.length, b.length, `${label}: length`);
  for (let i = 0; i < a.length; i++) {
    const x = a[i] >>> 0, y = b[i] >>> 0;
    if (x === y) continue;
    const fx = bitsf(x), fy = bitsf(y);
    const ok = Number.isFinite(fx) && Number.isFinite(fy)
      && Math.abs(fx - fy) <= 2e-5 * Math.max(1, Math.abs(fx), Math.abs(fy));
    assert.ok(ok, `${label}[${i}]: ${x.toString(16)} (${fx}) vs ${y.toString(16)} (${fy})`);
  }
}

const flatten = (slots) => slots.flat().map((x) => x >>> 0);

test('JS packer matches the Rust golden output for every preset', { skip: !existsSync(goldenPath) }, () => {
  const golden = JSON.parse(readFileSync(goldenPath, 'utf8'));
  const presets = JSON.parse(readFileSync(presetsPath, 'utf8'));
  assert.equal(golden.length, presets.length);
  for (const g of golden) {
    const desc = presets.find((p) => p.name === g.name);
    const s = g.slots;
    const p = pack(desc, g.position, g.rotation, s);
    const req = requirements(desc);
    assert.deepEqual(req, g.requirements, `${g.name}: requirements`);
    const com = centerOfMass(desc);
    com.forEach((c, i) => assert.ok(Math.abs(c - g.centerOfMass[i]) < 1e-5, `${g.name}: COM`));
    sameWords(p.bodies.flatMap((b) => Array.from(bodyWords(b))), g.bodies, `${g.name} bodies`);
    sameWords(p.colliders.flatMap((c) => Array.from(colliderWords(c))), g.colliders, `${g.name} colliders`);
    sameWords(flatten(p.vehicle), g.vehicle.flat(), `${g.name} vehicle`);
    assert.equal(p.wheels.length, g.wheels.length);
    p.wheels.forEach((w, i) => sameWords(flatten(w), g.wheels[i].flat(), `${g.name} wheel ${i}`));
    sameWords(p.curvePoints.flat().map((x) => new Uint32Array(new Float32Array([x]).buffer)[0]), g.curvePoints, `${g.name} curves`);
    sameWords(p.diffs.flat(2), g.diffs.flat(2), `${g.name} diffs`);
    sameWords(p.antiRolls.flat(), g.antiRolls.flat(), `${g.name} antiRolls`);
    sameWords(p.parts.flat(2), g.parts.flat(2), `${g.name} parts`);
  }
});
