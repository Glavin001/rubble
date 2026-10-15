// Our Y-up scene builders against pinned three-avbd's bench scenes, recorded in
// fixtures/three-avbd-reference.json (regenerate with
// fixtures/generate-three-avbd-reference.ts; nothing here reads inspiration/).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { brickQuaternion, rollingAngularVelocity, groundHalfExtent, boxInvInertia, sphereMassProperties, buildRing, buildGables } from '../scene-builders.mjs';

const reference = JSON.parse(readFileSync(new URL('./fixtures/three-avbd-reference.json', import.meta.url), 'utf8'));

function close(actual, expected, context, tol = 1e-9) {
  assert.equal(actual.length, expected.length, context);
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) <= tol * Math.max(1, Math.abs(expected[i])),
      `${context}, component ${i}: actual=${actual[i]}, reference=${expected[i]}. Swapping Y/Z also reverses axial vectors.`);
  }
}
// Engine Y-up -> three-avbd Z-up: polar vectors swap y/z; axial vectors also flip.
const polar = v => [v[0], v[2], v[1]];
const axial = v => [-v[0], -v[2], -v[1]];

for (const [name, key, count, build] of [
  ['ring 9k', 'ring9k', 9131, (brick, ball) => buildRing(brick, ball, 12, 20, 2, 10)],
  ['ring 110k', 'ring110k', 110441, (brick, ball) => buildRing(brick, ball, 80, 40, 4, 10)],
  ['gables 506k', 'gables506k', 505922, (brick, ball) => buildGables(brick, ball, 16, 68, 30)],
]) {
  test(`${name}: every sampled body and the scene totals match pinned three-avbd after coordinate conversion`, () => {
    const ref = reference.scenes[key];
    const samples = new Map(ref.samples.map(s => [s.index, s]));
    const sums = Object.fromEntries(Object.entries(ref.sums).map(([k, v]) => [k, v.map(() => 0)]));
    let index = 1; // ground is supplied separately by the demo
    const add = (p, q, mass, invInertia, v = [0, 0, 0], w = [0, 0, 0]) => {
      const body = {
        positionLin: polar(p), positionAng: [...axial(q), q[3]], mass,
        moment: polar(invInertia).map(x => 1 / x), velocityLin: polar(v), velocityAng: axial(w),
      };
      for (const [k, total] of Object.entries(sums)) {
        const value = k === 'mass' ? [body.mass] : body[k];
        value.forEach((x, c) => { total[c] += Number.isFinite(x) ? x : 0; });
      }
      const s = samples.get(index);
      if (s) for (const k of ['positionLin', 'positionAng', 'moment', 'velocityLin', 'velocityAng'])
        close(body[k], s[k], `${name} body ${index} ${k}`);
      if (s) close([mass], [s.mass], `${name} body ${index} mass`);
      index++;
    };
    build((p, angle, mass) => add(p, brickQuaternion(angle), mass, boxInvInertia(mass, 0.5, 0.25, 0.25)),
      (p, r, density, v, w) => {
        const { mass, invInertia } = sphereMassProperties(r, density);
        add(p, [0, 0, 0, 1], mass, [invInertia, invInertia, invInertia], v, w);
      });
    assert.equal(index - 1, ref.count, `${name}: missing or extra bricks/impactor vs pinned three-avbd`);
    assert.equal(index - 1, count, `${name}: changed benchmark workload`);
    // Totals catch a wrong body anywhere, not only at sampled indices.
    for (const [k, total] of Object.entries(sums)) close(total, ref.sums[k], `${name} sum of ${k}`, 1e-9);
  });
}

test('rolling sphere has zero contact-point velocity at the floor', () => {
  for (const radius of [4, 8, 30 / 7]) {
    const w = rollingAngularVelocity(30, radius);
    // v_contact = v_center + omega cross (0, -r, 0).
    close([0, 0, 30 - w[0] * radius], [0, 0, 0], `radius ${radius}: sphere spins against travel`);
  }
});

test('ring ground dimensions match the reference slab', () => {
  for (const { radius, size, positionLin } of reference.ground) {
    close([groundHalfExtent(radius + 10), 0.5, groundHalfExtent(radius + 10)],
      polar(size).map(x => x / 2), `radius ${radius}: ground half extents`);
    close(polar(positionLin), [0, 0, 0], 'ground center');
  }
});
