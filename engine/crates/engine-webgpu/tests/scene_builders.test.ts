// Run with: node --experimental-transform-types --test engine/crates/engine-webgpu/tests/scene_builders.test.ts
// Import the pinned implementation itself: no duplicated ring-generation oracle.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { brickRing, brickGables } from '../../../../inspiration/pinned/three-avbd/src/avbd3d/bench-scenes.ts';
import { Solver } from '../../../../inspiration/pinned/three-avbd/src/avbd3d/ref/solver.ts';
import { brickQuaternion, rollingAngularVelocity, groundHalfExtent, boxInvInertia, sphereMassProperties, buildRing, buildGables } from '../scene-builders.mjs';

function close(actual: ArrayLike<number>, expected: ArrayLike<number>, context: string) {
  assert.equal(actual.length, expected.length, context);
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) < 1e-9,
      `${context}, component ${i}: actual=${actual[i]}, reference=${expected[i]}. Swapping Y/Z also reverses axial vectors.`);
  }
}
const polar = (v: ArrayLike<number>) => [v[0], v[2], v[1]];
const axial = (v: ArrayLike<number>) => [-v[0], -v[2], -v[1]];

for (const [name, count, build, reference] of [
  ['ring 9k', 9131, (brick, ball) => buildRing(brick, ball, 12, 20, 2, 10), s => brickRing(s, 12, 20, 2, 10)],
  ['ring 110k', 110441, (brick, ball) => buildRing(brick, ball, 80, 40, 4, 10), s => brickRing(s)],
  ['gables 506k', 505922, (brick, ball) => buildGables(brick, ball, 16, 68, 30), s => brickGables(s)],
] as const) {
  test(`${name}: every body matches pinned three-avbd after coordinate conversion`, () => {
    const solver = new Solver();
    reference(solver);
    let index = 1; // ground is supplied separately by the demo
    const check = (p, q, mass, inertia, v = [0, 0, 0], w = [0, 0, 0]) => {
      const b = solver.bodies[index++];
      assert.ok(b, `${name}: extra body ${index - 1}`);
      close(p, polar(b.positionLin), `${name} body ${index - 1} position`);
      close(q, [...axial(b.positionAng), b.positionAng[3]], `${name} body ${index - 1} orientation`);
      close(inertia, polar(b.moment).map(x => 1 / x), `${name} body ${index - 1} inverse inertia`);
      close([mass], [b.mass], `${name} body ${index - 1} mass`);
      close(v, polar(b.velocityLin), `${name} body ${index - 1} velocity`);
      close(w, axial(b.velocityAng), `${name} body ${index - 1} angular velocity`);
    };
    build((p, angle, mass) => check(p, brickQuaternion(angle), mass, boxInvInertia(mass, 0.5, 0.25, 0.25)),
      (p, r, density, v, w) => {
        const { mass, invInertia } = sphereMassProperties(r, density);
        check(p, [0, 0, 0, 1], mass, [invInertia, invInertia, invInertia], v, w);
      });
    assert.equal(index, solver.bodies.length, `${name}: missing bricks or impactor`);
    assert.equal(index - 1, count, `${name}: changed benchmark workload`);
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
  for (const radius of [12, 80]) {
    const solver = new Solver();
    brickRing(solver, radius, 2, 1, 10);
    close([groundHalfExtent(radius + 10), 0.5, groundHalfExtent(radius + 10)],
      polar(solver.bodies[0].size).map(x => x / 2), `radius ${radius}: ground half extents`);
    close(polar(solver.bodies[0].positionLin), [0, 0, 0], 'ground center');
  }
});
