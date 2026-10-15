// Physical and geometric invariants of the benchmark scene builders (Y-up).
// Ring: running-bond courses of 1 x 0.5 x 0.5 bricks on concentric rings, a
// stepped profile (one ring fewer every `tier` courses) and a rolling impactor.
// Gables: stepped brick gables on a grid, each course centred over the joints
// below, and two rolling impactors.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { brickQuaternion, rollingAngularVelocity, groundHalfExtent, boxInvInertia, sphereMassProperties, buildRing, buildGables } from '../scene-builders.mjs';

const BRICK = [1, 0.5, 0.5]; // full extents; half extents 0.5, 0.25, 0.25
const GROUND_TOP = 0.5;      // ground slab centred at the origin, 1 m thick
const EPS = 1e-9;

const near = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b}`);
const rotate = (q, v) => { // unit quaternion (x, y, z, w)
  const [x, y, z, w] = q;
  const t = [2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])];
  return [v[0] + w * t[0] + (y * t[2] - z * t[1]), v[1] + w * t[1] + (z * t[0] - x * t[2]), v[2] + w * t[2] + (x * t[1] - y * t[0])];
};
// Solid box inertia from full extents (written independently of boxInvInertia).
const boxInertia = (m, [a, b, c]) => [m * (b * b + c * c) / 12, m * (a * a + c * c) / 12, m * (a * a + b * b) / 12];

function collect(build) {
  const bricks = [], balls = [];
  build((p, angle, mass, density) => bricks.push({ p, angle, mass, density }),
    (p, r, density, v, w) => balls.push({ p, r, density, v, w }));
  return { bricks, balls };
}

function checkBrickBody({ p, angle, mass }, label) {
  assert.ok(p.every(Number.isFinite), `${label}: finite position`);
  near(mass, 0.25, EPS, `${label}: brick mass`);
  const q = brickQuaternion(angle);
  near(Math.hypot(...q), 1, 1e-12, `${label}: unit orientation`);
  near(q[0], 0, EPS, `${label}: yaw only (x)`); near(q[2], 0, EPS, `${label}: yaw only (z)`);
  const inv = boxInvInertia(mass, BRICK[0] / 2, BRICK[1] / 2, BRICK[2] / 2);
  boxInertia(mass, BRICK).forEach((I, k) => near(inv[k] * I, 1, 1e-12, `${label}: inverse inertia axis ${k}`));
  return q;
}

function checkBall(ball, bounds, label) {
  const { mass, invInertia } = sphereMassProperties(ball.r, ball.density);
  near(mass, ball.density * 4 / 3 * Math.PI * ball.r ** 3, 1e-9 * mass, `${label}: sphere mass`);
  near(invInertia * 0.4 * mass * ball.r * ball.r, 1, 1e-12, `${label}: solid-sphere inertia`);
  near(ball.p[1] - ball.r, GROUND_TOP, EPS, `${label}: resting on the ground`);
  // Rolls without slipping: v + w x (0, -r, 0) = 0 at the contact point.
  const contact = [ball.v[0] + ball.w[2] * ball.r, ball.v[1], ball.v[2] - ball.w[0] * ball.r];
  contact.forEach((x, k) => near(x, 0, 1e-9, `${label}: contact velocity axis ${k}`));
  assert.ok(ball.v[2] > 0 && ball.p[2] + ball.r < bounds.minZ, `${label}: starts outside the structure, rolling towards it`);
}

for (const [name, radius, courses, tier, rows, count] of [
  ['ring 9k', 12, 20, 2, 10, 9131],
  ['ring 110k', 80, 40, 4, 10, 110441],
]) {
  test(`${name}: running-bond brick rings sit on their course and ring radii without overlap`, () => {
    const { bricks, balls } = collect((brick, ball) => buildRing(brick, ball, radius, courses, tier, rows));
    assert.equal(bricks.length + balls.length, count, `${name}: changed benchmark workload`);
    assert.equal(balls.length, 1);
    const rings = new Map(); // "course,ring" -> angles
    let minZ = Infinity;
    for (const [i, b] of bricks.entries()) {
      const label = `${name} brick ${i}`;
      const q = checkBrickBody(b, label);
      const course = (b.p[1] - 0.75) / 0.5;
      near(course, Math.round(course), 1e-9, `${label}: course height`);
      const c = Math.round(course);
      assert.ok(c >= 0 && c < courses, `${label}: course ${c}`);
      // Bottom course rests on the ground; each course stacks one brick height up.
      near(b.p[1] - BRICK[1] / 2, GROUND_TOP + BRICK[1] * c, 1e-9, `${label}: stacked height`);
      const r = Math.hypot(b.p[0], b.p[2]);
      const j = Math.round((r - radius - 0.25) / 0.52);
      near(r, radius + 0.25 + 0.52 * j, 1e-9, `${label}: ring radius`);
      // Stepped profile: course c keeps max(1, rows - floor(c / tier)) rings.
      assert.ok(j >= 0 && j < Math.max(1, rows - Math.floor(c / tier)), `${label}: ring ${j} beyond the step profile`);
      // Long axis tangent to the ring, short axis radial.
      const radial = [b.p[0] / r, 0, b.p[2] / r];
      const along = rotate(q, [1, 0, 0]);
      near(Math.abs(along[0] * radial[0] + along[2] * radial[2]), 0, 1e-9, `${label}: long axis tangent`);
      const key = `${c},${j}`;
      if (!rings.has(key)) rings.set(key, []);
      rings.get(key).push(Math.atan2(b.p[2], b.p[0]));
      minZ = Math.min(minZ, b.p[2] - 0.5);
    }
    for (const [key, angles] of rings) {
      const [c, j] = key.split(',').map(Number);
      const n = angles.length, r = radius + 0.25 + 0.52 * j, inner = r - BRICK[2] / 2;
      angles.sort((a, b) => a - b);
      // Evenly spaced, and as many bricks as fit: inner-face arc per brick >= 1.02
      // (1 m brick + 2 cm joint), one more brick would not fit.
      for (let k = 1; k < n; k++) near(angles[k] - angles[k - 1], 2 * Math.PI / n, 1e-9, `${name} course ${c} ring ${j}: even spacing`);
      assert.ok(2 * Math.PI * inner / n >= 1.02 - 1e-9 && 2 * Math.PI * inner / (n + 1) < 1.02, `${name} course ${c} ring ${j}: ${n} bricks is not the most that fit`);
      // Neighbouring centres are at least a brick length apart (no overlap).
      assert.ok(2 * r * Math.sin(Math.PI / n) >= BRICK[0], `${name} course ${c} ring ${j}: neighbours overlap`);
      // Absolute phase: even courses start at angle 0, odd courses half a brick on,
      // which with radius and height pins every brick position exactly.
      const step = 2 * Math.PI / n;
      const offset = (((angles[0] - (c % 2) * step / 2) % step) + step) % step; // circular
      near(Math.min(offset, step - offset), 0, 1e-9, `${name} course ${c} ring ${j}: phase`);
      // Running bond: course c+1 is rotated by half a brick against course c.
      const above = rings.get(`${c + 1},${j}`);
      if (above && above.length === n) {
        const shift = ((above[0] - angles[0]) % (2 * Math.PI / n) + 2 * Math.PI / n) % (2 * Math.PI / n);
        near(shift, Math.PI / n, 1e-9, `${name} course ${c}->${c + 1} ring ${j}: half-brick offset`);
      }
    }
    // Every course has exactly its stepped set of rings 0 .. depth-1.
    for (let c = 0; c < courses; c++) {
      const depth = Math.max(1, rows - Math.floor(c / tier));
      for (let j = 0; j < depth + 1; j++) assert.equal(rings.has(`${c},${j}`), j < depth, `${name} course ${c}: ring ${j} presence`);
    }
    // Rings are 0.52 m apart: a 2 cm radial joint between 0.5 m deep bricks.
    near(0.52 - BRICK[2], 0.02, 1e-12, 'radial joint');
    const ball = balls[0];
    near(ball.r, courses / 5, EPS, `${name}: impactor radius`);
    near(ball.p[0], 0, EPS, `${name}: impactor aimed at the ring centre`);
    checkBall(ball, { minZ }, `${name} impactor`);
  });
}

test('gables 506k: stepped gables stand on the joints below, never overlap, and two impactors line up with gable columns', () => {
  const columns = 16, rows = 68, base = 30;
  const { bricks, balls } = collect((brick, ball) => buildGables(brick, ball, columns, rows, base));
  assert.equal(bricks.length, columns * rows * base * (base + 1) / 2, 'triangular gables: base + (base-1) + ... + 1 per gable');
  assert.equal(bricks.length + balls.length, 505922, 'changed benchmark workload');
  const byGable = new Map();
  let minZ = Infinity;
  for (const [i, b] of bricks.entries()) {
    checkBrickBody(b, `gables brick ${i}`);
    near(b.angle, 0, EPS, `gables brick ${i}: aligned with X`);
    const k = Math.round((b.p[1] - 0.75) / 0.5);
    near(b.p[1] - BRICK[1] / 2, GROUND_TOP + BRICK[1] * k, 1e-9, `gables brick ${i}: stacked height`);
    const key = `${Math.round(b.p[0] / (base + 2) + (columns - 1) / 2)},${b.p[2]}`; // column index, row z
    if (!byGable.has(key)) byGable.set(key, []);
    byGable.get(key).push({ x: b.p[0], k });
    minZ = Math.min(minZ, b.p[2] - 0.25);
  }
  assert.equal(byGable.size, columns * rows);
  const centres = [];
  for (const [key, list] of byGable) {
    const courses = new Map();
    for (const { x, k } of list) (courses.get(k) ?? courses.set(k, []).get(k)).push(x);
    const xs0 = courses.get(0).sort((a, b) => a - b);
    const centre = (xs0[0] + xs0.at(-1)) / 2;
    centres.push(centre);
    for (let k = 0; k < base; k++) {
      const xs = (courses.get(k) ?? []).sort((a, b) => a - b);
      assert.equal(xs.length, base - k, `gable ${key} course ${k}: brick count`);
      near((xs[0] + xs.at(-1)) / 2, centre, 1e-9, `gable ${key} course ${k}: centred`);
      for (let n = 1; n < xs.length; n++) near(xs[n] - xs[n - 1], 1.01, 1e-9, `gable ${key} course ${k}: 1 cm joints`);
      // Every brick above course 0 bridges the joint between two bricks below.
      if (k > 0) {
        const below = courses.get(k - 1);
        for (const x of xs) assert.equal(below.filter(y => Math.abs(Math.abs(x - y) - 0.505) < 1e-9).length, 2, `gable ${key} course ${k}: brick at ${x} is not supported by two bricks`);
      }
    }
  }
  // Neighbouring gables never overlap: X pitch exceeds the widest course, Z pitch the depth.
  const width = (base - 1) * 1.01 + BRICK[0];
  assert.ok(base + 2 > width, 'X pitch leaves a gap between gables');
  assert.ok(5 > BRICK[2], 'Z pitch leaves a gap between gable rows');
  assert.equal(balls.length, 2);
  for (const [i, ball] of balls.entries()) {
    near(ball.r, base / 7, EPS, `gables impactor ${i}: radius`);
    assert.ok(centres.some(c => Math.abs(c - ball.p[0]) < 1e-9), `gables impactor ${i}: not aligned with a gable column`);
    checkBall(ball, { minZ }, `gables impactor ${i}`);
  }
});

test('ground slab covers every body and the impactor run-up, with a 100 m minimum', () => {
  near(groundHalfExtent(10), 100, 0, 'minimum half extent');
  for (const [radius, courses, rows] of [[12, 20, 10], [80, 40, 10]]) {
    const { bricks, balls } = collect((brick, ball) => buildRing(brick, ball, radius, courses, 2, rows));
    const half = groundHalfExtent(radius + 10);
    for (const b of [...bricks, ...balls]) assert.ok(Math.abs(b.p[0]) < half && Math.abs(b.p[2]) < half, `radius ${radius}: body off the ground slab`);
  }
});

test('rolling angular velocity spins against the direction of travel', () => {
  for (const radius of [4, 8, 30 / 7]) {
    const w = rollingAngularVelocity(30, radius);
    near(30 - w[0] * radius, 0, 1e-12, `radius ${radius}: contact-point speed`);
    assert.equal(w[1], 0); assert.equal(w[2], 0);
  }
});
