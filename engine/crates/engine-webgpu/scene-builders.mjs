// Benchmark scenes in the engine's Y-up frame (tests/scene_builders.test.mjs
// checks their geometric and physical invariants). Brick yaw is a rotation
// about +Y; impactors roll without slipping (v + w x (0, -r, 0) = 0).
export const brickQuaternion = angle => [0, -Math.sin(angle / 2), 0, Math.cos(angle / 2)];
export const rollingAngularVelocity = (speed, radius) => [speed / radius, 0, 0];

export const groundHalfExtent = extent => Math.max(100, 2 * extent);

export function boxInvInertia(mass, hx, hy, hz) {
  return [3 / (mass * (hy * hy + hz * hz)),
    3 / (mass * (hx * hx + hz * hz)), 3 / (mass * (hx * hx + hy * hy))];
}
export function sphereMassProperties(radius, density) {
  const mass = density * (4 / 3) * Math.PI * radius ** 3;
  return { mass, invInertia: 1 / (0.4 * mass * radius * radius) };
}

export function buildRing(addBrick, addBall, radius, courses, tier, rows) {
  const mass = 0.25;
  for (let c = 0; c < courses; c++) {
    const depth = Math.max(1, rows - Math.floor(c / tier));
    for (let j = 0; j < depth; j++) {
      const r = radius + 0.25 + 0.52 * j;
      const n = Math.floor((2 * Math.PI * (r - 0.25)) / 1.02);
      for (let i = 0; i < n; i++) {
        const a = ((i + (c % 2) * 0.5) / n) * 2 * Math.PI;
        addBrick([r * Math.cos(a), 0.75 + 0.5 * c, r * Math.sin(a)], a + Math.PI / 2, mass, 1);
      }
    }
  }
  const ballR = courses / 5;
  addBall([0, 0.5 + ballR, -(radius + 0.52 * rows + ballR + 2)], ballR, 10, [0, 0, 30], rollingAngularVelocity(30, ballR));
}
export function buildGables(addBrick, addBall, columns, rows, base) {
  const pitchX = base + 2, pitchY = 5, mass = 0.25;
  for (let cx = 0; cx < columns; cx++) {
    for (let cy = 0; cy < rows; cy++) {
      const x0 = (cx - (columns - 1) / 2) * pitchX;
      const z = (cy - (rows - 1) / 2) * pitchY;
      for (let k = 0; k < base; k++) {
        const m = base - k;
        for (let i = 0; i < m; i++) {
          addBrick([x0 + (i - (m - 1) / 2) * 1.01, 0.75 + 0.5 * k, z], 0, mass, 1);
        }
      }
    }
  }
  const ballR = base / 7;
  for (const cx of [columns / 2 - 2, columns / 2 + 1]) {
    const x = (cx - (columns - 1) / 2) * pitchX;
    const z = -((rows - 1) / 2) * pitchY - ballR - 12;
    addBall([x, 0.5 + ballR, z], ballR, 50, [0, 0, 30], rollingAngularVelocity(30, ballR));
  }
}
