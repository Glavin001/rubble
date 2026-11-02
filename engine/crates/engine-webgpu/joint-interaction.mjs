// Interaction for Rust-exported joint scenes on World (engine-world-wasm): the
// page only calls World.
//
// Ctrl/Alt-drag grabs a body with a soft drag joint (three-avbd mouse spring:
// a ball joint from the scene's static mouse body, 5000 N/m, no angular rows).
// F launches the projectile, M toggles every velocity motor, and position
// servos oscillate around their exported targets. Joint and body handles come
// from `World.importSnapshot`; the motors' and servos' drive values from the
// snapshot's metadata and records.

// engine-scenes scene_export's DRAG_STIFFNESS (three-avbd): its drag record's rows.
const DRAG_STIFFNESS = 5000;

export function installJointControls({ world, snap, handles, canvas, camera, ccd = true }) {
  const meta = snap.meta;
  const L = meta.jointLayout;
  const radius = new Float32Array(meta.renderBodies);
  {
    const cv = new DataView(snap.colliders.buffer, snap.colliders.byteOffset, snap.colliders.byteLength);
    for (let i = 0; i < meta.renderBodies && i * 64 + 64 <= snap.colliders.byteLength; i++) radius[i] = cv.getFloat32(i * 64 + 60, true);
  }
  // Each drive's other value (a motor's target, a servo's speed), read from its record.
  const jv = new DataView(snap.joints.buffer, snap.joints.byteOffset, snap.joints.byteLength);
  const drive = (id) => ({ target: jv.getFloat32(id * L.stride + L.driveTarget, true), velocity: jv.getFloat32(id * L.stride + L.driveVelocity, true) });
  const joint = (id) => {
    const h = handles.joints[id];
    if (h === null || h === undefined) throw new Error(`joint record ${id} has no joint (broken in the snapshot)`);
    return h;
  };
  const motors = meta.motors.map((m) => ({ ...m, handle: joint(m.id), target: drive(m.id).target }));
  const servos = meta.servos.map((s) => ({ ...s, handle: joint(s.id), velocity: drive(s.id).velocity }));
  const mouse = handles.bodies[meta.mouseBody];
  const projectile = handles.bodies[meta.projectile];
  // A parked projectile: continuous collision once it is launched dynamic.
  if (ccd) world.bodySetBullet(projectile, true);
  const renderSlots = Array.from({ length: meta.renderBodies }, (_, i) => world.bodySlot(handles.bodies[i]));
  // Grabbable: the dynamic records, and the projectile once launched.
  const bv = new DataView(snap.bodies.buffer, snap.bodies.byteOffset, snap.bodies.byteLength);
  const dynamic = Uint8Array.from({ length: meta.renderBodies }, (_, i) => (bv.getFloat32(i * 112 + 12, true) > 0 ? 1 : 0));

  let motorsOn = true;
  let grab = null;
  let busy = false;

  function ray(e) {
    const rect = canvas.getBoundingClientRect();
    const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const ny = 1 - ((e.clientY - rect.top) / rect.height) * 2;
    const eye = camera.eye();
    const target = camera.target();
    const norm = (a) => { const l = Math.hypot(...a) || 1; return a.map((v) => v / l); };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const fwd = norm(target.map((v, i) => v - eye[i]));
    const right = norm(cross(fwd, [0, 1, 0]));
    const up = cross(right, fwd);
    const t = Math.tan(camera.fovY / 2);
    const aspect = rect.width / Math.max(rect.height, 1);
    const dir = norm(fwd.map((v, i) => v + nx * t * aspect * right[i] + ny * t * up[i]));
    return { eye, dir };
  }

  async function startGrab(e) {
    if (busy || grab) return;
    busy = true;
    try {
      const { eye, dir } = ray(e);
      // Position xyz, rotation xyzw, velocities, awake (readBodies' 14 floats).
      const b = await world.readBodies(renderSlots);
      let best = -1, bestT = Infinity;
      for (let i = 1; i < meta.renderBodies; i++) {
        if (!dynamic[i]) continue;
        const o = i * 14;
        const c = [b[o], b[o + 1], b[o + 2]];
        const oc = eye.map((v, k) => v - c[k]);
        const p = oc[0] * dir[0] + oc[1] * dir[1] + oc[2] * dir[2];
        const h = p * p - (oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - radius[i] * radius[i]);
        if (h < 0) continue;
        const t = -p - Math.sqrt(h);
        if (t > 0 && t < bestT) { bestT = t; best = i; }
      }
      if (best < 0) return;
      const o = best * 14;
      const pos = [b[o], b[o + 1], b[o + 2]];
      const q = [b[o + 3], b[o + 4], b[o + 5], b[o + 6]];
      const hit = eye.map((v, k) => v + dir[k] * bestT);
      // Local anchor = conj(q) * (hit - pos).
      const d = hit.map((v, k) => v - pos[k]);
      const [x, y, z, w] = [-q[0], -q[1], -q[2], q[3]];
      const tx = 2 * (y * d[2] - z * d[1]), ty = 2 * (z * d[0] - x * d[2]), tz = 2 * (x * d[1] - y * d[0]);
      const local = [d[0] + w * tx + (y * tz - z * ty), d[1] + w * ty + (z * tx - x * tz), d[2] + w * tz + (x * ty - y * tx)];
      world.bodySetTransform(mouse, hit, [0, 0, 0, 1]);
      // A static mouse body never wakes a sleeper: wake the grabbed body.
      world.bodySetAwake(handles.bodies[best], true);
      const handle = world.createBallJoint({
        bodyA: mouse, bodyB: handles.bodies[best], anchorA: [0, 0, 0], anchorB: local,
        linearStiffness: DRAG_STIFFNESS, angularStiffness: 0, torqueArm: 1,
      });
      grab = { id: e.pointerId, depth: bestT, body: best, handle };
    } finally {
      busy = false;
    }
  }

  function moveGrab(e) {
    if (!grab || grab.id !== e.pointerId) return false;
    const { eye, dir } = ray(e);
    world.bodySetTransform(mouse, eye.map((v, k) => v + dir[k] * grab.depth), [0, 0, 0, 1]);
    return true;
  }

  function endGrab(e) {
    if (!grab || (e && grab.id !== e.pointerId)) return false;
    world.destroyJoint(grab.handle);
    grab = null;
    return true;
  }

  function fire() {
    const { eye, dir } = ray({ clientX: canvas.getBoundingClientRect().left + canvas.clientWidth / 2, clientY: canvas.getBoundingClientRect().top + canvas.clientHeight / 2 });
    const r = radius[meta.projectile] || 0.35;
    const mass = 8;
    const inertia = 0.4 * mass * r * r;
    const p = eye.map((c, k) => c + dir[k] * 2);
    world.bodySetType(projectile, 'dynamic', p, [0, 0, 0, 1], dir.map((c) => c * 30), [0, 0, 0]);
    world.bodySetMassData(projectile, mass, [0, 0, 0], [inertia, inertia, inertia]);
    dynamic[meta.projectile] = 1;
  }

  function toggleMotors() {
    motorsOn = !motorsOn;
    for (const m of motors) world.jointSetDrive(m.handle, motorsOn ? m.velocity : 0, m.target);
    return motorsOn;
  }

  // Servos sweep sinusoidally between ±target (3 s period).
  function beforeStep(t) {
    for (const s of servos) world.jointSetDrive(s.handle, s.velocity, s.target * Math.sin(t * 2 * Math.PI / 3));
  }

  // The page restored the scene's first state: the drag joint is gone, motors back on.
  function reset() {
    grab = null;
    motorsOn = true;
    dynamic[meta.projectile] = 0;
  }

  addEventListener('keydown', (e) => {
    if (e.key === 'f' || e.key === 'F') fire();
    else if (e.key === 'm' || e.key === 'M') toggleMotors();
  });

  return {
    startGrab, moveGrab, endGrab, beforeStep, fire, toggleMotors, reset,
    projectileSlot: world.bodySlot(projectile),
    legend: `Ctrl/Alt-drag grabs a body (soft drag joint) · F fires a ball · M toggles ${motors.length} motors · ${servos.length} servo(s) sweep`,
  };
}
