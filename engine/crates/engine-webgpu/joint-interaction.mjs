// Interaction for Rust-exported joint scenes (thin: only patches GPU records).
//
// Ctrl/Alt-drag grabs a body with the scene's soft drag joint (three-avbd
// mouse spring). F launches the projectile, M toggles every velocity motor,
// and position servos oscillate around their exported targets. All record
// offsets come from the snapshot metadata (engine-schema `Joint`).

const BODY_STRIDE = 112;

export function installJointControls({ device, bodyBuffer, jointBuffer, snap, maxBodies, canvas, camera }) {
  const meta = snap.meta;
  const L = meta.jointLayout;
  const scratch = new ArrayBuffer(16);
  const f32 = new Float32Array(scratch);
  const u32 = new Uint32Array(scratch);
  const readback = device.createBuffer({
    size: meta.renderBodies * BODY_STRIDE,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const radius = new Float32Array(meta.renderBodies);
  {
    const cv = new DataView(snap.colliders.buffer, snap.colliders.byteOffset, snap.colliders.byteLength);
    for (let i = 0; i < meta.renderBodies; i++) radius[i] = cv.getFloat32(i * 64 + 60, true);
  }
  const templateFlags = new DataView(snap.joints.buffer, snap.joints.byteOffset + meta.dragJoint * L.stride + L.flags, 4)
    .getUint32(0, true) & ~L.brokenFlag;

  let motorsOn = true;
  let grab = null;
  let busy = false;

  const writeU32 = (buf, offset, values) => device.queue.writeBuffer(buf, offset, new Uint32Array(values));
  const writeF32 = (buf, offset, values) => device.queue.writeBuffer(buf, offset, new Float32Array(values));
  const jointOff = (id, field) => id * L.stride + field;

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

  async function readBodies() {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(bodyBuffer, 0, readback, 0, meta.renderBodies * BODY_STRIDE);
    device.queue.submit([enc.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    return new DataView(copy);
  }

  function setMouse(p) {
    writeF32(bodyBuffer, meta.mouseBody * BODY_STRIDE, p);
    writeF32(bodyBuffer, (maxBodies + meta.mouseBody) * BODY_STRIDE, p);
  }

  async function startGrab(e) {
    if (busy) return;
    busy = true;
    try {
      const { eye, dir } = ray(e);
      const bodies = await readBodies();
      let best = -1, bestT = Infinity;
      for (let i = 1; i < meta.renderBodies; i++) {
        const o = i * BODY_STRIDE;
        if (!(bodies.getFloat32(o + 12, true) > 0)) continue; // dynamic only
        const c = [bodies.getFloat32(o, true), bodies.getFloat32(o + 4, true), bodies.getFloat32(o + 8, true)];
        const oc = eye.map((v, k) => v - c[k]);
        const b = oc[0] * dir[0] + oc[1] * dir[1] + oc[2] * dir[2];
        const h = b * b - (oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - radius[i] * radius[i]);
        if (h < 0) continue;
        const t = -b - Math.sqrt(h);
        if (t > 0 && t < bestT) { bestT = t; best = i; }
      }
      if (best < 0) return;
      const o = best * BODY_STRIDE;
      const pos = [bodies.getFloat32(o, true), bodies.getFloat32(o + 4, true), bodies.getFloat32(o + 8, true)];
      const q = [bodies.getFloat32(o + 16, true), bodies.getFloat32(o + 20, true), bodies.getFloat32(o + 24, true), bodies.getFloat32(o + 28, true)];
      const hit = eye.map((v, k) => v + dir[k] * bestT);
      // Local anchor = conj(q) * (hit - pos).
      const d = hit.map((v, k) => v - pos[k]);
      const [x, y, z, w] = [-q[0], -q[1], -q[2], q[3]];
      const tx = 2 * (y * d[2] - z * d[1]), ty = 2 * (z * d[0] - x * d[2]), tz = 2 * (x * d[1] - y * d[0]);
      const local = [d[0] + w * tx + (y * tz - z * ty), d[1] + w * ty + (z * tx - x * tz), d[2] + w * tz + (x * ty - y * tx)];
      setMouse(hit);
      // Wake the grabbed body (a static mouse body never wakes a sleeper).
      const flags = bodies.getUint32(o + 44, true);
      writeU32(bodyBuffer, o + 44, [((flags & ~4) | 8) >>> 0]);
      writeF32(jointBuffer, jointOff(meta.dragJoint, L.anchorB), local);
      writeU32(jointBuffer, jointOff(meta.dragJoint, L.bodyA), [meta.mouseBody, best, templateFlags]);
      grab = { id: e.pointerId, depth: bestT, body: best };
    } finally {
      busy = false;
    }
  }

  function moveGrab(e) {
    if (!grab || grab.id !== e.pointerId) return false;
    const { eye, dir } = ray(e);
    setMouse(eye.map((v, k) => v + dir[k] * grab.depth));
    return true;
  }

  function endGrab(e) {
    if (!grab || (e && grab.id !== e.pointerId)) return false;
    writeU32(jointBuffer, jointOff(meta.dragJoint, L.bodyA), [0, 0, (templateFlags | L.brokenFlag) >>> 0]);
    grab = null;
    return true;
  }

  function fire() {
    const { eye, dir } = ray({ clientX: canvas.getBoundingClientRect().left + canvas.clientWidth / 2, clientY: canvas.getBoundingClientRect().top + canvas.clientHeight / 2 });
    const r = radius[meta.projectile] || 0.35;
    const mass = 8;
    const inertia = 0.4 * mass * r * r;
    const rec = new ArrayBuffer(BODY_STRIDE);
    const v = new DataView(rec);
    const p = eye.map((c, k) => c + dir[k] * 2);
    const vel = dir.map((c) => c * 30);
    [p[0], p[1], p[2], 1 / mass, 0, 0, 0, 1, 1 / inertia, 1 / inertia, 1 / inertia].forEach((x, k) => v.setFloat32(k * 4, x, true));
    v.setUint32(44, 1 | 8, true);
    vel.forEach((x, k) => v.setFloat32(48 + k * 4, x, true));
    [p[0], p[1], p[2], 0, 0, 0, 0, 1].forEach((x, k) => v.setFloat32(80 + k * 4, x, true));
    device.queue.writeBuffer(bodyBuffer, meta.projectile * BODY_STRIDE, rec);
    device.queue.writeBuffer(bodyBuffer, (maxBodies + meta.projectile) * BODY_STRIDE, rec);
  }

  function toggleMotors() {
    motorsOn = !motorsOn;
    for (const m of meta.motors) writeF32(jointBuffer, jointOff(m.id, L.driveVelocity), [motorsOn ? m.velocity : 0]);
    return motorsOn;
  }

  // Servos sweep sinusoidally between ±target (3 s period).
  function beforeStep(t) {
    for (const s of meta.servos) writeF32(jointBuffer, jointOff(s.id, L.driveTarget), [s.target * Math.sin(t * 2 * Math.PI / 3)]);
  }

  addEventListener('keydown', (e) => {
    if (e.key === 'f' || e.key === 'F') fire();
    else if (e.key === 'm' || e.key === 'M') toggleMotors();
  });

  return {
    startGrab, moveGrab, endGrab, beforeStep, fire, toggleMotors,
    legend: `Ctrl/Alt-drag grabs a body (soft drag joint) · F fires a ball · M toggles ${meta.motors.length} motors · ${meta.servos.length} servo(s) sweep`,
  };
}
