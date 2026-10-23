// Renders vibe-land's dune buggy (generated/dune-buggy/buggy-mesh.*, baked by
// engine/scenes/dune-buggy) on every `dune_buggy` vehicle of a world-host world.
//
// Each of the mesh's ~50 primitives is one instanced draw over every part that
// uses it, on every buggy. A part is drawn
// - on the chassis while its Rubble part is attached (part state in gVehicleData),
// - on that part's own body once its mount broke (P_PARK body, minus the part's
//   centre of mass, as detachPart re-bases its colliders),
// - with its wheel's free body once the wheel came off (wheel and hub parts),
// - not at all once disabled.
// Suspension, steering and wheel spin come from buggy-rig.mjs, posed in the
// chassis frame from each buggy's wheel records (a small pipelined readback): the
// chassis and loose bodies themselves are read by the vertex shader from the
// current GPU state, so nothing lags the physics but the suspension's own pose.
//
// Hybrid buggies (engine_softbody::buggy, `addHybrid`): the vehicle is the soft
// body's rigid core (wheels, rig, engine, gearbox, steering column); a frame
// tube spans its two nodes as they are now; other frame parts ride the frame of
// three nodes they are bound to; bolted parts follow their own bodies.

import { BuggyRig, CORNER_IDS, mul } from './buggy-rig.mjs';

const NONE = 0xffffffff;
// Instance flags (Inst.info.w).
const F_ABS = 1;    // the dynamic slot is the whole local matrix (beams, spring)
const F_WHEEL = 2;  // travels with its wheel's free body once detached
const F_RIG = 4;    // posed by the rig (dynamic slot)
// Instance modes (Inst.slot.z).
const M_VEHICLE = 0, M_BODY = 2, M_TUBE = 3, M_FRAME = 4;
// 4 × vec4f rest columns, vec4f colour, 3 × vec4u, 4 × vec4f extra matrix.
const INST_WORDS = 48;
const BUGGY_WORDS = 8;
const MAT_FLOATS = 16;

// Recipe frame → Rubble design frame: R(x, y, z) = (-x, y, -z).
const R = [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1];
const translation = (p) => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, p[0], p[1], p[2], 1];
const hex = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16) / 255);

/**
 * @param {object} o
 * @param {GPUDevice} o.device
 * @param {GPUTextureFormat} o.format
 * @param {object} o.world      world-host world (bufs, vehicles, vlayout)
 * @param {GPUBuffer} o.frameBuf  vehicle.html's Frame uniform (viewProj, eye, light)
 * @param {object} o.L           vehicle layout constants
 * @param {string} o.shadeWgsl   WGSL defining qrot() and shade(c, n, w) against `frame`
 * @param {string} [o.base]      URL of generated/dune-buggy/
 */
export async function createBuggyRenderer({ device, format, world, frameBuf, L, shadeWgsl, base = './generated/dune-buggy/' }) {
  const [mesh, bin] = await Promise.all([
    fetch(base + 'buggy-mesh.json', { cache: 'no-store' }).then((r) => { if (!r.ok) throw new Error(`buggy-mesh.json: HTTP ${r.status}`); return r.json(); }),
    fetch(base + 'buggy-mesh.bin', { cache: 'no-store' }).then((r) => { if (!r.ok) throw new Error(`buggy-mesh.bin: HTTP ${r.status}`); return r.arrayBuffer(); }),
  ]);
  const geo = device.createBuffer({ size: bin.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(geo, 0, bin);
  const palette = mesh.palette.map((p) => [...hex(p.color), p.metalness]);
  const rigDef = mesh.rig;

  // Per-buggy dynamic matrix slots: 0 static, 1..4 detached-wheel frames,
  // 5 steering wheel, 6..13 wheel/knuckle per corner, then one per beam/spring.
  const SLOT_STATIC = 0, SLOT_WHEEL_DET = 1, SLOT_STEERING = 5, SLOT_CORNER = 6;
  const absInstances = [];
  const slotOf = mesh.instances.map((inst) => {
    switch (inst.role) {
      case null: case undefined: return SLOT_STATIC;
      case 'steering': return SLOT_STEERING;
      case 'hub': case 'wheel': return SLOT_CORNER + 2 * inst.corner;
      case 'knuckle': return SLOT_CORNER + 2 * inst.corner + 1;
      default: absInstances.push(inst); return SLOT_CORNER + 8 + absInstances.length - 1;
    }
  });
  const slotsPerBuggy = SLOT_CORNER + 8 + absInstances.length;
  const instancesByTemplate = mesh.templates.map(() => []);
  mesh.instances.forEach((inst, i) => instancesByTemplate[inst.template].push(i));

  const buggies = []; // {h, entry, rig, origin, wheelRecSlot, neutralL[4], dyn: Float32Array}
  let instBuf = null, buggyBuf = null, dynBuf = null, bindGroup = null, draws = [];

  const shader = /* wgsl */`
struct Body { position: vec3f, invMass: f32, orientation: vec4f, invInertia: vec3f, flags: u32,
  linearVel: vec3f, sleepTimer: f32, angularVel: vec3f, generation: u32, initialPos: vec3f, initialPad: f32, initialRot: vec4f };
struct Frame { viewProj: mat4x4f, eye: vec4f, light: vec4f };
struct Inst { c0: vec4f, c1: vec4f, c2: vec4f, c3: vec4f, color: vec4f, info: vec4u, slot: vec4u, nodes: vec4u,
  e0: vec4f, e1: vec4f, e2: vec4f, e3: vec4f };
struct Buggy { vehicle: vec4u, extra: vec4u };
@group(0) @binding(0) var<storage, read> bodies: array<Body>;
@group(0) @binding(1) var<uniform> frame: Frame;
@group(0) @binding(2) var<storage, read> vdata: array<vec4u>;
@group(0) @binding(3) var<storage, read> insts: array<Inst>;
@group(0) @binding(4) var<storage, read> buggies: array<Buggy>;
@group(0) @binding(5) var<storage, read> dyn: array<mat4x4f>;
${shadeWgsl}
fn vf(i: u32) -> vec4f { return bitcast<vec4f>(vdata[i]); }
// Shortest-arc rotation taking +Z onto unit d.
fn quatFromZ(d: vec3f) -> vec4f {
  let r = d.z + 1.0;
  if (r < 1e-6) { return vec4f(1.0, 0.0, 0.0, 0.0); }
  return normalize(vec4f(-d.y, d.x, 0.0, r));
}

struct Out { @builtin(position) clip: vec4f, @location(0) n: vec3f, @location(1) c: vec4f, @location(2) w: vec3f };

@vertex fn vsBuggy(@location(0) pos: vec3f, @location(1) nrm: vec3f, @builtin(instance_index) ii: u32) -> Out {
  let inst = insts[ii];
  let bg = buggies[inst.info.x];
  let vr = vdata[1u].x + bg.vehicle.x * ${L.VEH_STRIDE}u;
  let chassis = vdata[vr + ${L.V_CFG}u].x;
  let rest = mat4x4f(inst.c0, inst.c1, inst.c2, inst.c3);
  let d = dyn[bg.vehicle.y + inst.slot.x];
  var m = d * rest;
  if ((inst.info.w & ${F_ABS}u) != 0u) { m = d; }
  var p = (m * vec4f(pos, 1.0)).xyz;
  var n = (m * vec4f(nrm, 0.0)).xyz;
  var body = chassis;
  var hidden = false;
  if ((inst.info.w & ${F_WHEEL}u) != 0u) {
    let wr = vdata[1u].y + (vdata[vr + ${L.V_CFG}u].y + inst.info.z) * ${L.WHEEL_STRIDE}u;
    let wd = vdata[wr + ${L.W_C0}u].z;
    hidden = (wd & ${L.WD_DISABLED}u) != 0u;
    if ((wd & ${L.WD_DETACHED}u) != 0u) {
      // The free wheel body: centred on the hub, unsteered and unspun at detach.
      body = vdata[wr + ${L.W_PARK}u].x;
      let dm = dyn[bg.vehicle.y + inst.slot.y] * rest;
      p = (dm * vec4f(pos, 1.0)).xyz;
      n = (dm * vec4f(nrm, 0.0)).xyz;
    }
  } else {
    let pr = vdata[2u].z + (vdata[vr + ${L.V_POOL2}u].x + inst.info.y) * ${L.PART_STRIDE}u;
    let com = vdata[pr + ${L.P_COM}u];
    let state = com.w;
    if (state == ${L.PS_DETACHED}u) {
      body = vdata[pr + ${L.P_PARK}u].x;
      p = p - bitcast<vec3f>(com.xyz);
    } else if (state != ${L.PS_ATTACHED}u) {
      hidden = true;
    }
  }
  let b = bodies[body];
  var o: Out;
  o.w = b.position + qrot(b.orientation, p);
  o.n = qrot(b.orientation, n);
  let mode = inst.slot.z;
  let extra = mat4x4f(inst.e0, inst.e1, inst.e2, inst.e3);
  if (mode == ${M_BODY}u) {
    // A bolted part on its own body: extra = design → body local.
    let rb = bodies[inst.slot.w];
    o.w = rb.position + qrot(rb.orientation, (extra * vec4f(pos, 1.0)).xyz);
    o.n = qrot(rb.orientation, (extra * vec4f(nrm, 0.0)).xyz);
    hidden = false;
  } else if (mode == ${M_TUBE}u) {
    // A frame tube skinned along its node chain (2–4 nodes): the unit template's
    // z ∈ [−½, ½] runs along the chain by length fraction (e1: the nodes'
    // fractions, e0.y: node count); its section (radius e0.x) follows the span.
    let t = clamp(pos.z + 0.5, 0.0, 1.0);
    let count = u32(inst.e0.y);
    var k = 0u;
    if (count > 2u && t > inst.e1.y) { k = 1u; }
    if (count > 3u && t > inst.e1.z) { k = 2u; }
    let ids = array<u32, 4>(inst.nodes.x, inst.nodes.y, inst.nodes.z, inst.nodes.w);
    let fr = array<f32, 4>(inst.e1.x, inst.e1.y, inst.e1.z, inst.e1.w);
    let pa = bodies[ids[k]].position;
    let pb = bodies[ids[k + 1u]].position;
    let u = (t - fr[k]) / max(fr[k + 1u] - fr[k], 1e-6);
    let d = pb - pa;
    let q = quatFromZ(d / max(length(d), 1e-4));
    let r = inst.e0.x;
    o.w = mix(pa, pb, u) + qrot(q, vec3f(pos.x * r, pos.y * r, 0.0));
    o.n = qrot(q, nrm);
    hidden = false;
  } else if (mode == ${M_FRAME}u) {
    // Riding three nodes: extra = design → their rest frame (origin node 0).
    let p0 = bodies[inst.nodes.x].position;
    let e1 = normalize(bodies[inst.nodes.y].position - p0);
    let e3 = normalize(cross(e1, bodies[inst.nodes.z].position - p0));
    let e2 = cross(e3, e1);
    let f = mat3x3f(e1, e2, e3);
    o.w = p0 + f * (extra * vec4f(pos, 1.0)).xyz;
    o.n = f * (extra * vec4f(nrm, 0.0)).xyz;
    hidden = false;
  }
  o.c = inst.color;
  o.clip = select(frame.viewProj * vec4f(o.w, 1.0), vec4f(0.0, 0.0, 2.0, 1.0), hidden);
  return o;
}

@fragment fn fsBuggy(i: Out, @builtin(front_facing) front: bool) -> @location(0) vec4f {
  var n = normalize(i.n);
  if (!front) { n = -n; }
  // Metals pick up a little more of the sky and a tighter highlight.
  let base = shade(i.c.rgb, n, i.w);
  let v = normalize(frame.eye.xyz - i.w);
  let l = normalize(frame.light.xyz);
  let spec = pow(max(dot(reflect(-l, n), v), 0.0), 64.0) * 0.5 * i.c.a;
  return vec4f(base + vec3f(spec), 1.0);
}
`;
  const mod = device.createShaderModule({ label: 'buggy-render', code: shader });
  const info = await mod.getCompilationInfo();
  const err = info.messages.find((m) => m.type === 'error');
  if (err) throw new Error(`buggy shader: ${err.message} @${err.lineNum}:${err.linePos}`);
  const pipeline = device.createRenderPipeline({
    label: 'buggy',
    layout: 'auto',
    vertex: { module: mod, entryPoint: 'vsBuggy', buffers: [{ arrayStride: mesh.vertexStride, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' }] }] },
    fragment: { module: mod, entryPoint: 'fsBuggy', targets: [{ format }] },
    // Recipe primitives mix windings (the spring's mirrored frame): shade both faces.
    primitive: { cullMode: 'none' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });

  // ---- wheel-record readback for the rig (pipelined, a few frames deep) ----
  const RB = 3;
  const wheelBytes = 4 * L.WHEEL_STRIDE * 16;
  // Per buggy: its four wheel records, then its V_BRK slot (latched broken masks).
  const recBytes = wheelBytes + 16;
  let rbFree = [];
  let rbSize = 0;

  function rebuild() {
    const nB = buggies.length;
    if (!nB) { draws = []; return; }
    const total = mesh.instances.length * nB;
    const words = new ArrayBuffer(total * INST_WORDS * 4);
    const f = new Float32Array(words), u = new Uint32Array(words);
    let k = 0;
    draws = [];
    for (let t = 0; t < mesh.templates.length; t++) {
      const list = instancesByTemplate[t];
      if (!list.length) continue;
      const first = k;
      for (let b = 0; b < nB; b++) for (const i of list) {
        const inst = mesh.instances[i];
        const o = k * INST_WORDS;
        const m = inst.matrix;
        f.set([m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0, m[9], m[10], m[11], 1], o);
        f.set(palette[inst.material], o + 16);
        const wheel = mesh.classes[inst.cls] === 'wheel';
        const flags = (slotOf[i] >= SLOT_CORNER + 8 ? F_ABS : 0) | (wheel ? F_WHEEL : 0) | (inst.role ? F_RIG : 0);
        const bind = buggies[b].bind(i);
        u.set([b, bind.part, inst.corner < 0 ? NONE : inst.corner, flags], o + 20);
        u.set([slotOf[i], wheel ? SLOT_WHEEL_DET + inst.corner : SLOT_STATIC, bind.mode, bind.body ?? 0], o + 24);
        u.set([...(bind.nodes ?? []), 0, 0, 0, 0].slice(0, 4), o + 28);
        if (bind.extra) f.set(bind.extra, o + 32);
        k++;
      }
      const tp = mesh.templates[t];
      draws.push({ first, count: k - first, indexCount: tp.indexCount, firstIndex: tp.indexOffset / 4, vertexOffset: tp.vertexOffset });
    }
    instBuf?.destroy(); buggyBuf?.destroy(); dynBuf?.destroy();
    instBuf = device.createBuffer({ size: words.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(instBuf, 0, words);
    const bw = new Uint32Array(nB * BUGGY_WORDS);
    buggies.forEach((g, b) => bw.set([g.h, b * slotsPerBuggy, 0, 0], b * BUGGY_WORDS));
    buggyBuf = device.createBuffer({ size: bw.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buggyBuf, 0, bw);
    dynBuf = device.createBuffer({ size: nB * slotsPerBuggy * MAT_FLOATS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    for (const g of buggies) writeDyn(g);
    bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [[0, world.bufs[0]], [1, frameBuf], [2, world.bufs[31]], [3, instBuf], [4, buggyBuf], [5, dynBuf]]
        .map(([binding, buffer]) => ({ binding, resource: { buffer } })),
    });
    rbFree.forEach((b) => b.destroy());
    rbSize = nB * recBytes;
    rbFree = Array.from({ length: RB }, () => device.createBuffer({ size: rbSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
  }

  /** Pose the rig of buggy `g` and upload its matrix slots. */
  function writeDyn(g) {
    const toBody = mul(translation(g.origin.map((x) => -x)), R);
    const d = g.dyn;
    const put = (slot, m) => d.set(m, slot * MAT_FLOATS);
    put(SLOT_STATIC, toBody);
    CORNER_IDS.forEach((id, c) => {
      // Detached wheel body frame: centred on the design hub, axes of the chassis.
      const hub = rigDef.corners[id].hub;
      put(SLOT_WHEEL_DET + c, mul(translation([hub[0], -hub[1], hub[2]]), R));
    });
    put(SLOT_STEERING, mul(toBody, g.rig.steeringDelta));
    CORNER_IDS.forEach((id, c) => {
      put(SLOT_CORNER + 2 * c, mul(toBody, g.rig.states[id].wheel));
      put(SLOT_CORNER + 2 * c + 1, mul(toBody, g.rig.states[id].steer));
    });
    absInstances.forEach((inst, j) => {
      let m = g.rig.matrixFor(inst);
      if (inst.role === 'spring') {
        // The helix was modelled at the neutral coil: stretch it along its axis.
        const s = g.rig.states[CORNER_IDS[inst.corner]].coilLength / g.rig.neutralCoil(inst.corner);
        m = mul(m, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, s, 0, 0, 0, 0, 1]);
      }
      put(SLOT_CORNER + 8 + j, mul(toBody, m));
    });
    if (dynBuf) device.queue.writeBuffer(dynBuf, buggies.indexOf(g) * slotsPerBuggy * MAT_FLOATS * 4, d);
  }

  const buggyOf = (h) => {
    const entry = world.vehicles.entries[h];
    // Neutral suspension length per wheel: the hardpoint sits that far above the
    // rig's neutral hub (presets.rs dune_buggy).
    const neutralL = CORNER_IDS.map((id, c) => entry.desc.wheels[c].position[1] - rigDef.corners[id].hub[1]);
    return { h, entry, rig: new BuggyRig(rigDef), origin: entry.origin, neutralL, dyn: new Float32Array(slotsPerBuggy * MAT_FLOATS), wheelSlot: entry.slots.wheelBase };
  };
  const vehicleBodies = (entry) => ({ firstBody: entry.chassis, bodyCount: 2 + entry.wheelSlots + entry.desc.parts.length });
  const volume = (s) => s.type === 'box' ? 8 * s.half[0] * s.half[1] * s.half[2]
    : s.type === 'sphere' ? 4 / 3 * Math.PI * s.radius ** 3 : Math.PI * s.radius ** 2 * (2 * s.half_length + 4 / 3 * s.radius);
  // Volume-weighted collider centre: SoftRigidPart::center, its body origin.
  const centerOf = (cols) => {
    const V = cols.reduce((a, c) => a + volume(c.shape), 0) || 1;
    return [0, 1, 2].map((k) => cols.reduce((a, c) => a + c.position[k] * volume(c.shape) / V, 0));
  };
  const transpose3 = (m) => [m[0], m[4], m[8], 0, m[1], m[5], m[9], 0, m[2], m[6], m[10], 0, 0, 0, 0, 1];
  const restOf = (i) => { const m = mesh.instances[i].matrix; return [m[0], m[1], m[2], 0, m[3], m[4], m[5], 0, m[6], m[7], m[8], 0, m[9], m[10], m[11], 1]; };

  return {
    mesh,
    get count() { return buggies.length; },
    /** Draw rigid vehicle `h` (a dune_buggy preset) with the mesh. */
    add(h) {
      const desc = world.vehicles.entries[h].desc;
      if (desc.parts.length !== mesh.parts.length || desc.parts.some((p, i) => p.name !== mesh.parts[i]))
        throw new Error(`vehicle ${h} (${desc.name}) is not the baked dune buggy`);
      const g = buggyOf(h);
      g.bind = (i) => ({ mode: M_VEHICLE, part: mesh.instances[i].part });
      g.covers = () => [vehicleBodies(g.entry)];
      buggies.push(g);
      rebuild();
    },
    /** Draw hybrid soft vehicle `s` (engine_softbody::buggy; `desc` is its
     * SoftVehicleDesc) with the mesh: its core carries wheels, rig and drivetrain. */
    addHybrid(s, desc) {
      const soft = world.soft.entries[s];
      if (!soft.core) throw new Error('hybrid buggy without its core');
      if (soft.nodes !== mesh.frameNodes.length) throw new Error(`soft vehicle ${s} has ${soft.nodes} nodes, the mesh ${mesh.frameNodes.length}`);
      const g = buggyOf(soft.core.vehicle);
      const coreParts = g.entry.desc.parts.map((p) => p.name);
      const rigid = Object.fromEntries(soft.rigidParts.map((r, k) => [r.name, { body: r.body, center: centerOf(desc.rigid_parts[k].colliders) }]));
      const node = (k) => soft.firstBody + k;
      g.bind = (i) => {
        const inst = mesh.instances[i];
        const name = mesh.parts[inst.part];
        if (mesh.classes[inst.cls] === 'wheel') return { mode: M_VEHICLE, part: 0 };
        if (inst.role === 'steering') return { mode: M_VEHICLE, part: Math.max(0, coreParts.indexOf('steering_column')) };
        if (inst.role) return { mode: M_VEHICLE, part: 0 };
        if (coreParts.includes(name)) return { mode: M_VEHICLE, part: coreParts.indexOf(name) };
        if (inst.bind?.mode === 'tube') {
          const m = inst.matrix, ns = inst.bind.nodes, at = inst.bind.at;
          const ids = [0, 1, 2, 3].map((j) => node(ns[Math.min(j, ns.length - 1)]));
          const fr = [0, 1, 2, 3].map((j) => at[Math.min(j, at.length - 1)]);
          return { mode: M_TUBE, part: 0, nodes: ids, extra: [Math.hypot(m[0], m[1], m[2]), ns.length, 0, 0, ...fr] };
        }
        if (inst.bind?.mode === 'frame') {
          const [a, b, c] = inst.bind.nodes.map((k) => mesh.frameNodes[k]);
          const sub = (x, y) => x.map((v, k) => v - y[k]);
          const norm = (x) => { const l = Math.hypot(...x) || 1; return x.map((v) => v / l); };
          const cross = (x, y) => [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
          const e1 = norm(sub(b, a)), e3 = norm(cross(e1, sub(c, a))), e2 = cross(e3, e1);
          const frame = [...e1, 0, ...e2, 0, ...e3, 0, 0, 0, 0, 1];
          return { mode: M_FRAME, part: 0, nodes: inst.bind.nodes.map(node), extra: mul(mul(mul(transpose3(frame), translation(a.map((x) => -x))), R), restOf(i)) };
        }
        const r = rigid[name];
        if (r) return { mode: M_BODY, part: 0, body: r.body, extra: mul(mul(translation(r.center.map((x) => -x)), R), restOf(i)) };
        throw new Error(`hybrid buggy: no binding for ${inst.name} (${name})`);
      };
      g.covers = () => [
        vehicleBodies(g.entry),
        { firstBody: soft.firstBody, bodyCount: soft.nodes },
        ...soft.rigidParts.map((r) => ({ firstBody: r.body, bodyCount: 1 })),
      ];
      buggies.push(g);
      rebuild();
    },
    /** Bodies (chassis, anchor, wheel and part bodies; soft nodes and bolted parts)
     * whose generic collider shapes this mesh replaces, and the vehicle handles
     * whose generic wheels it replaces. */
    covered() {
      return buggies.map((g) => ({ vehicle: g.h, bodies: g.covers() }));
    },
    /** Queue the wheel-record copies after a step (pass the frame's encoder). */
    encodeReadback(enc) {
      if (!buggies.length || !rbFree.length) return null;
      const rb = rbFree.pop();
      const l = world.vlayout;
      buggies.forEach((g, b) => {
        enc.copyBufferToBuffer(world.bufs[31], (l.wheelBase + g.wheelSlot * L.WHEEL_STRIDE) * 16, rb, b * recBytes, wheelBytes);
        enc.copyBufferToBuffer(world.bufs[31], (l.vehicleBase + g.h * L.VEH_STRIDE + L.V_BRK) * 16, rb, b * recBytes + wheelBytes, 16);
      });
      return rb;
    },
    /** After `rb` (from encodeReadback) was submitted: map it, re-pose the rigs. */
    async consume(rb) {
      if (!rb) return;
      await rb.mapAsync(GPUMapMode.READ);
      if (rb.size !== rbSize) { rb.destroy(); return; } // buggies added meanwhile
      const words = new Uint32Array(rb.getMappedRange().slice(0));
      rb.unmap();
      rbFree.push(rb);
      const f = new Float32Array(words.buffer);
      buggies.forEach((g, b) => {
        const wheels = {};
        CORNER_IDS.forEach((id, c) => {
          const w = b * (recBytes / 4) + c * L.WHEEL_STRIDE * 4;
          const flags = words[w + L.W_C0 * 4 + 2];
          const gone = (flags & (L.WD_DETACHED | L.WD_DISABLED)) !== 0;
          const h = rigDef.corners[id];
          wheels[id] = gone
            ? { travelM: h.minTravel, steeringRad: 0, rotationRad: 0 }
            : {
              travelM: g.neutralL[c] - f[w + L.W_O_POS * 4 + 3],
              steeringRad: f[w + L.W_C5 * 4 + 3],
              // Rubble spins about +X (left) forward; the recipe's +X is Rubble's -X.
              rotationRad: -f[w + L.W_W * 4 + 1],
            };
        });
        g.rig.applyPose(wheels);
        writeDyn(g);
        const brk = b * (recBytes / 4) + wheelBytes / 4;
        g.partsBroken = words[brk + 2];
        g.wheelsBroken = words[brk + 3];
      });
    },
    /** Parts and wheels broken off under load so far, over every buggy (latched). */
    broken() {
      const bits = (x) => { let n = 0; for (x >>>= 0; x; x &= x - 1) n++; return n; };
      return buggies.reduce((s, g) => ({ parts: s.parts + bits(g.partsBroken ?? 0), wheels: s.wheels + bits(g.wheelsBroken ?? 0) }), { parts: 0, wheels: 0 });
    },
    draw(rp) {
      if (!draws.length) return;
      rp.setPipeline(pipeline);
      rp.setBindGroup(0, bindGroup);
      rp.setIndexBuffer(geo, 'uint32');
      for (const d of draws) {
        // Primitives are packed back to back: bind each one's vertices at its offset.
        rp.setVertexBuffer(0, geo, d.vertexOffset);
        rp.drawIndexed(d.indexCount, d.count, d.firstIndex, 0, d.first);
      }
    },
  };
}
