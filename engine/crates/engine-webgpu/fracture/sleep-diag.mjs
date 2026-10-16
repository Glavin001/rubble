// Sleep diagnostics (headless-page sleepdiag=<ticks>): why bodies are still
// awake. Every `every` ticks it records total mechanical energy (which must
// never rise once the shot's impact is over: contacts and friction only
// remove energy), the awake bodies split by why they are awake, the fastest
// bodies, the deepest contacts, and how many contacts kept their identity
// (pair + feature) from the previous tick.
//
// Body record (BodyState, 28 words): position 0-2, invMass 3, orientation 4-7,
// invInertia 8-10 (local), flags 11, linearVel 12-14, sleepTimer 15,
// angularVel 16-18 (world), generation 19.
const SLEEPING = 4, DYNAMIC = 1;

function rotateInv(q, v) { // q^-1 * v
  const [x, y, z, w] = q, ix = -x, iy = -y, iz = -z;
  const tx = 2 * (iy * v[2] - iz * v[1]), ty = 2 * (iz * v[0] - ix * v[2]), tz = 2 * (ix * v[1] - iy * v[0]);
  return [v[0] + w * tx + (iy * tz - iz * ty), v[1] + w * ty + (iz * tx - ix * tz), v[2] + w * tz + (ix * ty - iy * tx)];
}

export function bodyStats(f32, u32, scene, gravityY) {
  const n = scene.bodies.length, out = [];
  let ke = 0, pe = 0;
  for (let i = 0; i < n; i++) {
    const o = i * 28, invMass = f32[o + 3];
    if (!(invMass > 0) || !(u32[o + 11] & DYNAMIC)) continue;
    const m = 1 / invMass, v = [f32[o + 12], f32[o + 13], f32[o + 14]], w = [f32[o + 16], f32[o + 17], f32[o + 18]];
    const wl = rotateInv([f32[o + 4], f32[o + 5], f32[o + 6], f32[o + 7]], w);
    let rot = 0; for (let k = 0; k < 3; k++) { const ii = f32[o + 8 + k]; if (ii > 0) rot += wl[k] * wl[k] / ii; }
    const kin = 0.5 * m * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + 0.5 * rot;
    ke += kin; pe += m * -gravityY * f32[o + 1];
    out.push({i, m, lin: Math.hypot(...v), ang: Math.hypot(...w), timer: f32[o + 15], asleep: (u32[o + 11] & SLEEPING) !== 0, y: f32[o + 1], kin});
  }
  return {bodies: out, ke, pe};
}

// Contact identity: collider pair + feature id.
const contactKeys = (r) => { const s = new Set(); for (let i = 0; i < r.counters.contact; i++) s.add(`${r.meta[i * 8]},${r.meta[i * 8 + 1]},${r.meta[i * 8 + 2]}`); return s; };

// Why each awake body is awake, from its own velocity and its contacts:
//   flying       no contacts
//   opposed      two contacts with the same partner push in opposite directions
//                (normals' dot < -0.5): a piece jammed between a compound's children
//   deep         a contact overlaps more than 1 cm
//   groundSlide  moving, touching only static bodies (ground)
//   moving       moving, among dynamic neighbours
//   held         below the thresholds for sleepTime, kept awake by the island gate
//   settling     below the thresholds, timer still running
export function classifyAwake(awake, contacts, colliderBody, isStatic, isSlow, sleepTime) {
  const per = new Map(), f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
  const asF = (x) => { u32[0] = x; return f32[0]; };
  for (let k = 0; k < contacts.counters.contact; k++) {
    const a = colliderBody[contacts.meta[k * 8]], b = colliderBody[contacts.meta[k * 8 + 1]], c0 = contacts.points[k * 16 + 3];
    const n = [asF(contacts.meta[k * 8 + 4]), asF(contacts.meta[k * 8 + 5]), asF(contacts.meta[k * 8 + 6])];
    for (const [self, other, s] of [[a, b, 1], [b, a, -1]]) {
      let e = per.get(self); if (!e) { e = {minC0: 0, dyn: false, normals: new Map()}; per.set(self, e); }
      e.minC0 = Math.min(e.minC0, c0); if (!isStatic(other)) e.dyn = true;
      const list = e.normals.get(other) ?? []; list.push(n.map((x) => x * s)); e.normals.set(other, list);
    }
  }
  const out = {flying: 0, opposed: 0, deep: 0, groundSlide: 0, moving: 0, held: 0, settling: 0}, examples = {};
  for (const b of awake) {
    const e = per.get(b.i), slow = isSlow(b);
    let kind;
    if (slow) kind = b.timer >= sleepTime ? 'held' : 'settling';
    else if (!e) kind = 'flying';
    else {
      let opp = false;
      for (const list of e.normals.values()) { for (let x = 0; x < list.length && !opp; x++) for (let y = x + 1; y < list.length; y++) if (list[x][0] * list[y][0] + list[x][1] * list[y][1] + list[x][2] * list[y][2] < -0.5) { opp = true; break; } if (opp) break; }
      kind = opp ? 'opposed' : e.minC0 < -0.01 ? 'deep' : !e.dyn ? 'groundSlide' : 'moving';
    }
    out[kind]++; (examples[kind] ??= []).length < 4 && examples[kind].push(b.i);
  }
  return {counts: out, examples};
}

// Farthest collider point from each body's origin (as the shader's colliderExtent).
export function bodyExtents(scene) {
  const ext = new Float32Array(scene.bodies.length);
  for (const c of scene.colliders) {
    const p = c.params, r = c.shape === 0 ? p[0] : c.shape === 1 ? p[0] + p[1] : c.shape === 3 ? p[3] : Math.hypot(p[0], p[1], p[2]);
    ext[c.body] = Math.max(ext[c.body], Math.hypot(...(c.localPos ?? [0, 0, 0])) + r);
  }
  return ext;
}

// Point-speed sleep test (params.flags 8192) folds rotation into the linear
// threshold; otherwise the speeds are compared with sleepLin and sleepAng.
function sleepTest(scene) {
  const {sleepLin, sleepAng} = scene.params;
  if (!(scene.params.flags & 8192)) return (b) => b.lin < sleepLin && b.ang < sleepAng;
  const ext = bodyExtents(scene);
  return (b) => b.lin + b.ang * ext[b.i] < sleepLin;
}

// Islands of awake bodies joined by contacts between two dynamic bodies (static
// bodies do not join islands): how many, how large, and how many of their bodies
// sit in an island that still has a moving body.
export function awakeIslands(awake, contacts, colliderBody, isStatic, isSlow) {
  const parent = new Map(awake.map((b) => [b.i, b.i]));
  const find = (x) => { while (parent.get(x) !== x) { const g = parent.get(parent.get(x)); parent.set(x, g); x = g; } return x; };
  for (let k = 0; k < contacts.counters.contact; k++) {
    const a = colliderBody[contacts.meta[k * 8]], b = colliderBody[contacts.meta[k * 8 + 1]];
    if (isStatic(a) || isStatic(b) || !parent.has(a) || !parent.has(b)) continue;
    const ra = find(a), rb = find(b); if (ra !== rb) parent.set(ra, rb);
  }
  const size = new Map(), moving = new Map();
  for (const b of awake) { const r = find(b.i); size.set(r, (size.get(r) ?? 0) + 1); if (!isSlow(b)) moving.set(r, (moving.get(r) ?? 0) + 1); }
  const sizes = [...size.values()].sort((x, y) => y - x);
  let inMoving = 0; for (const [r, n] of size) if (moving.has(r)) inMoving += n;
  return {islands: size.size, largest: sizes.slice(0, 5), withMover: moving.size, bodiesInMovingIslands: inMoving};
}

export async function runSleepDiag(sim, scene, {ticks = 2400, every = 120, top = 12} = {}) {
  const g = scene.params.gravityY ?? -9.81, sleepLin = scene.params.sleepLin, sleepAng = scene.params.sleepAng, sleepTime = scene.params.sleepTime;
  const isSlow = sleepTest(scene);
  const colliderBody = scene.colliders.map((c) => c.body);
  const isStatic = (i) => !(scene.bodies[i].invMass > 0);
  const samples = [];
  let tick = 0, prevKeys = null, prevPos = null;
  const snap = async () => { const f = await sim.snapshot(); return {f32: f, u32: new Uint32Array(f.buffer)}; };
  while (tick < ticks) {
    const n = Math.min(4, every - (tick % every));
    await sim.step(n, {wait: false}); tick += n;
    if (tick % every) continue;
    await sim.flush();
    // Two consecutive ticks' contacts: identity survival from one to the next.
    const before = await sim.readContacts(); await sim.step(1); tick++;
    const after = await sim.readContacts();
    const ka = contactKeys(before), kb = contactKeys(after);
    let kept = 0; for (const k of kb) if (ka.has(k)) kept++;
    const {f32, u32} = await snap();
    const st = bodyStats(f32, u32, scene, g);
    const awake = st.bodies.filter((b) => !b.asleep);
    const moving = awake.filter((b) => !isSlow(b));
    const still = awake.filter(isSlow);
    const held = still.filter((b) => b.timer >= sleepTime); // a candidate the island gate kept awake
    // Contacts of awake bodies: depth (c0N < 0 = overlap beyond the margin).
    const pts = after.points, meta = after.meta, nC = after.counters.contact;
    let deep1 = 0, deep5 = 0, minC0 = 0;
    for (let i = 0; i < nC; i++) { const c0 = pts[i * 16 + 3]; if (c0 < -0.01) deep1++; if (c0 < -0.05) deep5++; if (c0 < minC0) minC0 = c0; }
    const speedHist = {'<0.03': 0, '0.03-0.1': 0, '0.1-0.3': 0, '0.3-1': 0, '1-3': 0, '>3': 0};
    for (const b of awake) { const s = b.lin; speedHist[s < 0.03 ? '<0.03' : s < 0.1 ? '0.03-0.1' : s < 0.3 ? '0.1-0.3' : s < 1 ? '0.3-1' : s < 3 ? '1-3' : '>3']++; }
    const spinHist = {'<0.05': 0, '0.05-0.5': 0, '0.5-2': 0, '2-10': 0, '>10': 0};
    for (const b of awake) { const s = b.ang; spinHist[s < 0.05 ? '<0.05' : s < 0.5 ? '0.05-0.5' : s < 2 ? '0.5-2' : s < 10 ? '2-10' : '>10']++; }
    const why = classifyAwake(awake, after, colliderBody, isStatic, isSlow, sleepTime);
    const islands = awakeIslands(awake, after, colliderBody, isStatic, isSlow);
    // Drift since the previous sample of bodies that are slow now: creep that
    // never trips the threshold but moves supports until something lets go.
    const pos = new Float32Array(scene.bodies.length * 3);
    for (const b of st.bodies) for (let k = 0; k < 3; k++) pos[b.i * 3 + k] = f32[b.i * 28 + k];
    let drift = null;
    if (prevPos) {
      drift = {'>1mm': 0, '>1cm': 0, '>5cm': 0, slowAwake: 0, examples: []};
      for (const b of still) {
        const d = Math.hypot(pos[b.i * 3] - prevPos[b.i * 3], pos[b.i * 3 + 1] - prevPos[b.i * 3 + 1], pos[b.i * 3 + 2] - prevPos[b.i * 3 + 2]);
        drift.slowAwake++;
        if (d > 0.001) drift['>1mm']++;
        if (d > 0.01) { drift['>1cm']++; if (drift.examples.length < 6) drift.examples.push([b.i, +d.toFixed(3)]); }
        if (d > 0.05) drift['>5cm']++;
      }
    }
    prevPos = pos;
    const settlingDetail = still.filter((b) => b.timer < sleepTime).slice(0, 8)
      .map((b) => ({i: b.i, timer: +b.timer.toFixed(3), lin: +b.lin.toFixed(4), ang: +b.ang.toFixed(3), m: +b.m.toFixed(3)}));
    samples.push({t: +(tick * (scene.params.dt ?? 1 / 120)).toFixed(2), why: why.counts, islands, drift, settlingDetail, whyExamples: why.examples, awake: awake.length, moving: moving.length, stillAwake: still.length, gateHeld: held.length,
      ke: +st.ke.toFixed(1), mechanical: +(st.ke + st.pe).toFixed(1), contacts: nC, keySurvival: +(kept / Math.max(1, kb.size)).toFixed(4),
      deepOver1cm: deep1, deepOver5cm: deep5, minC0: +minC0.toFixed(4), speedHist, spinHist});
    prevKeys = kb;
  }
  // Final: the worst offenders, with their contacts.
  await sim.flush();
  const last = await sim.readContacts();
  const {f32, u32} = await snap();
  const st = bodyStats(f32, u32, scene, g);
  const awake = st.bodies.filter((b) => !b.asleep);
  const perBody = new Map();
  for (let i = 0; i < last.counters.contact; i++) {
    const a = colliderBody[last.meta[i * 8]], b = colliderBody[last.meta[i * 8 + 1]], c0 = last.points[i * 16 + 3];
    const lam = last.points[i * 16 + 8];
    for (const body of [a, b]) { const e = perBody.get(body) ?? {contacts: 0, minC0: 0, partners: new Set(), lambdaN: 0}; e.contacts++; e.minC0 = Math.min(e.minC0, c0); e.partners.add(body === a ? b : a); e.lambdaN += Math.abs(lam); perBody.set(body, e); }
  }
  const describe = (b) => { const e = perBody.get(b.i); return {body: b.i, lin: +b.lin.toFixed(3), ang: +b.ang.toFixed(3), y: +b.y.toFixed(3), mass: +b.m.toFixed(3), timer: +b.timer.toFixed(2), contacts: e?.contacts ?? 0, partners: e ? e.partners.size : 0, minC0: e ? +e.minC0.toFixed(4) : 0}; };
  const byLin = [...awake].sort((x, y) => y.lin - x.lin).slice(0, top).map(describe);
  const bySpin = [...awake].sort((x, y) => y.ang - x.ang).slice(0, top).map(describe);
  const contactless = awake.filter((b) => !perBody.has(b.i)).length;
  // Deep contacts between pairs that were bonded in the bake (broken bonds).
  let deepPairs = 0, deepBonded = 0;
  if (scene.joints?.length) {
    const bonded = new Set(scene.joints.map((j) => `${Math.min(j.bodyA, j.bodyB)},${Math.max(j.bodyA, j.bodyB)}`));
    const seen = new Set();
    for (let i = 0; i < last.counters.contact; i++) {
      if (last.points[i * 16 + 3] > -0.005) continue;
      const a = colliderBody[last.meta[i * 8]], b = colliderBody[last.meta[i * 8 + 1]], key = `${Math.min(a, b)},${Math.max(a, b)}`;
      if (seen.has(key)) continue; seen.add(key); deepPairs++; if (bonded.has(key)) deepBonded++;
    }
  }
  return {samples, final: {awake: awake.length, contactlessAwake: contactless, deepPairs, deepBonded, fastest: byLin, spinniest: bySpin, byFragment: awakeByFragment(scene, awake, isSlow)}};
}

// Awake and moving bodies by source material and mass band, so asset-side
// causes (tiny fragments, extreme mass ratios) show up per fragment kind.
// Copies of one house repeat its local body ids, so local = (i-1) % perHouse + 1.
function awakeByFragment(scene, awake, isSlow) {
  if (!scene.byId || !scene.library) return null;
  const local = new Map(), perHouse = scene.byId.size;
  for (const {id, f} of scene.byId.values()) local.set(id, f);
  const band = (m) => m < 0.06 ? '<60g' : m < 0.5 ? '60-500g' : m < 5 ? '0.5-5kg' : m < 50 ? '5-50kg' : '>50kg';
  const out = {}, houses = {};
  for (const b of awake) {
    if (b.i >= scene.projectileStart) continue;
    const house = Math.floor((b.i - 1) / perHouse); houses[house] = (houses[house] ?? 0) + 1;
    const f = local.get((b.i - 1) % perHouse + 1); if (!f) continue;
    const mat = scene.library.material[f.materialIds[0]]?.id ?? String(f.materialIds[0]);
    const key = `${mat} ${band(f.mass)}`, e = out[key] ??= {awake: 0, moving: 0};
    e.awake++; if (!isSlow(b)) e.moving++;
  }
  return {kinds: out, houses};
}

// Trace (headless-page sleeptrace=<tick>): run to `at`, pick the fastest and
// spinniest bodies (not the projectiles), then record them every tick: pose,
// velocity, energy, momentum with their contact partners, and each contact.
// Contact point (SolverPoint, 16 floats): anchorA 0-2, c0N 3, anchorB 4-6,
// c0T1 7, lambda 8-10, c0T2 11, penalty 12-14, pairAndStick 15. Meta (8
// words): colliderA, colliderB, featureId, generation, normal 4-6, friction.
export async function runSleepTrace(sim, scene, {at = 1200, ticks = 30, pick = 3, bodies: chosen = null, settling = false, hullView, hullBase} = {}) {
  const g = scene.params.gravityY ?? -9.81, colliderBody = scene.colliders.map((c) => c.body);
  for (let t = 0; t < at; t += 4) await sim.step(Math.min(4, at - t), {wait: false});
  await sim.flush();
  let f = await sim.snapshot(), u = new Uint32Array(f.buffer);
  const st = bodyStats(f, u, scene, g).bodies.filter((b) => !b.asleep && b.i < scene.projectileStart);
  // settling: the bodies that block their island (awake, timer < sleepTime).
  const ids = chosen ?? (settling
    ? st.filter((b) => b.timer < scene.params.sleepTime).sort((a, b) => a.timer - b.timer).slice(0, 2 * pick).map((b) => b.i)
    : [...new Set([...[...st].sort((a, b) => b.lin - a.lin).slice(0, pick), ...[...st].sort((a, b) => b.ang - a.ang).slice(0, pick)].map((b) => b.i))]);
  const hullOf = (c) => {
    if (c.shape !== 3) return {shape: c.shape, params: c.params};
    const v = hullView(scene.hullData, hullBase(c.params)), verts = Array.from({length: v.nv}, (_, i) => v.vertex(i));
    const lo = [0, 1, 2].map((k) => Math.min(...verts.map((p) => p[k]))), hi = [0, 1, 2].map((k) => Math.max(...verts.map((p) => p[k])));
    return {shape: 3, nv: v.nv, nf: v.nf, ne: v.ne, closed: v.closed, size: hi.map((x, k) => +(x - lo[k]).toFixed(3)), radius: +c.params[3].toFixed(3), localPos: c.localPos};
  };
  const info = ids.map((i) => {
    const b = scene.bodies[i], cols = scene.colliders.filter((c) => c.body === i);
    return {body: i, invMass: b.invMass, invI: b.invI, colliders: cols.map(hullOf)};
  });
  const rows = [];
  for (let t = 0; t < ticks; t++) {
    await sim.step(1);
    const c = await sim.readContacts();
    f = await sim.snapshot(); u = new Uint32Array(f.buffer);
    const tick = {t: at + t + 1, bodies: {}};
    for (const i of ids) {
      const o = i * 28, m = 1 / f[o + 3];
      const v = [f[o + 12], f[o + 13], f[o + 14]], w = [f[o + 16], f[o + 17], f[o + 18]];
      const contacts = [], partners = new Set();
      for (let k = 0; k < c.counters.contact; k++) {
        const a = colliderBody[c.meta[k * 8]], b = colliderBody[c.meta[k * 8 + 1]];
        if (a !== i && b !== i) continue;
        const other = a === i ? b : a; partners.add(other);
        const p = k * 16;
        contacts.push({other, otherShape: scene.colliders[c.meta[k * 8 + (a === i ? 1 : 0)]].shape, self: a === i ? 'A' : 'B', feat: (c.meta[k * 8 + 2] >>> 0).toString(16),
          n: [c.meta[k * 8 + 4], c.meta[k * 8 + 5], c.meta[k * 8 + 6]].map((x) => +new Float32Array(new Uint32Array([x]).buffer)[0].toFixed(3)),
          c0N: +c.points[p + 3].toFixed(4), lamN: +c.points[p + 8].toFixed(2), kN: +c.points[p + 12].toExponential(2), kT: +c.points[p + 13].toExponential(2), lamT: +Math.hypot(c.points[p + 9], c.points[p + 10]).toFixed(3), c0T: +Math.hypot(c.points[p + 7], c.points[p + 11]).toFixed(4), stick: (new Uint32Array(c.points.buffer)[p + 15] >>> 31)});
      }
      // Momentum of the body and of the body with its partners (dynamic only).
      let P = v.map((x) => x * m);
      for (const q of partners) { const oq = q * 28; if (f[oq + 3] > 0 && (u[oq + 11] & 1)) { const mq = 1 / f[oq + 3]; P = P.map((x, k) => x + mq * f[oq + 12 + k]); } }
      tick.bodies[i] = {p: [0, 1, 2].map((k) => +f[o + k].toFixed(4)), y: +f[o + 1].toFixed(4), timer: +f[o + 15].toFixed(3), moved: +f[o + 23].toFixed(4), v: v.map((x) => +x.toFixed(3)), w: w.map((x) => +x.toFixed(2)), flags: u[o + 11], groupMomentum: P.map((x) => +x.toFixed(3)), contacts};
    }
    rows.push(tick);
  }
  // Bonds of the traced bodies at the end: partner, broken, |λ| linear/angular.
  if (scene.joints?.length) {
    const st = await sim.jointState();
    for (const e of info) e.bonds = scene.joints.map((j, k) => ({j, k})).filter(({j}) => j.bodyA === e.body || j.bodyB === e.body).slice(0, 12)
      .map(({j, k}) => ({other: j.bodyA === e.body ? j.bodyB : j.bodyA, broken: (st.flags[k] & 1) !== 0, flags: st.flags[k],
        lamLin: +Math.hypot(...st.lambda[k].slice(0, 3)).toFixed(2), lamAng: +Math.hypot(...st.lambda[k].slice(3, 6)).toFixed(3)}));
  }
  return {info, rows};
}

// Burst watch (headless-page sleepwatch=<from> [to]): step one tick at a time
// and, whenever the number of moving awake bodies jumps from a quiet tick,
// record the bodies that started moving first, with their contacts.
export async function runSleepWatch(sim, scene, {from = 1200, to = 2400, quiet = 2, top = 6} = {}) {
  const g = scene.params.gravityY ?? -9.81, isSlow = sleepTest(scene);
  const colliderBody = scene.colliders.map((c) => c.body);
  for (let t = 0; t < from; t += 4) await sim.step(Math.min(4, from - t), {wait: false});
  await sim.flush();
  const bursts = [], series = [];
  let prev = new Set(), prevCount = 0;
  for (let tick = from; tick < to; tick++) {
    await sim.step(1);
    const f = await sim.snapshot(), u = new Uint32Array(f.buffer);
    const st = bodyStats(f, u, scene, g).bodies.filter((b) => !b.asleep);
    const moving = st.filter((b) => !isSlow(b));
    if (tick % 30 === 0) series.push([tick, st.length, moving.length]);
    if (prevCount <= quiet && moving.length > quiet && bursts.length < 12) {
      const c = await sim.readContacts();
      const fresh = moving.filter((b) => !prev.has(b.i)).sort((a, b) => b.kin - a.kin).slice(0, top);
      bursts.push({tick, awake: st.length, moving: moving.length, first: fresh.map((b) => {
        const partners = new Map();
        for (let k = 0; k < c.counters.contact; k++) {
          const a = colliderBody[c.meta[k * 8]], bb = colliderBody[c.meta[k * 8 + 1]];
          if (a !== b.i && bb !== b.i) continue;
          const o = a === b.i ? bb : a, e = partners.get(o) ?? {n: 0, minC0: 0, lamN: 0};
          e.n++; e.minC0 = Math.min(e.minC0, c.points[k * 16 + 3]); e.lamN += c.points[k * 16 + 8]; partners.set(o, e);
        }
        return {body: b.i, mass: +b.m.toFixed(3), lin: +b.lin.toFixed(3), ang: +b.ang.toFixed(3), y: +b.y.toFixed(3), timer: +b.timer.toFixed(2),
          partners: [...partners].map(([o, e]) => ({o, n: e.n, minC0: +e.minC0.toFixed(4), lamN: +e.lamN.toFixed(2), oAsleep: (u[o * 28 + 11] & 4) !== 0}))};
      })});
    }
    prev = new Set(moving.map((b) => b.i)); prevCount = moving.length;
  }
  return {bursts, series};
}

// Wake watch (headless-page wakewatch=<from> [to]): every tick, the bodies that
// went from asleep to awake, with each awake contact partner's speed and timer
// (the candidates for having woken it).
export async function runWakeWatch(sim, scene, {from = 1200, to = 1500, top = 10} = {}) {
  const g = scene.params.gravityY ?? -9.81, colliderBody = scene.colliders.map((c) => c.body), ext = bodyExtents(scene);
  for (let t = 0; t < from; t += 4) await sim.step(Math.min(4, from - t), {wait: false});
  await sim.flush();
  let f = await sim.snapshot(), u = new Uint32Array(f.buffer);
  const asleep = (u, i) => (u[i * 28 + 11] & 4) !== 0;
  let prev = new Uint8Array(scene.bodies.length);
  for (let i = 0; i < scene.bodies.length; i++) prev[i] = asleep(u, i) ? 1 : 0;
  const events = [], series = [];
  let totalWakes = 0, lastTimer = null, lastAsleep = null;
  for (let tick = from; tick < to; tick++) {
    await sim.step(1);
    const c = await sim.readContacts();
    f = await sim.snapshot(); u = new Uint32Array(f.buffer);
    const woke = [];
    for (let i = 0; i < scene.bodies.length; i++) { const s = asleep(u, i) ? 1 : 0; if (prev[i] && !s) woke.push(i); prev[i] = s; }
    totalWakes += woke.length;
    if (tick % 60 === 0) {
      let awakeN = 0, settling = 0, resets = 0;
      for (let i = 0; i < scene.bodies.length; i++) {
        const o = i * 28; if (!(f[o + 3] > 0) || asleep(u, i)) continue;
        awakeN++; const tm = f[o + 15]; if (tm < scene.params.sleepTime) settling++;
        if (lastTimer && tm < lastTimer[i] && !(lastAsleep && lastAsleep[i])) resets++;
      }
      series.push([tick, awakeN, settling, resets]);
      lastTimer = new Float32Array(scene.bodies.length); lastAsleep = new Uint8Array(scene.bodies.length);
      for (let i = 0; i < scene.bodies.length; i++) { lastTimer[i] = f[i * 28 + 15]; lastAsleep[i] = asleep(u, i) ? 1 : 0; }
    }
    if (woke.length && events.length < 30) {
      const set = new Set(woke.slice(0, top)), partners = new Map();
      for (let k = 0; k < c.counters.contact; k++) {
        const a = colliderBody[c.meta[k * 8]], b = colliderBody[c.meta[k * 8 + 1]];
        for (const [x, y] of [[a, b], [b, a]]) if (set.has(x) && !set.has(y)) { const l = partners.get(x) ?? new Set(); l.add(y); partners.set(x, l); }
      }
      const speed = (i) => { const o = i * 28; return Math.hypot(f[o + 12], f[o + 13], f[o + 14]) + Math.hypot(f[o + 16], f[o + 17], f[o + 18]) * ext[i]; };
      events.push({tick, woke: woke.length, sample: [...set].map((i) => ({i, partners: [...(partners.get(i) ?? [])].filter((p) => p > 0 && !asleep(u, p)).slice(0, 6)
        .map((p) => ({p, speed: +speed(p).toFixed(4), timer: +f[p * 28 + 15].toFixed(3), m: +(1 / f[p * 28 + 3]).toFixed(2)}))}))});
    }
  }
  return {totalWakes, series, events};
}
