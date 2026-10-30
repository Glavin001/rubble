// Deformable terrain (GPU MLS-MPM sand regions on a heightfield) for the WebGPU
// world host. Rust decides everything: engine_terrain::plan (the command plan
// WgpuSim replays natively) and the open region's lifecycle come from the wasm
// TerrainStep (crates/engine-vehicle-wasm/src/terrain.rs). This file only creates
// the buffers of bindings 37-51, builds the terrain_mpm kernels, executes the
// command words it is given and copies the counters back.
//
//   const terrain = await createTerrain(world, settings);
//   terrain.impact(groundCollider, field, point, radius, density, speed);
//   terrain.impact(groundCollider, field, point2, radius, density, speed); // more at once
//   terrain.areas; // the open areas (TerrainStep.areas JSON)
//   // each frame, before world.encodeStep(enc):
//   terrain.encodeFrame(enc);
//   world.encodeStep(enc); …; device.queue.submit([enc.finish()]); terrain.afterSubmit();
//
// With { surface: true } each coupled frame also rebuilds the live ground over
// every open area from its particles (engine_terrain::surface, render only)
// into terrain.surfaceBuffer (binding T.SURFACE): the heights a renderer draws
// over each window, and the line above which material is drawn as debris.
// With { patches } it also keeps each open area's patch on its sand.
import { buildKernels } from './world-host.mjs';
import { wasm } from './vehicle.mjs';

/** engine_terrain's plan and region lifecycle in wasm (predictions need no GPU). */
export const { TerrainStep } = wasm;
/** Bindings, counter words and record sizes (engine_terrain::layout). */
export const T = JSON.parse(TerrainStep.constants());
const KERNELS = JSON.parse(TerrainStep.kernels());
const SURFACE_KERNELS = JSON.parse(TerrainStep.surface_kernels());
const OP_DISPATCH = 1, OP_INDIRECT = 2;
const READBACKS = 8;

// Workgroups for `threads` invocations at 64 per workgroup (WgpuSim's `groups`).
const groups = (threads) => {
  const n = Math.ceil(Math.max(threads, 1) / 64);
  return [Math.min(n, 32768), Math.ceil(n / 32768), 1];
};

/** Decode `[blocks, block…]` words (terrain.rs) into [{ writes, cmds }]. */
export function decodeBlocks(words, at = 0) {
  const blocks = [];
  const n = words[at++];
  for (let b = 0; b < n; b++) {
    const nw = words[at++], nc = words[at++];
    const writes = [], cmds = [];
    for (let i = 0; i < nw; i++) {
      const binding = words[at], offset = words[at + 1], len = words[at + 2];
      writes.push({ binding, offset, words: words.slice(at + 3, at + 3 + len) });
      at += 3 + len;
    }
    for (let i = 0; i < nc; i++, at += 3) cmds.push({ op: words[at], kernel: words[at + 1], value: words[at + 2] });
    blocks.push({ writes, cmds });
  }
  return blocks;
}

/**
 * Terrain for `world` (world-host.mjs). `settings`: TerrainStep JSON (config,
 * sand, region_margin, z, speed_headroom, settle_speed, settle_frames).
 */
export async function createTerrain(world, settings = {}, { wgslBase = './generated/wgsl/', status = () => {}, surface = false, patch = null, patches = null } = {}) {
  const device = world.device;
  // patches: [{ collider, nx, nz }] per area slot, static heightfields the
  // surface pass keeps on each open area's live sand (wheel rays and
  // contacts meet it; engine_terrain::surface). Give one per max_regions.
  // patch: one, for slot 0.
  patches = patches || (patch ? [patch] : []);
  const newPlan = () => {
    const p = new TerrainStep(JSON.stringify(settings));
    if (patches.length) p.set_surface_patches(Uint32Array.from(patches.flatMap((q) => [q.collider, q.nx, q.nz])));
    return p;
  };
  let plan = newPlan();
  const sizes = plan.buffer_sizes();
  const limit = Math.min(device.limits.maxBufferSize, device.limits.maxStorageBufferBindingSize);
  for (let i = 0; i < sizes.length; i += 3) {
    const [binding, bytes, need] = [sizes[i], sizes[i + 1], sizes[i + 2]];
    if (need > limit) throw new Error(`terrain binding ${binding} needs ${need} bytes; this device allows ${limit} (lower the terrain config's limits)`);
    world.bufs[binding]?.destroy?.();
    world.bufs[binding] = device.createBuffer({
      label: `b${binding} terrain`, size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | (binding === T.ARGS ? GPUBufferUsage.INDIRECT : 0),
    });
  }
  // The look-ahead (terrain_watch.slang, binding 52) is not run by this host;
  // no terrain_mpm kernel binds it.
  if (surface) {
    const bytes = plan.surface_bytes();
    if (bytes > limit) throw new Error(`terrain surface needs ${bytes} bytes; this device allows ${limit} (lower surface_cells)`);
    world.bufs[T.SURFACE]?.destroy?.();
    world.bufs[T.SURFACE] = device.createBuffer({ label: `b${T.SURFACE} terrain surface`, size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
  }
  const pipes = await buildKernels(device, world.bufs, KERNELS, { wgslBase, status });
  const surfacePipes = surface ? await buildKernels(device, world.bufs, SURFACE_KERNELS, { wgslBase, status }) : null;
  const bufs = world.bufs;
  const counterBytes = T.COUNTER_WORDS * 4;
  const free = Array.from({ length: READBACKS }, () => device.createBuffer({ size: counterBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
  const [flagBinding, flagOffset, flagBytes] = plan.flags_clear();

  const terrain = {
    plan, settings: JSON.parse(plan.settings()), T,
    region: null, // the first open area (TerrainStep.region JSON)
    areas: [], // every open area (TerrainStep.areas JSON)
    report: null, // the last frame's report
    error: null, // a flag the GPU raised (the frame's error), or a host failure
    lastKind: 0, lastSubsteps: 0, lastStep: -1,
    onReport: null, onBake: null,
    pending: [],
    /** The live surface (binding T.SURFACE), or null without { surface: true }. */
    surfaceBuffer: surface ? world.bufs[T.SURFACE] : null,
    /** Rebuild the surface each coupled frame (when built with it). */
    surfaceOn: surface,
  };

  const write = (w) => device.queue.writeBuffer(bufs[w.binding], w.offset, w.words);
  const pass = (enc, block, timestampWrites, kernels = pipes, label = 'terrain') => {
    const p = enc.beginComputePass({ label, timestampWrites });
    for (const c of block.cmds) {
      const k = kernels[c.kernel];
      p.setPipeline(k.pipeline);
      p.setBindGroup(0, k.bind);
      if (c.op === OP_DISPATCH) p.dispatchWorkgroups(...groups(c.value));
      else if (c.op === OP_INDIRECT) p.dispatchWorkgroupsIndirect(bufs[T.ARGS], c.value);
      else throw new Error(`terrain command op ${c.op}`);
    }
    p.end();
  };
  // Blocks run one submission each, writes first (WgpuSim::terrain_submit).
  const submitBlocks = (words, at = 0, kernels = pipes) => submitDecoded(decodeBlocks(words, at), kernels);
  const submitDecoded = (blocks, kernels = pipes) => {
    for (const block of blocks) {
      block.writes.forEach(write);
      const enc = device.createCommandEncoder({ label: 'terrain' });
      pass(enc, block, undefined, kernels);
      device.queue.submit([enc.finish()]);
    }
  };
  // A static body's surface changed under its contacts (WgpuSim::mark_surface_moved):
  // its generation becomes this step, so no manifold against the old surface is reused.
  const surfaceMoved = (collider) => {
    const body = world.colliderWords[collider * 16];
    world.bodyWords[body * 28 + 19] = world.steps >>> 0;
    device.queue.writeBuffer(bufs[0], body * 112 + 76, Uint32Array.of(world.steps >>> 0));
  };
  const fieldWords = (f) => Float32Array.of(f.nx, f.nz, f.dx, f.dz, ...(f.origin || [0, 0, 0]));
  const refresh = () => {
    terrain.region = JSON.parse(plan.region());
    terrain.areas = JSON.parse(plan.areas());
  };
  const opened = (collider, words) => {
    if (!terrain.areas.length) {
      terrain.error = null;
      terrain.report = null;
    }
    submitBlocks(words);
    surfaceMoved(collider);
    terrain.collider = collider;
    refresh();
    return terrain.region;
  };

  /** Predictions without opening a region (JSON from π-scaling). */
  terrain.predictImpact = (radius, density, speed, gravity = -world.params.gravityY) => JSON.parse(plan.predict_impact(radius, density, speed, gravity));
  terrain.predictExplosion = (kg, depth, gravity = -world.params.gravityY) => JSON.parse(plan.predict_explosion(kg, depth, gravity));
  /** A vertical impact at `point` on heightfield collider `collider` (`field` = {nx, nz, dx, dz, origin}). */
  terrain.impact = (collider, field, point, radius, density, speed) =>
    opened(collider, plan.impact(collider, fieldWords(field), Float32Array.from(point), radius, density, speed, -world.params.gravityY));
  /** A buried charge of `kg` TNT `depth` m under `point`. */
  terrain.explode = (collider, field, point, kg, depth) =>
    opened(collider, plan.explode(collider, fieldWords(field), Float32Array.from(point), kg, depth, -world.params.gravityY));
  /** A sourceless region over [lo, hi] with cells of dx. */
  terrain.openBox = (collider, field, lo, hi, dx) =>
    opened(collider, plan.open_box(collider, fieldWords(field), Float32Array.from(lo), Float32Array.from(hi), dx));
  // The heightfield is the ground again: the surface's valid word off, and
  // the patches (if any) sunk out of reach.
  const surfaceOff = () => {
    if (!surface) return;
    device.queue.writeBuffer(bufs[T.SURFACE], 4 * T.SURFACE_VALID_WORD, new Uint32Array(1));
    if (patches.length) { submitBlocks(plan.surface_retire(), 0, surfacePipes); patches.forEach((q) => surfaceMoved(q.collider)); }
  };
  const baked = () => {
    surfaceMoved(terrain.collider);
    const before = terrain.areas.map((a) => a.slot);
    refresh();
    /** The area slots the last bake closed (surfaceSwitchError's `slots`). */
    terrain.bakedSlots = before.filter((s) => !terrain.areas.some((a) => a.slot === s));
    // The surface shows every open area: off once none is left (the next
    // surface pass sinks a baked area's patch while others stay open).
    if (!terrain.areas.length) surfaceOff();
    terrain.onBake?.();
  };
  /** Bake every open area now. */
  terrain.close = () => { submitBlocks(plan.close()); baked(); };
  /** Bake open area `slot` now; the others go on. */
  terrain.closeArea = (slot) => { submitBlocks(plan.close_area(slot)); baked(); };

  /**
   * This frame's terrain work into `enc`, ahead of world.encodeStep(enc): the
   * coupled substeps and a copy of their counters (or a bake, submitted on its
   * own). Returns the frame's kind (0 none, 1 coupled, 2 baked, 3 some areas
   * baked and the others coupled).
   * `timestampWrites` (optional) times the coupled pass, `surfaceTimestampWrites`
   * the surface pass that follows it.
   */
  terrain.encodeFrame = (enc, { timestampWrites, surfaceTimestampWrites } = {}) => {
    if (terrain.error) throw terrain.error;
    const words = plan.frame(world.params.dt, world.params.gravityY, world.colliderCount);
    const kind = words[0], step = words[1];
    terrain.lastKind = kind;
    if (kind === 2) {
      submitBlocks(words, 2);
      baked();
    } else if (kind === 1 || kind === 3) {
      const blocks = decodeBlocks(words, 2);
      const block = blocks.pop();
      // Kind 3: the settled areas' bakes first, each its own submission.
      if (blocks.length) { submitDecoded(blocks); baked(); }
      block.writes.forEach(write);
      pass(enc, block, timestampWrites);
      terrain.lastSubsteps = block.cmds.filter((c) => KERNELS[c.kernel] === 'mpm_p2g').length;
      if (surface && terrain.surfaceOn) terrain.encodeSurface(enc, { timestampWrites: surfaceTimestampWrites });
      // The rigid step applies the frame's reactions (impulse records on the GPU).
      world.terrainImpulses = true;
      const rb = free.pop();
      if (!rb) throw new Error('terrain counter readbacks all in flight (await terrain.settle())');
      enc.copyBufferToBuffer(bufs[T.COUNTERS], 0, rb, 0, counterBytes);
      enc.clearBuffer(bufs[flagBinding], flagOffset, flagBytes);
      terrain.pending.push({ rb, step });
    }
    return kind;
  };
  /** The surface pass for the open areas as they are now, into `enc` (encodeFrame
   * runs it after each coupled frame; call it yourself to rebuild it at other times). */
  terrain.encodeSurface = (enc, { timestampWrites } = {}) => {
    if (!surface) throw new Error('createTerrain(…, { surface: true }) builds the surface pass');
    const words = plan.surface(-world.params.gravityY);
    if (!words[0]) return false;
    const [block] = decodeBlocks(words);
    block.writes.forEach(write);
    pass(enc, block, timestampWrites, surfacePipes, 'terrain surface');
    // The patches' surfaces moved under their contacts.
    patches.forEach((q) => surfaceMoved(q.collider));
    return true;
  };
  // After the frame's submission: map its counters; reports arrive in order.
  const reading = [];
  let generation = 0;
  terrain.afterSubmit = () => {
    const gen = generation;
    for (const { rb, step } of terrain.pending.splice(0)) {
      reading.push(rb.mapAsync(GPUMapMode.READ).then(() => {
        const words = new Uint32Array(rb.getMappedRange().slice(0));
        rb.unmap();
        free.push(rb);
        if (gen !== generation) return; // a copy from before a reset
        try {
          terrain.report = JSON.parse(plan.report(step, words));
          terrain.lastStep = step;
          if (terrain.region) refresh();
          terrain.onReport?.(terrain.report, step);
        } catch (e) {
          terrain.error = e instanceof Error ? e : new Error(String(e));
        }
      }));
    }
  };
  /** Wait for every counter copy in flight (before a frame when all are busy, or for tests). */
  terrain.settle = async () => { await Promise.all(reading.splice(0)); };
  terrain.busy = () => free.length === 0;
  /** Start over (a scene reset): a fresh plan with the same settings, no
   * region, counters zeroed. The caller restores the heightfield. */
  terrain.reset = () => {
    generation++;
    plan.free();
    plan = newPlan();
    terrain.plan = plan;
    terrain.region = null; terrain.areas = []; terrain.report = null; terrain.error = null;
    terrain.lastKind = 0; terrain.lastSubsteps = 0; terrain.lastStep = -1;
    terrain.pending.length = 0;
    device.queue.writeBuffer(bufs[T.COUNTERS], 0, new Uint32Array(T.COUNTER_WORDS));
    surfaceOff();
  };

  // ---- reads for rendering and tools ----
  const readWords = async (buffer, offset, bytes) => {
    const rb = device.createBuffer({ size: Math.max(16, bytes), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = device.createCommandEncoder();
    if (bytes) enc.copyBufferToBuffer(buffer, offset, rb, 0, bytes);
    device.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const w = new Uint32Array(rb.getMappedRange(0, bytes).slice(0));
    rb.unmap(); rb.destroy();
    return w;
  };
  /** Particle `i`'s record (32 words: xm, vj, c0-c2, f0-f2). */
  terrain.readParticle = (i) => readWords(bufs[T.PARTICLES], i * T.PARTICLE_BYTES, T.PARTICLE_BYTES);
  /** Parcels [0, n): (x, y, z, quanta bits), (vx, vy, vz, mass; 0 once landed). */
  terrain.readParcels = (n) => readWords(bufs[T.PARCELS], 0, n * T.PARCEL_BYTES);
  /** Parcel `k` alone (8 words). */
  terrain.readParcel = (k) => readWords(bufs[T.PARCELS], k * T.PARCEL_BYTES, T.PARCEL_BYTES);
  /** The counters now (blocking on the GPU queue). */
  terrain.readCounters = () => readWords(bufs[T.COUNTERS], 0, counterBytes);
  return terrain;
}
