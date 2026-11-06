// The playground's render layers (engine/docs/playground-plan.md phase 4): the pipelines a
// page draws a world through, each a Slang shader (engine/slang/render, exported to WGSL by
// wgsl_export). JS only creates the pipelines, writes the frame and binds the world's
// buffers in place. A preset's look is data: `view.look`.

import { loadShaderModules } from './shader-loader.mjs';

const WGSL = (file) => new URL(`./generated/wgsl/${file}`, import.meta.url).href;

/** Colours (`look.colors`) and lighting (`look.light`), as `slang/render/common.slang`. */
export const COLORS = { position: 0, sleep: 1, speed: 2, shapes: 3 };
export const LIGHT = { demo: 0, sky: 1 };

// The frame every layer reads (common.slang `Frame`, 144 bytes).
function frameWriter(device) {
  const buf = device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const data = new ArrayBuffer(144), f = new Float32Array(data), u = new Uint32Array(data);
  return {
    buf,
    /**
     * `view`: { viewProj (16 floats, column-major), eye [x, y, z], look: { colors, light,
     * lightDir, fog: { distance, color } }, projectile (a body slot, or none), ground: { half,
     * y } | null, boundScale (2: exact bounds; the terrain page pads them 2.02), heightfield
     * (its hull arena base) }.
     */
    write(view) {
      const look = view.look ?? {};
      f.set(view.viewProj, 0);
      f.set([view.eye[0], view.eye[1], view.eye[2], 0], 16);
      f.set([...(look.lightDir ?? [0.35, 0.85, 0.4]), 0], 20);
      u[24] = look.colors ?? 0;
      u[25] = look.light ?? 0;
      u[26] = view.projectile ?? 0xffffffff;
      u[27] = view.heightfield ?? 0;
      f[28] = view.ground?.half ?? 0;
      f[29] = view.ground?.y ?? 0;
      f[30] = view.boundScale ?? 2;
      f[31] = look.fog?.distance ?? 1;
      f.set([...(look.fog?.color ?? [0, 0, 0]), 0], 32);
      device.queue.writeBuffer(buf, 0, data);
    },
  };
}

const depthState = (format) => ({ format, depthWriteEnabled: true, depthCompare: 'less' });
const storage = { type: 'read-only-storage' };
const ALL = GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT;

/**
 * The rigid-body layer (`rigid_bodies.slang`): every collider (boxes; spheres and capsules
 * ray-hit) and an optional ground square. `draw(pass, world, view)`: `view` as the frame
 * (above) plus `count`, the colliders after the ground's to draw.
 */
export async function createRigidLayer(device, format, { depthFormat = 'depth24plus', status } = {}) {
  const [{ module }] = await loadShaderModules(device, [{ name: 'rigid_bodies', file: 'rigid_bodies.wgsl' }], { urlOf: WGSL, status });
  const layout = device.createBindGroupLayout({
    label: 'rigid layer',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: storage },
      { binding: 1, visibility: ALL, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: storage },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const [bodies, ground] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'rigid bodies', layout: pipelineLayout,
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { cullMode: 'back' }, depthStencil: depthState(depthFormat),
    }),
    device.createRenderPipelineAsync({
      label: 'rigid ground', layout: pipelineLayout,
      vertex: { module, entryPoint: 'vsGround' },
      fragment: { module, entryPoint: 'fsGround', targets: [{ format }] },
      primitive: { cullMode: 'none' }, depthStencil: depthState(depthFormat),
    }),
  ]);
  const frame = frameWriter(device);
  let bind = null, bound = [];
  return {
    draw(pass, world, view) {
      const bufs = [world.bodyBuffer(), world.colliderBuffer()];
      if (!bind || bufs.some((b, i) => b !== bound[i])) {
        bound = bufs;
        bind = device.createBindGroup({ layout, entries: [
          { binding: 0, resource: { buffer: bufs[0] } },
          { binding: 1, resource: { buffer: frame.buf } },
          { binding: 2, resource: { buffer: bufs[1] } },
        ] });
      }
      frame.write(view);
      pass.setBindGroup(0, bind);
      if (view.ground) { pass.setPipeline(ground); pass.draw(6); }
      if (view.count > 0) { pass.setPipeline(bodies); pass.draw(36, view.count); }
    },
  };
}

/**
 * The heightfield layer (`heightfield.slang`): heightfield shape `shape`'s surface from its
 * physics records (the hull arena), `nx × nz` samples. `draw(pass, world, view)`.
 */
export async function createHeightfieldLayer(device, format, { depthFormat = 'depth24plus', status } = {}) {
  const [{ module }] = await loadShaderModules(device, [{ name: 'heightfield', file: 'heightfield.wgsl' }], { urlOf: WGSL, status });
  const layout = device.createBindGroupLayout({
    label: 'heightfield layer',
    entries: [
      { binding: 0, visibility: ALL, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: storage },
    ],
  });
  const pipeline = await device.createRenderPipelineAsync({
    label: 'heightfield', layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    primitive: { cullMode: 'none' }, depthStencil: depthState(depthFormat),
  });
  const frame = frameWriter(device);
  let bind = null, bound = null;
  return {
    /** `view.heightfield`: the shape's hull arena base; `view.cells`: (nx − 1)(nz − 1). */
    draw(pass, world, view) {
      const hulls = world.hullArenaBuffer();
      if (!bind || hulls !== bound) {
        bound = hulls;
        bind = device.createBindGroup({ layout, entries: [
          { binding: 0, resource: { buffer: frame.buf } },
          { binding: 1, resource: { buffer: hulls } },
        ] });
      }
      frame.write(view);
      pass.setBindGroup(0, bind);
      pass.setPipeline(pipeline);
      pass.draw(view.cells * 6);
    },
  };
}
