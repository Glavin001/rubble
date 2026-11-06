// The playground's render layers (engine/docs/playground-plan.md phase 4): the pipelines a
// page draws a world through, each a Slang shader (engine/slang/render, exported to WGSL by
// wgsl_export). JS only creates the pipelines and binds the world's buffers in place.

import { loadShaderModules } from './shader-loader.mjs';

const WGSL = (file) => new URL(`./generated/wgsl/${file}`, import.meta.url).href;

/**
 * The rigid-body layer (`rigid_bodies.slang`): every body's collider (boxes; spheres and
 * capsules ray-hit) and a ground square. `draw(pass, view)` with `view = { viewProj (16
 * floats, column-major), eye ([x, y, z]), mode (0 position tint, 1 awake/asleep, 2 speed),
 * projectile (body slot of the scene's ball, or 0xffffffff), ground: { half, y } | null,
 * count (bodies after the ground) }`.
 */
export async function createRigidLayer(device, format, { depthFormat = 'depth24plus', status } = {}) {
  const [{ module }] = await loadShaderModules(device, [{ name: 'rigid_bodies', file: 'rigid_bodies.wgsl' }], { urlOf: WGSL, status });
  const storage = { type: 'read-only-storage' };
  const layout = device.createBindGroupLayout({
    label: 'rigid layer',
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: storage },
      { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: storage },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const depthStencil = { format: depthFormat, depthWriteEnabled: true, depthCompare: 'less' };
  const [bodies, ground] = await Promise.all([
    device.createRenderPipelineAsync({
      label: 'rigid bodies', layout: pipelineLayout,
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format }] },
      primitive: { cullMode: 'back' }, depthStencil,
    }),
    device.createRenderPipelineAsync({
      label: 'rigid ground', layout: pipelineLayout,
      vertex: { module, entryPoint: 'vsGround' },
      fragment: { module, entryPoint: 'fsGround', targets: [{ format }] },
      primitive: { cullMode: 'none' }, depthStencil,
    }),
  ]);
  const frameBuf = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const eyeBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const frame = new ArrayBuffer(80), frameF = new Float32Array(frame), frameU = new Uint32Array(frame);
  const eye = new Float32Array(4);
  let bind = null, bound = [];
  return {
    /** Draw `world`'s bodies (its buffers bound in place; rebound when one is replaced). */
    draw(pass, world, view) {
      const bufs = [world.bodyBuffer(), world.colliderBuffer()];
      if (!bind || bufs.some((b, i) => b !== bound[i])) {
        bound = bufs;
        bind = device.createBindGroup({ layout, entries: [
          { binding: 0, resource: { buffer: bufs[0] } },
          { binding: 1, resource: { buffer: frameBuf } },
          { binding: 2, resource: { buffer: bufs[1] } },
          { binding: 3, resource: { buffer: eyeBuf } },
        ] });
      }
      frameF.set(view.viewProj);
      frameU[16] = view.mode ?? 0;
      frameU[17] = view.projectile ?? 0xffffffff;
      frameF[18] = view.ground?.half ?? 0;
      frameF[19] = view.ground?.y ?? 0;
      device.queue.writeBuffer(frameBuf, 0, frame);
      eye.set([view.eye[0], view.eye[1], view.eye[2], 0]);
      device.queue.writeBuffer(eyeBuf, 0, eye);
      pass.setBindGroup(0, bind);
      if (view.ground) { pass.setPipeline(ground); pass.draw(6); }
      if (view.count > 0) { pass.setPipeline(bodies); pass.draw(36, view.count); }
    },
  };
}
