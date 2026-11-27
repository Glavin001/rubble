// The crash lab page's renderer set-up and frame drawing (crash-lab.html): 4× MSAA into the
// canvas, a 2,048² sun shadow map around the camera target, the frame uniform (the camera,
// the sun, the view options, the frame's lead), the World's colliders (the sim's static
// shapes, the wheels' stand-in boxes), and per frame: the lab renderer (crash-lab-render.mjs:
// tubes, nodes, solid skin, studio), the textured body (crash-lab-body.mjs) or, in the plain
// views, the solver's own collision skin and beams (crash-lab-shader.mjs). State: page.draw.

import { createCrashLabRenderer } from './crash-lab-render.mjs';
import { WHEEL_RIM_MATERIAL } from './vehicle.mjs';
import { SCENE_WGSL } from './crash-lab-shader.mjs';
import { createBody } from './crash-lab-body.mjs';
import { norm3, mat4Inv, lightViewProj } from './view-math.mjs';

const SAMPLES = 4;
const SHADOW = 2048;
const LIGHT = norm3([0.45, 0.8, 0.35]);

// A unit cube (36 vertices: position, normal) and the 800 m ground square.
function meshes() {
  const cube = [];
  for (const [u, v, w] of [[0, 1, 2], [1, 2, 0], [2, 0, 1]]) for (const s of [-1, 1]) {
    const n = [0, 0, 0]; n[w] = s;
    const cs = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => { const p = [0, 0, 0]; p[u] = a * 0.5; p[v] = b * 0.5; p[w] = s * 0.5; return p; });
    const tri = s > 0 ? [[0, 1, 2], [0, 2, 3]] : [[0, 2, 1], [0, 3, 2]];
    for (const t of tri) for (const k of t) cube.push(...cs[k], ...n);
  }
  const G = 400;
  const ground = [[-G, -G], [G, -G], [G, G], [-G, -G], [G, G], [-G, G]].flatMap(([x, z]) => [x, 0.001, z]);
  return { cube, ground };
}

/** page.draw: pipelines, the lab and body renderers, page.drawScene and draw.writeFrame. */
export async function createDraw(page) {
  const { device, format, world, cfg, qs } = page;
  const mod = device.createShaderModule({ label: 'crash-lab-scene', code: SCENE_WGSL });
  const info = await mod.getCompilationInfo();
  const cerr = info.messages.find((m) => m.type === 'error');
  if (cerr) throw new Error(`render shader: ${cerr.message} @${cerr.lineNum}`);
  const { cube, ground } = meshes();
  const vbo = (arr) => { const b = device.createBuffer({ size: arr.length * 4, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, new Float32Array(arr)); return b; };
  const cubeVbo = vbo(cube), groundVbo = vbo(ground);
  const frameBuf = device.createBuffer({ size: 240, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const shadowTex = device.createTexture({ size: [SHADOW, SHADOW], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const shadowView = shadowTex.createView();
  const shadowSampler = device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
  const shMod = device.createShaderModule({ label: 'crash-lab-shadow', code: SCENE_WGSL.replaceAll('frame.viewProj *', 'frame.lightVP *') });
  const shadowPipe = (vs, buffers) => device.createRenderPipeline({
    layout: 'auto', vertex: { module: shMod, entryPoint: vs, buffers }, primitive: { cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2.5 },
  });
  const vtx6 = [{ arrayStride: 24, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }, { shaderLocation: 1, offset: 12, format: 'float32x3' }] }];
  const depthStencil = { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' };
  const pipe = (vs, fs, buffers, cull = 'back', topology = 'triangle-list') => device.createRenderPipeline({
    layout: 'auto', vertex: { module: mod, entryPoint: vs, buffers }, fragment: { module: mod, entryPoint: fs, targets: [{ format }] },
    primitive: { cullMode: cull, topology }, depthStencil, multisample: { count: SAMPLES },
  });
  const colPipe = pipe('vsCol', 'fsCol', vtx6);
  const groundPipe = pipe('vsGround', 'fsGround', [{ arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }], 'none');
  const xSkinPipe = pipe('vsXSkin', 'fsSkin', [], 'none');
  const xBeamPipe = pipe('vsXBeam', 'fsBeam', [], 'none', 'line-list');
  const skyPipe = device.createRenderPipeline({
    layout: 'auto', vertex: { module: mod, entryPoint: 'vsSky' }, fragment: { module: mod, entryPoint: 'fsSky', targets: [{ format }] },
    primitive: { cullMode: 'none' }, depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' }, multisample: { count: SAMPLES },
  });
  const bind = (p, entries) => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: entries.map(([b, r]) => ({ binding: b, resource: r instanceof GPUBuffer ? { buffer: r } : r })) });
  const SH = [[9, shadowView], [10, shadowSampler]];
  const skyBind = bind(skyPipe, [[1, frameBuf]]);
  const colShadow = shadowPipe('vsCol', vtx6);
  const xSkinShadow = shadowPipe('vsXSkin', []);
  const draw = page.draw = { colDirty: true, frameBuf };
  // The World's buffers can be replaced when they grow: rebind when one is a new object.
  let bound = [], groundBind = null;
  const rebind = () => {
    const bufs = [world.bodyBuffer(), world.colliderBuffer()];
    if (bufs.every((b, i) => b === bound[i])) return;
    bound = bufs;
    colBind = null;
    groundBind = bind(groundPipe, [[0, bufs[0]], [1, frameBuf], ...SH]);
  };
  // Collider instances: only drawable ones (not wheel rims or hidden groups), rebuilt when
  // the colliders change.
  let colBind = null, colShadowBind = null, colIdxBuf = null, colIdxN = 0, colIdxCount = -1;
  const refreshColliders = () => {
    const n = world.colliderCount();
    if (colBind && !draw.colDirty && n === colIdxCount) return;
    draw.colDirty = false;
    colIdxCount = n;
    const mats = world.colliderMaterials(), list = [];
    for (let i = 1; i < n; i++) {
      const mat = mats[i];
      if (mat === WHEEL_RIM_MATERIAL || ((mat | 0) >> 16) < 0) continue;
      list.push(i);
    }
    colIdxN = list.length;
    colIdxBuf?.destroy();
    colIdxBuf = device.createBuffer({ size: Math.max(16, list.length * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    if (list.length) device.queue.writeBuffer(colIdxBuf, 0, Uint32Array.from(list));
    colBind = bind(colPipe, [[0, bound[0]], [1, frameBuf], [2, bound[1]], [11, colIdxBuf], ...SH]);
    colShadowBind = bind(colShadow, [[0, bound[0]], [1, frameBuf], [2, bound[1]], [11, colIdxBuf]]);
  };
  // The solver's bind groups (the plain views' skin and beams), made again for a new sim.
  let xBinds = null;
  const xBindings = () => {
    const sim = page.sim;
    if (xBinds?.sim === sim) return xBinds;
    const n = sim.nodesBuffer(), m = sim.miscBuffer();
    xBinds = {
      sim,
      skin: bind(xSkinPipe, [[1, frameBuf], [13, n], [14, m], [15, sim.trisBuffer()], ...SH]),
      shadow: bind(xSkinShadow, [[1, frameBuf], [13, n], [14, m], [15, sim.trisBuffer()]]),
      beam: bind(xBeamPipe, [[1, frameBuf], [13, n], [14, m], [16, sim.beamRecordsBuffer()], [17, sim.beamsBuffer()]]),
    };
    return xBinds;
  };
  const labR = draw.labR = cfg.lab ? createCrashLabRenderer({ device, format, sampleCount: SAMPLES, frameBuf, shadowView, shadowSampler }) : null;
  labR?.setSim(page.sim, page.layout);
  createBody(page, { sampleCount: SAMPLES });
  // The showcase body reflected in the lab's floor (?labReflect=0 off, =0.2 its reflectance).
  const LAB_REFLECT = qs.has('labReflect') ? Number(qs.get('labReflect')) || 0 : 0.16;
  rebind();
  // The frame uniform and the render targets (after the camera moved: page.updateCamera).
  const frameF = new Float32Array(60);
  let depth = null, msaa = null;
  const canvas = page.canvas;
  draw.writeFrame = () => {
    if (!depth || depth.width !== canvas.width || depth.height !== canvas.height) {
      depth?.destroy(); msaa?.destroy();
      depth = device.createTexture({ size: [canvas.width, canvas.height], format: 'depth24plus', sampleCount: SAMPLES, usage: GPUTextureUsage.RENDER_ATTACHMENT });
      msaa = device.createTexture({ size: [canvas.width, canvas.height], format, sampleCount: SAMPLES, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
    const { vp, eye } = page.labView;
    frameF.set(vp, 0);
    frameF.set([...eye, 0], 16);
    frameF.set([...LIGHT, 0], 20);
    frameF.set([page.beamView ? 1 : 0, cfg.lab && labR.opts.studio ? 1 : 0, page.manual ? 0 : page.lead, 0], 24);
    frameF.set(lightViewProj(page.cam.target, LIGHT), 28);
    frameF.set(mat4Inv(vp), 44);
    device.queue.writeBuffer(frameBuf, 0, frameF);
  };
  const { bodyR, bodyFor } = draw;
  page.drawScene = (enc) => {
    const sim = page.sim, layout = page.layout, lab = page.lab, LAB = cfg.lab;
    rebind();
    if (lab) lab.beforeDraw(enc, page.labDt, page.manual ? null : page.lead);
    if (draw.bodyOn() && draw.syncBody()) {
      // The simulated time shown since the last frame (live or replayed; 0 when paused or
      // scrubbed back: no debris, no wear).
      const step = lab?.shownStep() ?? 0;
      const dt = bodyFor.step === null ? 0 : (step - bodyFor.step) / 2000;
      bodyFor.step = step;
      bodyR.update(enc, { dt, floorY: labR.opts.floorY, boxes: layout.boxes });
    }
    // Sun shadow map.
    const sp = enc.beginRenderPass({ colorAttachments: [], depthStencilAttachment: { view: shadowView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' } });
    refreshColliders();
    if (colIdxN) { sp.setPipeline(colShadow); sp.setBindGroup(0, colShadowBind); sp.setVertexBuffer(0, cubeVbo); sp.draw(36, colIdxN); }
    if (sim && LAB) labR.drawShadow(sp);
    else if (sim) { sp.setPipeline(xSkinShadow); sp.setBindGroup(0, xBindings().shadow); sp.draw(layout.triangles * 3); }
    sp.end();
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: msaa.createView(), resolveTarget: page.context.getCurrentTexture().createView(), clearValue: [0.55, 0.62, 0.72, 1], loadOp: 'clear', storeOp: 'discard' }],
      depthStencilAttachment: { view: depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    const view = page.labView;
    if (LAB && labR.opts.studio) {
      // The body reflected in the studio floor: the floor then leaves no depth behind.
      const refl = sim && draw.bodyOn() && bodyFor.pos ? LAB_REFLECT : 0;
      if (labR.opts.reflect !== refl > 0) labR.set({ reflect: refl > 0 });
      labR.drawBackground(rp);
      if (refl > 0) bodyR.drawReflection(rp, { viewProj: view.vp, eye: view.eye, light: LIGHT, strength: refl, floorY: labR.opts.floorY });
    } else {
      rp.setPipeline(skyPipe); rp.setBindGroup(0, skyBind); rp.draw(3);
      rp.setPipeline(groundPipe); rp.setBindGroup(0, groundBind); rp.setVertexBuffer(0, groundVbo); rp.draw(6);
    }
    if (colIdxN) { rp.setPipeline(colPipe); rp.setBindGroup(0, colBind); rp.setVertexBuffer(0, cubeVbo); rp.draw(36, colIdxN); }
    // The lab's skeleton, nodes, solid skin and grab line; the textured body (from
    // labR.positionsBuffer(): live and replayed alike), else the solver's skin and beams.
    if (sim && LAB) labR.drawOpaque(rp);
    if (sim && draw.bodyOn() && bodyFor.pos) bodyR.draw(rp, { viewProj: view.vp, eye: view.eye, light: LIGHT, height: canvas.height });
    else if (sim) {
      const xb = xBindings();
      rp.setPipeline(xSkinPipe); rp.setBindGroup(0, xb.skin); rp.draw(layout.triangles * 3);
      rp.setPipeline(xBeamPipe); rp.setBindGroup(0, xb.beam); rp.draw(4 * layout.beams);
    }
    if (LAB) labR.drawTransparent(rp);
    rp.end();
  };
}
