// The explicit vehicles' nodes as drawn between World steps: interpolated between the
// last two stepped states, never carried forward along their velocities. Rendering only.
//
// Why: a World step (1/60 s) is 33 of the solver's 2 kHz steps, and in a crash a node's
// velocity at the end of a step says little about the next one (the metal rings at
// hundreds of hertz). Carried forward by `v · lead` (up to a whole step, the page's
// slow motion), the drawn car ran into the wall before it touched it and jumped back at
// every step: measured on the generated sedan's 56 km/h offset wall, CPU reference
// (`engine-softbody-lab --example wall_crash_probe`), the paint drawn half a step ahead
// was up to 248 mm from the true state there and 216 mm inside the wall, where the true
// paint is at most 50 mm in and the nodes never. Halfway between the two states: at most
// 87 mm from the truth and 23 mm inside. Each drawn state is a blend of two true
// states, so nothing a convex obstacle keeps out is drawn inside it, and nothing jumps.
// The cost: the drawing shows the World one step late (`shownLead`, negative).
//
//   const blend = createNodeBlend(device, { nodes, misc, nodeVehicles });
//   blend.keep(queue)            // before each World step (the state the step starts from)
//   blend.invalidate()           // after a restart or anything else that moves the nodes
//   blend.update(enc, alpha)     // per drawn frame: alpha in [0, 1] of the way to the last step
//   blend.out                    // the node buffer to draw from (the solver's layout)
//
// The solver's layout (ExplicitSim nodesBuffer): 3 vec4<f32> a node (position relative
// to its vehicle's origin, mass; velocity; force); misc: vec4<u32> a vehicle, its
// origin's f32 bits. The output keeps that layout and the current origins, so the
// renderers bind it in place of the solver's buffer with no lead.

const WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> cur: array<vec4f>;
@group(0) @binding(1) var<storage, read> prev: array<vec4f>;
@group(0) @binding(2) var<storage, read> misc: array<vec4u>;
@group(0) @binding(3) var<storage, read> prevMisc: array<vec4u>;
@group(0) @binding(4) var<storage, read> nodeVeh: array<u32>;
@group(0) @binding(5) var<storage, read_write> outNodes: array<vec4f>;
// x: nodes, y: alpha (f32 bits).
@group(0) @binding(6) var<uniform> u: vec4u;
// A node that moved farther than this in one step (180 m/s) was moved by a restart or
// a teleport, not by the physics: drawn where it is now.
const JUMP = 3.0;
@compute @workgroup_size(64) fn blend(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= u.x) { return; }
  let v = nodeVeh[i] & 0xffffu;
  let o1 = bitcast<vec3f>(misc[v].xyz);
  let o0 = bitcast<vec3f>(prevMisc[v].xyz);
  let c = cur[i * 3u];
  // The previous position relative to the current origin.
  let p0 = prev[i * 3u].xyz + o0 - o1;
  let a = bitcast<f32>(u.y);
  let p = select(mix(p0, c.xyz, a), c.xyz, distance(p0, c.xyz) > JUMP);
  outNodes[i * 3u] = vec4f(p, c.w);
  outNodes[i * 3u + 1u] = cur[i * 3u + 1u];
}
`;

/**
 * @param {GPUDevice} device
 * @param {{ nodes: GPUBuffer, misc: GPUBuffer, nodeVehicles: Uint32Array }} o the World's
 *   explicit node and misc buffers and each node's vehicle (`explicitLabRecords().nodeVehicles`)
 */
export function createNodeBlend(device, { nodes, misc, nodeVehicles }) {
  const count = nodeVehicles.length;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const prev = device.createBuffer({ size: nodes.size, usage });
  const prevMisc = device.createBuffer({ size: misc.size, usage });
  const out = device.createBuffer({ size: nodes.size, usage });
  const veh = device.createBuffer({ size: Math.max(16, nodeVehicles.byteLength), usage });
  if (nodeVehicles.byteLength) device.queue.writeBuffer(veh, 0, nodeVehicles);
  const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: WGSL }), entryPoint: 'blend' } });
  const bind = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [nodes, prev, misc, prevMisc, veh, out, uni].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const words = new Uint32Array(4), alphaBits = new Float32Array(words.buffer, 4, 1);
  let valid = false;
  return {
    out,
    /** Whether the kept state is the one the last step started from. */
    get valid() { return valid; },
    /** Keep the current state (call before each World step, on the same queue). */
    keep(queue) {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(nodes, 0, prev, 0, nodes.size);
      enc.copyBufferToBuffer(misc, 0, prevMisc, 0, misc.size);
      queue.submit([enc.finish()]);
      valid = true;
    },
    /** The kept state is not the step's start any more (a restart): draw the current one. */
    invalidate() { valid = false; },
    /** The drawn nodes, `alpha` of the way from the kept state to the current one (1 when
     * the kept state is not valid). */
    update(enc, alpha) {
      const a = valid ? Math.min(1, Math.max(0, alpha)) : 1;
      words[0] = count; alphaBits[0] = a;
      device.queue.writeBuffer(uni, 0, words);
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(count / 64));
      pass.end();
      return a;
    },
    destroy() { for (const b of [prev, prevMisc, out, veh, uni]) b.destroy(); },
  };
}
