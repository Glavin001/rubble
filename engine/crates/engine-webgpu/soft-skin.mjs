// GPU skinning of render meshes bound to soft bodies (engine kernel
// slang/kernels/soft_skin.slang → generated/wgsl/soft_skin.wgsl; the native twin
// is engine_webgpu::SoftSkinner). After each step, encode() rebuilds the mesh
// from the body buffer with no readback:
//
//   positions  vec4f per vertex: xyz, w = 1 when its frame is torn
//   normals    vec4f per vertex: smooth normal, w = largest crumple strain around it
//   faces      vec4f per triangle: area-weighted normal, w = strain against rest,
//              or −1 when torn (drop it)
//
// bindings: SoftHost.bindMesh() words (8 per vertex); rest: Float32Array xyz
// (the vertices as bound, for strain); triangles: Uint32Array, 3 or 4 words per
// triangle (a, b, c[, user word kept in the record for your renderer).

export async function createSoftSkinner(device, bodies, { bindings, rest, triangles, stride = 3 }, { url = new URL('./generated/wgsl/soft_skin.wgsl', import.meta.url) } = {}) {
  const src = await (await fetch(url)).text();
  const module = device.createShaderModule({ label: 'soft_skin', code: src });
  const err = (await module.getCompilationInfo()).messages.find((m) => m.type === 'error');
  if (err) throw new Error(`soft_skin.wgsl: ${err.message}`);
  const pipe = (entryPoint) => device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
  const pipes = [pipe('soft_skin_vertices'), pipe('soft_skin_faces'), pipe('soft_skin_normals')];
  const nv = bindings.length / 8, nt = triangles.length / stride;
  const tris = new Uint32Array(nt * 4);
  for (let t = 0; t < nt; t++) for (let k = 0; k < stride; k++) tris[t * 4 + k] = triangles[t * stride + k];
  const rest4 = new Float32Array(nv * 4);
  for (let v = 0; v < nv; v++) rest4.set(rest.subarray(v * 3, v * 3 + 3), v * 4);
  // Vertex → triangles (CSR): starts, then the lists.
  const deg = new Uint32Array(nv);
  for (let t = 0; t < nt; t++) for (let c = 0; c < 3; c++) deg[tris[t * 4 + c]]++;
  const adj = new Uint32Array(nv + 1 + nt * 3);
  let acc = nv + 1;
  for (let v = 0; v < nv; v++) { adj[v] = acc; acc += deg[v]; }
  adj[nv] = acc;
  const fill = adj.slice(0, nv);
  for (let t = 0; t < nt; t++) for (let c = 0; c < 3; c++) adj[fill[tris[t * 4 + c]]++] = t;
  const buf = (data, usage) => {
    const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
    if (data.byteLength) device.queue.writeBuffer(b, 0, data);
    return b;
  };
  const S = GPUBufferUsage.STORAGE, OUT = S | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_SRC;
  const bindB = buf(bindings, S), restB = buf(rest4, S), trisB = buf(tris, S), adjB = buf(adj, S);
  const positions = buf(new Float32Array(nv * 4), OUT), faces = buf(new Float32Array(nt * 4), OUT), normals = buf(new Float32Array(nv * 4), OUT);
  const params = new ArrayBuffer(16);
  new Uint32Array(params, 0, 2).set([nv, nt]);
  new Float32Array(params, 8, 1)[0] = 1.8;
  const paramsB = buf(new Uint8Array(params), GPUBufferUsage.UNIFORM);
  const asleepB = buf(new Uint32Array(nv), S);
  let first = true;
  const group = (p, list) => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: list.map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
  const groups = [
    group(pipes[0], [[0, bodies], [1, bindB], [5, positions], [8, paramsB], [9, asleepB]]),
    group(pipes[1], [[2, restB], [3, trisB], [5, positions], [6, faces], [8, paramsB], [9, asleepB]]),
    group(pipes[2], [[4, adjB], [6, faces], [7, normals], [8, paramsB]]),
  ];
  return {
    positions, normals, faces, triangles: trisB, vertices: nv, triangleCount: nt,
    /** Skin from the current body buffer (encode after the step). */
    encode(enc) {
      const pass = enc.beginComputePass({ label: 'soft_skin' });
      [nv, nt, nv].forEach((n, i) => { pass.setPipeline(pipes[i]); pass.setBindGroup(0, groups[i]); pass.dispatchWorkgroups(Math.max(1, Math.ceil(n / 64))); });
      pass.end();
      // Every vertex skinned once: from now on sleeping ones keep their output.
      if (first) { first = false; device.queue.writeBuffer(paramsB, 12, Uint32Array.of(1)); }
    },
    destroy() { [bindB, restB, trisB, adjB, positions, faces, normals, paramsB, asleepB].forEach((b) => b.destroy()); },
  };
}
