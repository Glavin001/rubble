// GPU storage permutation only. Scene IDs remain in construction order for
// accuracy checks, impactor tracking and reset. Keep the renderer's ground at 0.
export function mortonSceneOrder(bodies, colliders) {
  if (!bodies.length || bodies[0].invMass !== 0) throw new Error('Morton upload requires static ground at scene body 0.');
  if (bodies.length !== colliders.length) throw new Error('Morton demo upload requires one collider per body.');
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let id = 0; id < bodies.length; id++) {
    if (colliders[id].body !== id) throw new Error(`Morton upload: collider ${id} belongs to body ${colliders[id].body}; compound mapping is not supported by this demo renderer.`);
    for (let k = 0; k < 3; k++) {
      const v = bodies[id].p[k];
      if (!Number.isFinite(v)) throw new Error(`Morton upload: body ${id}, position axis ${k} is not finite.`);
      lo[k] = Math.min(lo[k], v); hi[k] = Math.max(hi[k], v);
    }
  }
  const spread = v => {
    let x = v & 0x3ff;
    x = (x | x << 16) & 0x30000ff;
    x = (x | x << 8) & 0x300f00f;
    x = (x | x << 4) & 0x30c30c3;
    return (x | x << 2) & 0x9249249;
  };
  const codes = bodies.map(b => {
    const q = k => Math.min(1023, Math.floor((b.p[k] - lo[k]) / Math.max(hi[k] - lo[k], 1e-9) * 1024));
    // Reference uses Z-up. Our Y/Z reflection must also permute Morton axes.
    return (spread(q(0)) | spread(q(2)) << 1 | spread(q(1)) << 2) >>> 0;
  });
  const gpuToScene = Uint32Array.from(bodies.map((_,id)=>id).slice(1).sort((a,b)=>codes[a]-codes[b] || a-b));
  const order = new Uint32Array(bodies.length);
  order.set(gpuToScene,1);
  const sceneToGpu = new Uint32Array(bodies.length);
  order.forEach((scene,gpu)=>{ sceneToGpu[scene]=gpu; });
  return {gpuToScene:order,sceneToGpu};
}

export function restoreSceneOrder(bytes, gpuToScene, stride = 112) {
  const n = gpuToScene.length;
  if (!Number.isSafeInteger(stride) || stride <= 0 || bytes.byteLength < n * stride) {
    throw new Error(`Scene readback: ${bytes.byteLength} bytes cannot hold ${n} records of stride ${stride}.`);
  }
  const seen = new Uint8Array(n), result = new Uint8Array(n * stride);
  const source = new Uint8Array(bytes, 0, n * stride);
  for (let gpu = 0; gpu < n; gpu++) {
    const scene = gpuToScene[gpu];
    if (!Number.isInteger(scene) || scene < 0 || scene >= n || seen[scene]) throw new Error(`Scene readback: GPU body ${gpu} has invalid or duplicate scene identity ${scene}.`);
    seen[scene] = 1;
    result.set(source.subarray(gpu * stride,(gpu + 1) * stride),scene * stride);
  }
  return result.buffer;
}
