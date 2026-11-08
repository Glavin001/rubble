// demo.html's readbacks in scene order: World's slots hold the scene's bodies in its
// upload order (engine-scenes `web::upload_order`, Morton with ?order=morton), and
// `gpuToScene[slot]` names the scene body at each slot.
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
