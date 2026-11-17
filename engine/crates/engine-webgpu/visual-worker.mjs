// Builds a glTF-skinned vehicle's visual (the World wasm's jbeamVisualSkinned) in a
// worker of its own, so the wasm heap the build grows (about 530 MB for the generated
// sedan: its glb and rskin copied in, the visual built, its lists flattened) is freed
// when the worker ends. A wasm heap never shrinks: built on the page's own wasm, those
// pages stay the tab's for as long as it runs (lab-load-memory.md), which a phone does
// not have to spare. The same wasm and the same call: the same visual.
//
// The page posts { wasm, glue, gpuWait, spec, glb, skin } (glb and skin transferred);
// the worker answers { result } with every typed array transferred, or { error }.
// The wasm's glue imports the bare name 'rubble-gpu-wait', which a page's import map
// resolves; a worker has no import map, so the glue is loaded with that name replaced
// by gpuWait's URL.
self.onmessage = async ({ data }) => {
  try {
    const src = (await (await fetch(data.glue)).text()).replace(/from\s*["']rubble-gpu-wait["']/, `from ${JSON.stringify(data.gpuWait)}`);
    const url = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
    const glue = await import(url);
    URL.revokeObjectURL(url);
    await glue.default({ module_or_path: data.wasm });
    const result = glue.jbeamVisualSkinned(data.spec, [], data.glb, data.skin);
    const moved = new Set();
    const walk = (x) => {
      if (!x || typeof x !== 'object') return;
      if (ArrayBuffer.isView(x)) { moved.add(x.buffer); return; }
      for (const v of Array.isArray(x) ? x : Object.values(x)) walk(v);
    };
    walk(result);
    self.postMessage({ result }, [...moved]);
  } catch (e) {
    self.postMessage({ error: String(e?.stack ?? e) });
  }
};
