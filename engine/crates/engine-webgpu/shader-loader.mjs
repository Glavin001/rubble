// Kernel loading shared by the WebGPU pages (demo.html, world-host.mjs, fracture/).
// WGSL is fetched with the default cache mode, so the browser revalidates by ETag:
// it picks up a redeploy and does not re-download unchanged shaders. Fetching and
// validation run in parallel. The status line only names a kernel whose pipeline is
// slow, which is usually a miss in the browser's own shader cache.

const SLOW_MS = 50;

// kernels: [{ name, file }] → [{ src, module }] in the same order.
export async function loadShaderModules(device, kernels, { urlOf, status = () => {}, missing } = {}) {
  status(`compiling ${kernels.length} shaders…`);
  return Promise.all(kernels.map(async ({ name, file }) => {
    const url = urlOf(file);
    // One retry: on a phone or a proxied link, one of ~100 parallel fetches can fail
    // in transit (a network error or a 5xx from the proxy) though the file is there.
    const get = () => fetch(url).catch((e) => ({ ok: false, status: e.message }));
    let response = await get();
    if (!response.ok) response = await get();
    if (!response.ok) throw new Error(`${missing ? missing(file) : url} (HTTP ${response.status})`);
    const src = await response.text();
    const module = device.createShaderModule({ label: name, code: src });
    // Force frontend validation now so createComputePipeline cannot hang forever
    // on a deferred compile error (Chrome async path has been wedging here).
    if (module.getCompilationInfo) {
      const err = (await module.getCompilationInfo()).messages.find((m) => m.type === 'error');
      if (err) throw new Error(`${name}: ${err.message}`);
    }
    return { src, module };
  }));
}

// Sequential pipeline builds. Call `create(name, descriptor)` per kernel, then
// `await finish()`. Validation errors are collected without a GPU round trip per
// kernel, and the event loop is only yielded once SLOW_MS has passed.
export function pipelineBuilder(device, total, status = () => {}) {
  const checks = [];
  let done = 0, lastYield = performance.now();
  status(`creating ${total} pipelines…`);
  return {
    async create(name, descriptor) {
      const t = performance.now();
      // Sync create: createComputePipelineAsync was hanging on sleep_island_commit
      // after the large narrowphase / solve_primal modules on Chrome Metal.
      device.pushErrorScope('validation');
      const pipeline = device.createComputePipeline(descriptor);
      checks.push(device.popErrorScope().then((e) => e && new Error(`${name}: ${e.message}`)));
      const now = performance.now(), ms = now - t;
      done++;
      status(ms >= SLOW_MS ? `pipeline ${name} ${ms.toFixed(0)}ms (${done}/${total})` : `pipelines ${done}/${total}`);
      if (now - lastYield >= SLOW_MS) {
        // setTimeout, not rAF: background tabs throttle rAF and freeze load between kernels.
        await new Promise((r) => setTimeout(r, 0));
        lastYield = performance.now();
      }
      return pipeline;
    },
    async finish() {
      const err = (await Promise.all(checks)).find(Boolean);
      if (err) throw err;
    },
  };
}
