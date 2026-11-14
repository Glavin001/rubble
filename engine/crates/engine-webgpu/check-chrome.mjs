// How the browser checks launch Chrome, and the stock-Chrome guard they share.
//
// Default: headless with --enable-unsafe-webgpu (Linux CI needs it for WebGPU).
// --stock: the Chrome a user runs: a visible window, none of the WebGPU or
// throttling flags (only --disable-backgrounding-occluded-windows), a fresh profile. It matters because a windowed Chrome arms its
// GPU watchdog and headless Chrome does not: a synchronous createComputePipeline
// whose Metal compile outlasts the watchdog (~30 s) kills the GPU process and loses
// every WebGPU device ("A valid external Instance reference no longer exists"),
// while a headless check passes. How long a compile takes depends on the shader
// cache (cold on a user's first load, warm on the checker's), so --stock also
// fails any synchronous compute pipeline creation of a large module (over
// SYNC_LIMIT bytes of WGSL) that was not compiled first with
// createComputePipelineAsync: the hazard itself, cache or not.
export const SYNC_LIMIT = 64 * 1024;

/** Chrome's arguments: `stock` (a user's Chrome) or the headless checker's. */
export function chromeArgs({ stock, port, profile, width, height }) {
  const common = [`--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, `--window-size=${width},${height}`, '--no-first-run', '--no-default-browser-check'];
  // Stock keeps one non-GPU flag: an occluded checker window would otherwise get
  // no animation frames (macOS occlusion), and the runs step through frames.
  return stock
    ? [...common, '--disable-backgrounding-occluded-windows', 'about:blank']
    : ['--headless=new', ...common, '--enable-unsafe-webgpu', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', 'about:blank'];
}

/**
 * The page probe (Page.addScriptToEvaluateOnNewDocument): records in
 * `window.__syncCompiles` every synchronous compute pipeline of a module over
 * SYNC_LIMIT bytes not compiled async first, and `window.__adapterLimits`.
 */
export const SYNC_COMPILE_PROBE = `(() => {
  if (!self.GPUDevice) return;
  const LIMIT = ${SYNC_LIMIT};
  const code = new WeakMap(), warmed = new Set();
  window.__syncCompiles = [];
  const sm = GPUDevice.prototype.createShaderModule;
  GPUDevice.prototype.createShaderModule = function (d) { const m = sm.call(this, d); code.set(m, d.code); return m; };
  const ca = GPUDevice.prototype.createComputePipelineAsync;
  GPUDevice.prototype.createComputePipelineAsync = function (d) { const c = code.get(d.compute.module); if (c) warmed.add(c); return ca.call(this, d); };
  const cs = GPUDevice.prototype.createComputePipeline;
  GPUDevice.prototype.createComputePipeline = function (d) {
    const c = code.get(d.compute.module);
    if (c && c.length > LIMIT && !warmed.has(c)) window.__syncCompiles.push((d.label || d.compute.entryPoint || '?') + ' (' + Math.round(c.length / 1024) + ' KB)');
    return cs.call(this, d);
  };
  const ra = GPU.prototype.requestAdapter;
  GPU.prototype.requestAdapter = async function (o) {
    const a = await ra.call(this, o);
    if (a) { const l = {}; for (const k in a.limits) l[k] = a.limits[k]; window.__adapterLimits = l; window.__adapterFeatures = [...a.features]; }
    return a;
  };
})();`;
