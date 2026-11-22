// Preloaded by `run.sh --probe --preload tools/vehicle-lab/sync-pipelines.js …`: times
// every synchronous createComputePipeline / createRenderPipeline. In a windowed Chrome a
// synchronous creation compiles on the GPU process's main thread, and one longer than the
// GPU watchdog kills the GPU process (compute_spec.rs). Headless Chrome arms no watchdog,
// so this finds such a creation safely. Read back with `window.__syncPipes`, logged on
// the console as `[sync-pipe]`.
(() => {
  const log = (window.__syncPipes = []);
  const wrap = (name) => {
    const f = GPUDevice.prototype[name];
    GPUDevice.prototype[name] = function (desc) {
      const t = performance.now();
      const r = f.call(this, desc);
      const ms = performance.now() - t;
      const label = desc?.label ?? desc?.compute?.entryPoint ?? '?';
      log.push([name, label, Math.round(ms), Math.round(t)]);
      if (ms > 200) console.warn(`[sync-pipe] ${name} ${label} ${(ms / 1000).toFixed(1)} s`);
      return r;
    };
  };
  if (typeof GPUDevice !== 'undefined') { wrap('createComputePipeline'); wrap('createRenderPipeline'); }
})();
