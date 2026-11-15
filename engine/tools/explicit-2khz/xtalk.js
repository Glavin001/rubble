// CPU<->GPU traffic of the running page, per second (vehicle-lab-record.mjs --probe --eval
// tools/explicit-2khz/xtalk.js): every queue.writeBuffer (uploads), every
// copyBufferToBuffer into a mappable buffer (readbacks), every mapAsync and how long it
// took to resolve, and the page's World steps a second; one line a second into
// window.__abLog (the probe prints it at the end). Measurement only: the wrappers pass
// every call through unchanged.
(() => {
  const s = { up: 0, upBytes: 0, upBig: 0, rb: 0, rbBytes: 0, maps: 0, mapMs: 0, mapMax: 0, steps: 0 };
  const q = GPUQueue.prototype;
  const wb = q.writeBuffer;
  q.writeBuffer = function (buf, off, data, dataOff, size) {
    const n = size !== undefined ? size * (data.BYTES_PER_ELEMENT || 1) : (data.byteLength - (dataOff || 0) * (data.BYTES_PER_ELEMENT || 1));
    s.up++;
    s.upBytes += n;
    if (n >= 4096) s.upBig++;
    return wb.apply(this, arguments);
  };
  const e = GPUCommandEncoder.prototype;
  const cp = e.copyBufferToBuffer;
  e.copyBufferToBuffer = function (src, so, dst, doff, size) {
    if (dst.usage & GPUBufferUsage.MAP_READ) {
      s.rb++;
      s.rbBytes += size ?? dst.size;
    }
    return cp.apply(this, arguments);
  };
  const b = GPUBuffer.prototype;
  const ma = b.mapAsync;
  b.mapAsync = function (mode) {
    const t = performance.now();
    s.maps++;
    return ma.apply(this, arguments).then((r) => {
      const d = performance.now() - t;
      s.mapMs += d;
      s.mapMax = Math.max(s.mapMax, d);
      return r;
    });
  };
  window.__abLog = window.__abLog || [];
  setInterval(() => {
    const st = window.__state?.() || {};
    window.__abLog.push(`xtalk t ${(+(st.t ?? 0)).toFixed(1)}: World steps/s ${(+(st.simRate ?? 0)).toFixed(0)}, refused ${st.refused ?? '?'}; uploads ${s.up} (${(s.upBytes / 1024).toFixed(1)} KB, ${s.upBig} of 4 KB or more); readback copies ${s.rb} (${(s.rbBytes / 1024).toFixed(1)} KB); mapAsync ${s.maps}, mean ${(s.mapMs / Math.max(s.maps, 1)).toFixed(1)} ms, max ${s.mapMax.toFixed(1)} ms`);
    Object.keys(s).forEach((k) => (s[k] = 0));
  }, 1000);
  return 'xtalk on';
})();
