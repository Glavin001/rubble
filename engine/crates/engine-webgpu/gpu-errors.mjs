// A WebGPU error in words a person can act on. A browser's uncaptured error names
// neither the object that failed nor how much was asked for, and WebKit's
// GPUOutOfMemoryError can arrive with an empty message (a page that prints the error
// shows "[object GPUOutOfMemoryError]"). Here: the error's kind, the browser's text
// when it gave one, what the page was creating (the phase, the last buffers and
// textures created, with labels and sizes), what was live on the device and the
// device's buffer limits.
//
//   trackGpuAllocations();                       // once, before any device is made
//   device.addEventListener('uncapturederror', (e) => fail(gpuErrorText(e.error, device)));
//   await gpuPhase(device, 'building the car renderer', () => { ... });  // throws a GpuError
//
// An uncaptured error reaches the page after the fact, so its text names what was
// created last; a phase (its own error scopes) names the phase and what it created.
//
// Bookkeeping only: nothing is created, sized or freed differently.

const MB = 2 ** 20;
const mb = (b) => (b >= 10 * MB ? `${Math.round(b / MB)} MB` : b >= MB ? `${(b / MB).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} kB`);

const KINDS = {
  GPUOutOfMemoryError: 'GPU out of memory',
  GPUValidationError: 'GPU validation error',
  GPUInternalError: 'GPU internal error',
};

// Per device: live bytes, the latest creations (a ring of RECENT) and every creation's
// record (up to LOG, for a phase's list).
const RECENT = 6, LOG = 20000;
const stats = new WeakMap();
const sizes = new WeakMap();
const statsOf = (device) => {
  let s = stats.get(device);
  if (!s) stats.set(device, (s = { buffers: 0, textures: 0, count: 0, largest: null, recent: [], log: [], phase: '' }));
  return s;
};

/** A texture descriptor's bytes (all mips and samples; formats the pages use, 4 B a texel otherwise). */
export function textureBytes(d) {
  const s = Array.isArray(d.size) ? d.size : [d.size.width, d.size.height ?? 1, d.size.depthOrArrayLayers ?? 1];
  const w = s[0], h = s[1] ?? 1, l = s[2] ?? 1;
  const f = d.format ?? '';
  const bpp = /rgba32/.test(f) ? 16 : /rgba16|rg32/.test(f) ? 8 : /depth24plus-stencil8|depth32float-stencil8/.test(f) ? 5
    : /^(r8|stencil8)/.test(f) ? 1 : /^(rg8|r16|depth16)/.test(f) ? 2 : 4;
  let total = 0;
  for (let m = 0, mw = w, mh = h; m < (d.mipLevelCount ?? 1); m++, mw = Math.max(1, mw >> 1), mh = Math.max(1, mh >> 1)) total += mw * mh * l * bpp;
  return total * (d.sampleCount ?? 1);
}

let installed = false;
/**
 * Counts every buffer and texture created on any device (labels, sizes, live bytes,
 * less destroy()), for `gpuErrorText`. Idempotent; a no-op without WebGPU.
 */
export function trackGpuAllocations() {
  if (installed || typeof GPUDevice === 'undefined') return;
  installed = true;
  const note = (device, kind, label, bytes, obj) => {
    const s = statsOf(device);
    s[kind] += bytes; s.count++;
    const rec = { kind: kind === 'buffers' ? 'buffer' : 'texture', label: label || '(no label)', bytes };
    if (!s.largest || bytes > s.largest.bytes) s.largest = rec;
    s.recent.push(rec); if (s.recent.length > RECENT) s.recent.shift();
    if (s.log.length < LOG) s.log.push(rec);
    if (obj) sizes.set(obj, [device, kind, bytes]);
  };
  const cb = GPUDevice.prototype.createBuffer, ct = GPUDevice.prototype.createTexture;
  GPUDevice.prototype.createBuffer = function (d) { const b = cb.call(this, d); note(this, 'buffers', d?.label, Number(d?.size) || 0, b); return b; };
  GPUDevice.prototype.createTexture = function (d) { const t = ct.call(this, d); note(this, 'textures', d?.label, textureBytes(d), t); return t; };
  for (const C of [globalThis.GPUBuffer, globalThis.GPUTexture]) {
    if (!C) continue;
    const ds = C.prototype.destroy;
    C.prototype.destroy = function () {
      const z = sizes.get(this);
      if (z) { statsOf(z[0])[z[1]] -= z[2]; sizes.delete(this); }
      return ds.call(this);
    };
  }
}

/** What `trackGpuAllocations` has seen on `device`: `{ buffers, textures, count, largest, recent, phase }`. */
export function gpuAllocations(device) {
  const s = statsOf(device);
  return { ...s, recent: [...s.recent], log: undefined };
}

/** A GPU error caught by `gpuPhase`: `message` is `gpuErrorText`'s, `gpuError` the browser's. */
export class GpuError extends Error {
  constructor(message, gpuError) { super(message); this.name = 'GpuError'; this.gpuError = gpuError; this.outOfMemory = gpuError?.constructor?.name === 'GPUOutOfMemoryError'; }
}

/**
 * Runs `f` (sync or async) as the named phase of `device`'s work, inside its own
 * out-of-memory and validation error scopes: an error there throws a `GpuError` naming
 * the phase and the buffers and textures it created (largest first). Returns `f`'s result.
 */
export async function gpuPhase(device, name, f) {
  const s = statsOf(device), was = s.phase, since = s.count;
  s.phase = name;
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  let r, thrown = null;
  try { r = await f(); } catch (e) { thrown = e; }
  // (Scopes pop at once, in order; their errors arrive later.)
  const v = device.popErrorScope(), o = device.popErrorScope();
  s.phase = was;
  const err = (await o.catch(() => null)) ?? (await v.catch(() => null));
  if (err) throw new GpuError(gpuErrorText(err, device, { phase: name, since }), err);
  if (thrown) throw thrown;
  return r;
}

/** The device's (or adapter's) buffer limits as words. */
export function limitsText(limits) {
  if (!limits) return '';
  const parts = [];
  if (limits.maxBufferSize) parts.push(`largest buffer ${mb(limits.maxBufferSize)} (maxBufferSize)`);
  if (limits.maxStorageBufferBindingSize) parts.push(`largest storage binding ${mb(limits.maxStorageBufferBindingSize)} (maxStorageBufferBindingSize)`);
  return parts.join(', ');
}

/**
 * A GPU error (a `GPUError`, an `uncapturederror` event's `error`, or anything else)
 * as one plain paragraph: its kind, the browser's text (or that it gave none), what
 * the page was doing and creating, what was live on the device, the device's limits.
 */
export function gpuErrorText(error, device = null, { phase = '', since = -1 } = {}) {
  const type = error && typeof error === 'object' && error.constructor?.name && error.constructor.name !== 'Object' ? error.constructor.name : '';
  const kind = KINDS[type] ?? (type ? `GPU error (${type})` : 'GPU error');
  const msg = typeof error === 'string' ? error : (error?.message ?? '').trim();
  const lines = [`${kind}${type && KINDS[type] ? ` (${type})` : ''}: ${msg || 'the browser gave no message'}.`];
  if (device) {
    const s = statsOf(device);
    if (phase || s.phase) lines.push(`It happened while ${phase || s.phase}.`);
    const made = since >= 0 ? s.log.slice(since) : [];
    if (made.length) {
      const top = [...made].sort((a, b) => b.bytes - a.bytes).slice(0, 5);
      lines.push(`That created ${made.length} buffers and textures, ${mb(made.reduce((t, r) => t + r.bytes, 0))} in all; the largest: ${top.map((r) => `${r.kind} "${r.label}" ${mb(r.bytes)}`).join(', ')}.`);
    }
    if (s.count) {
      lines.push(`On the GPU: ${mb(s.buffers)} of buffers and ${mb(s.textures)} of textures live (${s.count} created)` +
        (s.largest ? `; the largest: ${s.largest.kind} "${s.largest.label}", ${mb(s.largest.bytes)}` : '') + '.');
      if (!made.length) lines.push(`Created last: ${s.recent.map((r) => `${r.kind} "${r.label}" ${mb(r.bytes)}`).join(', ')}.`);
    }
    const l = limitsText(device.limits);
    if (l) lines.push(`This GPU allows: ${l}.`);
  }
  if (type === 'GPUOutOfMemoryError') lines.push('The device ran out of memory for what the page asked for: fewer cars or a lighter car detail may fit.');
  return lines.join(' ');
}
