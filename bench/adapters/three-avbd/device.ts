// A WebGPU device through Dawn for the three-avbd adapter, as three-avbd's tests-gpu/device.ts
// makes one (same limits and features), from this folder's pinned `webgpu` package unless
// BENCH_DAWN=pinned selects three-avbd's own (its 0.6 build needs macOS 26+).
import { createRequire } from 'node:module';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const from = process.env.BENCH_DAWN === 'pinned' ? resolve(here, '../../../inspiration/pinned/three-avbd/package.json') : resolve(here, 'package.json');
const webgpu = createRequire(from)('webgpu');
Object.assign(globalThis, webgpu.globals);

export const dawn = `webgpu@${createRequire(from)('webgpu/package.json').version}`;
export const errors: string[] = [];
export let adapterLabel = '';

/** Dawn 0.2 hands adapter strings back as decimal char codes run together ("97112112108101"). */
function decode(s: string): string {
  return s.replace(/\d{4,}/g, (run) => {
    let out = '';
    for (let i = 0; i < run.length; ) {
      const three = Number(run.slice(i, i + 3));
      const take = run[i] === '1' && three >= 100 && three <= 126 ? 3 : 2;
      out += String.fromCharCode(Number(run.slice(i, i + take)));
      i += take;
    }
    return out;
  }).replace(/-(\d\d)$/, (_, c) => `-${String.fromCharCode(Number(c))}`);
}
export const { device, skip } = await (async (): Promise<{ device: GPUDevice | null; skip: string | false }> => {
  try {
    const gpu = webgpu.create([]);
    const adapter: GPUAdapter | null = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { device: null, skip: 'no WebGPU adapter available' };
    const requiredFeatures = (['timestamp-query'] as GPUFeatureName[]).filter((f) => adapter.features.has(f));
    const requiredLimits = {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    };
    const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
    (globalThis as { __dawn?: unknown }).__dawn = [gpu, adapter, device];
    // Dawn 0.2 cannot take onuncapturederror; run.ts wraps the run in error scopes instead
    try {
      device.onuncapturederror = (e) => errors.push(e.error.message);
    } catch {}
    adapterLabel = [adapter.info?.vendor, adapter.info?.architecture, adapter.info?.device, adapter.info?.description].map((v) => decode(String(v ?? ''))).filter(Boolean).join(' · ');
    return { device, skip: false };
  } catch (e) {
    return { device: null, skip: `WebGPU unavailable: ${e instanceof Error ? e.message : String(e)}` };
  }
})();
