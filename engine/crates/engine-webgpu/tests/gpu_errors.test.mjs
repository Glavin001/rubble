import test from 'node:test';
import assert from 'node:assert/strict';

// A WebGPU error in words (gpu-errors.mjs): WebKit's out-of-memory error with an empty
// message printed as "[object GPUOutOfMemoryError]" on an iPhone. Stand-ins for the
// browser's classes (node has no WebGPU).
class GPUOutOfMemoryError { constructor(message) { this.message = message; } }
class GPUValidationError { constructor(message) { this.message = message; } }
class GPUBuffer { destroy() {} }
class GPUTexture { destroy() {} }
class GPUDevice {
  constructor(limits) { this.limits = limits; this.scopes = []; }
  createBuffer(d) { return new GPUBuffer(d); }
  createTexture(d) { return new GPUTexture(d); }
  pushErrorScope(filter) { this.scopes.push({ filter, error: null }); }
  popErrorScope() { const s = this.scopes.pop(); return Promise.resolve(s.error); }
  // What the browser does with an error raised inside the scopes.
  raise(error) { const want = error instanceof GPUOutOfMemoryError ? 'out-of-memory' : 'validation'; for (let i = this.scopes.length - 1; i >= 0; i--) if (this.scopes[i].filter === want) { this.scopes[i].error ??= error; return; } }
}
Object.assign(globalThis, { GPUDevice, GPUBuffer, GPUTexture });
const { trackGpuAllocations, gpuErrorText, gpuPhase, gpuAllocations, textureBytes } = await import('../gpu-errors.mjs');
trackGpuAllocations();
trackGpuAllocations(); // once is enough; twice changes nothing

const MB = 2 ** 20;
const IOS = { maxBufferSize: 256 * MB, maxStorageBufferBindingSize: 256 * MB };

test('an out-of-memory error with no message says what it is, what was created and the limits', () => {
  const device = new GPUDevice(IOS);
  device.createBuffer({ label: 'b33', size: 16 * MB });
  device.createBuffer({ label: 'showcase-binds', size: 157 * MB });
  device.createTexture({ label: 'msaa', size: [1000, 1000], format: 'bgra8unorm', sampleCount: 4 });
  const text = gpuErrorText(new GPUOutOfMemoryError(''), device);
  assert.doesNotMatch(text, /\[object/);
  assert.match(text, /^GPU out of memory \(GPUOutOfMemoryError\): the browser gave no message\./);
  assert.match(text, /showcase-binds" 157 MB/);
  assert.match(text, /largest: buffer "showcase-binds"/);
  assert.match(text, /173 MB of buffers/);
  assert.match(text, /largest buffer 256 MB \(maxBufferSize\), largest storage binding 256 MB \(maxStorageBufferBindingSize\)/);
});

test('a validation error keeps the browser\'s text; destroyed buffers are no longer counted', () => {
  const device = new GPUDevice(IOS);
  const b = device.createBuffer({ label: 'big', size: 100 * MB });
  device.createBuffer({ label: 'small', size: 2 * MB });
  b.destroy();
  assert.equal(gpuAllocations(device).buffers, 2 * MB);
  const text = gpuErrorText(new GPUValidationError('Buffer size exceeds the limit'), device);
  assert.match(text, /^GPU validation error \(GPUValidationError\): Buffer size exceeds the limit\./);
  assert.match(text, /2\.0 MB of buffers/);
  // Not a GPUError at all: its text, plainly.
  assert.match(gpuErrorText('lost'), /^GPU error: lost\./);
});

test('a phase names itself and what it created, largest first, and throws', async () => {
  const device = new GPUDevice(IOS);
  device.createBuffer({ label: 'before', size: 50 * MB });
  await assert.rejects(gpuPhase(device, 'uploading the car', () => {
    device.createBuffer({ label: 'showcase-attrs', size: 19 * MB });
    device.createBuffer({ label: 'showcase-binds', size: 157 * MB });
    device.raise(new GPUOutOfMemoryError(''));
  }), (e) => {
    assert.equal(e.name, 'GpuError');
    assert.equal(e.outOfMemory, true);
    assert.match(e.message, /while uploading the car\./);
    assert.match(e.message, /That created 2 buffers and textures, 176 MB in all; the largest: buffer "showcase-binds" 157 MB, buffer "showcase-attrs" 19 MB\./);
    return true;
  });
  // A phase without errors returns its result and leaves the scopes as they were.
  assert.equal(await gpuPhase(device, 'nothing', async () => 7), 7);
  assert.equal(device.scopes.length, 0);
});

test('texture bytes count mips and samples', () => {
  assert.equal(textureBytes({ size: [4, 4], format: 'rgba8unorm', mipLevelCount: 3 }), (16 + 4 + 1) * 4);
  assert.equal(textureBytes({ size: [10, 10], format: 'depth32float', sampleCount: 4 }), 1600);
});
