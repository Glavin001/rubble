// Dispatch and barrier costs for the contact-group design
// (engine/docs/measurements/contact-groups-design.md). One source for the native runner
// (engine-webgpu examples/contact_dispatch_cost.rs) and headless Chrome
// (engine/tools/contact-groups/dispatch.mjs). WG_WORDS is substituted by the runner:
// the size of the workgroup array (the vehicle kernels hold about 7,900 words).
//
// Bindings as the explicit frame's: a uniform and ten storage buffers, every one used.
struct Params { k: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> gp: Params;
@group(0) @binding(1) var<storage, read_write> b1: array<u32>;
@group(0) @binding(2) var<storage, read_write> b2: array<u32>;
@group(0) @binding(3) var<storage, read_write> b3: array<u32>;
@group(0) @binding(4) var<storage, read_write> b4: array<u32>;
@group(0) @binding(5) var<storage, read_write> b5: array<u32>;
@group(0) @binding(6) var<storage, read_write> b6: array<u32>;
@group(0) @binding(7) var<storage, read_write> b7: array<u32>;
@group(0) @binding(8) var<storage, read_write> b8: array<u32>;
@group(0) @binding(9) var<storage, read_write> b9: array<u32>;
@group(0) @binding(10) var<storage, read_write> b10: array<u32>;

var<workgroup> w: array<u32, WG_WORDS>;

fn touch(i: u32) -> u32 {
  return b2[i % 64u] + b3[i % 64u] + b4[i % 64u] + b5[i % 64u] + b6[i % 64u] + b7[i % 64u]
    + b8[i % 64u] + b9[i % 64u] + b10[i % 64u];
}

// A dependent dispatch: every lane reads its word of the last dispatch, passes it
// through the workgroup array and writes it back plus one.
@compute @workgroup_size(512)
fn dispatch(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>) {
  let i = g.x * 512u + l;
  w[l] = b1[i] + (touch(l) + gp.k) * 0u;
  workgroupBarrier();
  b1[i] = w[(l + 1u) % 512u] + 1u;
}

// gp.k barriers as the vehicle kernels' sync() (a storage barrier, then a workgroup
// barrier), each after a store to the workgroup array and one to the storage buffer.
@compute @workgroup_size(512)
fn syncs(@builtin(local_invocation_index) l: u32, @builtin(workgroup_id) g: vec3<u32>) {
  let i = g.x * 512u + l;
  var x = b1[i] + touch(l) * 0u;
  w[l] = x;
  workgroupBarrier();
  for (var k = 0u; k < min(gp.k, 100000u); k++) {
    x = x + w[(l + 1u + k) % 512u];
    b1[i] = x;
    storageBarrier();
    workgroupBarrier();
    w[l] = x;
  }
  b1[i] = x;
}
