// Shared by browser and native wgpu diagnostics. 2D dispatches tile at 32768
// groups so the 32 MiB case stays within WebGPU's per-axis workgroup limit.
@group(0) @binding(0) var<storage, read_write> values: array<u32>;
override groupSize: u32 = 64u;
@compute @workgroup_size(groupSize)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x + id.y * (32768u * groupSize);
    values[index] = values[index] + 1u;
}
