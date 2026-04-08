/// WGSL kernels for building per-body contact adjacency entirely on the GPU.
pub const ADJACENCY_RESET_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> body_contact_counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> active_body_flags: array<u32>;
@group(0) @binding(2) var<uniform>             params:              vec4<u32>; // x = num_bodies

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.x {
        return;
    }
    body_contact_counts[idx] = 0u;
    active_body_flags[idx] = 0u;
}
"#;

pub const ADJACENCY_COUNT_WGSL: &str = r#"
struct Contact {
    point:          vec4<f32>,
    normal:         vec4<f32>,
    tangent:        vec4<f32>,
    local_anchor_a: vec4<f32>,
    local_anchor_b: vec4<f32>,
    lambda:         vec4<f32>,
    penalty:        vec4<f32>,
    body_a:         u32,
    body_b:         u32,
    feature_id:     u32,
    flags:          u32,
};

@group(0) @binding(0) var<storage, read>       contacts:            array<Contact>;
@group(0) @binding(1) var<storage, read_write> body_contact_counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> active_body_flags:   array<atomic<u32>>;
@group(0) @binding(3) var<storage, read>       contact_count_buf:   array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x;
    let contact_count = contact_count_buf[0];
    if ci >= contact_count {
        return;
    }

    let c = contacts[ci];
    atomicAdd(&body_contact_counts[c.body_a], 1u);
    atomicAdd(&body_contact_counts[c.body_b], 1u);
    atomicStore(&active_body_flags[c.body_a], 1u);
    atomicStore(&active_body_flags[c.body_b], 1u);
}
"#;

pub const ADJACENCY_INIT_RANGES_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       body_contact_counts:  array<u32>;
@group(0) @binding(1) var<storage, read>       body_contact_offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> body_contact_ranges:  array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> body_contact_heads:   array<u32>;
@group(0) @binding(4) var<uniform>             params:               vec4<u32>; // x = num_bodies

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.x {
        return;
    }

    let offset = body_contact_offsets[idx];
    let count = body_contact_counts[idx];
    body_contact_ranges[idx] = vec2<u32>(offset, count);
    body_contact_heads[idx] = offset;
}
"#;

pub const ADJACENCY_SCATTER_WGSL: &str = r#"
struct Contact {
    point:          vec4<f32>,
    normal:         vec4<f32>,
    tangent:        vec4<f32>,
    local_anchor_a: vec4<f32>,
    local_anchor_b: vec4<f32>,
    lambda:         vec4<f32>,
    penalty:        vec4<f32>,
    body_a:         u32,
    body_b:         u32,
    feature_id:     u32,
    flags:          u32,
};

@group(0) @binding(0) var<storage, read>       contacts:              array<Contact>;
@group(0) @binding(1) var<storage, read_write> body_contact_heads:    array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> body_contact_indices:  array<u32>;
@group(0) @binding(3) var<storage, read>       contact_count_buf:     array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x;
    let contact_count = contact_count_buf[0];
    if ci >= contact_count {
        return;
    }

    let c = contacts[ci];
    let slot_a = atomicAdd(&body_contact_heads[c.body_a], 1u);
    let slot_b = atomicAdd(&body_contact_heads[c.body_b], 1u);
    body_contact_indices[slot_a] = ci;
    body_contact_indices[slot_b] = ci;
}
"#;
