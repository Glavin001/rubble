//! WGSL source for GPU body graph coloring over prebuilt body-contact adjacency.

/// Reset kernel: initialize active bodies as uncolored, inactive bodies as sentinels,
/// assign priorities, and seed the body-order values with the identity permutation.
pub const COLORING_RESET_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> body_colors:     array<u32>;
@group(0) @binding(1) var<storage, read_write> body_priorities: array<u32>;
@group(0) @binding(2) var<storage, read_write> body_order:      array<u32>;
@group(0) @binding(3) var<storage, read>       active_body_flags: array<u32>;
@group(0) @binding(4) var<uniform>             params:          vec4<u32>; // x=num_bodies, z=seed

const WORKGROUP_SIZE: u32 = 64u;
const UNCOLORED: u32 = 0xFFFFFFFFu;
const INACTIVE_COLOR: u32 = 0xFFFFFFFEu;

fn hash(key: u32, seed: u32) -> u32 {
    var h = key ^ seed;
    h *= 0xcc9e2d51u;
    h = (h << 15u) | (h >> 17u);
    h *= 0x1b873593u;
    return h;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let num_bodies = params.x;
    if idx >= num_bodies { return; }
    body_colors[idx] = select(INACTIVE_COLOR, UNCOLORED, active_body_flags[idx] != 0u);
    body_priorities[idx] = hash(idx, params.z);
    body_order[idx] = idx;
}
"#;

/// Step kernel: one iteration of adjacency-based Jones-Plassmann/Luby body coloring.
/// Uses compact body_contact_neighbors buffer instead of full Contact structs
/// to reduce memory bandwidth (~4 bytes vs ~128 bytes per neighbor lookup).
pub const COLORING_STEP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> body_colors:     array<u32>;
@group(0) @binding(1) var<storage, read>       body_priorities: array<u32>;
@group(0) @binding(2) var<storage, read>       body_contact_ranges: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read>       body_contact_neighbors: array<u32>;
@group(0) @binding(4) var<uniform>             params:          vec4<u32>; // x=num_bodies, z=current_color
@group(0) @binding(5) var<storage, read_write> unfinished:      atomic<u32>;

const WORKGROUP_SIZE: u32 = 64u;
const UNCOLORED: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let body_idx = gid.x;
    let num_bodies = params.x;
    let curr_color = params.z;
    if body_idx >= num_bodies { return; }

    if body_colors[body_idx] != UNCOLORED {
        return;
    }

    let my_priority = body_priorities[body_idx];
    var is_local_max = true;
    let range = body_contact_ranges[body_idx];
    let range_end = range.x + range.y;
    for (var slot = range.x; slot < range_end; slot = slot + 1u) {
        let neighbor = body_contact_neighbors[slot];

        let neighbor_color = body_colors[neighbor];
        if neighbor_color != UNCOLORED {
            continue;
        }

        let neighbor_priority = body_priorities[neighbor];
        if neighbor_priority > my_priority || (neighbor_priority == my_priority && neighbor > body_idx) {
            is_local_max = false;
            break;
        }
    }

    if is_local_max {
        body_colors[body_idx] = curr_color;
    } else {
        atomicAdd(&unfinished, 1u);
    }
}
"#;
