//! WGSL source for GPU body graph coloring using Luby's algorithm.
//!
//! Colors bodies so no two bodies sharing a contact have the same color,
//! enabling parallel AVBD primal dispatch per color group.
//!
//! Pipeline: reset → iterate (step + check convergence) → build body order
//!
//! Adapted from wgrapier's coloring approach but simplified for Rubble's
//! body-centric coloring (wgrapier colors constraints instead).

/// Reset kernel: initialize all bodies as uncolored and assign random priorities.
pub const COLORING_RESET_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> body_colors:     array<u32>;
@group(0) @binding(1) var<storage, read_write> body_priorities: array<u32>;
@group(0) @binding(2) var<uniform>             params:          vec4<u32>; // x=num_bodies, y=num_contacts, z=seed

const WORKGROUP_SIZE: u32 = 64u;
const UNCOLORED: u32 = 0xFFFFFFFFu;

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
    body_colors[idx] = UNCOLORED;
    body_priorities[idx] = hash(idx, params.z);
}
"#;

/// Step kernel: one iteration of Luby's body coloring.
/// Each uncolored body checks if it has the highest priority among its
/// uncolored neighbors (connected via contacts). If so, assign current color.
pub const COLORING_STEP_WGSL: &str = r#"
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

@group(0) @binding(0) var<storage, read_write> body_colors:     array<u32>;
@group(0) @binding(1) var<storage, read>       body_priorities: array<u32>;
@group(0) @binding(2) var<storage, read>       contacts:        array<Contact>;
@group(0) @binding(3) var<uniform>             params:          vec4<u32>; // x=num_bodies, y=num_contacts, z=current_color

const WORKGROUP_SIZE: u32 = 64u;
const UNCOLORED: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let body_idx = gid.x;
    let num_bodies = params.x;
    let num_contacts = params.y;
    let curr_color = params.z;
    if body_idx >= num_bodies { return; }

    if body_colors[body_idx] != UNCOLORED {
        return; // already colored
    }

    let my_priority = body_priorities[body_idx];
    var is_local_max = true;

    // Check all contacts to find neighbors.
    // This is O(contacts) per body but runs in parallel across all bodies.
    for (var ci = 0u; ci < num_contacts; ci = ci + 1u) {
        var neighbor = UNCOLORED;
        if contacts[ci].body_a == body_idx {
            neighbor = contacts[ci].body_b;
        } else if contacts[ci].body_b == body_idx {
            neighbor = contacts[ci].body_a;
        }

        if neighbor == UNCOLORED {
            continue;
        }

        // Only compete with uncolored neighbors
        if body_colors[neighbor] != UNCOLORED {
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
    }
}
"#;

// NOTE: Body order building (sorting by color, computing group offsets) is currently
// done on CPU after downloading the body_colors buffer. This is a small O(n) operation
// on just u32-per-body data, much cheaper than the previous full contact download.
// A future optimization could move this to GPU using a radix sort on colors.
