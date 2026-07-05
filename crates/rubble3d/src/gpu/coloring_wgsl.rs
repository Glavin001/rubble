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

/// Step kernel: one iteration of Luby's body coloring — *mark phase*.
///
/// Each uncolored body checks if it has the highest priority among its
/// uncolored neighbors (connected via contacts) and, if so, marks itself a
/// winner. It does NOT write `body_colors`: writing colors in the same
/// dispatch that reads neighbors' colors is a data race (a neighbor's
/// just-written color may or may not be visible), which produced improper
/// colorings — two adjacent bodies sharing a color — and made whole-sim
/// results vary run to run. The separate apply kernel below commits winners.
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

@group(0) @binding(0) var<storage, read>       body_colors:     array<u32>;
@group(0) @binding(1) var<storage, read>       body_priorities: array<u32>;
@group(0) @binding(2) var<storage, read>       contacts:        array<Contact>;
@group(0) @binding(3) var<uniform>             params:          vec4<u32>; // x=num_bodies, y=num_contacts, z=current_color
@group(0) @binding(4) var<storage, read_write> winners:         array<u32>;
@group(0) @binding(5) var<storage, read>       body_contact_ranges:  array<vec2<u32>>;
@group(0) @binding(6) var<storage, read>       body_contact_indices: array<u32>;

const WORKGROUP_SIZE: u32 = 64u;
const UNCOLORED: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let body_idx = gid.x;
    let num_bodies = params.x;
    if body_idx >= num_bodies { return; }

    winners[body_idx] = 0u;
    if body_colors[body_idx] != UNCOLORED {
        return; // already colored
    }

    let my_priority = body_priorities[body_idx];
    var is_local_max = true;

    // Walk only this body's contacts via the CSR adjacency (built immediately
    // before coloring). The previous all-contacts scan was O(bodies*contacts)
    // per round — recoloring every frame made that the dominant solve cost.
    let range = body_contact_ranges[body_idx];
    let range_end = range.x + range.y;
    for (var slot = range.x; slot < range_end; slot = slot + 1u) {
        let c = contacts[body_contact_indices[slot]];
        var neighbor = c.body_a;
        if neighbor == body_idx {
            neighbor = c.body_b;
        }
        if neighbor == body_idx {
            continue;
        }

        // Only compete with uncolored neighbors (stable read: this dispatch
        // never writes body_colors).
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
        winners[body_idx] = 1u;
    }
}
"#;

/// Apply kernel: commit this round's winners to `body_colors`. Runs as a
/// separate dispatch after the mark phase so color reads and writes never
/// overlap (each invocation touches only its own slot here).
pub const COLORING_APPLY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> body_colors: array<u32>;
@group(0) @binding(1) var<storage, read>       winners:     array<u32>;
@group(0) @binding(2) var<uniform>             params:      vec4<u32>; // x=num_bodies, z=current_color

const WORKGROUP_SIZE: u32 = 64u;
const UNCOLORED: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let body_idx = gid.x;
    if body_idx >= params.x { return; }
    if winners[body_idx] == 1u && body_colors[body_idx] == UNCOLORED {
        body_colors[body_idx] = params.z;
    }
}
"#;

// NOTE: Body order building (sorting by color, computing group offsets) is currently
// done on CPU after downloading the body_colors buffer. This is a small O(n) operation
// on just u32-per-body data, much cheaper than the previous full contact download.
// A future optimization could move this to GPU using a radix sort on colors.
