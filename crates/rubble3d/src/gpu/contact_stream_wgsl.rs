//! WGSL kernels for the GPU-resident contact stream (M2(d)):
//! deterministic sort-key extraction + gather, and CSR adjacency construction
//! (count / ranges / incidence emission), replacing the per-frame CPU contact
//! download + sort + adjacency build.

/// Extract one 32-bit sort-key word per contact, indexed through the current
/// permutation (`vals[i]` = original contact slot). Chained stable LSD passes
/// over the words (hash, feature, body_b, body_a — least significant first)
/// reproduce the CPU total order `(body_a, body_b, feature_id, pos_hash)`.
///
/// `params.x` = contact count, `params.y` = word selector:
///   0 = pos_hash (bitcast of `lambda.w`, written at narrowphase emit)
///   1 = feature_id
///   2 = body_b
///   3 = body_a
pub const SORT_KEY_EXTRACT_WGSL: &str = r#"
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

@group(0) @binding(0) var<storage, read>       contacts: array<Contact>;
@group(0) @binding(1) var<storage, read>       vals:     array<u32>;
@group(0) @binding(2) var<storage, read_write> keys:     array<u32>;
@group(0) @binding(3) var<uniform>             params:   vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.x {
        return;
    }
    let c = contacts[vals[i]];
    var key: u32;
    switch params.y {
        case 0u: { key = bitcast<u32>(c.lambda.w); }
        case 1u: { key = c.feature_id; }
        case 2u: { key = c.body_b; }
        default: { key = c.body_a; }
    }
    keys[i] = key;
}
"#;

/// Identity-fill for the permutation values buffer.
pub const SORT_IOTA_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> vals:   array<u32>;
@group(0) @binding(1) var<uniform>             params: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.x {
        return;
    }
    vals[i] = i;
}
"#;

/// Gather contacts through the final permutation into the scratch buffer
/// (`scratch[i] = contacts[vals[i]]`); a buffer copy moves them back.
pub const SORT_GATHER_WGSL: &str = r#"
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

@group(0) @binding(0) var<storage, read>       contacts: array<Contact>;
@group(0) @binding(1) var<storage, read>       vals:     array<u32>;
@group(0) @binding(2) var<storage, read_write> scratch:  array<Contact>;
@group(0) @binding(3) var<uniform>             params:   vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.x {
        return;
    }
    scratch[i] = contacts[vals[i]];
}
"#;

/// Clear per-body contact counts and active flags. `params.x` = num_bodies.
pub const ADJ_CLEAR_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> counts: array<u32>;
@group(0) @binding(1) var<storage, read_write> active_flags: array<u32>;
@group(0) @binding(2) var<uniform>             params: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.x {
        return;
    }
    counts[i] = 0u;
    active_flags[i] = 0u;
}
"#;

/// Count contacts per body and mark contact-touched bodies active.
/// `params.x` = contact count.
pub const ADJ_COUNT_WGSL: &str = r#"
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

@group(0) @binding(0) var<storage, read>       contacts: array<Contact>;
@group(0) @binding(1) var<storage, read_write> counts:   array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> active_flags:   array<u32>;
@group(0) @binding(3) var<uniform>             params:   vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.x {
        return;
    }
    let c = contacts[i];
    atomicAdd(&counts[c.body_a], 1u);
    atomicAdd(&counts[c.body_b], 1u);
    active_flags[c.body_a] = 1u;
    active_flags[c.body_b] = 1u;
}
"#;

/// Assemble `(offset, count)` ranges from the scanned offsets and raw counts
/// into the solver's `body_contact_ranges` layout. `params.x` = num_bodies.
pub const ADJ_RANGES_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read>       offsets: array<u32>;
@group(0) @binding(1) var<storage, read>       counts:  array<u32>;
@group(0) @binding(2) var<storage, read_write> ranges:  array<vec2<u32>>;
@group(0) @binding(3) var<uniform>             params:  vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.x {
        return;
    }
    ranges[i] = vec2<u32>(offsets[i], counts[i]);
}
"#;

/// Emit the (body, contact) incidence list: two entries per contact. A stable
/// radix sort of `keys=body` with `vals=contact_idx` then yields exactly the
/// CPU adjacency's `body_contact_indices` (contact order within each body).
/// `params.x` = contact count.
pub const ADJ_INCIDENCE_WGSL: &str = r#"
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

@group(0) @binding(0) var<storage, read>       contacts: array<Contact>;
@group(0) @binding(1) var<storage, read_write> keys:     array<u32>;
@group(0) @binding(2) var<storage, read_write> vals:     array<u32>;
@group(0) @binding(3) var<uniform>             params:   vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.x {
        return;
    }
    let c = contacts[i];
    keys[2u * i] = c.body_a;
    vals[2u * i] = i;
    keys[2u * i + 1u] = c.body_b;
    vals[2u * i + 1u] = i;
}
"#;
