/// Build sortable warmstart hashes for the previous-contact buffer.
pub const WARMSTART_BUILD_KEYS_WGSL: &str = r#"
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

// Feature ID prefix that should skip warmstarting (e.g., stacked box specials)
const FEATURE_SKIP_PREFIX: u32 = 0x32000000u;
const FEATURE_PREFIX_MASK: u32 = 0xFF000000u;

fn hash_u32(x: u32) -> u32 {
    var h = x;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return h;
}

fn warm_key(body_a: u32, body_b: u32, feature_id: u32) -> u32 {
    let lo = min(body_a, body_b);
    let hi = max(body_a, body_b);
    return hash_u32(lo ^ (hi * 0x9e3779b9u) ^ (feature_id * 0x85ebca6bu));
}

@group(0) @binding(0) var<storage, read>       prev_contacts: array<Contact>;
@group(0) @binding(1) var<storage, read_write> prev_hashes:   array<u32>;
@group(0) @binding(2) var<storage, read_write> prev_indices:  array<u32>;
@group(0) @binding(3) var<storage, read>       prev_count_buf: array<u32>;

const WORKGROUP_SIZE: u32 = 64u;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let prev_count = prev_count_buf[0];
    if idx >= prev_count {
        return;
    }

    let pc = prev_contacts[idx];
    prev_hashes[idx] = warm_key(pc.body_a, pc.body_b, pc.feature_id);
    prev_indices[idx] = idx;
}
"#;

/// WGSL source for GPU warmstart impulse transfer between frames.
pub const WARMSTART_MATCH_WGSL: &str = r#"
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

struct WarmstartParams {
    prev_count:     u32,
    new_count:      u32,
    alpha:          f32,
    gamma:          f32,
};

const CONTACT_FLAG_STICKING: u32 = 1u;
const CONTACT_FLAG_WARMSTARTED: u32 = 2u;
const FEATURE_SKIP_PREFIX: u32 = 0x32000000u;
const FEATURE_PREFIX_MASK: u32 = 0xFF000000u;

fn hash_u32(x: u32) -> u32 {
    var h = x;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return h;
}

fn warm_key(body_a: u32, body_b: u32, feature_id: u32) -> u32 {
    let lo = min(body_a, body_b);
    let hi = max(body_a, body_b);
    return hash_u32(lo ^ (hi * 0x9e3779b9u) ^ (feature_id * 0x85ebca6bu));
}

@group(0) @binding(0) var<storage, read>       prev_contacts: array<Contact>;
@group(0) @binding(1) var<storage, read>       prev_hashes:   array<u32>;
@group(0) @binding(2) var<storage, read>       prev_indices:  array<u32>;
@group(0) @binding(3) var<storage, read_write> new_contacts:  array<Contact>;
@group(0) @binding(4) var<uniform>             params:        WarmstartParams;

const WORKGROUP_SIZE: u32 = 64u;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.new_count {
        return;
    }

    let nc = new_contacts[idx];

    // Skip contacts with special feature ID prefix
    if (nc.feature_id & FEATURE_PREFIX_MASK) == FEATURE_SKIP_PREFIX {
        return;
    }

    let key_lo = min(nc.body_a, nc.body_b);
    let key_hi = max(nc.body_a, nc.body_b);
    let key_fid = nc.feature_id;
    let target_hash = warm_key(nc.body_a, nc.body_b, key_fid);

    var left = 0u;
    var right = params.prev_count;
    var found = params.prev_count;
    loop {
        if left >= right {
            break;
        }
        let mid = left + ((right - left) >> 1u);
        let mid_hash = prev_hashes[mid];
        if mid_hash < target_hash {
            left = mid + 1u;
        } else {
            right = mid;
            if mid_hash == target_hash {
                found = mid;
            }
        }
    }

    if found == params.prev_count {
        return;
    }

    var scan = found;
    loop {
        if scan > 0u && prev_hashes[scan - 1u] == target_hash {
            scan = scan - 1u;
        } else {
            break;
        }
    }

    loop {
        if scan >= params.prev_count || prev_hashes[scan] != target_hash {
            break;
        }
        let pc = prev_contacts[prev_indices[scan]];
        let prev_lo = min(pc.body_a, pc.body_b);
        let prev_hi = max(pc.body_a, pc.body_b);
        if prev_lo == key_lo && prev_hi == key_hi && pc.feature_id == key_fid {
            let scale = params.alpha * params.gamma;
            new_contacts[idx].lambda = pc.lambda * scale;
            new_contacts[idx].penalty = pc.penalty * params.gamma;
            new_contacts[idx].flags = pc.flags | CONTACT_FLAG_WARMSTARTED;
            if (pc.flags & CONTACT_FLAG_STICKING) != 0u {
                new_contacts[idx].local_anchor_a = pc.local_anchor_a;
                new_contacts[idx].local_anchor_b = pc.local_anchor_b;
            }
            break;
        }
        scan = scan + 1u;
    }
}
"#;
