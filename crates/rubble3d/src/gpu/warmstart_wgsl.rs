/// WGSL source to clear the warmstart hash map (fill keys with sentinel 0xFFFFFFFF).
pub const WARMSTART_HASHMAP_CLEAR_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> hashmap_keys: array<u32>;
@group(0) @binding(1) var<storage, read>       capacity_buf: array<u32>;

const WORKGROUP_SIZE: u32 = 256u;
const EMPTY_KEY: u32 = 0xFFFFFFFFu;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let cap = capacity_buf[0];
    if idx >= cap {
        return;
    }
    hashmap_keys[idx] = EMPTY_KEY;
}
"#;

/// WGSL source to insert previous contacts into the hash map.
pub const WARMSTART_HASHMAP_INSERT_WGSL: &str = r#"
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

const FEATURE_SKIP_PREFIX: u32 = 0x32000000u;
const FEATURE_PREFIX_MASK: u32 = 0xFF000000u;
const EMPTY_KEY: u32 = 0xFFFFFFFFu;
const MAX_PROBES: u32 = 64u;

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
    var k = hash_u32(lo ^ (hi * 0x9e3779b9u) ^ (feature_id * 0x85ebca6bu));
    // Reserve 0xFFFFFFFF as empty sentinel
    if k == EMPTY_KEY {
        k = 0xFFFFFFFEu;
    }
    return k;
}

@group(0) @binding(0) var<storage, read>       prev_contacts:  array<Contact>;
@group(0) @binding(1) var<storage, read_write> hashmap_keys:   array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> hashmap_values: array<u32>;
@group(0) @binding(3) var<storage, read>       params_buf:     array<u32>; // [prev_count, capacity]

const WORKGROUP_SIZE: u32 = 64u;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let prev_count = params_buf[0];
    let capacity = params_buf[1];
    if idx >= prev_count {
        return;
    }

    let pc = prev_contacts[idx];

    // Skip contacts with special feature ID prefix
    if (pc.feature_id & FEATURE_PREFIX_MASK) == FEATURE_SKIP_PREFIX {
        return;
    }

    let key = warm_key(pc.body_a, pc.body_b, pc.feature_id);
    var slot = key % capacity;
    var probe = 0u;

    loop {
        if probe >= MAX_PROBES {
            break;
        }
        let prev = atomicCompareExchangeWeak(&hashmap_keys[slot], EMPTY_KEY, key);
        if prev.exchanged {
            // Won an empty slot
            hashmap_values[slot] = idx;
            return;
        }
        if prev.old_value != EMPTY_KEY {
            // Slot genuinely occupied — advance to next slot
            slot = (slot + 1u) % capacity;
            probe = probe + 1u;
        }
        // If old_value == EMPTY_KEY but !exchanged, it was a spurious CAS failure — retry same slot
    }
}
"#;

/// WGSL source for GPU warmstart impulse transfer between frames.
/// Uses hash map probe instead of binary search.
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
    prev_count:       u32,
    new_count:        u32,
    alpha:            f32,
    gamma:            f32,
    hashmap_capacity: u32,
    k_start:          f32,
    _pad1:            u32,
    _pad2:            u32,
};

const CONTACT_FLAG_STICKING: u32 = 1u;
const CONTACT_FLAG_WARMSTARTED: u32 = 2u;
const FEATURE_SKIP_PREFIX: u32 = 0x32000000u;
const FEATURE_PREFIX_MASK: u32 = 0xFF000000u;
const EMPTY_KEY: u32 = 0xFFFFFFFFu;
const MAX_PROBES: u32 = 64u;

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
    var k = hash_u32(lo ^ (hi * 0x9e3779b9u) ^ (feature_id * 0x85ebca6bu));
    if k == EMPTY_KEY {
        k = 0xFFFFFFFEu;
    }
    return k;
}

@group(0) @binding(0) var<storage, read>       prev_contacts:  array<Contact>;
@group(0) @binding(1) var<storage, read>       hashmap_keys:   array<u32>;
@group(0) @binding(2) var<storage, read>       hashmap_values: array<u32>;
@group(0) @binding(3) var<storage, read_write> new_contacts:   array<Contact>;
@group(0) @binding(4) var<uniform>             params:         WarmstartParams;

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
    let target_key = warm_key(nc.body_a, nc.body_b, key_fid);
    let capacity = params.hashmap_capacity;
    var slot = target_key % capacity;

    for (var probe = 0u; probe < MAX_PROBES; probe = probe + 1u) {
        let stored_key = hashmap_keys[slot];
        if stored_key == EMPTY_KEY {
            // Empty slot — key not in table
            return;
        }
        if stored_key == target_key {
            // Hash match — verify exact body pair + feature_id
            let prev_idx = hashmap_values[slot];
            let pc = prev_contacts[prev_idx];
            let prev_lo = min(pc.body_a, pc.body_b);
            let prev_hi = max(pc.body_a, pc.body_b);
            if prev_lo == key_lo && prev_hi == key_hi && pc.feature_id == key_fid {
                let scale = params.alpha * params.gamma;
                new_contacts[idx].lambda = pc.lambda * scale;
                let k_floor = vec4<f32>(params.k_start, params.k_start, params.k_start, 0.0);
                new_contacts[idx].penalty = max(pc.penalty * params.gamma, k_floor);
                new_contacts[idx].flags = pc.flags | CONTACT_FLAG_WARMSTARTED;
                if (pc.flags & CONTACT_FLAG_STICKING) != 0u {
                    new_contacts[idx].local_anchor_a = pc.local_anchor_a;
                    new_contacts[idx].local_anchor_b = pc.local_anchor_b;
                }
                return;
            }
        }
        slot = (slot + 1u) % capacity;
    }
}
"#;
