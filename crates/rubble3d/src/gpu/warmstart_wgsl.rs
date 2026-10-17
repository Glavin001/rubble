/// WGSL source for GPU warmstart impulse transfer between frames.
///
/// Matches new contacts against previous-frame contacts by body pair and feature
/// ID, then transfers cached solver state (impulses, penalties, flags) with
/// decay.
///
/// Historically this was an O(new x prev) linear scan (each new contact scanned
/// every previous contact). That is quadratic in contact count and collapses at
/// the 10k-body target (~50k contacts => billions of iterations/frame). W4a-3
/// replaces it with an open-addressing hash map keyed by
/// `(min(body_a,body_b), max(body_a,body_b), feature_id)`:
///
///   1. `WARMSTART_MAP_INSERT_WGSL` builds the map from `prev_contacts`.
///   2. `WARMSTART_MATCH_WGSL` looks up each new contact in O(1) expected.
///
/// Determinism: inserters only `atomicCompareExchangeWeak` a 32-bit hash word to
/// claim a bucket — they never read another inserter's non-atomic key writes, so
/// there is no intra-dispatch visibility hazard. The full key is written after
/// the claim and verified on lookup in the *following* dispatch (writes visible
/// across the pass boundary). For unique keys (structured feature IDs, e.g. the
/// box/pyramid scenes) each key claims one deterministic base bucket, so the
/// result is bit-identical to the old linear scan. Only contacts that genuinely
/// share `(lo,hi,feature_id)` (e.g. capsule cap points with a constant feature
/// id) resolve to a scheduling-dependent occupant — the same approximation the
/// linear scan already made (it took whichever matched first), and covered by
/// the oracle's tolerance rather than the bitwise determinism probe.
///
/// Shared prelude: the `Contact` layout, warmstart params, feature-skip
/// constants, and the composite-key hash. Concatenated into both kernels.
const WARMSTART_COMMON: &str = r#"
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
    cap_mask:       u32,
    _pad0:          u32,
    _pad1:          u32,
    _pad2:          u32,
};

const CONTACT_FLAG_STICKING: u32 = 1u;
const CONTACT_FLAG_WARMSTARTED: u32 = 2u;

// Feature ID prefix that should skip warmstarting (e.g., stacked box specials).
const FEATURE_SKIP_PREFIX: u32 = 0x32000000u;
const FEATURE_PREFIX_MASK: u32 = 0xFF000000u;

// Bounded linear-probe budget. With capacity >= 2*contacts (load factor <= 0.5)
// probe chains are short; the bound just guarantees termination on a pathological
// fill (a warm-start miss is harmless — the contact simply starts cold).
const WS_MAX_PROBE: u32 = 64u;

// 32-bit hash of the composite key, forced nonzero so 0 can mark an empty
// bucket. u32 arithmetic wraps mod 2^32 per the WGSL spec.
fn ws_full_hash(lo: u32, hi: u32, fid: u32) -> u32 {
    var h = lo * 0x9e3779b1u;
    h = (h ^ hi) * 0x85ebca77u;
    h = (h ^ fid) * 0xc2b2ae3du;
    h = h ^ (h >> 15u);
    if h == 0u {
        h = 1u;
    }
    return h;
}
"#;

/// Build the warm-start pair map from `prev_contacts` (one thread per previous
/// contact). Claims a bucket via CAS on `map_hash`, then writes the full key +
/// previous-contact index into `map_entry` (lo, hi, feature_id, prev_index).
pub fn warmstart_map_insert_wgsl() -> String {
    format!(
        "{WARMSTART_COMMON}{body}",
        body = r#"
@group(0) @binding(0) var<storage, read>       prev_contacts: array<Contact>;
@group(0) @binding(1) var<uniform>             params:        WarmstartParams;
@group(0) @binding(2) var<storage, read_write> map_hash:      array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> map_entry:     array<vec4<u32>>;
@group(0) @binding(4) var<storage, read>       prev_count_buf: array<u32>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    // params.prev_count is the dispatch upper bound (exact on native, an upper
    // bound on web where the count stays GPU-resident); the true previous count
    // comes from the GPU counter (W5 — no count readback on the hot path).
    if i >= min(prev_count_buf[0], params.prev_count) {
        return;
    }
    let c = prev_contacts[i];
    if (c.feature_id & FEATURE_PREFIX_MASK) == FEATURE_SKIP_PREFIX {
        return;
    }
    let lo = min(c.body_a, c.body_b);
    let hi = max(c.body_a, c.body_b);
    let fid = c.feature_id;
    let fh = ws_full_hash(lo, hi, fid);

    var b = fh & params.cap_mask;
    var probe = 0u;
    loop {
        if probe >= WS_MAX_PROBE {
            return;
        }
        let res = atomicCompareExchangeWeak(&map_hash[b], 0u, fh);
        if res.exchanged {
            // Won this empty bucket — publish the full key + value.
            map_entry[b] = vec4<u32>(lo, hi, fid, i);
            return;
        }
        if res.old_value != 0u {
            // Occupied (by this key's duplicate or a hash collision) — probe on.
            b = (b + 1u) & params.cap_mask;
            probe = probe + 1u;
        }
        // else: spurious CAS failure (old_value == 0) — retry the same bucket.
    }
}
"#
    )
}

/// Look up each new contact in the pair map and transfer cached solver state
/// (one thread per new contact). Non-atomic reads of `map_hash`/`map_entry`
/// (written in the preceding insert dispatch — visible across the pass boundary).
pub fn warmstart_match_wgsl() -> String {
    format!(
        "{WARMSTART_COMMON}{body}",
        body = r#"
@group(0) @binding(0) var<storage, read>       prev_contacts: array<Contact>;
@group(0) @binding(1) var<storage, read_write> new_contacts:  array<Contact>;
@group(0) @binding(2) var<uniform>             params:        WarmstartParams;
@group(0) @binding(3) var<storage, read>       count_buf:     array<u32>;
@group(0) @binding(4) var<storage, read>       map_hash:      array<u32>;
@group(0) @binding(5) var<storage, read>       map_entry:     array<vec4<u32>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    // params.new_count is the dispatch upper bound (live count on native,
    // capacity on web); the live count comes from the GPU counter.
    if idx >= min(count_buf[0], params.new_count) {
        return;
    }

    let nc = new_contacts[idx];
    if (nc.feature_id & FEATURE_PREFIX_MASK) == FEATURE_SKIP_PREFIX {
        return;
    }

    let lo = min(nc.body_a, nc.body_b);
    let hi = max(nc.body_a, nc.body_b);
    let fid = nc.feature_id;
    let fh = ws_full_hash(lo, hi, fid);

    var b = fh & params.cap_mask;
    var probe = 0u;
    loop {
        if probe >= WS_MAX_PROBE {
            return;
        }
        let slot = map_hash[b];
        if slot == 0u {
            // Empty bucket — the key was never inserted (no previous match).
            return;
        }
        if slot == fh {
            let e = map_entry[b];
            if e.x == lo && e.y == hi && e.z == fid {
                let pc = prev_contacts[e.w];
                // Transfer cached solver state with decay.
                // Penalty: paper warm-start is k <- max(k_start, gamma*k); the
                // fresh contact's penalty already carries k_start (and contact
                // prep re-applies the m/dt^2 floor + caps after this), so take
                // the componentwise max with the decayed carried stiffness.
                let scale = params.alpha * params.gamma;
                new_contacts[idx].lambda = pc.lambda * scale;
                new_contacts[idx].penalty = max(nc.penalty, pc.penalty * params.gamma);
                new_contacts[idx].flags = pc.flags | CONTACT_FLAG_WARMSTARTED;

                // Preserve local anchors if the contact was sticking (friction lock).
                if (pc.flags & CONTACT_FLAG_STICKING) != 0u {
                    new_contacts[idx].local_anchor_a = pc.local_anchor_a;
                    new_contacts[idx].local_anchor_b = pc.local_anchor_b;
                }
                return;
            }
        }
        b = (b + 1u) & params.cap_mask;
        probe = probe + 1u;
    }
}
"#
    )
}
