/// WGSL source for GPU warmstart impulse transfer between frames.
///
/// Matches new contacts against previous-frame contacts by body pair and feature ID,
/// then transfers cached solver state (impulses, penalties, flags) with decay.
///
/// Inspired by wgrapier's warmstart pattern but adapted for Rubble's simpler contact model
/// where matching uses (min(body_a,body_b), max(body_a,body_b), feature_id) as the key.
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

// Feature ID prefix that should skip warmstarting (e.g., stacked box specials)
const FEATURE_SKIP_PREFIX: u32 = 0x32000000u;
const FEATURE_PREFIX_MASK: u32 = 0xFF000000u;

@group(0) @binding(0) var<storage, read>       prev_contacts: array<Contact>;
@group(0) @binding(1) var<storage, read_write> new_contacts:  array<Contact>;
@group(0) @binding(2) var<uniform>             params:        WarmstartParams;

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

    // Build matching key: (min(body_a, body_b), max(body_a, body_b), feature_id)
    let key_lo = min(nc.body_a, nc.body_b);
    let key_hi = max(nc.body_a, nc.body_b);
    let key_fid = nc.feature_id;

    // Linear scan through previous contacts for a match.
    // For typical contact counts (<1000), this is fast enough on GPU.
    // Future optimization: sort by body pair key for binary search.
    for (var j = 0u; j < params.prev_count; j = j + 1u) {
        let pc = prev_contacts[j];
        let prev_lo = min(pc.body_a, pc.body_b);
        let prev_hi = max(pc.body_a, pc.body_b);

        if prev_lo == key_lo && prev_hi == key_hi && pc.feature_id == key_fid {
            // Match found — transfer cached solver state with decay
            let scale = params.alpha * params.gamma;
            new_contacts[idx].lambda = pc.lambda * scale;
            new_contacts[idx].penalty = pc.penalty * params.gamma;
            new_contacts[idx].flags = pc.flags | CONTACT_FLAG_WARMSTARTED;

            // Preserve local anchors if contact was sticking (friction lock)
            if (pc.flags & CONTACT_FLAG_STICKING) != 0u {
                new_contacts[idx].local_anchor_a = pc.local_anchor_a;
                new_contacts[idx].local_anchor_b = pc.local_anchor_b;
            }
            break;
        }
    }
}
"#;
