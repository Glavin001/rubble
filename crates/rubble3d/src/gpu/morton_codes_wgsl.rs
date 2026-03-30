/// WGSL source for Morton code computation from AABB centroids.
///
/// For each body's AABB, computes a 30-bit Morton code from the centroid
/// position (10 bits per axis), normalized to scene bounds.
pub const MORTON_CODES_WGSL: &str = r#"
struct Aabb {
    min_pt: vec4<f32>,
    max_pt: vec4<f32>,
};

struct SimParams {
    gravity:           vec4<f32>,
    dt:                f32,
    num_bodies:        u32,
    solver_iterations: u32,
    _pad:              u32,
};

struct SceneBounds {
    scene_min: vec4<f32>,
    scene_max: vec4<f32>,
};

@group(0) @binding(0) var<storage, read>       aabbs:       array<Aabb>;
@group(0) @binding(1) var<storage, read_write> morton_codes: array<u32>;
@group(0) @binding(2) var<uniform>             params:      SimParams;
@group(0) @binding(3) var<uniform>             bounds:      SceneBounds;

// Expand a 10-bit integer into 30 bits (interleave with 2 zero-bits between each).
fn expand_bits(v_in: u32) -> u32 {
    var v = v_in;
    v = (v | (v << 16u)) & 0x030000FFu;
    v = (v | (v <<  8u)) & 0x0300F00Fu;
    v = (v | (v <<  4u)) & 0x030C30C3u;
    v = (v | (v <<  2u)) & 0x09249249u;
    return v;
}

fn morton_3d(x: u32, y: u32, z: u32) -> u32 {
    return expand_bits(x) | (expand_bits(y) << 1u) | (expand_bits(z) << 2u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_bodies {
        return;
    }

    let aabb = aabbs[idx];
    let centroid = (aabb.min_pt.xyz + aabb.max_pt.xyz) * 0.5;

    // Normalize to [0, 1] within scene bounds.
    let scene_extent = bounds.scene_max.xyz - bounds.scene_min.xyz;
    let safe_extent = max(scene_extent, vec3<f32>(1e-6, 1e-6, 1e-6));
    let normalized = clamp((centroid - bounds.scene_min.xyz) / safe_extent, vec3<f32>(0.0), vec3<f32>(1.0));

    // Map to 10-bit integer range [0, 1023].
    let ix = u32(normalized.x * 1023.0);
    let iy = u32(normalized.y * 1023.0);
    let iz = u32(normalized.z * 1023.0);

    morton_codes[idx] = morton_3d(ix, iy, iz);
}
"#;
