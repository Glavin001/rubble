/// WGSL source for O(N^2) broadphase pair detection (2D).
///
/// For each pair of bodies (i < j), tests 2D AABB overlap and writes the pair
/// to the output buffer using an atomic counter.
pub const BROADPHASE_PAIRS_2D_WGSL: &str = r#"
struct Aabb2D {
    min_pt: vec4<f32>,
    max_pt: vec4<f32>,
};

struct SimParams2D {
    gravity:           vec4<f32>,
    dt:                f32,
    num_bodies:        u32,
    solver_iterations: u32,
    _pad:              u32,
};

struct Pair {
    a: u32,
    b: u32,
};

@group(0) @binding(0) var<storage, read>       aabbs:      array<Aabb2D>;
@group(0) @binding(1) var<storage, read_write> pairs:      array<Pair>;
@group(0) @binding(2) var<storage, read_write> pair_count: atomic<u32>;
@group(0) @binding(3) var<uniform>             params:     SimParams2D;

fn aabb_overlap_2d(a: Aabb2D, b: Aabb2D) -> bool {
    return a.min_pt.x <= b.max_pt.x && a.max_pt.x >= b.min_pt.x &&
           a.min_pt.y <= b.max_pt.y && a.max_pt.y >= b.min_pt.y;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.num_bodies;
    let total_pairs = n * (n - 1u) / 2u;
    let idx = gid.x;
    if idx >= total_pairs {
        return;
    }

    // Triangular index decoding: find i,j such that idx maps to pair (i,j), i<j
    let nf = f32(n);
    let t = 2.0 * nf - 1.0;
    let disc = t * t - 8.0 * f32(idx);
    let i_f = floor((t - sqrt(max(disc, 0.0))) * 0.5);
    let i = u32(i_f);
    let row_start = i * n - i * (i + 1u) / 2u;
    let j = idx - row_start + i + 1u;

    if i >= n || j >= n || i >= j {
        return;
    }

    let a = aabbs[i];
    let b = aabbs[j];

    if aabb_overlap_2d(a, b) {
        let slot = atomicAdd(&pair_count, 1u);
        let max_pairs = n * 8u;
        if slot < max_pairs {
            pairs[slot].a = i;
            pairs[slot].b = j;
        }
    }
}
"#;
