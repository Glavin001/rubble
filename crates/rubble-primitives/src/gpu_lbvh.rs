//! GPU-native Linear BVH broadphase using compute shaders.
//!
//! Pipeline: compute scene bounds → Morton codes → radix sort → build Karras tree
//! → refit AABBs → find pairs.
//! The scene-bounds reduction, Morton code computation, leaf gathering, Karras build,
//! refit, and pair finding run as compute shaders. The [`GpuLbvh::build_and_query_raw`]
//! family can still download sorted keys and build a CPU BVH for debugging or callers
//! that need overlap pairs on the host.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use rubble_gpu::{
    round_up_workgroups, BroadphaseBreakdownMs, ComputeKernel, GpuAtomicCounter, GpuBuffer,
    GpuContext,
};
use rubble_math::Aabb3D;
#[cfg(not(target_arch = "wasm32"))]
use std::collections::VecDeque;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use crate::GpuRadixSort;

const WG: u32 = 256;

#[repr(u32)]
#[derive(Clone, Copy)]
enum BroadphaseTimingMarker {
    BoundsStart = 0,
    BoundsEnd = 1,
    SortStart = 2,
    SortEnd = 3,
    BuildStart = 4,
    BuildEnd = 5,
    TraverseStart = 6,
    TraverseEnd = 7,
}

const BROADPHASE_QUERY_COUNT: u32 = 8;

#[cfg(not(target_arch = "wasm32"))]
fn precise_gpu_timing_enabled() -> bool {
    matches!(
        std::env::var("RUBBLE_PRECISE_GPU_TIMING").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "on" | "ON" | "yes" | "YES")
    )
}

#[cfg(not(target_arch = "wasm32"))]
struct PendingBroadphaseTimingReadback {
    staging: wgpu::Buffer,
    ready: Arc<AtomicBool>,
    success: Arc<AtomicBool>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy)]
struct PreciseBroadphaseTimings {
    bounds_ms: f32,
    sort_ms: f32,
    build_ms: f32,
    traverse_ms: f32,
}

#[cfg(not(target_arch = "wasm32"))]
struct GpuBroadphaseProfiler {
    query_set: wgpu::QuerySet,
    timestamp_period_ns: f32,
    pending: VecDeque<PendingBroadphaseTimingReadback>,
    latest: Option<PreciseBroadphaseTimings>,
}

#[cfg(not(target_arch = "wasm32"))]
impl GpuBroadphaseProfiler {
    fn new(ctx: &GpuContext) -> Option<Self> {
        if !precise_gpu_timing_enabled() {
            return None;
        }
        let required =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if !ctx.device.features().contains(required) {
            return None;
        }
        let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("lbvh broadphase timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: BROADPHASE_QUERY_COUNT,
        });
        Some(Self {
            query_set,
            timestamp_period_ns: ctx.queue.get_timestamp_period(),
            pending: VecDeque::new(),
            latest: None,
        })
    }

    fn mark(&self, ctx: &GpuContext, marker: BroadphaseTimingMarker) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lbvh timestamp marker"),
            });
        encoder.write_timestamp(&self.query_set, marker as u32);
        ctx.queue.submit(Some(encoder.finish()));
    }

    fn finish_frame(&mut self, ctx: &GpuContext) {
        let byte_size = BROADPHASE_QUERY_COUNT as u64 * std::mem::size_of::<u64>() as u64;
        let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lbvh timestamp resolve"),
            size: byte_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lbvh timestamp staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lbvh timestamp resolve encoder"),
            });
        encoder.resolve_query_set(&self.query_set, 0..BROADPHASE_QUERY_COUNT, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &staging, 0, byte_size);
        ctx.queue.submit(Some(encoder.finish()));

        let ready = Arc::new(AtomicBool::new(false));
        let success = Arc::new(AtomicBool::new(false));
        let ready_cb = ready.clone();
        let success_cb = success.clone();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                success_cb.store(result.is_ok(), Ordering::SeqCst);
                ready_cb.store(true, Ordering::SeqCst);
            });
        self.pending.push_back(PendingBroadphaseTimingReadback {
            staging,
            ready,
            success,
        });
        while self.pending.len() > 3 {
            let Some(front) = self.pending.front() else {
                break;
            };
            if !front.ready.load(Ordering::SeqCst) {
                break;
            }
            if let Some(old) = self.pending.pop_front() {
                old.staging.unmap();
            }
        }
    }

    fn collect_ready(&mut self, ctx: &GpuContext) {
        let _ = ctx.device.poll(wgpu::PollType::Poll);
        loop {
            let Some(front) = self.pending.front() else {
                break;
            };
            if !front.ready.load(Ordering::SeqCst) {
                break;
            }
            let pending = self.pending.pop_front().unwrap();
            if pending.success.load(Ordering::SeqCst) {
                let mapped = pending.staging.slice(..).get_mapped_range();
                let ticks: &[u64] = bytemuck::cast_slice(&mapped);
                if ticks.len() >= BROADPHASE_QUERY_COUNT as usize {
                    self.latest = Some(PreciseBroadphaseTimings {
                        bounds_ms: self.delta_ms(
                            ticks,
                            BroadphaseTimingMarker::BoundsStart,
                            BroadphaseTimingMarker::BoundsEnd,
                        ),
                        sort_ms: self.delta_ms(
                            ticks,
                            BroadphaseTimingMarker::SortStart,
                            BroadphaseTimingMarker::SortEnd,
                        ),
                        build_ms: self.delta_ms(
                            ticks,
                            BroadphaseTimingMarker::BuildStart,
                            BroadphaseTimingMarker::BuildEnd,
                        ),
                        traverse_ms: self.delta_ms(
                            ticks,
                            BroadphaseTimingMarker::TraverseStart,
                            BroadphaseTimingMarker::TraverseEnd,
                        ),
                    });
                }
                drop(mapped);
            }
            pending.staging.unmap();
        }
    }

    fn apply_latest(&self, breakdown: &mut BroadphaseBreakdownMs) {
        if let Some(latest) = self.latest {
            breakdown.bounds_ms = latest.bounds_ms;
            breakdown.sort_ms = latest.sort_ms;
            breakdown.build_ms = latest.build_ms;
            breakdown.traverse_ms = latest.traverse_ms;
            breakdown.precise = true;
        }
    }

    fn delta_ms(
        &self,
        ticks: &[u64],
        start: BroadphaseTimingMarker,
        end: BroadphaseTimingMarker,
    ) -> f32 {
        let start_tick = ticks[start as usize];
        let end_tick = ticks[end as usize];
        if end_tick <= start_tick {
            return 0.0;
        }
        ((end_tick - start_tick) as f32 * self.timestamp_period_ns) / 1_000_000.0
    }
}

// ---------------------------------------------------------------------------
// GPU types
// ---------------------------------------------------------------------------

/// Output pair from broadphase.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BroadPair {
    pub a: u32,
    pub b: u32,
}

/// BVH node stored in a GPU buffer. Used by tree build, refit, and pair-finding kernels.
///
/// Layout: [0..n-1] internal nodes, [n-1..2n-1] leaf nodes.
/// For internal nodes: `left`/`right` are child indices into this array.
/// For leaf nodes: `left` stores the sorted body index.
/// `parent` points to the parent internal node. `refit_count` is used by
/// the bottom-up refit kernel (atomic counter, 0→1→2).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BvhNodeGpu {
    pub aabb_min: [f32; 4],
    pub aabb_max: [f32; 4],
    pub left: i32,
    pub right: i32,
    pub parent: u32,
    pub refit_count: u32,
}

// ---------------------------------------------------------------------------
// Scene-bounds reduction shader
// ---------------------------------------------------------------------------

const SCENE_BOUNDS_REDUCE_WGSL: &str = r#"
struct Aabb {
    min_pt: vec4<f32>,
    max_pt: vec4<f32>,
};

var<workgroup> shared_min: array<vec4<f32>, 256>;
var<workgroup> shared_max: array<vec4<f32>, 256>;

@group(0) @binding(0) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(1) var<storage, read_write> out_bounds: array<Aabb>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let idx = gid.x;
    let local_idx = lid.x;
    let count = params.x;

    if idx < count {
        let aabb = aabbs[idx];
        shared_min[local_idx] = aabb.min_pt;
        shared_max[local_idx] = aabb.max_pt;
    } else {
        shared_min[local_idx] = vec4<f32>(1e30, 1e30, 1e30, 0.0);
        shared_max[local_idx] = vec4<f32>(-1e30, -1e30, -1e30, 0.0);
    }
    workgroupBarrier();

    var stride = 128u;
    loop {
        if local_idx < stride {
            shared_min[local_idx] = min(shared_min[local_idx], shared_min[local_idx + stride]);
            shared_max[local_idx] = max(shared_max[local_idx], shared_max[local_idx + stride]);
        }
        workgroupBarrier();
        if stride == 1u {
            break;
        }
        stride = stride >> 1u;
    }

    if local_idx == 0u {
        out_bounds[wid.x].min_pt = shared_min[0];
        out_bounds[wid.x].max_pt = shared_max[0];
    }
}
"#;

// ---------------------------------------------------------------------------
// Morton code compute shader (3D, 30-bit)
// ---------------------------------------------------------------------------

const MORTON_3D_WGSL: &str = r#"
struct Aabb {
    min_pt: vec4<f32>,
    max_pt: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(1) var<storage, read> scene_bounds: array<Aabb>;
@group(0) @binding(2) var<storage, read_write> morton_keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> body_indices: array<u32>;
@group(0) @binding(4) var<uniform> params: vec4<u32>;

fn expand10(v_in: u32) -> u32 {
    var v = v_in & 0x3FFu;
    v = (v | (v << 16u)) & 0x030000FFu;
    v = (v | (v << 8u))  & 0x0300F00Fu;
    v = (v | (v << 4u))  & 0x030C30C3u;
    v = (v | (v << 2u))  & 0x09249249u;
    return v;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let num_bodies = params.x;
    if idx >= num_bodies { return; }
    let aabb = aabbs[idx];
    let scene = scene_bounds[0];
    let extent = scene.max_pt.xyz - scene.min_pt.xyz;
    let inv_extent_x = select(0.0, 1.0 / extent.x, extent.x > 1e-10);
    let inv_extent_y = select(0.0, 1.0 / extent.y, extent.y > 1e-10);
    let inv_extent_z = select(0.0, 1.0 / extent.z, extent.z > 1e-10);
    let cx = (aabb.min_pt.x + aabb.max_pt.x) * 0.5;
    let cy = (aabb.min_pt.y + aabb.max_pt.y) * 0.5;
    let cz = (aabb.min_pt.z + aabb.max_pt.z) * 0.5;
    let nx = clamp((cx - scene.min_pt.x) * inv_extent_x, 0.0, 1.0);
    let ny = clamp((cy - scene.min_pt.y) * inv_extent_y, 0.0, 1.0);
    let nz = clamp((cz - scene.min_pt.z) * inv_extent_z, 0.0, 1.0);
    let ix = u32(nx * 1023.0);
    let iy = u32(ny * 1023.0);
    let iz = u32(nz * 1023.0);
    morton_keys[idx] = (expand10(ix) << 2u) | (expand10(iy) << 1u) | expand10(iz);
    body_indices[idx] = idx;
}
"#;

// ---------------------------------------------------------------------------
// Sorted leaf AABB gather shader
// ---------------------------------------------------------------------------

const GATHER_SORTED_LEAF_AABBS_WGSL: &str = r#"
struct Aabb {
    min_pt: vec4<f32>,
    max_pt: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> aabbs: array<Aabb>;
@group(0) @binding(1) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> sorted_leaf_aabbs: array<Aabb>;
@group(0) @binding(3) var<uniform> params: vec4<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let num_bodies = params.x;
    if idx >= num_bodies { return; }
    sorted_leaf_aabbs[idx] = aabbs[sorted_indices[idx]];
}
"#;

// ---------------------------------------------------------------------------
// Pair-finding compute shader
// ---------------------------------------------------------------------------

/// Find pairs shader variant for the CPU-built tree (uses internal_nodes + negative leaf encoding).
const FIND_PAIRS_WGSL: &str = r#"
struct Aabb {
    min_pt: vec4<f32>,
    max_pt: vec4<f32>,
};

struct BvhNode {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    left: i32,
    right: i32,
    parent: u32,
    refit_count: u32,
};

struct Pair {
    a: u32,
    b: u32,
};

@group(0) @binding(0) var<storage, read> leaf_aabbs: array<Aabb>;
@group(0) @binding(1) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(2) var<storage, read> nodes: array<BvhNode>;
@group(0) @binding(3) var<storage, read_write> pairs: array<Pair>;
@group(0) @binding(4) var<storage, read_write> pair_count: atomic<u32>;
@group(0) @binding(5) var<uniform> params: vec4<u32>; // x = num_leaves, y = max_pairs

fn aabb_overlap(a_min: vec3<f32>, a_max: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>) -> bool {
    return a_min.x <= b_max.x && a_max.x >= b_min.x
        && a_min.y <= b_max.y && a_max.y >= b_min.y
        && a_min.z <= b_max.z && a_max.z >= b_min.z;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let leaf_idx = gid.x;
    let n = params.x;
    let max_pairs = params.y;
    if leaf_idx >= n { return; }

    let body_i = sorted_indices[leaf_idx];
    let ai_min = leaf_aabbs[leaf_idx].min_pt.xyz;
    let ai_max = leaf_aabbs[leaf_idx].max_pt.xyz;

    var stack: array<i32, 64>;
    var sp = 0u;
    stack[0] = 0;
    sp = 1u;

    while sp > 0u {
        sp = sp - 1u;
        let node_idx = stack[sp];
        if node_idx < 0 {
            let j = u32(-node_idx - 1);
            if j != leaf_idx {
                let body_j = sorted_indices[j];
                if body_i < body_j {
                    if aabb_overlap(ai_min, ai_max, leaf_aabbs[j].min_pt.xyz, leaf_aabbs[j].max_pt.xyz) {
                        let slot = atomicAdd(&pair_count, 1u);
                        if slot < max_pairs {
                            pairs[slot].a = body_i;
                            pairs[slot].b = body_j;
                        }
                    }
                }
            }
        } else {
            let nd = nodes[u32(node_idx)];
            if aabb_overlap(ai_min, ai_max, nd.aabb_min.xyz, nd.aabb_max.xyz) {
                if sp + 2u <= 64u {
                    stack[sp] = nd.left; sp = sp + 1u;
                    stack[sp] = nd.right; sp = sp + 1u;
                }
            }
        }
    }
}
"#;

/// Find pairs shader for the GPU-built tree (uses tree_nodes with absolute indices).
/// Tree layout: [0..n-1] internal nodes, [n-1..2n-1] leaf nodes.
/// Node.left for a leaf stores the sorted body index.
const FIND_PAIRS_GPU_TREE_WGSL: &str = r#"
struct BvhNode {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    left: i32,
    right: i32,
    parent: u32,
    refit_count: u32,
};

struct Pair {
    a: u32,
    b: u32,
};

@group(0) @binding(0) var<storage, read> tree: array<BvhNode>;
@group(0) @binding(1) var<storage, read_write> pairs: array<Pair>;
@group(0) @binding(2) var<storage, read_write> pair_count: atomic<u32>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = num_bodies, y = max_pairs

fn aabb_overlap(a_min: vec3<f32>, a_max: vec3<f32>, b_min: vec3<f32>, b_max: vec3<f32>) -> bool {
    return a_min.x <= b_max.x && a_max.x >= b_min.x
        && a_min.y <= b_max.y && a_max.y >= b_min.y
        && a_min.z <= b_max.z && a_max.z >= b_min.z;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let leaf_local = gid.x;
    let n = params.x;
    let max_pairs = params.y;
    let first_leaf = n - 1u;
    if leaf_local >= n { return; }

    let leaf_id = first_leaf + leaf_local;
    let body_i = u32(tree[leaf_id].left); // leaf stores sorted body index
    let ai_min = tree[leaf_id].aabb_min.xyz;
    let ai_max = tree[leaf_id].aabb_max.xyz;

    var stack: array<u32, 64>;
    var sp = 0u;
    stack[0] = 0u; // start at root
    sp = 1u;

    while sp > 0u {
        sp -= 1u;
        let node_id = stack[sp];

        if node_id >= first_leaf {
            // Leaf node
            if node_id != leaf_id {
                let body_j = u32(tree[node_id].left);
                if body_i < body_j {
                    if aabb_overlap(ai_min, ai_max, tree[node_id].aabb_min.xyz, tree[node_id].aabb_max.xyz) {
                        let slot = atomicAdd(&pair_count, 1u);
                        if slot < max_pairs {
                            pairs[slot].a = body_i;
                            pairs[slot].b = body_j;
                        }
                    }
                }
            }
        } else {
            // Internal node
            if aabb_overlap(ai_min, ai_max, tree[node_id].aabb_min.xyz, tree[node_id].aabb_max.xyz) {
                if sp + 2u <= 64u {
                    stack[sp] = u32(tree[node_id].left); sp += 1u;
                    stack[sp] = u32(tree[node_id].right); sp += 1u;
                }
            }
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// GPU Karras tree build shader (adapted from wgparry lbvh.wgsl)
// ---------------------------------------------------------------------------

const KARRAS_BUILD_WGSL: &str = r#"
struct BvhNode {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    left: i32,
    right: i32,
    parent: u32,
    refit_count: u32,
};

@group(0) @binding(0) var<storage, read> morton_keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> tree: array<BvhNode>;
@group(0) @binding(2) var<uniform> params: vec4<u32>; // x = num_bodies

fn prefix_len(curr_key: u32, curr_index: i32, other_index: i32, n: i32) -> i32 {
    if other_index < 0 || other_index > n - 1 {
        return -1;
    }
    let other_key = i32(morton_keys[u32(other_index)]);
    let morton_prefix = countLeadingZeros(i32(curr_key) ^ other_key);
    let fallback = 32 + countLeadingZeros(curr_index ^ other_index);
    return select(fallback, morton_prefix, i32(curr_key) != other_key);
}

fn div_ceil_i(a: i32, b: i32) -> i32 {
    return (a + b - 1) / b;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let num_bodies = params.x;
    let num_internal = num_bodies - 1u;
    let first_leaf = num_internal;
    let i = gid.x;
    if i >= num_internal { return; }

    let ii = i32(i);
    let n = i32(num_bodies);
    let curr_key = morton_keys[i];

    // Determine direction of the range
    let d = sign(prefix_len(curr_key, ii, ii + 1, n) - prefix_len(curr_key, ii, ii - 1, n));

    // Compute upper bound for range length
    let delta_min = prefix_len(curr_key, ii, ii - d, n);
    var lmax = 2;
    while prefix_len(curr_key, ii, ii + lmax * d, n) > delta_min {
        lmax *= 2;
    }

    // Binary search for the other end
    var l = 0;
    var t = lmax / 2;
    while t >= 1 {
        if prefix_len(curr_key, ii, ii + (l + t) * d, n) > delta_min {
            l += t;
        }
        t /= 2;
    }
    let j = ii + l * d;

    // Find split position
    let delta_node = prefix_len(curr_key, ii, j, n);
    var s = 0;
    t = div_ceil_i(l, 2);
    loop {
        if prefix_len(curr_key, ii, ii + (s + t) * d, n) > delta_node {
            s += t;
        }
        if t <= 1 { break; }
        t = div_ceil_i(t, 2);
    }
    let gamma = ii + s * d + min(d, 0);

    // Output child and parent pointers
    let left = select(gamma, i32(first_leaf) + gamma, min(ii, j) == gamma);
    let right = select(gamma + 1, i32(first_leaf) + gamma + 1, max(ii, j) == gamma + 1);
    tree[i].left = left;
    tree[i].right = right;
    tree[i].refit_count = 0u;
    tree[u32(left)].parent = i;
    tree[u32(right)].parent = i;
}
"#;

// ---------------------------------------------------------------------------
// GPU bottom-up refit shader (adapted from wgparry lbvh.wgsl)
// ---------------------------------------------------------------------------

const REFIT_WGSL: &str = r#"
struct Aabb {
    min_pt: vec4<f32>,
    max_pt: vec4<f32>,
};

// Note: refit_count uses atomic<u32> because this shader does atomicAdd
// for bottom-up synchronization. Other shaders that share the same GPU
// buffer declare refit_count as plain u32 (they don't use atomics).
struct BvhNode {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    left: i32,
    right: i32,
    parent: u32,
    refit_count: atomic<u32>,
};

@group(0) @binding(0) var<storage, read> leaf_aabbs: array<Aabb>;
@group(0) @binding(1) var<storage, read> sorted_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> tree: array<BvhNode>;
@group(0) @binding(3) var<uniform> params: vec4<u32>; // x = num_bodies

// Single-workgroup refit for web compatibility (uniform control flow).
// Uses workgroupBarrier to synchronize between tree levels.
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let num_bodies = params.x;
    let num_internal = num_bodies - 1u;
    let first_leaf = num_internal;
    let num_threads = 256u;
    let num_iterations = (num_bodies + num_threads - 1u) / num_threads;

    for (var iter = 0u; iter < num_iterations; iter++) {
        let i = lid.x + iter * num_threads;
        var thread_is_active = i < num_bodies;

        // Set leaf AABB
        var curr_id = 0u;
        if thread_is_active {
            let leaf_id = first_leaf + i;
            tree[leaf_id].aabb_min = leaf_aabbs[i].min_pt;
            tree[leaf_id].aabb_max = leaf_aabbs[i].max_pt;
            tree[leaf_id].left = i32(sorted_indices[i]); // store body index
            curr_id = tree[leaf_id].parent;
        }

        // Propagate AABBs bottom-up using atomic synchronization
        for (var level = 0u; level < 32u; level++) {
            if thread_is_active {
                let refit_count = atomicAdd(&tree[curr_id].refit_count, 1u);
                if refit_count == 0u {
                    // Sibling hasn't arrived yet; stop here
                    thread_is_active = false;
                } else {
                    // Both children ready; merge AABBs
                    let left_id = u32(tree[curr_id].left);
                    let right_id = u32(tree[curr_id].right);
                    tree[curr_id].aabb_min = min(tree[left_id].aabb_min, tree[right_id].aabb_min);
                    tree[curr_id].aabb_max = max(tree[left_id].aabb_max, tree[right_id].aabb_max);

                    if curr_id == 0u {
                        thread_is_active = false;
                    } else {
                        curr_id = tree[curr_id].parent;
                    }
                }
            }
            workgroupBarrier();
        }
    }
}
"#;

// ---------------------------------------------------------------------------
// Small count params uniform layout
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CountParams {
    num_bodies: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ---------------------------------------------------------------------------
// CPU-side Karras tree build (reused from lbvh.rs pattern)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Child {
    Internal(usize),
    Leaf(usize),
}

fn delta(codes: &[u32], i: usize, j: usize) -> i32 {
    if j >= codes.len() {
        return -1;
    }
    let xor = codes[i] ^ codes[j];
    if xor == 0 {
        32 + ((i as u32 ^ j as u32).leading_zeros() as i32)
    } else {
        xor.leading_zeros() as i32
    }
}

fn karras_node(codes: &[u32], i: usize) -> (usize, usize, usize) {
    let n = codes.len();
    let d_left = if i == 0 { -1 } else { delta(codes, i, i - 1) };
    let d_right = if i + 1 >= n {
        -1
    } else {
        delta(codes, i, i + 1)
    };
    let d: i32 = if d_right > d_left { 1 } else { -1 };
    let delta_min = if d > 0 { d_left } else { d_right };

    let mut l_max: usize = 2;
    loop {
        let j = i as i64 + l_max as i64 * d as i64;
        if j < 0 || j >= n as i64 || delta(codes, i, j as usize) <= delta_min {
            break;
        }
        l_max *= 2;
    }

    let mut l: usize = 0;
    let mut t = l_max >> 1;
    while t >= 1 {
        let j = i as i64 + (l + t) as i64 * d as i64;
        if j >= 0 && j < n as i64 && delta(codes, i, j as usize) > delta_min {
            l += t;
        }
        t >>= 1;
    }

    let j_other = (i as i64 + l as i64 * d as i64) as usize;
    let range_left = i.min(j_other);
    let range_right = i.max(j_other);

    let delta_node = delta(codes, range_left, range_right);
    let mut s: usize = 0;
    let mut t = ((range_right - range_left + 1) as u64).next_power_of_two() as usize / 2;
    if t == 0 {
        t = 1;
    }
    loop {
        let candidate = range_left + s + t;
        if candidate < range_right && delta(codes, range_left, candidate) > delta_node {
            s += t;
        }
        if t == 1 {
            break;
        }
        t = t.div_ceil(2);
    }

    (range_left, range_right, range_left + s)
}

fn aabb_union(a: &Aabb3D, b: &Aabb3D) -> Aabb3D {
    Aabb3D::new(
        a.min_point().min(b.min_point()),
        a.max_point().max(b.max_point()),
    )
}

fn validate_internal_tree(left: &[Child], right: &[Child]) -> bool {
    if left.is_empty() {
        return true;
    }

    let n = left.len();
    let mut state = vec![0u8; n];
    let mut seen = 0usize;
    let mut stack = vec![(0usize, false)];

    while let Some((idx, expanded)) = stack.pop() {
        if idx >= n {
            return false;
        }

        if expanded {
            if state[idx] != 1 {
                return false;
            }
            state[idx] = 2;
            continue;
        }

        match state[idx] {
            0 => {
                state[idx] = 1;
                seen += 1;
                stack.push((idx, true));
                if let Child::Internal(ci) = right[idx] {
                    stack.push((ci, false));
                }
                if let Child::Internal(ci) = left[idx] {
                    stack.push((ci, false));
                }
            }
            1 => return false,
            2 => return false,
            _ => unreachable!(),
        }
    }

    seen == n
}

fn refit_iterative(left: &[Child], right: &[Child], leaves: &[Aabb3D]) -> Option<Vec<Aabb3D>> {
    if left.is_empty() {
        return Some(Vec::new());
    }

    let n = left.len();
    let mut internal = vec![Aabb3D::new(Vec3::ZERO, Vec3::ZERO); n];
    let mut state = vec![0u8; n];
    let mut stack = vec![(0usize, false)];

    while let Some((idx, expanded)) = stack.pop() {
        if idx >= n {
            return None;
        }

        if expanded {
            let la = match left[idx] {
                Child::Leaf(i) => *leaves.get(i)?,
                Child::Internal(i) => {
                    if state.get(i).copied()? != 2 {
                        return None;
                    }
                    internal[i]
                }
            };
            let ra = match right[idx] {
                Child::Leaf(i) => *leaves.get(i)?,
                Child::Internal(i) => {
                    if state.get(i).copied()? != 2 {
                        return None;
                    }
                    internal[i]
                }
            };
            internal[idx] = aabb_union(&la, &ra);
            state[idx] = 2;
            continue;
        }

        match state[idx] {
            0 => {
                state[idx] = 1;
                stack.push((idx, true));
                if let Child::Internal(ci) = right[idx] {
                    stack.push((ci, false));
                }
                if let Child::Internal(ci) = left[idx] {
                    stack.push((ci, false));
                }
            }
            1 => return None,
            2 => {}
            _ => unreachable!(),
        }
    }

    Some(internal)
}

fn children_to_gpu_nodes(
    left: &[Child],
    right: &[Child],
    internal_aabbs: &[Aabb3D],
) -> Vec<BvhNodeGpu> {
    (0..left.len())
        .map(|i| {
            let left_val = match left[i] {
                Child::Leaf(l) => -(l as i32 + 1),
                Child::Internal(l) => l as i32,
            };
            let right_val = match right[i] {
                Child::Leaf(r) => -(r as i32 + 1),
                Child::Internal(r) => r as i32,
            };
            let aabb = internal_aabbs[i];
            BvhNodeGpu {
                aabb_min: [
                    aabb.min_point().x,
                    aabb.min_point().y,
                    aabb.min_point().z,
                    0.0,
                ],
                aabb_max: [
                    aabb.max_point().x,
                    aabb.max_point().y,
                    aabb.max_point().z,
                    0.0,
                ],
                left: left_val,
                right: right_val,
                parent: 0,
                refit_count: 0,
            }
        })
        .collect()
}

fn build_balanced_tree_cpu(leaf_aabbs: &[Aabb3D]) -> Vec<BvhNodeGpu> {
    if leaf_aabbs.len() <= 1 {
        return Vec::new();
    }

    #[derive(Clone, Copy)]
    struct Task {
        start: usize,
        end: usize,
        parent: Option<(usize, bool)>,
    }

    let mut left = Vec::<Child>::new();
    let mut right = Vec::<Child>::new();
    let mut stack = vec![Task {
        start: 0,
        end: leaf_aabbs.len(),
        parent: None,
    }];

    while let Some(task) = stack.pop() {
        let len = task.end - task.start;
        if len == 0 {
            continue;
        }

        if len == 1 {
            if let Some((parent, is_left)) = task.parent {
                let child = Child::Leaf(task.start);
                if is_left {
                    left[parent] = child;
                } else {
                    right[parent] = child;
                }
            }
            continue;
        }

        let node_idx = left.len();
        left.push(Child::Leaf(task.start));
        right.push(Child::Leaf(task.start));

        if let Some((parent, is_left)) = task.parent {
            let child = Child::Internal(node_idx);
            if is_left {
                left[parent] = child;
            } else {
                right[parent] = child;
            }
        }

        let mid = task.start + len / 2;
        stack.push(Task {
            start: mid,
            end: task.end,
            parent: Some((node_idx, false)),
        });
        stack.push(Task {
            start: task.start,
            end: mid,
            parent: Some((node_idx, true)),
        });
    }

    let mut internal_aabbs = vec![Aabb3D::new(Vec3::ZERO, Vec3::ZERO); left.len()];
    for idx in (0..left.len()).rev() {
        let la = match left[idx] {
            Child::Leaf(i) => leaf_aabbs[i],
            Child::Internal(i) => internal_aabbs[i],
        };
        let ra = match right[idx] {
            Child::Leaf(i) => leaf_aabbs[i],
            Child::Internal(i) => internal_aabbs[i],
        };
        internal_aabbs[idx] = aabb_union(&la, &ra);
    }

    children_to_gpu_nodes(&left, &right, &internal_aabbs)
}

/// Build Karras tree + refit on CPU from already-sorted leaf AABBs.
fn build_tree_cpu(sorted_codes: &[u32], leaf_aabbs: &[Aabb3D]) -> Vec<BvhNodeGpu> {
    let n = sorted_codes.len();
    if n <= 1 {
        return Vec::new();
    }
    let num_internal = n - 1;

    let mut internal_left: Vec<Child> = Vec::with_capacity(num_internal);
    let mut internal_right: Vec<Child> = Vec::with_capacity(num_internal);

    for i in 0..num_internal {
        let (range_left, range_right, split) = karras_node(sorted_codes, i);
        let left = if split == range_left {
            Child::Leaf(split)
        } else {
            Child::Internal(split)
        };
        let right = if split + 1 == range_right {
            Child::Leaf(split + 1)
        } else {
            Child::Internal(split + 1)
        };
        internal_left.push(left);
        internal_right.push(right);
    }

    if validate_internal_tree(&internal_left, &internal_right) {
        if let Some(internal_aabbs) = refit_iterative(&internal_left, &internal_right, leaf_aabbs) {
            children_to_gpu_nodes(&internal_left, &internal_right, &internal_aabbs)
        } else {
            build_balanced_tree_cpu(leaf_aabbs)
        }
    } else {
        build_balanced_tree_cpu(leaf_aabbs)
    }
}

// ---------------------------------------------------------------------------
// GpuLbvh
// ---------------------------------------------------------------------------

/// GPU-accelerated Linear BVH broadphase.
///
/// All stages run on GPU: scene bounds reduction, Morton codes, radix sort,
/// Karras tree build, bottom-up refit, and pair finding.
/// A CPU fallback path is retained for validation (behind `build_tree_cpu`).
pub struct GpuLbvh {
    scene_bounds_reduce_kernel: ComputeKernel,
    morton_kernel: ComputeKernel,
    gather_sorted_leaf_aabbs_kernel: ComputeKernel,
    find_pairs_kernel: ComputeKernel,
    // GPU-only tree kernels (Karras build + refit + GPU-tree traversal).
    // Lazily compiled on first call to `query_on_device_gpu` to avoid loading
    // unused shaders at startup (some stricter WebGPU backends validate atomics
    // in storage struct members differently — only compile when actually used).
    find_pairs_gpu_tree_kernel: Option<ComputeKernel>,
    karras_build_kernel: Option<ComputeKernel>,
    refit_kernel: Option<ComputeKernel>,
    radix_sort: GpuRadixSort,
    scene_bounds: GpuBuffer<Aabb3D>,
    bounds_scratch_a: GpuBuffer<Aabb3D>,
    bounds_scratch_b: GpuBuffer<Aabb3D>,
    morton_keys: GpuBuffer<u32>,
    body_indices: GpuBuffer<u32>,
    /// Combined tree buffer: [0..n-1] internal nodes, [n-1..2n-1] leaf nodes.
    tree_nodes: GpuBuffer<BvhNodeGpu>,
    internal_nodes: GpuBuffer<BvhNodeGpu>,
    leaf_aabbs_sorted: GpuBuffer<Aabb3D>,
    pairs_out: GpuBuffer<BroadPair>,
    pair_counter: GpuAtomicCounter,
    count_params_buf: wgpu::Buffer,
    find_params_buf: wgpu::Buffer,
    tree_build_params_buf: wgpu::Buffer,
    #[cfg(not(target_arch = "wasm32"))]
    broadphase_profiler: Option<GpuBroadphaseProfiler>,
}

impl GpuLbvh {
    /// Create a new GPU LBVH broadphase for up to `max_bodies` bodies.
    pub fn new(ctx: &GpuContext, max_bodies: usize) -> Self {
        let scene_bounds_reduce_kernel =
            ComputeKernel::from_wgsl(ctx, SCENE_BOUNDS_REDUCE_WGSL, "main");
        let morton_kernel = ComputeKernel::from_wgsl(ctx, MORTON_3D_WGSL, "main");
        let gather_sorted_leaf_aabbs_kernel =
            ComputeKernel::from_wgsl(ctx, GATHER_SORTED_LEAF_AABBS_WGSL, "main");
        let find_pairs_kernel = ComputeKernel::from_wgsl(ctx, FIND_PAIRS_WGSL, "main");
        // Lazily compiled — see field doc comments.
        let find_pairs_gpu_tree_kernel = None;
        let karras_build_kernel = None;
        let refit_kernel = None;
        let radix_sort = GpuRadixSort::new(ctx, max_bodies);
        let max_bounds_partials = round_up_workgroups(max_bodies.max(1) as u32, WG) as usize;

        let scene_bounds = GpuBuffer::new(ctx, 1);
        let bounds_scratch_a = GpuBuffer::new(ctx, max_bounds_partials.max(1));
        let bounds_scratch_b = GpuBuffer::new(ctx, max_bounds_partials.max(1));
        let morton_keys = GpuBuffer::new(ctx, max_bodies.max(1));
        let body_indices = GpuBuffer::new(ctx, max_bodies.max(1));
        // Tree: 2*n - 1 nodes (n-1 internal + n leaves)
        let tree_nodes = GpuBuffer::new(ctx, (max_bodies * 2).max(1));
        let internal_nodes = GpuBuffer::new(ctx, max_bodies.max(1));
        let leaf_aabbs_sorted = GpuBuffer::new(ctx, max_bodies.max(1));
        let max_pairs = max_bodies * 8;
        let pairs_out = GpuBuffer::new(ctx, max_pairs.max(1));
        let pair_counter = GpuAtomicCounter::new(ctx);

        let count_params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lbvh count params"),
            size: std::mem::size_of::<CountParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let find_params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("find_pairs params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tree_build_params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tree build params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        #[cfg(not(target_arch = "wasm32"))]
        let broadphase_profiler = GpuBroadphaseProfiler::new(ctx);

        Self {
            scene_bounds_reduce_kernel,
            morton_kernel,
            gather_sorted_leaf_aabbs_kernel,
            find_pairs_kernel,
            find_pairs_gpu_tree_kernel,
            karras_build_kernel,
            refit_kernel,
            radix_sort,
            scene_bounds,
            bounds_scratch_a,
            bounds_scratch_b,
            morton_keys,
            body_indices,
            tree_nodes,
            internal_nodes,
            leaf_aabbs_sorted,
            pairs_out,
            pair_counter,
            count_params_buf,
            find_params_buf,
            tree_build_params_buf,
            #[cfg(not(target_arch = "wasm32"))]
            broadphase_profiler,
        }
    }

    fn mark_precise_breakdown(&self, ctx: &GpuContext, marker: BroadphaseTimingMarker) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(profiler) = &self.broadphase_profiler {
            profiler.mark(ctx, marker);
        }
    }

    fn finish_precise_breakdown(&mut self, ctx: &GpuContext) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(profiler) = &mut self.broadphase_profiler {
            profiler.finish_frame(ctx);
        }
    }

    pub fn refresh_precise_breakdown(
        &mut self,
        ctx: &GpuContext,
        breakdown: &mut BroadphaseBreakdownMs,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(profiler) = &mut self.broadphase_profiler {
            profiler.collect_ready(ctx);
            profiler.apply_latest(breakdown);
        }
    }

    fn write_count_params(&self, ctx: &GpuContext, count: u32) {
        let params = CountParams {
            num_bodies: count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.queue
            .write_buffer(&self.count_params_buf, 0, bytemuck::bytes_of(&params));
    }

    fn dispatch_scene_bounds_reduce_pass(
        &self,
        ctx: &GpuContext,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        count: u32,
    ) {
        self.write_count_params(ctx, count);
        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene_bounds_reduce"),
            layout: self.scene_bounds_reduce_kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.count_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("scene_bounds_reduce"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.scene_bounds_reduce_kernel.pipeline());
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(round_up_workgroups(count, WG), 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    fn reduce_scene_bounds(&mut self, ctx: &GpuContext, aabb_buf: &wgpu::Buffer, num_bodies: u32) {
        #[derive(Clone, Copy)]
        enum BoundsInput<'a> {
            Raw(&'a wgpu::Buffer),
            ScratchA,
            ScratchB,
        }

        self.scene_bounds.set_len(1);

        let mut input = BoundsInput::Raw(aabb_buf);
        let mut remaining = num_bodies;
        let mut write_to_a = true;

        loop {
            let partial_count = round_up_workgroups(remaining, WG);
            let output_is_final = partial_count == 1;

            if !output_is_final {
                if write_to_a {
                    self.bounds_scratch_a
                        .grow_if_needed(ctx, partial_count as usize);
                    self.bounds_scratch_a.set_len(partial_count);
                } else {
                    self.bounds_scratch_b
                        .grow_if_needed(ctx, partial_count as usize);
                    self.bounds_scratch_b.set_len(partial_count);
                }
            }

            let input_buffer = match input {
                BoundsInput::Raw(buffer) => buffer,
                BoundsInput::ScratchA => self.bounds_scratch_a.buffer(),
                BoundsInput::ScratchB => self.bounds_scratch_b.buffer(),
            };
            let output_buffer = if output_is_final {
                self.scene_bounds.buffer()
            } else if write_to_a {
                self.bounds_scratch_a.buffer()
            } else {
                self.bounds_scratch_b.buffer()
            };

            self.dispatch_scene_bounds_reduce_pass(ctx, input_buffer, output_buffer, remaining);

            if output_is_final {
                break;
            }

            input = if write_to_a {
                BoundsInput::ScratchA
            } else {
                BoundsInput::ScratchB
            };
            remaining = partial_count;
            write_to_a = !write_to_a;
        }
    }

    fn dispatch_morton(&mut self, ctx: &GpuContext, aabb_buf: &wgpu::Buffer, num_bodies: u32) {
        let n = num_bodies as usize;
        self.write_count_params(ctx, num_bodies);
        self.morton_keys.grow_if_needed(ctx, n);
        self.body_indices.grow_if_needed(ctx, n);
        self.morton_keys.set_len(num_bodies);
        self.body_indices.set_len(num_bodies);

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("morton"),
            layout: self.morton_kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: aabb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.scene_bounds.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.morton_keys.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.body_indices.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.count_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("morton"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.morton_kernel.pipeline());
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(round_up_workgroups(num_bodies, WG), 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    fn dispatch_sorted_leaf_gather(
        &mut self,
        ctx: &GpuContext,
        aabb_buf: &wgpu::Buffer,
        num_bodies: u32,
    ) {
        self.write_count_params(ctx, num_bodies);
        self.leaf_aabbs_sorted
            .grow_if_needed(ctx, num_bodies as usize);
        self.leaf_aabbs_sorted.set_len(num_bodies);

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gather_sorted_leaf_aabbs"),
            layout: self.gather_sorted_leaf_aabbs_kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: aabb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.body_indices.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.leaf_aabbs_sorted.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.count_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gather_sorted_leaf_aabbs"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.gather_sorted_leaf_aabbs_kernel.pipeline());
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(round_up_workgroups(num_bodies, WG), 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    /// Dispatch GPU Karras tree build: construct BVH topology from sorted Morton codes.
    fn dispatch_karras_build(&mut self, ctx: &GpuContext, num_bodies: u32) {
        if num_bodies <= 1 {
            return;
        }
        let num_internal = num_bodies - 1;
        // Grow tree buffer: 2*n - 1 nodes (internal + leaves)
        let tree_size = (num_bodies as usize * 2).saturating_sub(1);
        self.tree_nodes.grow_if_needed(ctx, tree_size);

        // Lazy compile on first use
        if self.karras_build_kernel.is_none() {
            self.karras_build_kernel =
                Some(ComputeKernel::from_wgsl(ctx, KARRAS_BUILD_WGSL, "main"));
        }
        let kernel = self.karras_build_kernel.as_ref().unwrap();

        let params: [u32; 4] = [num_bodies, 0, 0, 0];
        ctx.queue.write_buffer(
            &self.tree_build_params_buf,
            0,
            bytemuck::cast_slice(&params),
        );

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("karras_build"),
            layout: kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.morton_keys.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.tree_nodes.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.tree_build_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("karras_build"),
                timestamp_writes: None,
            });
            pass.set_pipeline(kernel.pipeline());
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(round_up_workgroups(num_internal, WG), 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    /// Dispatch GPU bottom-up refit: propagate leaf AABBs up through the tree.
    /// Uses a single workgroup with atomic synchronization for web compatibility.
    fn dispatch_refit(&mut self, ctx: &GpuContext, num_bodies: u32) {
        if num_bodies <= 1 {
            return;
        }

        // Lazy compile on first use
        if self.refit_kernel.is_none() {
            self.refit_kernel = Some(ComputeKernel::from_wgsl(ctx, REFIT_WGSL, "main"));
        }
        let kernel = self.refit_kernel.as_ref().unwrap();

        let params: [u32; 4] = [num_bodies, 0, 0, 0];
        ctx.queue.write_buffer(
            &self.tree_build_params_buf,
            0,
            bytemuck::cast_slice(&params),
        );

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("refit"),
            layout: kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.leaf_aabbs_sorted.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.body_indices.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.tree_nodes.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.tree_build_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("refit"),
                timestamp_writes: None,
            });
            pass.set_pipeline(kernel.pipeline());
            pass.set_bind_group(0, &bg, &[]);
            // Single workgroup for web-compatible uniform control flow
            pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    fn dispatch_find_pairs(&mut self, ctx: &GpuContext, num_bodies: u32) -> u32 {
        let max_pairs = (num_bodies as usize * 8) as u32;
        self.pair_counter.reset(ctx);
        self.pairs_out.grow_if_needed(ctx, max_pairs as usize);

        let find_params: [u32; 4] = [num_bodies, max_pairs, 0, 0];
        ctx.queue
            .write_buffer(&self.find_params_buf, 0, bytemuck::cast_slice(&find_params));

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("find_pairs"),
            layout: self.find_pairs_kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.leaf_aabbs_sorted.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.body_indices.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.internal_nodes.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.pairs_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.pair_counter.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.find_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("find_pairs"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.find_pairs_kernel.pipeline());
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(round_up_workgroups(num_bodies, 64), 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
        max_pairs
    }

    /// Batch gather + karras build + refit + find_pairs into a single encoder/submit.
    /// Saves 3 queue.submit() calls vs calling each dispatch method individually.
    fn dispatch_build_traverse_batched(
        &mut self,
        ctx: &GpuContext,
        aabb_buf: &wgpu::Buffer,
        num_bodies: u32,
    ) -> u32 {
        if num_bodies <= 1 {
            return 0;
        }

        let max_pairs = (num_bodies as usize * 8) as u32;
        let num_internal = num_bodies - 1;

        // Pre-allocate / grow all buffers
        self.leaf_aabbs_sorted
            .grow_if_needed(ctx, num_bodies as usize);
        self.leaf_aabbs_sorted.set_len(num_bodies);
        let tree_size = (num_bodies as usize * 2).saturating_sub(1);
        self.tree_nodes.grow_if_needed(ctx, tree_size);
        self.pair_counter.reset(ctx);
        self.pairs_out.grow_if_needed(ctx, max_pairs as usize);

        // Lazy-compile kernels
        if self.karras_build_kernel.is_none() {
            self.karras_build_kernel =
                Some(ComputeKernel::from_wgsl(ctx, KARRAS_BUILD_WGSL, "main"));
        }
        if self.refit_kernel.is_none() {
            self.refit_kernel = Some(ComputeKernel::from_wgsl(ctx, REFIT_WGSL, "main"));
        }
        if self.find_pairs_gpu_tree_kernel.is_none() {
            self.find_pairs_gpu_tree_kernel = Some(ComputeKernel::from_wgsl(
                ctx,
                FIND_PAIRS_GPU_TREE_WGSL,
                "main",
            ));
        }

        // Write params to different buffers (all visible at submit time)
        self.write_count_params(ctx, num_bodies);
        ctx.queue.write_buffer(
            &self.tree_build_params_buf,
            0,
            bytemuck::cast_slice(&[num_bodies, 0u32, 0, 0]),
        );
        let find_params: [u32; 4] = [num_bodies, max_pairs, 0, 0];
        ctx.queue
            .write_buffer(&self.find_params_buf, 0, bytemuck::cast_slice(&find_params));

        // Build bind groups
        let gather_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gather_sorted_leaf_aabbs"),
            layout: self.gather_sorted_leaf_aabbs_kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: aabb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.body_indices.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.leaf_aabbs_sorted.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.count_params_buf.as_entire_binding(),
                },
            ],
        });

        let karras_kernel = self.karras_build_kernel.as_ref().unwrap();
        let karras_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("karras_build"),
            layout: karras_kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.morton_keys.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.tree_nodes.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.tree_build_params_buf.as_entire_binding(),
                },
            ],
        });

        let refit_kernel = self.refit_kernel.as_ref().unwrap();
        let refit_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("refit"),
            layout: refit_kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.leaf_aabbs_sorted.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.body_indices.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.tree_nodes.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.tree_build_params_buf.as_entire_binding(),
                },
            ],
        });

        let find_kernel = self.find_pairs_gpu_tree_kernel.as_ref().unwrap();
        let find_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("find_pairs_gpu_tree"),
            layout: find_kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.tree_nodes.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.pairs_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.pair_counter.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.find_params_buf.as_entire_binding(),
                },
            ],
        });

        // Record all 4 passes on one encoder
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lbvh_build_traverse_batch"),
            });

        // 1. Gather sorted leaf AABBs
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gather_sorted_leaf_aabbs"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.gather_sorted_leaf_aabbs_kernel.pipeline());
            pass.set_bind_group(0, &gather_bg, &[]);
            pass.dispatch_workgroups(round_up_workgroups(num_bodies, WG), 1, 1);
        }

        // 2. Karras BVH build
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("karras_build"),
                timestamp_writes: None,
            });
            pass.set_pipeline(karras_kernel.pipeline());
            pass.set_bind_group(0, &karras_bg, &[]);
            pass.dispatch_workgroups(round_up_workgroups(num_internal, WG), 1, 1);
        }

        // 3. Bottom-up refit
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("refit"),
                timestamp_writes: None,
            });
            pass.set_pipeline(refit_kernel.pipeline());
            pass.set_bind_group(0, &refit_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // 4. Find pairs via tree traversal
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("find_pairs_gpu_tree"),
                timestamp_writes: None,
            });
            pass.set_pipeline(find_kernel.pipeline());
            pass.set_bind_group(0, &find_bg, &[]);
            pass.dispatch_workgroups(round_up_workgroups(num_bodies, 64), 1, 1);
        }

        ctx.queue.submit(Some(encoder.finish()));
        max_pairs
    }

    /// Find pairs using the GPU-built tree (tree_nodes buffer with absolute indices).
    fn dispatch_find_pairs_gpu_tree(&mut self, ctx: &GpuContext, num_bodies: u32) -> u32 {
        let max_pairs = (num_bodies as usize * 8) as u32;
        self.pair_counter.reset(ctx);
        self.pairs_out.grow_if_needed(ctx, max_pairs as usize);

        // Lazy compile on first use
        if self.find_pairs_gpu_tree_kernel.is_none() {
            self.find_pairs_gpu_tree_kernel = Some(ComputeKernel::from_wgsl(
                ctx,
                FIND_PAIRS_GPU_TREE_WGSL,
                "main",
            ));
        }
        let kernel = self.find_pairs_gpu_tree_kernel.as_ref().unwrap();

        let find_params: [u32; 4] = [num_bodies, max_pairs, 0, 0];
        ctx.queue
            .write_buffer(&self.find_params_buf, 0, bytemuck::cast_slice(&find_params));

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("find_pairs_gpu_tree"),
            layout: kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.tree_nodes.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.pairs_out.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.pair_counter.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.find_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("find_pairs_gpu_tree"),
                timestamp_writes: None,
            });
            pass.set_pipeline(kernel.pipeline());
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(round_up_workgroups(num_bodies, 64), 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
        max_pairs
    }

    /// Run the full GPU broadphase pipeline on a 3D AABB buffer.
    pub fn build_and_query(
        &mut self,
        ctx: &GpuContext,
        aabbs: &GpuBuffer<Aabb3D>,
        num_bodies: u32,
    ) -> Vec<[u32; 2]> {
        self.build_and_query_raw(ctx, aabbs.buffer(), num_bodies)
    }

    /// Run the full GPU broadphase pipeline on any AABB buffer.
    ///
    /// `aabb_buf` is the raw GPU buffer (must have the same layout as `array<Aabb>` in WGSL:
    /// each element is `{min_pt: vec4<f32>, max_pt: vec4<f32>}`).
    /// Both `Aabb3D` and `Aabb2D` satisfy this layout (32 bytes, Vec4 min + Vec4 max).
    pub fn build_and_query_raw(
        &mut self,
        ctx: &GpuContext,
        aabb_buf: &wgpu::Buffer,
        num_bodies: u32,
    ) -> Vec<[u32; 2]> {
        let mut breakdown = BroadphaseBreakdownMs::default();
        self.build_and_query_raw_with_breakdown(ctx, aabb_buf, num_bodies, &mut breakdown)
    }

    /// Run the broadphase and leave the resulting pair buffer resident on the GPU.
    ///
    /// Returns the dispatched pair capacity (`num_bodies * 8`) used as the upper bound
    /// for the pair-finding kernel. The actual overlap count is stored in
    /// [`GpuLbvh::pair_counter_buffer`].
    pub fn query_on_device_raw(
        &mut self,
        ctx: &GpuContext,
        aabb_buf: &wgpu::Buffer,
        num_bodies: u32,
    ) -> u32 {
        let mut breakdown = BroadphaseBreakdownMs::default();
        self.query_on_device_raw_with_breakdown(ctx, aabb_buf, num_bodies, &mut breakdown)
    }

    /// Fully-GPU broadphase pipeline: no CPU readback for tree construction.
    ///
    /// Pipeline: bounds reduction → Morton codes → radix sort → leaf gather →
    /// GPU Karras tree build → GPU bottom-up refit → GPU pair finding.
    /// All stages stay on GPU. Returns pair capacity (actual count in pair_counter).
    pub fn query_on_device_gpu(
        &mut self,
        ctx: &GpuContext,
        aabb_buf: &wgpu::Buffer,
        num_bodies: u32,
    ) -> u32 {
        let mut breakdown = BroadphaseBreakdownMs::default();
        self.query_on_device_gpu_with_breakdown(ctx, aabb_buf, num_bodies, &mut breakdown)
    }

    /// Same as [`GpuLbvh::query_on_device_gpu`], with coarse timing buckets filled in.
    ///
    /// GPU Karras build and refit are included in `build_ms`; pair-finding command submission
    /// is included in `traverse_ms`. The function does not force a queue wait so downstream
    /// stages can consume the pair counter directly on GPU without a CPU sync.
    pub fn query_on_device_gpu_with_breakdown(
        &mut self,
        ctx: &GpuContext,
        aabb_buf: &wgpu::Buffer,
        num_bodies: u32,
        breakdown: &mut BroadphaseBreakdownMs,
    ) -> u32 {
        #[cfg(target_arch = "wasm32")]
        use rubble_gpu::web_time::Instant;
        #[cfg(not(target_arch = "wasm32"))]
        use std::time::Instant;

        if num_bodies <= 1 {
            return 0;
        }

        let t_bounds = Instant::now();
        self.mark_precise_breakdown(ctx, BroadphaseTimingMarker::BoundsStart);
        self.reduce_scene_bounds(ctx, aabb_buf, num_bodies);
        self.mark_precise_breakdown(ctx, BroadphaseTimingMarker::BoundsEnd);
        breakdown.bounds_ms += t_bounds.elapsed().as_secs_f32() * 1000.0;

        let t_sort = Instant::now();
        self.mark_precise_breakdown(ctx, BroadphaseTimingMarker::SortStart);
        self.dispatch_morton(ctx, aabb_buf, num_bodies);
        self.radix_sort
            .sort_key_value_in_place(ctx, &mut self.morton_keys, &mut self.body_indices);
        self.mark_precise_breakdown(ctx, BroadphaseTimingMarker::SortEnd);
        breakdown.sort_ms += t_sort.elapsed().as_secs_f32() * 1000.0;

        let t_build = Instant::now();
        self.mark_precise_breakdown(ctx, BroadphaseTimingMarker::BuildStart);
        let max_pairs = self.dispatch_build_traverse_batched(ctx, aabb_buf, num_bodies);
        self.mark_precise_breakdown(ctx, BroadphaseTimingMarker::BuildEnd);
        breakdown.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;
        // traverse timing merged into build since they share one submit
        self.finish_precise_breakdown(ctx);
        max_pairs
    }

    /// Run the full GPU broadphase pipeline on any AABB buffer and accumulate
    /// coarse timing buckets for the current hybrid CPU/GPU implementation.
    pub fn build_and_query_raw_with_breakdown(
        &mut self,
        ctx: &GpuContext,
        aabb_buf: &wgpu::Buffer,
        num_bodies: u32,
        breakdown: &mut BroadphaseBreakdownMs,
    ) -> Vec<[u32; 2]> {
        #[cfg(target_arch = "wasm32")]
        use rubble_gpu::web_time::Instant;
        #[cfg(not(target_arch = "wasm32"))]
        use std::time::Instant;

        if num_bodies <= 1 {
            return Vec::new();
        }

        let t_bounds = Instant::now();
        self.reduce_scene_bounds(ctx, aabb_buf, num_bodies);
        breakdown.bounds_ms += t_bounds.elapsed().as_secs_f32() * 1000.0;

        let t_sort = Instant::now();
        self.dispatch_morton(ctx, aabb_buf, num_bodies);
        self.radix_sort
            .sort_key_value_in_place(ctx, &mut self.morton_keys, &mut self.body_indices);
        breakdown.sort_ms += t_sort.elapsed().as_secs_f32() * 1000.0;

        let t_build = Instant::now();
        self.dispatch_sorted_leaf_gather(ctx, aabb_buf, num_bodies);
        breakdown.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;

        let t_readback = Instant::now();
        let (sorted_codes, leaf_aabbs) =
            self.morton_keys.download_with(ctx, &self.leaf_aabbs_sorted);
        breakdown.readback_ms += t_readback.elapsed().as_secs_f32() * 1000.0;

        let t_build = Instant::now();
        let gpu_nodes = build_tree_cpu(&sorted_codes, &leaf_aabbs);
        if gpu_nodes.is_empty() {
            breakdown.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;
            return Vec::new();
        }

        self.internal_nodes.upload(ctx, &gpu_nodes);
        breakdown.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;

        let t_traverse = Instant::now();
        let max_pairs = self.dispatch_find_pairs(ctx, num_bodies);
        breakdown.traverse_ms += t_traverse.elapsed().as_secs_f32() * 1000.0;

        let t_readback = Instant::now();
        let (pairs, pair_count) =
            self.pairs_out
                .download_with_counter(ctx, &self.pair_counter, max_pairs);
        let pair_count = pair_count.min(max_pairs);
        if pair_count == 0 {
            breakdown.readback_ms += t_readback.elapsed().as_secs_f32() * 1000.0;
            return Vec::new();
        }

        let mut result: Vec<[u32; 2]> = pairs
            .into_iter()
            .take(pair_count as usize)
            .map(|p| {
                let (a, b) = if p.a < p.b { (p.a, p.b) } else { (p.b, p.a) };
                [a, b]
            })
            .collect();
        result.sort_unstable();
        result.dedup();
        breakdown.readback_ms += t_readback.elapsed().as_secs_f32() * 1000.0;
        result
    }

    /// Variant of [`GpuLbvh::query_on_device_raw`] that accumulates timing buckets.
    ///
    /// Uses the fully GPU LBVH path ([`GpuLbvh::query_on_device_gpu_with_breakdown`]) so
    /// Morton codes and leaf AABBs are not read back for a CPU tree build.
    pub fn query_on_device_raw_with_breakdown(
        &mut self,
        ctx: &GpuContext,
        aabb_buf: &wgpu::Buffer,
        num_bodies: u32,
        breakdown: &mut BroadphaseBreakdownMs,
    ) -> u32 {
        self.query_on_device_gpu_with_breakdown(ctx, aabb_buf, num_bodies, breakdown)
    }

    /// Async variant for WASM/WebGPU callers.
    #[cfg(target_arch = "wasm32")]
    pub async fn build_and_query_raw_async(
        &mut self,
        ctx: &GpuContext,
        aabb_buf: &wgpu::Buffer,
        num_bodies: u32,
    ) -> Vec<[u32; 2]> {
        let mut breakdown = BroadphaseBreakdownMs::default();
        self.build_and_query_raw_async_with_breakdown(ctx, aabb_buf, num_bodies, &mut breakdown)
            .await
    }

    /// Async variant of [`GpuLbvh::query_on_device_raw`].
    #[cfg(target_arch = "wasm32")]
    pub async fn query_on_device_raw_async(
        &mut self,
        ctx: &GpuContext,
        aabb_buf: &wgpu::Buffer,
        num_bodies: u32,
    ) -> u32 {
        let mut breakdown = BroadphaseBreakdownMs::default();
        self.query_on_device_raw_async_with_breakdown(ctx, aabb_buf, num_bodies, &mut breakdown)
            .await
    }

    /// Async variant of [`build_and_query_raw_with_breakdown`].
    #[cfg(target_arch = "wasm32")]
    pub async fn build_and_query_raw_async_with_breakdown(
        &mut self,
        ctx: &GpuContext,
        aabb_buf: &wgpu::Buffer,
        num_bodies: u32,
        breakdown: &mut BroadphaseBreakdownMs,
    ) -> Vec<[u32; 2]> {
        use rubble_gpu::web_time::Instant;

        if num_bodies <= 1 {
            return Vec::new();
        }

        let t_bounds = Instant::now();
        self.reduce_scene_bounds(ctx, aabb_buf, num_bodies);
        breakdown.bounds_ms += t_bounds.elapsed().as_secs_f32() * 1000.0;

        let t_sort = Instant::now();
        self.dispatch_morton(ctx, aabb_buf, num_bodies);
        self.radix_sort
            .sort_key_value_in_place(ctx, &mut self.morton_keys, &mut self.body_indices);
        breakdown.sort_ms += t_sort.elapsed().as_secs_f32() * 1000.0;

        let t_build = Instant::now();
        self.dispatch_sorted_leaf_gather(ctx, aabb_buf, num_bodies);
        breakdown.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;

        let t_readback = Instant::now();
        let (sorted_codes, leaf_aabbs) = self
            .morton_keys
            .download_with_async(ctx, &self.leaf_aabbs_sorted)
            .await;
        breakdown.readback_ms += t_readback.elapsed().as_secs_f32() * 1000.0;

        let t_build = Instant::now();
        let gpu_nodes = build_tree_cpu(&sorted_codes, &leaf_aabbs);
        if gpu_nodes.is_empty() {
            breakdown.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;
            return Vec::new();
        }

        self.internal_nodes.upload(ctx, &gpu_nodes);
        breakdown.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;

        let t_traverse = Instant::now();
        let max_pairs = self.dispatch_find_pairs(ctx, num_bodies);
        breakdown.traverse_ms += t_traverse.elapsed().as_secs_f32() * 1000.0;

        let t_readback = Instant::now();
        let (pairs, pair_count) = self
            .pairs_out
            .download_with_counter_async(ctx, &self.pair_counter, max_pairs)
            .await;
        let pair_count = pair_count.min(max_pairs);
        if pair_count == 0 {
            breakdown.readback_ms += t_readback.elapsed().as_secs_f32() * 1000.0;
            return Vec::new();
        }

        let mut result: Vec<[u32; 2]> = pairs
            .into_iter()
            .take(pair_count as usize)
            .map(|p| {
                let (a, b) = if p.a < p.b { (p.a, p.b) } else { (p.b, p.a) };
                [a, b]
            })
            .collect();
        result.sort_unstable();
        result.dedup();
        breakdown.readback_ms += t_readback.elapsed().as_secs_f32() * 1000.0;
        result
    }

    /// Async variant of [`GpuLbvh::query_on_device_raw_with_breakdown`].
    #[cfg(target_arch = "wasm32")]
    pub async fn query_on_device_raw_async_with_breakdown(
        &mut self,
        ctx: &GpuContext,
        aabb_buf: &wgpu::Buffer,
        num_bodies: u32,
        breakdown: &mut BroadphaseBreakdownMs,
    ) -> u32 {
        self.query_on_device_gpu_with_breakdown(ctx, aabb_buf, num_bodies, breakdown)
    }

    pub fn pair_buffer(&self) -> &wgpu::Buffer {
        self.pairs_out.buffer()
    }

    pub fn read_pair_count(&self, ctx: &GpuContext, max_pairs: u32) -> u32 {
        self.pair_counter.read(ctx).min(max_pairs)
    }

    #[cfg(target_arch = "wasm32")]
    pub async fn read_pair_count_async(&self, ctx: &GpuContext, max_pairs: u32) -> u32 {
        self.pair_counter.read_async(ctx).await.min(max_pairs)
    }

    pub fn pair_counter_buffer(&self) -> &wgpu::Buffer {
        self.pair_counter.buffer()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rubble_gpu::GpuBuffer;
    use rubble_math::Aabb3D;

    fn try_ctx() -> Option<GpuContext> {
        crate::test_gpu()
    }

    fn aabb(min: [f32; 3], max: [f32; 3]) -> Aabb3D {
        Aabb3D::new(Vec3::from(min), Vec3::from(max))
    }

    #[test]
    fn single_body_returns_no_pairs() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut lbvh = GpuLbvh::new(&ctx, 16);
        let aabbs = vec![aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])];
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, 1);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query(&ctx, &buf, 1);
        assert!(pairs.is_empty());
    }

    #[test]
    fn two_overlapping_bodies() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut lbvh = GpuLbvh::new(&ctx, 16);
        let aabbs = vec![
            aabb([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
            aabb([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]),
        ];
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, 2);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query(&ctx, &buf, 2);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], [0, 1]);
    }

    #[test]
    fn two_separated_bodies_no_pairs() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut lbvh = GpuLbvh::new(&ctx, 16);
        let aabbs = vec![
            aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            aabb([5.0, 5.0, 5.0], [6.0, 6.0, 6.0]),
        ];
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, 2);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query(&ctx, &buf, 2);
        assert!(pairs.is_empty());
    }

    #[test]
    fn three_bodies_partial_overlap() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut lbvh = GpuLbvh::new(&ctx, 16);
        // A overlaps B, B overlaps C, A does NOT overlap C
        let aabbs = vec![
            aabb([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]), // A
            aabb([1.5, 1.5, 1.5], [3.5, 3.5, 3.5]), // B
            aabb([3.0, 3.0, 3.0], [5.0, 5.0, 5.0]), // C
        ];
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, 3);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query(&ctx, &buf, 3);
        assert_eq!(pairs.len(), 2);
        assert!(pairs.contains(&[0, 1]));
        assert!(pairs.contains(&[1, 2]));
        // A-C should NOT be a pair
        assert!(!pairs.contains(&[0, 2]));
    }

    #[test]
    fn all_overlapping_returns_all_pairs() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut lbvh = GpuLbvh::new(&ctx, 16);
        // All overlap each other (big box)
        let aabbs = vec![
            aabb([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]),
            aabb([1.0, 1.0, 1.0], [11.0, 11.0, 11.0]),
            aabb([2.0, 2.0, 2.0], [12.0, 12.0, 12.0]),
            aabb([3.0, 3.0, 3.0], [13.0, 13.0, 13.0]),
        ];
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, 4);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query(&ctx, &buf, 4);
        // 4 choose 2 = 6 pairs
        assert_eq!(pairs.len(), 6);
    }

    #[test]
    fn no_duplicate_pairs() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut lbvh = GpuLbvh::new(&ctx, 16);
        let aabbs = vec![
            aabb([0.0, 0.0, 0.0], [5.0, 5.0, 5.0]),
            aabb([1.0, 1.0, 1.0], [6.0, 6.0, 6.0]),
            aabb([2.0, 2.0, 2.0], [7.0, 7.0, 7.0]),
        ];
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, 3);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query(&ctx, &buf, 3);
        // All pairs should be unique and ordered (a < b)
        for p in &pairs {
            assert!(p[0] < p[1], "pair should be ordered: {:?}", p);
        }
        let mut deduped = pairs.clone();
        deduped.dedup();
        assert_eq!(pairs.len(), deduped.len(), "no duplicates expected");
    }

    #[test]
    fn build_and_query_raw_works() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut lbvh = GpuLbvh::new(&ctx, 16);
        let aabbs = vec![
            aabb([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
            aabb([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]),
        ];
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, 2);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query_raw(&ctx, buf.buffer(), 2);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0], [0, 1]);
    }

    #[test]
    fn many_bodies_stress() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let n = 64;
        let mut lbvh = GpuLbvh::new(&ctx, n);
        // Line of non-overlapping unit cubes along X
        let aabbs: Vec<Aabb3D> = (0..n)
            .map(|i| {
                let x = i as f32 * 3.0; // gap of 2 between each
                aabb([x, 0.0, 0.0], [x + 1.0, 1.0, 1.0])
            })
            .collect();
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, n);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query(&ctx, &buf, n as u32);
        assert!(
            pairs.is_empty(),
            "separated bodies should have no pairs, got {}",
            pairs.len()
        );
    }

    #[test]
    fn touching_aabbs_overlap() {
        let Some(ctx) = try_ctx() else {
            eprintln!("SKIP: No GPU");
            return;
        };
        let mut lbvh = GpuLbvh::new(&ctx, 16);
        // Edge-touching (min.x of B == max.x of A)
        let aabbs = vec![
            aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            aabb([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
        ];
        let mut buf = GpuBuffer::<Aabb3D>::new(&ctx, 2);
        buf.upload(&ctx, &aabbs);
        let pairs = lbvh.build_and_query(&ctx, &buf, 2);
        // Touching AABBs satisfy <=/>= so they overlap
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn karras_tree_build_correctness() {
        // Test CPU tree build directly with known Morton codes
        let codes = vec![0u32, 1, 2, 3];
        let aabbs = vec![
            aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            aabb([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
            aabb([2.0, 0.0, 0.0], [3.0, 1.0, 1.0]),
            aabb([3.0, 0.0, 0.0], [4.0, 1.0, 1.0]),
        ];
        let nodes = build_tree_cpu(&codes, &aabbs);

        // 4 leaves → 3 internal nodes
        assert_eq!(nodes.len(), 3);

        // Root node AABB should encompass all leaves
        let root = &nodes[0];
        assert!(root.aabb_min[0] <= 0.0);
        assert!(root.aabb_max[0] >= 4.0);
        assert!(root.aabb_min[1] <= 0.0);
        assert!(root.aabb_max[1] >= 1.0);
    }

    // -----------------------------------------------------------------------
    // WGSL shader validation — catches shader bugs before they reach WebGPU
    // -----------------------------------------------------------------------

    fn validate_wgsl(name: &str, source: &str) {
        let module = match naga::front::wgsl::parse_str(source) {
            Ok(m) => m,
            Err(e) => panic!("{name}: WGSL parse error:\n{e}"),
        };
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        if let Err(e) = validator.validate(&module) {
            panic!("{name}: WGSL validation error:\n{e}");
        }
    }

    #[test]
    fn validate_scene_bounds_reduce_wgsl() {
        validate_wgsl("SCENE_BOUNDS_REDUCE_WGSL", SCENE_BOUNDS_REDUCE_WGSL);
    }

    #[test]
    fn validate_morton_3d_wgsl() {
        validate_wgsl("MORTON_3D_WGSL", MORTON_3D_WGSL);
    }

    #[test]
    fn validate_gather_sorted_leaf_aabbs_wgsl() {
        validate_wgsl(
            "GATHER_SORTED_LEAF_AABBS_WGSL",
            GATHER_SORTED_LEAF_AABBS_WGSL,
        );
    }

    #[test]
    fn validate_find_pairs_wgsl() {
        validate_wgsl("FIND_PAIRS_WGSL", FIND_PAIRS_WGSL);
    }

    #[test]
    fn validate_find_pairs_gpu_tree_wgsl() {
        validate_wgsl("FIND_PAIRS_GPU_TREE_WGSL", FIND_PAIRS_GPU_TREE_WGSL);
    }

    #[test]
    fn validate_karras_build_wgsl() {
        validate_wgsl("KARRAS_BUILD_WGSL", KARRAS_BUILD_WGSL);
    }

    #[test]
    fn validate_refit_wgsl() {
        validate_wgsl("REFIT_WGSL", REFIT_WGSL);
    }
}
