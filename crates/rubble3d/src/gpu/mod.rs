//! GPU compute pipeline for 3D physics simulation using AVBD solver.
//!
//! Orchestrates the full simulation step on the GPU using compute shaders:
//! predict → AABB compute → broadphase → narrowphase → AVBD solver → velocity extraction.
//!
//! ## Shader Backends
//!
//! The pipeline supports two shader compilation paths:
//!
//! 1. **WGSL** (default): Inline WGSL string constants compiled at runtime via wgpu.
//!    Zero build-time dependencies. Works on all wgpu-supported platforms.
//!
//! 2. **rust-gpu / SPIR-V** (via `spirv` feature): Physics kernels written in Rust
//!    (see `rubble-shaders` crate), compiled to SPIR-V by `rust-gpu`, loaded via
//!    `ComputeKernel::from_spirv()`. Enables multi-GPU target support — the same
//!    Rust shader source produces SPIR-V that runs on Vulkan, Metal (via MoltenVK),
//!    and DX12 (via vkd3d) without per-vendor shader maintenance.
//!
//! The narrowphase uses WGSL in both modes (heavy use of atomics and shape dispatch).
//! The performance-critical AVBD solver, predict, and extract kernels are available
//! in both WGSL and rust-gpu variants.

mod avbd_solve_wgsl;
mod coloring_wgsl;
mod contact_stream_wgsl;
mod extract_velocity_wgsl;
mod narrowphase_wgsl;
mod predict_wgsl;
mod warmstart_wgsl;

pub use avbd_solve_wgsl::{
    avbd_persistent_wgsl, AVBD_DUAL_WGSL, AVBD_PRIMAL_WGSL, CONTACT_PREP_WGSL, RESTITUTION_WGSL,
};
pub use coloring_wgsl::{
    COLORING_APPLY_WGSL, COLORING_RESET_WGSL, COLORING_STEP_WGSL, COLOR_COUNT_WGSL,
    COLOR_SCAN_ARGS_WGSL, COLOR_SCATTER_WGSL,
};
pub use contact_stream_wgsl::{
    ADJ_CLEAR_WGSL, ADJ_COUNT_WGSL, ADJ_INCIDENCE_WGSL, ADJ_RANGES_WGSL, SORT_GATHER_WGSL,
    SORT_IOTA_WGSL, SORT_KEY_EXTRACT_WGSL,
};
pub use extract_velocity_wgsl::EXTRACT_VELOCITY_WGSL;
pub use narrowphase_wgsl::{NARROWPHASE_ARGS_WGSL, NARROWPHASE_WGSL};
pub use predict_wgsl::PREDICT_WGSL;
pub use warmstart_wgsl::{warmstart_map_insert_wgsl, warmstart_match_wgsl};

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};
use rubble_gpu::{
    round_up_workgroups, BroadphaseBreakdownMs, ComputeKernel, GpuAtomicCounter, GpuBuffer,
    GpuContext, PingPongBuffer,
};
use rubble_math::{
    Aabb3D, BodyHandle, CollisionEvent, Contact3D, RigidBodyProps3D, RigidBodyState3D,
};
use rubble_primitives::{GpuLbvh, GpuPrefixScan, GpuRadixSort, RadixRecordTemps, ScanRecordTemps};
use rubble_shapes3d::{
    BoxData, CapsuleData, CompoundChildGpu, CompoundShapeGpu, ConvexHullData, ConvexVertex3D,
    SphereData,
};
use std::collections::HashSet;
const WORKGROUP_SIZE: u32 = 64;
/// Scenes at or below this body count solve in ONE persistent-kernel dispatch
/// (D8/W3); 256 = WebGPU's guaranteed max workgroup size.
const PERSISTENT_MAX_BODIES: u32 = 256;
const AVBD_WARMSTART_ALPHA: f32 = 0.99;
/// Number of Luby coloring iterations. O(log n) rounds suffices for bounded-degree
/// graphs. 32 handles any practical contact graph with room to spare.
const COLORING_MAX_ITERATIONS: u32 = 32;
const MAX_CONTACT_PENALTY: f32 = 1.0e6;

#[derive(Default)]
struct CachedBindGroup<K> {
    key: Option<K>,
    bind_group: Option<wgpu::BindGroup>,
}

#[derive(Default)]
struct CachedBindGroupVec<K> {
    key: Option<K>,
    bind_groups: Vec<wgpu::BindGroup>,
}

// ---------------------------------------------------------------------------
// GPU-side uniform structs
// ---------------------------------------------------------------------------

/// GPU simulation parameters. Must match the WGSL `SimParams` layout exactly.
///
/// Shaders that don't consume the trailing `solver2` field may declare the
/// 64-byte prefix only — WGSL uniform binding requires the buffer to be at
/// least the struct size, and trailing extra bytes are legal.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SimParamsGpu {
    pub gravity: [f32; 4],
    pub solver: [f32; 4], // (dt, beta, k_start, k_max_normal)
    pub counts: [u32; 4], // (num_bodies, solver_iterations, pair_count*, contact_count*)
    // *counts.z and counts.w are patched ON GPU each frame (buffer copies from
    //  the clamped pair/contact counters) — the CPU shadow values are stale.
    /// Narrowphase/solver quality knobs:
    /// x = contact_offset (prediction distance for speculative contacts)
    /// y = restitution_threshold (velocity below which restitution = 0)
    /// z = penetration_slop (allowable penetration before correction)
    /// w = contact_stabilization (AVBD alpha)
    pub quality: [f32; 4],
    /// x = k_max_tangential (friction stiffness cap)
    /// y = max_speed (post-solve linear velocity clamp, 0 = off)
    /// z = max_ang_speed (post-solve angular velocity clamp, 0 = off)
    /// w = dynamic_mu_scale (dynamic friction = static friction * this)
    pub solver2: [f32; 4],
}

/// Broadphase pair (two body indices).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuPair {
    pub a: u32,
    pub b: u32,
}

/// Solve range for graph-colored dispatch: (offset, count) into the sorted contact buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SolveRangeGpu {
    pub offset: u32,
    pub count: u32,
}

/// Uniform buffer layout for plane data in the narrowphase shader.
/// Uses a uniform binding instead of storage to stay within the
/// `maxStorageBuffersPerShaderStage` limit on constrained adapters (e.g. SwiftShader).
const MAX_PLANES: usize = 16;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PlaneParamsGpu {
    pub num_planes: u32,
    pub _pad: [u32; 3],
    pub planes: [[f32; 4]; MAX_PLANES],
}

/// GPU warmstart parameters. Must match the WGSL `WarmstartParams` layout exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct WarmstartParamsGpu {
    pub prev_count: u32,
    pub new_count: u32,
    pub alpha: f32,
    pub gamma: f32,
    /// `map_cap - 1` (capacity is a power of two) for bucket indexing (W4a-3).
    pub cap_mask: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

// ---------------------------------------------------------------------------
// Compile-time GPU layout validation
// ---------------------------------------------------------------------------

const _: () = assert!(std::mem::size_of::<SimParamsGpu>() == 80);
const _: () = assert!(std::mem::size_of::<GpuPair>() == 8);
const _: () = assert!(std::mem::size_of::<SolveRangeGpu>() == 8);
const _: () = assert!(std::mem::size_of::<WarmstartParamsGpu>() == 32);

// ---------------------------------------------------------------------------
// AABB compute shader
// ---------------------------------------------------------------------------

const AABB_COMPUTE_WGSL: &str = r#"
struct Body {
    position_inv_mass: vec4<f32>,
    orientation:       vec4<f32>,
    lin_vel:           vec4<f32>,
    ang_vel:           vec4<f32>,
};

struct BodyProps {
    inv_inertia_row0: vec4<f32>,
    inv_inertia_row1: vec4<f32>,
    inv_inertia_row2: vec4<f32>,
    friction:         f32,
    shape_type:       u32,
    shape_index:      u32,
    flags:            u32,
};

struct SphereData {
    radius: f32,
    _pad0:  f32,
    _pad1:  f32,
    _pad2:  f32,
};

struct BoxDataGpu {
    half_extents: vec4<f32>,
};

struct Aabb {
    min_pt: vec4<f32>,
    max_pt: vec4<f32>,
};

struct SimParams {
    gravity: vec4<f32>,
    solver:  vec4<f32>,
    counts:  vec4<u32>,
    quality: vec4<f32>,
};

const SHAPE_SPHERE:      u32 = 0u;
const SHAPE_BOX:         u32 = 1u;
const SHAPE_CAPSULE:     u32 = 2u;
const SHAPE_CONVEX_HULL: u32 = 3u;
const SHAPE_PLANE:       u32 = 4u;

struct ConvexHullInfo {
    vertex_offset: u32,
    vertex_count:  u32,
    face_offset:   u32,
    face_count:    u32,
    edge_offset:   u32,
    edge_count:    u32,
    _pad0: u32,
    _pad1: u32,
};

struct ConvexVert {
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
};

struct CapsuleDataGpu {
    half_height: f32,
    radius: f32,
    _pad0: f32,
    _pad1: f32,
};

// Plane stored as vec4(nx, ny, nz, distance)

@group(0) @binding(0) var<storage, read>       bodies:       array<Body>;
@group(0) @binding(1) var<storage, read>       props:        array<BodyProps>;
@group(0) @binding(2) var<storage, read>       spheres:      array<SphereData>;
@group(0) @binding(3) var<storage, read>       boxes:        array<BoxDataGpu>;
@group(0) @binding(4) var<storage, read_write> aabbs:        array<Aabb>;
@group(0) @binding(5) var<uniform>             params:       SimParams;
@group(0) @binding(6) var<storage, read>       convex_hulls: array<ConvexHullInfo>;
@group(0) @binding(7) var<storage, read>       convex_verts: array<ConvexVert>;
@group(0) @binding(8) var<storage, read>       capsules:     array<CapsuleDataGpu>;
@group(0) @binding(9) var<storage, read>       plane_data:   array<vec4<f32>>;

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u
         + (s * s - dot(u, u)) * v
         + 2.0 * s * cross(u, v);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.counts.x {
        return;
    }

    let pos = bodies[idx].position_inv_mass.xyz;
    let rot = bodies[idx].orientation;
    let st = props[idx].shape_type;
    let si = props[idx].shape_index;

    var aabb_min: vec3<f32>;
    var aabb_max: vec3<f32>;

    if st == SHAPE_SPHERE {
        let r = spheres[si].radius;
        aabb_min = pos - vec3<f32>(r, r, r);
        aabb_max = pos + vec3<f32>(r, r, r);
    } else if st == SHAPE_BOX {
        let he = boxes[si].half_extents.xyz;
        let ax = abs(quat_rotate(rot, vec3<f32>(he.x, 0.0, 0.0)));
        let ay = abs(quat_rotate(rot, vec3<f32>(0.0, he.y, 0.0)));
        let az = abs(quat_rotate(rot, vec3<f32>(0.0, 0.0, he.z)));
        let extent = ax + ay + az;
        aabb_min = pos - extent;
        aabb_max = pos + extent;
    } else if st == SHAPE_CAPSULE {
        let cap = capsules[si];
        let hh = cap.half_height;
        let r = cap.radius;
        // Capsule axis is local Y. Rotate to world space.
        let world_axis = quat_rotate(rot, vec3<f32>(0.0, hh, 0.0));
        let a = pos + world_axis;
        let b = pos - world_axis;
        let rv = vec3<f32>(r, r, r);
        aabb_min = min(a, b) - rv;
        aabb_max = max(a, b) + rv;
    } else if st == SHAPE_CONVEX_HULL {
        let hull = convex_hulls[si];
        var mn = vec3<f32>(1e30, 1e30, 1e30);
        var mx = vec3<f32>(-1e30, -1e30, -1e30);
        for (var vi = 0u; vi < hull.vertex_count; vi = vi + 1u) {
            let cv = convex_verts[hull.vertex_offset + vi];
            let local_v = vec3<f32>(cv.x, cv.y, cv.z);
            let world_v = pos + quat_rotate(rot, local_v);
            mn = min(mn, world_v);
            mx = max(mx, world_v);
        }
        aabb_min = mn;
        aabb_max = mx;
    } else if st == SHAPE_PLANE {
        // Plane: huge AABB so it pairs with everything
        let pdata = plane_data[si];
        let n = pdata.xyz;
        let d = pdata.w;
        // Center the AABB on the plane's nearest point to origin
        let center = n * d;
        aabb_min = center - vec3<f32>(1e4, 1e4, 1e4);
        aabb_max = center + vec3<f32>(1e4, 1e4, 1e4);
    } else {
        // SHAPE_COMPOUND or unknown: use a conservative large AABB.
        // The CPU-side compound expansion will handle precise child overlap testing.
        aabb_min = pos - vec3<f32>(100.0, 100.0, 100.0);
        aabb_max = pos + vec3<f32>(100.0, 100.0, 100.0);
    }

    aabbs[idx].min_pt = vec4<f32>(aabb_min, 0.0);
    aabbs[idx].max_pt = vec4<f32>(aabb_max, 0.0);
}
"#;

const FREE_MOTION_WGSL: &str = r#"
struct Body {
    position_inv_mass: vec4<f32>,
    orientation:       vec4<f32>,
    lin_vel:           vec4<f32>,
    ang_vel:           vec4<f32>,
};

struct SimParams {
    gravity: vec4<f32>,
    solver:  vec4<f32>,
    counts:  vec4<u32>,
    quality: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> bodies:          array<Body>;
@group(0) @binding(1) var<storage, read>       inertial_states: array<Body>;
@group(0) @binding(2) var<storage, read>       active_bodies:   array<u32>;
@group(0) @binding(3) var<uniform>             params:          SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.counts.x {
        return;
    }
    if active_bodies[idx] == 0u {
        bodies[idx] = inertial_states[idx];
    }
}
"#;

// ---------------------------------------------------------------------------
// GpuPipeline
// ---------------------------------------------------------------------------

/// Orchestrates GPU compute dispatches for the full 3D AVBD physics step.
///
/// The pipeline uses velocity-based AVBD (Averaged Velocity-Based Dynamics):
/// 1. Predict positions from velocities + gravity
/// 2. Compute AABBs from predicted positions
/// 3. Broadphase: hybrid LBVH broadphase (GPU kernels with current CPU staging/readback)
/// 4. Narrowphase: generate contacts from overlapping pairs
/// 5. AVBD solve: apply velocity impulses using averaged velocities
/// 6. Extract velocities from position changes
pub struct GpuPipeline {
    ctx: GpuContext,

    // Kernels
    predict_kernel: ComputeKernel,
    aabb_kernel: ComputeKernel,
    narrowphase_kernel: ComputeKernel,
    free_motion_kernel: ComputeKernel,
    primal_kernel: ComputeKernel,
    dual_kernel: ComputeKernel,
    extract_kernel: ComputeKernel,
    /// Warm-start pair-map lookup (new contacts -> cached solver state) (W4a-3).
    warmstart_kernel: ComputeKernel,
    /// Warm-start pair-map builder (prev contacts -> map) (W4a-3).
    warmstart_insert_kernel: ComputeKernel,
    contact_prep_kernel: ComputeKernel,
    restitution_kernel: ComputeKernel,

    // Storage buffers
    body_states: PingPongBuffer<RigidBodyState3D>,
    /// Async snapshot ring for post-extract body states (W4b, wasm only): the
    /// step enqueues a non-blocking copy and returns the most recent completed
    /// snapshot (~1 frame lag) instead of blocking on a `mapAsync` round trip.
    #[cfg(target_arch = "wasm32")]
    states_ring: rubble_gpu::SnapshotRing<RigidBodyState3D>,
    body_props: GpuBuffer<RigidBodyProps3D>,
    old_states: GpuBuffer<RigidBodyState3D>,
    prev_step_states: GpuBuffer<RigidBodyState3D>,
    inertial_states: GpuBuffer<RigidBodyState3D>,
    /// Snapshot of body states right after predict — the pose the narrowphase
    /// saw and the frame the Taylor-cached constraint Jacobians are frozen in.
    solve_start_states: GpuBuffer<RigidBodyState3D>,
    /// Immutable copy of post-extract velocities for the Jacobi restitution pass.
    restitution_snapshot: GpuBuffer<RigidBodyState3D>,
    aabbs: GpuBuffer<Aabb3D>,
    contacts: GpuBuffer<Contact3D>,
    contact_count: GpuAtomicCounter,
    pairs: GpuBuffer<GpuPair>,
    spheres: GpuBuffer<SphereData>,
    boxes: GpuBuffer<BoxData>,
    capsules: GpuBuffer<CapsuleData>,
    convex_hulls: GpuBuffer<ConvexHullData>,
    convex_vertices: GpuBuffer<ConvexVertex3D>,
    planes: GpuBuffer<Vec4>,
    body_order: GpuBuffer<u32>,
    body_contact_ranges: GpuBuffer<[u32; 2]>,
    body_contact_indices: GpuBuffer<u32>,
    active_body_flags: GpuBuffer<u32>,
    lbvh_subset_aabbs: GpuBuffer<Aabb3D>,

    // GPU graph coloring
    gpu_coloring: GpuColoringState,

    // GPU contact stream (deterministic sort + CSR adjacency)
    contact_stream: ContactStreamState,

    // Persistent contact buffers for GPU-side warmstarting
    prev_contacts: GpuBuffer<Contact3D>,
    prev_contact_count: u32,
    /// GPU-resident previous-frame solved contact count (W5): lets the warm-start
    /// insert kernel clamp to the true count without a CPU count readback, so the
    /// web hot path never blocks on `contact_count`.
    prev_contact_count_buf: GpuBuffer<u32>,
    warmstart_params_uniform: wgpu::Buffer,
    warmstart_bg_cache: CachedBindGroup<[u64; 3]>,
    warmstart_insert_bg_cache: CachedBindGroup<[u64; 3]>,
    /// Warm-start pair map (W4a-3): open-addressing, power-of-two capacity.
    /// `ws_map_hash[b]` = 0 (empty) or a nonzero key hash; `ws_map_entry[b]` =
    /// (lo, hi, feature_id, prev_contact_index).
    ws_map_hash: GpuBuffer<u32>,
    ws_map_entry: GpuBuffer<[u32; 4]>,
    ws_map_cap: u32,

    // Compound shape data (for CPU-side pair expansion)
    compound_shapes_data: Vec<CompoundShapeGpu>,
    compound_children_data: Vec<CompoundChildGpu>,
    sphere_data_cpu: Vec<SphereData>,
    box_data_cpu: Vec<BoxData>,
    capsule_data_cpu: Vec<CapsuleData>,
    convex_hulls_cpu: Vec<ConvexHullData>,
    convex_vertices_cpu: Vec<ConvexVertex3D>,
    /// CPU-side compound shapes with BVH data for broadphase culling.
    compound_shapes_cpu: Vec<rubble_shapes3d::CompoundShape>,
    plane_data_cpu: Vec<Vec4>,
    body_props_cpu: Vec<RigidBodyProps3D>,

    // Uniform buffers
    sim_params: SimParamsGpu,
    warmstart_decay: f32,
    params_uniform: wgpu::Buffer,
    solve_range_buffers: Vec<wgpu::Buffer>,
    /// One-shot init guard for the static color-index uniforms (see
    /// `ensure_color_index_uniforms`).
    color_uniforms_written: bool,
    /// Indirect narrowphase (W1): clamps the GPU pair demand, writes dispatch
    /// args, and stages the clamped count for a copy into `counts.z` — no CPU
    /// pair-count readback on any target.
    narrowphase_args_kernel: ComputeKernel,
    narrowphase_dispatch_args: wgpu::Buffer,
    clamped_pair_count: wgpu::Buffer,
    narrowphase_args_params: wgpu::Buffer,
    /// Single-workgroup persistent solve (D8/W3): the whole iterate loop in
    /// one dispatch for scenes with <= 256 bodies.
    persistent_kernel: ComputeKernel,
    /// Uniform for the contact-count patch (args kernel input: x = capacity).
    contact_args_params: wgpu::Buffer,
    /// Uniform buffer for plane data in narrowphase (avoids a storage binding).
    plane_params_uniform: wgpu::Buffer,

    // Cached bind groups (reused while backing buffers stay stable).
    predict_bg_cache: CachedBindGroup<[u64; 4]>,
    aabb_bg_cache: CachedBindGroup<[u64; 9]>,
    narrowphase_bg_cache: CachedBindGroup<[u64; 9]>,
    free_motion_bg_cache: CachedBindGroup<[u64; 4]>,
    primal_bg_cache: CachedBindGroupVec<[u64; 8]>,
    persistent_bg_cache: CachedBindGroup<[u64; 9]>,
    dual_bg_cache: CachedBindGroup<[u64; 3]>,
    extract_bg_cache: CachedBindGroup<[u64; 3]>,
    contact_prep_bg_cache: CachedBindGroup<[u64; 2]>,
    restitution_bg_cache: CachedBindGroup<[u64; 5]>,

    /// True when any uploaded body has a nonzero restitution coefficient
    /// (packed in `props.inv_inertia_row2.w`); gates the restitution pass.
    any_restitution: bool,
    /// Normal-direction penalty stiffness cap (config `k_max_normal`).
    k_max_normal: f32,

    // GPU broadphase
    gpu_lbvh: GpuLbvh,
}

/// GPU coloring state for Luby-style parallel body graph coloring.
struct GpuColoringState {
    reset_kernel: ComputeKernel,
    step_kernel: ComputeKernel,
    apply_kernel: ComputeKernel,
    color_count_kernel: ComputeKernel,
    color_scan_args_kernel: ComputeKernel,
    color_scatter_kernel: ComputeKernel,
    body_colors: GpuBuffer<u32>,
    body_priorities: GpuBuffer<u32>,
    /// Per-round winner flags (mark phase output, apply phase input): keeps
    /// color reads and writes in separate dispatches — see COLORING_STEP_WGSL.
    winners: GpuBuffer<u32>,
    params_buf: wgpu::Buffer,
    /// Per-round uniforms (round index = current color) so the whole Luby
    /// iteration records into one submit instead of one submit per round.
    round_params: Vec<wgpu::Buffer>,
    round_step_bgs: Vec<wgpu::BindGroup>,
    round_apply_bgs: Vec<wgpu::BindGroup>,
    /// Cache key for the per-round bind groups (buffer identities via sizes).
    round_bg_key: Option<[u64; 5]>,
    /// Per-color body counts (compaction pass); doubles as scatter cursors.
    color_counts: GpuBuffer<u32>,
    /// Per-color indirect dispatch args (ceil(count/64), 1, 1) — 32 entries.
    dispatch_args: wgpu::Buffer,
    frame_counter: u32,
}

impl GpuColoringState {
    fn new(ctx: &GpuContext, max_bodies: usize) -> Self {
        let reset_kernel =
            ComputeKernel::from_wgsl(ctx, coloring_wgsl::COLORING_RESET_WGSL, "main");
        let step_kernel = ComputeKernel::from_wgsl(ctx, coloring_wgsl::COLORING_STEP_WGSL, "main");
        let apply_kernel =
            ComputeKernel::from_wgsl(ctx, coloring_wgsl::COLORING_APPLY_WGSL, "main");
        let color_count_kernel =
            ComputeKernel::from_wgsl(ctx, coloring_wgsl::COLOR_COUNT_WGSL, "main");
        let color_scan_args_kernel =
            ComputeKernel::from_wgsl(ctx, coloring_wgsl::COLOR_SCAN_ARGS_WGSL, "main");
        let color_scatter_kernel =
            ComputeKernel::from_wgsl(ctx, coloring_wgsl::COLOR_SCATTER_WGSL, "main");
        let body_colors = GpuBuffer::new(ctx, max_bodies.max(1));
        let body_priorities = GpuBuffer::new(ctx, max_bodies.max(1));
        let winners = GpuBuffer::new(ctx, max_bodies.max(1));
        let params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coloring params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // One tiny uniform per Luby round so all rounds record into a single
        // encoder/submit (the round's `current_color` differs per dispatch).
        let round_params = (0..COLORING_MAX_ITERATIONS)
            .map(|_| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("coloring round params"),
                    size: 16,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let color_counts = GpuBuffer::new(ctx, COLORING_MAX_ITERATIONS as usize);
        let dispatch_args = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("per-color dispatch args"),
            size: COLORING_MAX_ITERATIONS as u64 * 12,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            reset_kernel,
            step_kernel,
            apply_kernel,
            color_count_kernel,
            color_scan_args_kernel,
            color_scatter_kernel,
            body_colors,
            body_priorities,
            winners,
            params_buf,
            round_params,
            round_step_bgs: Vec::new(),
            round_apply_bgs: Vec::new(),
            round_bg_key: None,
            color_counts,
            dispatch_args,
            frame_counter: 0,
        }
    }
}

/// GPU contact-stream state (M2(d)): deterministic sort + CSR adjacency,
/// replacing the per-frame CPU contact download/sort/adjacency build.
struct ContactStreamState {
    key_extract_kernel: ComputeKernel,
    iota_kernel: ComputeKernel,
    gather_kernel: ComputeKernel,
    adj_clear_kernel: ComputeKernel,
    adj_count_kernel: ComputeKernel,
    adj_ranges_kernel: ComputeKernel,
    adj_incidence_kernel: ComputeKernel,
    sort_keys: GpuBuffer<u32>,
    sort_vals: GpuBuffer<u32>,
    contacts_scratch: GpuBuffer<Contact3D>,
    adj_counts: GpuBuffer<u32>,
    adj_offsets: GpuBuffer<u32>,
    incidence_keys: GpuBuffer<u32>,
    incidence_vals: GpuBuffer<u32>,
    /// Per-stage tiny uniforms (count / word-selector), rewritten per use.
    params_bufs: Vec<wgpu::Buffer>,
    radix: GpuRadixSort,
    radix_temps: RadixRecordTemps,
    scan: GpuPrefixScan,
    scan_temps: ScanRecordTemps,
}

impl ContactStreamState {
    fn new(ctx: &GpuContext, max_bodies: usize, max_contacts: usize) -> Self {
        let mk_params = || {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("contact stream params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            key_extract_kernel: ComputeKernel::from_wgsl(
                ctx,
                contact_stream_wgsl::SORT_KEY_EXTRACT_WGSL,
                "main",
            ),
            iota_kernel: ComputeKernel::from_wgsl(ctx, contact_stream_wgsl::SORT_IOTA_WGSL, "main"),
            gather_kernel: ComputeKernel::from_wgsl(
                ctx,
                contact_stream_wgsl::SORT_GATHER_WGSL,
                "main",
            ),
            adj_clear_kernel: ComputeKernel::from_wgsl(
                ctx,
                contact_stream_wgsl::ADJ_CLEAR_WGSL,
                "main",
            ),
            adj_count_kernel: ComputeKernel::from_wgsl(
                ctx,
                contact_stream_wgsl::ADJ_COUNT_WGSL,
                "main",
            ),
            adj_ranges_kernel: ComputeKernel::from_wgsl(
                ctx,
                contact_stream_wgsl::ADJ_RANGES_WGSL,
                "main",
            ),
            adj_incidence_kernel: ComputeKernel::from_wgsl(
                ctx,
                contact_stream_wgsl::ADJ_INCIDENCE_WGSL,
                "main",
            ),
            sort_keys: GpuBuffer::new(ctx, max_contacts.max(1)),
            sort_vals: GpuBuffer::new(ctx, max_contacts.max(1)),
            contacts_scratch: GpuBuffer::new(ctx, max_contacts.max(1)),
            adj_counts: GpuBuffer::new(ctx, max_bodies.max(1)),
            adj_offsets: GpuBuffer::new(ctx, max_bodies.max(1)),
            incidence_keys: GpuBuffer::new(ctx, (2 * max_contacts).max(1)),
            incidence_vals: GpuBuffer::new(ctx, (2 * max_contacts).max(1)),
            params_bufs: (0..8).map(|_| mk_params()).collect(),
            radix: GpuRadixSort::new(ctx, (2 * max_contacts).max(1)),
            radix_temps: RadixRecordTemps::new(ctx, (2 * max_contacts).max(1)),
            scan: GpuPrefixScan::new(ctx, max_bodies.max(1)),
            scan_temps: ScanRecordTemps::new(ctx),
        }
    }
}

impl GpuPipeline {
    /// Create a new GPU pipeline. Compiles all shaders and allocates buffers.
    pub fn new(ctx: GpuContext, max_bodies: usize) -> Self {
        let max_pairs = max_bodies * 8;
        let max_contacts = max_bodies * 8;

        let predict_kernel = ComputeKernel::from_wgsl(&ctx, PREDICT_WGSL, "main");
        let aabb_kernel = ComputeKernel::from_wgsl(&ctx, AABB_COMPUTE_WGSL, "main");
        let narrowphase_kernel = ComputeKernel::from_wgsl(&ctx, NARROWPHASE_WGSL, "main");
        let narrowphase_args_kernel = ComputeKernel::from_wgsl(&ctx, NARROWPHASE_ARGS_WGSL, "main");
        let persistent_kernel = ComputeKernel::from_wgsl(&ctx, &avbd_persistent_wgsl(), "main");
        let free_motion_kernel = ComputeKernel::from_wgsl(&ctx, FREE_MOTION_WGSL, "main");
        let primal_kernel = ComputeKernel::from_wgsl(&ctx, AVBD_PRIMAL_WGSL, "main");
        let dual_kernel = ComputeKernel::from_wgsl(&ctx, AVBD_DUAL_WGSL, "main");
        let extract_kernel = ComputeKernel::from_wgsl(&ctx, EXTRACT_VELOCITY_WGSL, "main");
        let warmstart_kernel = ComputeKernel::from_wgsl(&ctx, &warmstart_match_wgsl(), "main");
        let warmstart_insert_kernel =
            ComputeKernel::from_wgsl(&ctx, &warmstart_map_insert_wgsl(), "main");
        let contact_prep_kernel = ComputeKernel::from_wgsl(&ctx, CONTACT_PREP_WGSL, "main");
        let restitution_kernel = ComputeKernel::from_wgsl(&ctx, RESTITUTION_WGSL, "main");

        let body_states = PingPongBuffer::new(&ctx, max_bodies);
        #[cfg(target_arch = "wasm32")]
        let states_ring = rubble_gpu::SnapshotRing::new(&ctx, max_bodies);
        let body_props = GpuBuffer::new(&ctx, max_bodies);
        let old_states = GpuBuffer::new(&ctx, max_bodies);
        let prev_step_states = GpuBuffer::new(&ctx, max_bodies);
        let inertial_states = GpuBuffer::new(&ctx, max_bodies);
        let solve_start_states = GpuBuffer::new(&ctx, max_bodies);
        let restitution_snapshot = GpuBuffer::new(&ctx, max_bodies);
        let aabbs = GpuBuffer::new(&ctx, max_bodies);
        let contacts = GpuBuffer::new(&ctx, max_contacts);
        let contact_count = GpuAtomicCounter::new(&ctx);
        let pairs = GpuBuffer::new(&ctx, max_pairs);
        let spheres = GpuBuffer::new(&ctx, max_bodies.max(1));
        let boxes = GpuBuffer::new(&ctx, max_bodies.max(1));
        let capsules = GpuBuffer::new(&ctx, max_bodies.max(1));
        let convex_hulls = GpuBuffer::new(&ctx, max_bodies.max(1));
        let convex_vertices = GpuBuffer::new(&ctx, (max_bodies * 8).max(1));
        let planes = GpuBuffer::new(&ctx, 16);
        let body_order = GpuBuffer::new(&ctx, max_bodies);
        let body_contact_ranges = GpuBuffer::new(&ctx, max_bodies.max(1));
        let body_contact_indices = GpuBuffer::new(&ctx, (max_contacts * 2).max(1));
        let active_body_flags = GpuBuffer::new(&ctx, max_bodies.max(1));
        let lbvh_subset_aabbs = GpuBuffer::new(&ctx, max_bodies.max(1));

        let params_uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SimParams uniform"),
            size: std::mem::size_of::<SimParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let narrowphase_dispatch_args = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("narrowphase dispatch args"),
            size: 12,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let clamped_pair_count = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("clamped pair count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let narrowphase_args_params = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("narrowphase args params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let contact_args_params = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("contact args params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let plane_params_uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PlaneParams uniform"),
            size: std::mem::size_of::<PlaneParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let prev_contacts: GpuBuffer<Contact3D> = GpuBuffer::new(&ctx, max_contacts);
        let mut prev_contact_count_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1);
        prev_contact_count_buf.set_len(1);
        let warmstart_params_uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WarmstartParams uniform"),
            size: std::mem::size_of::<WarmstartParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Warm-start pair map (W4a-3): power-of-two capacity >= 2*max_contacts
        // keeps the open-addressing load factor <= 0.5.
        let ws_map_cap = (2 * max_contacts.max(1)).next_power_of_two() as u32;
        let ws_map_hash: GpuBuffer<u32> = GpuBuffer::new(&ctx, ws_map_cap as usize);
        let ws_map_entry: GpuBuffer<[u32; 4]> = GpuBuffer::new(&ctx, ws_map_cap as usize);

        let gpu_lbvh = GpuLbvh::new(&ctx, max_bodies);
        let gpu_coloring = GpuColoringState::new(&ctx, max_bodies);
        let contact_stream = ContactStreamState::new(&ctx, max_bodies, max_contacts);

        Self {
            ctx,
            predict_kernel,
            aabb_kernel,
            narrowphase_kernel,
            narrowphase_args_kernel,
            free_motion_kernel,
            primal_kernel,
            dual_kernel,
            extract_kernel,
            warmstart_kernel,
            warmstart_insert_kernel,
            contact_prep_kernel,
            restitution_kernel,
            body_states,
            #[cfg(target_arch = "wasm32")]
            states_ring,
            body_props,
            old_states,
            prev_step_states,
            inertial_states,
            solve_start_states,
            restitution_snapshot,
            aabbs,
            contacts,
            contact_count,
            pairs,
            spheres,
            boxes,
            capsules,
            convex_hulls,
            convex_vertices,
            planes,
            body_order,
            body_contact_ranges,
            body_contact_indices,
            active_body_flags,
            lbvh_subset_aabbs,
            gpu_coloring,
            contact_stream,
            prev_contacts,
            prev_contact_count: 0,
            prev_contact_count_buf,
            warmstart_params_uniform,
            warmstart_bg_cache: CachedBindGroup::default(),
            warmstart_insert_bg_cache: CachedBindGroup::default(),
            ws_map_hash,
            ws_map_entry,
            ws_map_cap,
            compound_shapes_data: Vec::new(),
            compound_children_data: Vec::new(),
            sphere_data_cpu: Vec::new(),
            box_data_cpu: Vec::new(),
            capsule_data_cpu: Vec::new(),
            convex_hulls_cpu: Vec::new(),
            convex_vertices_cpu: Vec::new(),
            compound_shapes_cpu: Vec::new(),
            plane_data_cpu: Vec::new(),
            body_props_cpu: Vec::new(),
            sim_params: SimParamsGpu {
                gravity: [0.0; 4],
                solver: [0.0, 10.0, 1.0e4, MAX_CONTACT_PENALTY],
                counts: [0, 0, 0, 0],
                quality: [0.02, 0.5, 0.005, 0.95], // contact_offset, restitution_threshold, penetration_slop, contact_stabilization (alpha)
                solver2: [1.0e6, 100.0, 64.0, 1.0], // k_max_tangential, max_speed, max_ang_speed, dynamic_mu_scale
            },
            warmstart_decay: 0.95,
            params_uniform,
            solve_range_buffers: Vec::new(),
            color_uniforms_written: false,
            narrowphase_dispatch_args,
            clamped_pair_count,
            narrowphase_args_params,
            persistent_kernel,
            contact_args_params,
            plane_params_uniform,
            predict_bg_cache: CachedBindGroup::default(),
            aabb_bg_cache: CachedBindGroup::default(),
            narrowphase_bg_cache: CachedBindGroup::default(),
            free_motion_bg_cache: CachedBindGroup::default(),
            primal_bg_cache: CachedBindGroupVec::default(),
            persistent_bg_cache: CachedBindGroup::default(),
            dual_bg_cache: CachedBindGroup::default(),
            extract_bg_cache: CachedBindGroup::default(),
            contact_prep_bg_cache: CachedBindGroup::default(),
            restitution_bg_cache: CachedBindGroup::default(),
            any_restitution: false,
            k_max_normal: MAX_CONTACT_PENALTY,
            gpu_lbvh,
        }
    }

    /// Try to create a GPU pipeline (async). Returns None if no GPU adapter is available.
    pub async fn try_new_async(max_bodies: usize) -> Option<Self> {
        let ctx = GpuContext::new().await.ok()?;
        Some(Self::new(ctx, max_bodies))
    }

    /// Try to create a GPU pipeline. Returns None if no GPU adapter is available.
    /// Not available on WASM targets — use [`try_new_async`](Self::try_new_async) instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn try_new(max_bodies: usize) -> Option<Self> {
        pollster::block_on(GpuContext::new())
            .ok()
            .map(|ctx| Self::new(ctx, max_bodies))
    }

    /// Upload body data from CPU arrays to GPU buffers and set simulation params.
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        &mut self,
        states: &[RigidBodyState3D],
        prev_step_states: &[RigidBodyState3D],
        props: &[RigidBodyProps3D],
        sphere_data: &[SphereData],
        box_data: &[BoxData],
        capsule_data: &[CapsuleData],
        hull_data: &[ConvexHullData],
        hull_vertices: &[ConvexVertex3D],
        plane_data: &[Vec4],
        compound_shapes: &[CompoundShapeGpu],
        compound_children: &[CompoundChildGpu],
        compound_shapes_cpu: &[rubble_shapes3d::CompoundShape],
        gravity: Vec3,
        dt: f32,
        solver_iterations: u32,
        beta: f32,
        k_start: f32,
        warmstart_decay: f32,
    ) {
        // Store compound data for CPU-side pair expansion
        self.compound_shapes_data = compound_shapes.to_vec();
        self.compound_children_data = compound_children.to_vec();
        self.sphere_data_cpu = sphere_data.to_vec();
        self.box_data_cpu = box_data.to_vec();
        self.capsule_data_cpu = capsule_data.to_vec();
        self.convex_hulls_cpu = hull_data.to_vec();
        self.convex_vertices_cpu = hull_vertices.to_vec();
        self.compound_shapes_cpu = compound_shapes_cpu.to_vec();
        self.plane_data_cpu = plane_data.to_vec();
        self.body_props_cpu = props.to_vec();
        self.body_states.upload(&self.ctx, states);
        self.old_states.upload(&self.ctx, states);
        self.prev_step_states.upload(&self.ctx, prev_step_states);
        self.inertial_states.upload(&self.ctx, states);
        self.body_props.upload(&self.ctx, props);

        if !sphere_data.is_empty() {
            self.spheres.upload(&self.ctx, sphere_data);
        }
        if !box_data.is_empty() {
            self.boxes.upload(&self.ctx, box_data);
        }
        if !capsule_data.is_empty() {
            self.capsules.upload(&self.ctx, capsule_data);
        }
        if !hull_data.is_empty() {
            self.convex_hulls.upload(&self.ctx, hull_data);
        }
        if !hull_vertices.is_empty() {
            self.convex_vertices.upload(&self.ctx, hull_vertices);
        }
        if !plane_data.is_empty() {
            self.planes.upload(&self.ctx, plane_data);
        }

        // Write plane data as a uniform for the narrowphase (avoids a storage binding)
        {
            let mut pp = PlaneParamsGpu {
                num_planes: plane_data.len().min(MAX_PLANES) as u32,
                _pad: [0; 3],
                planes: [[0.0; 4]; MAX_PLANES],
            };
            for (i, pd) in plane_data.iter().enumerate().take(MAX_PLANES) {
                pp.planes[i] = [pd.x, pd.y, pd.z, pd.w];
            }
            self.ctx
                .queue
                .write_buffer(&self.plane_params_uniform, 0, bytemuck::bytes_of(&pp));
        }

        self.aabbs.grow_if_needed(&self.ctx, states.len());

        self.any_restitution = props.iter().any(|p| p.inv_inertia_row2.w > 0.0);
        self.warmstart_decay = warmstart_decay;
        self.sim_params = SimParamsGpu {
            gravity: [gravity.x, gravity.y, gravity.z, 0.0],
            solver: [dt, beta, k_start, self.k_max_normal],
            counts: [states.len() as u32, solver_iterations, 0, 0],
            quality: self.sim_params.quality, // preserve quality params
            solver2: self.sim_params.solver2, // preserve extended solver params
        };
        self.ctx.queue.write_buffer(
            &self.params_uniform,
            0,
            bytemuck::bytes_of(&self.sim_params),
        );
    }

    /// Per-frame parameter refresh for clean frames (no CPU-side body/shape
    /// mutations since the last upload): the GPU already holds bit-identical
    /// state — `body_states.current()` persists across steps — so only the
    /// simulation parameters need updating. Skips the full re-upload that
    /// used to dominate `upload_ms`.
    #[allow(clippy::too_many_arguments)]
    pub fn refresh_frame_params(
        &mut self,
        gravity: Vec3,
        dt: f32,
        solver_iterations: u32,
        beta: f32,
        k_start: f32,
        warmstart_decay: f32,
        num_bodies: u32,
    ) {
        self.warmstart_decay = warmstart_decay;
        self.sim_params.gravity = [gravity.x, gravity.y, gravity.z, 0.0];
        self.sim_params.solver = [dt, beta, k_start, self.k_max_normal];
        self.sim_params.counts[0] = num_bodies;
        self.sim_params.counts[1] = solver_iterations;
        self.ctx.queue.write_buffer(
            &self.params_uniform,
            0,
            bytemuck::bytes_of(&self.sim_params),
        );
    }

    /// Set narrowphase/solver quality tuning knobs.
    pub fn set_quality_params(
        &mut self,
        contact_offset: f32,
        restitution_threshold: f32,
        penetration_slop: f32,
        contact_stabilization: f32,
    ) {
        self.sim_params.quality = [
            contact_offset,
            restitution_threshold,
            penetration_slop,
            contact_stabilization,
        ];
        self.ctx.queue.write_buffer(
            &self.params_uniform,
            0,
            bytemuck::bytes_of(&self.sim_params),
        );
    }

    /// Set the extended solver parameters: penalty caps, velocity clamps, and
    /// the dynamic-friction ratio.
    pub fn set_solver_extras(
        &mut self,
        k_max_normal: f32,
        k_max_tangential: f32,
        max_speed: f32,
        max_ang_speed: f32,
        dynamic_mu_scale: f32,
    ) {
        self.k_max_normal = k_max_normal;
        self.sim_params.solver[3] = k_max_normal;
        self.sim_params.solver2 = [k_max_tangential, max_speed, max_ang_speed, dynamic_mu_scale];
        self.ctx.queue.write_buffer(
            &self.params_uniform,
            0,
            bytemuck::bytes_of(&self.sim_params),
        );
    }

    /// Run predict → AABB → broadphase → narrowphase (shared by both step variants).
    ///
    /// When a broadphase pair involves a compound shape (SHAPE_COMPOUND = 5),
    /// the pair is expanded on the CPU into individual child-vs-body pairs.
    /// This avoids adding compound-specific bindings/logic to the GPU narrowphase.
    fn run_detection(&mut self, num_bodies: u32) -> Vec<Contact3D> {
        self.contact_count.reset(&self.ctx);

        self.dispatch_predict_and_bounds(num_bodies);

        let has_compounds = self.body_props_cpu[..num_bodies as usize]
            .iter()
            .any(|prop| prop.shape_type == rubble_math::SHAPE_COMPOUND);
        // Planes go through GPU LBVH broadphase (their large AABB handles pairing).
        // Only compound shapes still require CPU pair expansion.
        let requires_cpu_pair_stage = has_compounds;
        self.aabbs.set_len(num_bodies);

        let mut cpu_compound_contacts: Vec<Contact3D> = Vec::new();
        let mut pair_thread_count: u32 = 0;
        let mut pair_count: u32 = 0;

        if requires_cpu_pair_stage {
            let cpu_aabbs = self.aabbs.download(&self.ctx);
            let overlap_pairs = self.broadphase_pairs_3d(num_bodies, &cpu_aabbs);
            if !overlap_pairs.is_empty() {
                let props = &self.body_props_cpu[..num_bodies as usize];
                let mut non_compound_pairs: Vec<GpuPair> = Vec::with_capacity(overlap_pairs.len());
                let states = if has_compounds {
                    self.body_states.current_mut().set_len(num_bodies);
                    Some(self.body_states.download(&self.ctx))
                } else {
                    None
                };

                for p in &overlap_pairs {
                    let a = p[0];
                    let b = p[1];
                    if has_compounds {
                        let st_a = props[a as usize].shape_type;
                        let st_b = props[b as usize].shape_type;

                        let a_is_compound = st_a == rubble_math::SHAPE_COMPOUND;
                        let b_is_compound = st_b == rubble_math::SHAPE_COMPOUND;

                        if !a_is_compound && !b_is_compound {
                            non_compound_pairs.push(GpuPair { a, b });
                        } else {
                            self.generate_compound_contacts_cpu(
                                a,
                                b,
                                props,
                                states.as_ref().expect("compound path requires states"),
                                &mut cpu_compound_contacts,
                            );
                        }
                    } else {
                        non_compound_pairs.push(GpuPair { a, b });
                    }
                }

                if !non_compound_pairs.is_empty() {
                    // Sort pairs by (shape_type_a << 16 | shape_type_b) for SIMD-friendly
                    // narrowphase dispatch. Bodies with the same shape-pair type are grouped
                    // together, improving GPU warp/wavefront coherence.
                    non_compound_pairs.sort_unstable_by_key(|pair| {
                        let st_a = props[pair.a as usize].shape_type;
                        let st_b = props[pair.b as usize].shape_type;
                        let (lo, hi) = if st_a <= st_b {
                            (st_a, st_b)
                        } else {
                            (st_b, st_a)
                        };
                        (lo << 16) | hi
                    });
                    let count = non_compound_pairs.len() as u32;
                    self.pairs.upload(&self.ctx, &non_compound_pairs);
                    pair_count = count;
                }
            }
        } else {
            pair_thread_count =
                self.gpu_lbvh
                    .query_on_device_raw(&self.ctx, self.aabbs.buffer(), num_bodies);
        }

        if requires_cpu_pair_stage {
            self.dispatch_narrowphase(num_bodies, pair_count);
        } else if pair_thread_count > 0 {
            // No pair-count readback: args kernel + uniform patch + indirect.
            let pair_buffer = self.gpu_lbvh.pair_buffer().clone();
            self.dispatch_narrowphase_indirect(&pair_buffer, pair_thread_count);
        }

        // Buffer overflow recovery: if contact demand exceeded capacity, grow
        // and retry (native-only same-frame heal; the emission clamp keeps the
        // pre-retry buffer contents valid either way).
        let contact_count_val = self.contact_count.read(&self.ctx);
        let capacity = self.contacts.capacity();
        if contact_count_val > capacity {
            // Grow contacts buffer to 2x the needed size
            let new_cap = (contact_count_val as usize) * 2;
            self.contacts.grow_if_needed(&self.ctx, new_cap);
            // Reset counter and re-run narrowphase
            self.contact_count.reset(&self.ctx);
            if requires_cpu_pair_stage {
                self.dispatch_narrowphase(num_bodies, pair_count);
            } else if pair_thread_count > 0 {
                let pair_buffer = self.gpu_lbvh.pair_buffer().clone();
                self.dispatch_narrowphase_indirect(&pair_buffer, pair_thread_count);
            }
        }

        cpu_compound_contacts
    }

    /// Generate contacts on the CPU for pairs involving compound shapes.
    /// For each child of the compound, compute a sphere-based proximity test
    /// and emit contacts using the parent body index (so the solver applies forces
    /// to the correct rigid body).
    fn generate_compound_contacts_cpu(
        &self,
        body_a: u32,
        body_b: u32,
        props: &[RigidBodyProps3D],
        states: &[RigidBodyState3D],
        out: &mut Vec<Contact3D>,
    ) {
        let st_a = props[body_a as usize].shape_type;
        let st_b = props[body_b as usize].shape_type;
        let si_a = props[body_a as usize].shape_index;
        let si_b = props[body_b as usize].shape_index;

        let pos_a = states[body_a as usize].position();
        let rot_a = states[body_a as usize].quat();
        let pos_b = states[body_b as usize].position();
        let rot_b = states[body_b as usize].quat();

        let a_is_compound = st_a == rubble_math::SHAPE_COMPOUND;
        let b_is_compound = st_b == rubble_math::SHAPE_COMPOUND;

        // Collect children for body A
        let children_a: Vec<(Vec3, glam::Quat, u32, u32)> = if a_is_compound {
            self.get_compound_children_world(si_a, pos_a, rot_a)
        } else {
            vec![(pos_a, rot_a, st_a, si_a)]
        };

        // Collect children for body B
        let children_b: Vec<(Vec3, glam::Quat, u32, u32)> = if b_is_compound {
            self.get_compound_children_world(si_b, pos_b, rot_b)
        } else {
            vec![(pos_b, rot_b, st_b, si_b)]
        };

        // Try BVH-accelerated culling when compound shapes have BVH data.
        let bvh_b = if b_is_compound {
            self.compound_shapes_cpu
                .get(si_b as usize)
                .filter(|cs| cs.bvh_nodes.len() >= 2)
        } else {
            None
        };
        let bvh_a = if a_is_compound {
            self.compound_shapes_cpu
                .get(si_a as usize)
                .filter(|cs| cs.bvh_nodes.len() >= 2)
        } else {
            None
        };

        // Build candidate child-pair indices using BVH when possible.
        let candidate_pairs: Vec<(usize, usize)> = if bvh_b.is_some() || bvh_a.is_some() {
            let mut pairs = Vec::new();
            if let Some(cs_b) = bvh_b {
                let inv_rot_b = rot_b.conjugate();
                for (idx_a, &(child_pos_a, _, child_st_a, child_si_a)) in
                    children_a.iter().enumerate()
                {
                    let ext_a = self.child_extent(child_st_a, child_si_a);
                    let world_min_a = child_pos_a - Vec3::splat(ext_a);
                    let world_max_a = child_pos_a + Vec3::splat(ext_a);
                    let local_min =
                        Self::transform_aabb_to_local(world_min_a, world_max_a, pos_b, inv_rot_b);
                    let local_max = Self::transform_aabb_to_local_max(
                        world_min_a,
                        world_max_a,
                        pos_b,
                        inv_rot_b,
                    );
                    for idx_b in Self::traverse_compound_bvh(&cs_b.bvh_nodes, local_min, local_max)
                    {
                        if idx_b < children_b.len() {
                            pairs.push((idx_a, idx_b));
                        }
                    }
                }
            } else if let Some(cs_a) = bvh_a {
                let inv_rot_a = rot_a.conjugate();
                for (idx_b, &(child_pos_b, _, child_st_b, child_si_b)) in
                    children_b.iter().enumerate()
                {
                    let ext_b = self.child_extent(child_st_b, child_si_b);
                    let world_min_b = child_pos_b - Vec3::splat(ext_b);
                    let world_max_b = child_pos_b + Vec3::splat(ext_b);
                    let local_min =
                        Self::transform_aabb_to_local(world_min_b, world_max_b, pos_a, inv_rot_a);
                    let local_max = Self::transform_aabb_to_local_max(
                        world_min_b,
                        world_max_b,
                        pos_a,
                        inv_rot_a,
                    );
                    for idx_a in Self::traverse_compound_bvh(&cs_a.bvh_nodes, local_min, local_max)
                    {
                        if idx_a < children_a.len() {
                            pairs.push((idx_a, idx_b));
                        }
                    }
                }
            }
            pairs
        } else {
            // Fallback: O(n*m) brute-force when no BVH is available.
            let mut all_pairs = Vec::with_capacity(children_a.len() * children_b.len());
            for idx_a in 0..children_a.len() {
                for idx_b in 0..children_b.len() {
                    all_pairs.push((idx_a, idx_b));
                }
            }
            all_pairs
        };

        // Test candidate child pairs with sphere-based proximity.
        for (idx_a, idx_b) in candidate_pairs {
            let (child_pos_a, _, child_st_a, child_si_a) = children_a[idx_a];
            let (child_pos_b, _, child_st_b, child_si_b) = children_b[idx_b];

            if child_st_a == rubble_math::SHAPE_PLANE || child_st_b == rubble_math::SHAPE_PLANE {
                let (
                    plane_parent,
                    plane_si,
                    dynamic_parent,
                    dynamic_pos,
                    dynamic_st,
                    dynamic_si,
                    dynamic_idx,
                ) = if child_st_a == rubble_math::SHAPE_PLANE {
                    (
                        body_a,
                        child_si_a,
                        body_b,
                        child_pos_b,
                        child_st_b,
                        child_si_b,
                        idx_b as u32,
                    )
                } else {
                    (
                        body_b,
                        child_si_b,
                        body_a,
                        child_pos_a,
                        child_st_a,
                        child_si_a,
                        idx_a as u32,
                    )
                };

                if let Some(plane) = self.plane_data_cpu.get(plane_si as usize) {
                    let plane_normal = Vec3::new(plane.x, plane.y, plane.z).normalize_or_zero();
                    let plane_dist = plane.w;
                    let extent = self.child_extent(dynamic_st, dynamic_si);
                    let signed_distance = plane_normal.dot(dynamic_pos) - plane_dist;
                    let depth = signed_distance - extent;
                    if depth > 0.0 {
                        continue;
                    }

                    let plane_point = dynamic_pos - plane_normal * signed_distance;
                    let tangent = if plane_normal.z.abs() > 0.707 {
                        plane_normal.cross(Vec3::Y).normalize_or_zero()
                    } else {
                        plane_normal.cross(Vec3::Z).normalize_or_zero()
                    };
                    let world_a = plane_point + plane_normal * depth;
                    let world_b = plane_point;
                    let point = (world_a + world_b) * 0.5;

                    let pos_dyn = states[dynamic_parent as usize].position();
                    let rot_dyn = states[dynamic_parent as usize].quat();
                    let pos_plane = states[plane_parent as usize].position();
                    let rot_plane = states[plane_parent as usize].quat();
                    let local_dyn = rot_dyn.conjugate() * (world_a - pos_dyn);
                    let local_plane = rot_plane.conjugate() * (world_b - pos_plane);
                    let feature_id = 0x4000_0000u32 | (dynamic_idx & 0xFFFF);

                    out.push(Contact3D {
                        point: Vec4::new(point.x, point.y, point.z, depth),
                        normal: Vec4::new(plane_normal.x, plane_normal.y, plane_normal.z, 0.0),
                        tangent: Vec4::new(tangent.x, tangent.y, tangent.z, 0.0),
                        local_anchor_a: Vec4::new(local_dyn.x, local_dyn.y, local_dyn.z, 0.0),
                        local_anchor_b: Vec4::new(local_plane.x, local_plane.y, local_plane.z, 0.0),
                        lambda: Vec4::ZERO,
                        penalty: Vec4::new(
                            self.sim_params.solver[2],
                            self.sim_params.solver[2],
                            self.sim_params.solver[2],
                            0.0,
                        ),
                        body_a: dynamic_parent,
                        body_b: plane_parent,
                        feature_id,
                        flags: 0,
                    });
                }
                continue;
            }

            let ext_a = self.child_extent(child_st_a, child_si_a);
            let ext_b = self.child_extent(child_st_b, child_si_b);

            let diff = child_pos_b - child_pos_a;
            let sum_ext = ext_a + ext_b;

            if diff.length_squared() > sum_ext * sum_ext {
                continue;
            }

            let dist = diff.length();
            if dist < 1e-12 {
                continue;
            }
            let normal = diff / dist;
            let depth = dist - sum_ext;
            if depth > 0.0 {
                continue;
            }
            let point = child_pos_a + normal * (ext_a + depth * 0.5);
            let tangent = if normal.z.abs() > 0.707 {
                normal.cross(Vec3::Y).normalize_or_zero()
            } else {
                normal.cross(Vec3::Z).normalize_or_zero()
            };
            let world_a = point + normal * depth * 0.5;
            let world_b = point - normal * depth * 0.5;
            let local_a = rot_a.conjugate() * (world_a - pos_a);
            let local_b = rot_b.conjugate() * (world_b - pos_b);
            let feature_id =
                0x4100_0000u32 | (((idx_a as u32) & 0xFF) << 8) | ((idx_b as u32) & 0xFF);

            out.push(Contact3D {
                point: Vec4::new(point.x, point.y, point.z, depth),
                normal: Vec4::new(normal.x, normal.y, normal.z, 0.0),
                tangent: Vec4::new(tangent.x, tangent.y, tangent.z, 0.0),
                local_anchor_a: Vec4::new(local_a.x, local_a.y, local_a.z, 0.0),
                local_anchor_b: Vec4::new(local_b.x, local_b.y, local_b.z, 0.0),
                lambda: Vec4::ZERO,
                penalty: Vec4::new(
                    self.sim_params.solver[2],
                    self.sim_params.solver[2],
                    self.sim_params.solver[2],
                    0.0,
                ),
                body_a,
                body_b,
                feature_id,
                flags: 0,
            });
        }
    }

    /// Traverse a compound shape's BVH to find child indices whose local AABBs
    /// overlap the given query AABB (in the compound's local space).
    fn traverse_compound_bvh(
        bvh_nodes: &[rubble_math::BvhNode],
        query_min: Vec3,
        query_max: Vec3,
    ) -> Vec<usize> {
        let mut result = Vec::new();
        if bvh_nodes.is_empty() {
            return result;
        }
        let root = (bvh_nodes.len() - 1) as i32;
        let mut stack = vec![root];
        while let Some(idx) = stack.pop() {
            if idx < 0 {
                result.push((-idx - 1) as usize);
                continue;
            }
            let node = &bvh_nodes[idx as usize];
            let node_min = node.aabb_min.truncate();
            let node_max = node.aabb_max.truncate();
            if node_min.x <= query_max.x
                && node_max.x >= query_min.x
                && node_min.y <= query_max.y
                && node_max.y >= query_min.y
                && node_min.z <= query_max.z
                && node_max.z >= query_min.z
            {
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        result
    }

    /// Transform a world-space AABB to a compound's local space (min corner).
    fn transform_aabb_to_local(
        world_min: Vec3,
        world_max: Vec3,
        parent_pos: Vec3,
        inv_parent_rot: glam::Quat,
    ) -> Vec3 {
        let corners = [
            Vec3::new(world_min.x, world_min.y, world_min.z),
            Vec3::new(world_max.x, world_min.y, world_min.z),
            Vec3::new(world_min.x, world_max.y, world_min.z),
            Vec3::new(world_min.x, world_min.y, world_max.z),
            Vec3::new(world_max.x, world_max.y, world_min.z),
            Vec3::new(world_max.x, world_min.y, world_max.z),
            Vec3::new(world_min.x, world_max.y, world_max.z),
            Vec3::new(world_max.x, world_max.y, world_max.z),
        ];
        let mut local_min = Vec3::splat(f32::MAX);
        for &c in &corners {
            local_min = local_min.min(inv_parent_rot * (c - parent_pos));
        }
        local_min
    }

    /// Transform a world-space AABB to a compound's local space (max corner).
    fn transform_aabb_to_local_max(
        world_min: Vec3,
        world_max: Vec3,
        parent_pos: Vec3,
        inv_parent_rot: glam::Quat,
    ) -> Vec3 {
        let corners = [
            Vec3::new(world_min.x, world_min.y, world_min.z),
            Vec3::new(world_max.x, world_min.y, world_min.z),
            Vec3::new(world_min.x, world_max.y, world_min.z),
            Vec3::new(world_min.x, world_min.y, world_max.z),
            Vec3::new(world_max.x, world_max.y, world_min.z),
            Vec3::new(world_max.x, world_min.y, world_max.z),
            Vec3::new(world_min.x, world_max.y, world_max.z),
            Vec3::new(world_max.x, world_max.y, world_max.z),
        ];
        let mut local_max = Vec3::splat(f32::MIN);
        for &c in &corners {
            local_max = local_max.max(inv_parent_rot * (c - parent_pos));
        }
        local_max
    }

    /// Get world-space positions/rotations of compound children.
    fn get_compound_children_world(
        &self,
        compound_index: u32,
        parent_pos: Vec3,
        parent_rot: glam::Quat,
    ) -> Vec<(Vec3, glam::Quat, u32, u32)> {
        if (compound_index as usize) >= self.compound_shapes_data.len() {
            return vec![];
        }
        let compound = &self.compound_shapes_data[compound_index as usize];
        let mut result = Vec::with_capacity(compound.child_count as usize);
        for i in 0..compound.child_count {
            let ci = (compound.child_offset + i) as usize;
            if ci >= self.compound_children_data.len() {
                break;
            }
            let child = &self.compound_children_data[ci];
            let local_pos = child.local_position.truncate();
            let local_rot = glam::Quat::from_xyzw(
                child.local_rotation.x,
                child.local_rotation.y,
                child.local_rotation.z,
                child.local_rotation.w,
            );
            let world_pos = parent_pos + parent_rot * local_pos;
            let world_rot = parent_rot * local_rot;
            result.push((world_pos, world_rot, child.shape_type, child.shape_index));
        }
        result
    }

    /// Get a conservative bounding radius for a child shape.
    fn child_extent(&self, shape_type: u32, shape_index: u32) -> f32 {
        match shape_type {
            0 => self
                .sphere_data_cpu
                .get(shape_index as usize)
                .map(|sphere| sphere.radius)
                .unwrap_or(1.0),
            1 => self
                .box_data_cpu
                .get(shape_index as usize)
                .map(|box_data| box_data.half_extents.truncate().length())
                .unwrap_or(1.5),
            2 => self
                .capsule_data_cpu
                .get(shape_index as usize)
                .map(|capsule| capsule.half_height + capsule.radius)
                .unwrap_or(1.5),
            3 => self
                .convex_hulls_cpu
                .get(shape_index as usize)
                .map(|hull| {
                    let start = hull.vertex_offset as usize;
                    let end = start + hull.vertex_count as usize;
                    self.convex_vertices_cpu[start..end]
                        .iter()
                        .map(|vertex| Vec3::new(vertex.x, vertex.y, vertex.z).length())
                        .fold(0.0, f32::max)
                })
                .unwrap_or(2.0),
            4 => 1e4,
            _ => 2.0,
        }
    }

    /// Body-colored AVBD solve in position space.
    fn run_colored_solve(&mut self, num_bodies: u32, solver_iterations: u32, contact_count: u32) {
        // GPU CSR adjacency + active flags (also correct for zero contacts:
        // clears active so free-motion advances everything).
        self.build_adjacency_gpu(num_bodies, contact_count);
        self.dispatch_free_motion(num_bodies);
        if contact_count == 0 {
            return;
        }

        // Recolor + compact on GPU every frame (one submit): body_order gains a
        // 64-word (offset,count) header and per-color indirect dispatch args —
        // no CPU downloads anywhere in the solve (M2(c)). Identical path for
        // native and web.
        self.dispatch_gpu_coloring(num_bodies, contact_count);
        // Freeze the Taylor cache (C0 per row, penalty floor/caps, restitution
        // v_n) for the sorted contact stream before the first iteration.
        self.dispatch_contact_prep(contact_count);

        if num_bodies <= PERSISTENT_MAX_BODIES {
            // D8/W3: whole iterations x (colors + dual) loop in ONE dispatch of
            // one workgroup. The clamped contact count is patched into
            // SimParams.counts.w GPU-side first (same args-kernel + 4-byte-copy
            // trick as the narrowphase pair count).
            let args_params: [u32; 4] = [self.contacts.capacity(), 0, 0, 0];
            self.ctx.queue.write_buffer(
                &self.contact_args_params,
                0,
                bytemuck::cast_slice(&args_params),
            );
            let args_bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("contact_count_args"),
                    layout: self.narrowphase_args_kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.contact_count.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.narrowphase_dispatch_args.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.clamped_pair_count.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.contact_args_params.as_entire_binding(),
                        },
                    ],
                });
            let _ = self.persistent_bind_group();
            let persistent_bg = self.persistent_bg_cache.bind_group.as_ref().unwrap();
            let mut encoder =
                self.ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("avbd_solve_persistent_3d"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("contact_count_args"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(self.narrowphase_args_kernel.pipeline());
                pass.set_bind_group(0, &args_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            let counts_w = (std::mem::offset_of!(SimParamsGpu, counts) + 12) as u64;
            encoder.copy_buffer_to_buffer(
                &self.clamped_pair_count,
                0,
                &self.params_uniform,
                counts_w,
                4,
            );
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("avbd_persistent_3d"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(self.persistent_kernel.pipeline());
                pass.set_bind_group(0, persistent_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            self.ctx.queue.submit(Some(encoder.finish()));
            return;
        }

        self.ensure_color_index_uniforms();
        self.sync_primal_bind_groups(COLORING_MAX_ITERATIONS as usize);
        let _ = self.dual_bind_group();
        let primal_bind_groups = &self.primal_bg_cache.bind_groups;
        let dual_bind_group = self.dual_bg_cache.bind_group.as_ref().unwrap();
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("avbd_solve_batch_3d"),
            });
        for _ in 0..solver_iterations {
            // One compute pass PER COLOR so wgpu's pass-boundary hazard
            // tracking orders color N's body writes before color N+1's partner
            // reads (Gauss-Seidel across colors). Empty colors dispatch zero
            // workgroups via the GPU-written indirect args.
            for (color, bind_group) in primal_bind_groups
                .iter()
                .enumerate()
                .take(COLORING_MAX_ITERATIONS as usize)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("avbd_primal_3d"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(self.primal_kernel.pipeline());
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups_indirect(
                    &self.gpu_coloring.dispatch_args,
                    (color * 12) as u64,
                );
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("avbd_dual_3d"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(self.dual_kernel.pipeline());
                pass.set_bind_group(0, dual_bind_group, &[]);
                pass.dispatch_workgroups(round_up_workgroups(contact_count, WORKGROUP_SIZE), 1, 1);
            }
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Run the full GPU physics step and download updated states.
    pub fn step(&mut self, num_bodies: u32, solver_iterations: u32) -> Vec<RigidBodyState3D> {
        if num_bodies == 0 {
            return Vec::new();
        }

        let compound_contacts = self.run_detection(num_bodies);

        let gpu_count = self.contact_count.read(&self.ctx);
        let total = self.append_compound_contacts(gpu_count, &compound_contacts);
        if total > 0 {
            self.contacts.set_len(total);
            self.sort_contacts_gpu(total);
        }

        self.run_colored_solve(num_bodies, solver_iterations, total);

        self.dispatch_extract(num_bodies);
        self.dispatch_restitution(num_bodies, total > 0);
        self.body_states.download(&self.ctx)
    }

    /// Run the full GPU physics step with warm-starting support.
    /// Returns (updated_states, new_contacts) so the caller can track persistence.
    ///
    /// Warmstarting runs on GPU. When `warm_contacts` is provided, those contacts
    /// are uploaded to the GPU `prev_contacts` buffer and used for matching.
    /// When `None`, the internally maintained `prev_contacts` buffer (from the
    /// previous step on this pipeline) is used.
    pub fn step_with_contacts(
        &mut self,
        num_bodies: u32,
        solver_iterations: u32,
        warm_contacts: Option<&[Contact3D]>,
    ) -> (Vec<RigidBodyState3D>, Vec<Contact3D>) {
        if num_bodies == 0 {
            return (Vec::new(), Vec::new());
        }

        // If caller provides explicit warm contacts, upload them as prev_contacts
        // so the GPU warmstart kernel can match against them.
        if let Some(warm) = warm_contacts {
            self.upload_warm_contacts(warm);
        }

        let compound_contacts = self.run_detection(num_bodies);

        let gpu_count = self.contact_count.read(&self.ctx);
        let total = self.append_compound_contacts(gpu_count, &compound_contacts);
        if total > 0 {
            self.contacts.set_len(total);
            // Deterministic order first (the sort key rides in lambda.w and the
            // warm-start pass overwrites lambda), then GPU warm-start matching.
            self.sort_contacts_gpu(total);
            self.dispatch_gpu_warmstart(total);
        }

        self.run_colored_solve(num_bodies, solver_iterations, total);

        // Download contacts after solve (persistence/events/last_contacts —
        // goes away with the CAS pair map in the next M2 stage).
        let final_contacts = if total > 0 {
            let fc = self.download_contacts();
            // Swap solved contacts → prev_contacts for next frame's GPU warmstart
            let solved_count = fc.len() as u32;
            self.swap_contact_buffers(solved_count);
            fc
        } else {
            self.prev_contact_count = 0;
            Vec::new()
        };

        self.dispatch_extract(num_bodies);
        self.dispatch_restitution(num_bodies, total > 0);
        let states = self.body_states.download(&self.ctx);
        (states, final_contacts)
    }

    /// Timed version of `step_with_contacts` that populates per-phase timings.
    ///
    /// When `want_contacts` is false, the solved contacts are NOT downloaded
    /// (the returned contact vec is empty); warm-start persistence is still
    /// maintained on-device via `swap_contact_buffers`. Pass true only when the
    /// caller needs the contacts (collision events / debug queries) — W4a-2.
    pub fn step_with_contacts_timed(
        &mut self,
        num_bodies: u32,
        solver_iterations: u32,
        warm_contacts: Option<&[Contact3D]>,
        want_contacts: bool,
        timings: &mut rubble_gpu::StepTimingsMs,
    ) -> (Vec<RigidBodyState3D>, Vec<Contact3D>) {
        use std::time::Instant;

        if num_bodies == 0 {
            return (Vec::new(), Vec::new());
        }

        if let Some(warm) = warm_contacts {
            self.upload_warm_contacts(warm);
        }

        let compound_contacts = self.run_detection_timed(num_bodies, timings);

        let t_cf = Instant::now();
        let gpu_count = self.contact_count.read(&self.ctx);
        let total = self.append_compound_contacts(gpu_count, &compound_contacts);
        if total > 0 {
            self.contacts.set_len(total);
            // Deterministic order first (the sort key rides in lambda.w and the
            // warm-start pass overwrites lambda), then GPU warm-start matching.
            self.sort_contacts_gpu(total);
            self.dispatch_gpu_warmstart(total);
        }
        timings.contact_fetch_ms = t_cf.elapsed().as_secs_f32() * 1000.0;

        let t_solve = Instant::now();
        self.run_colored_solve(num_bodies, solver_iterations, total);
        timings.solve_ms = t_solve.elapsed().as_secs_f32() * 1000.0;

        // Post-solve: swap solved contacts -> prev_contacts on-device for next
        // frame's warm-start (always), and download them only when the caller
        // wants them. The download is a GPU->CPU readback, not solve work, so
        // it accounts to contact_fetch_ms (QW-0). `gpu_count` is the GPU-emitted
        // solved count that `download_contacts()` would re-read (W4a-2).
        let t_cf2 = Instant::now();
        let final_contacts = if total > 0 {
            self.swap_contact_buffers(gpu_count);
            if want_contacts {
                self.download_contacts()
            } else {
                Vec::new()
            }
        } else {
            self.prev_contact_count = 0;
            Vec::new()
        };
        timings.contact_fetch_ms += t_cf2.elapsed().as_secs_f32() * 1000.0;

        let t_ext = Instant::now();
        self.dispatch_extract(num_bodies);
        // Guard on the pre-solve count (`total`), not `final_contacts`: with
        // events disabled (W4a-2) the contacts aren't downloaded, so
        // `final_contacts` is empty even when the solve had contacts.
        self.dispatch_restitution(num_bodies, total > 0);
        let states = self.body_states.download(&self.ctx);
        timings.extract_ms = t_ext.elapsed().as_secs_f32() * 1000.0;

        (states, final_contacts)
    }

    fn run_detection_timed(
        &mut self,
        num_bodies: u32,
        timings: &mut rubble_gpu::StepTimingsMs,
    ) -> Vec<Contact3D> {
        use std::time::Instant;

        self.contact_count.reset(&self.ctx);

        let t0 = Instant::now();
        self.dispatch_predict_and_bounds(num_bodies);
        timings.predict_aabb_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t1 = Instant::now();
        let mut broadphase = BroadphaseBreakdownMs::default();
        let has_compounds = self.body_props_cpu[..num_bodies as usize]
            .iter()
            .any(|prop| prop.shape_type == rubble_math::SHAPE_COMPOUND);
        // Planes go through GPU LBVH broadphase (their large AABB handles pairing).
        // Only compound shapes still require CPU pair expansion.
        let requires_cpu_pair_stage = has_compounds;
        self.aabbs.set_len(num_bodies);

        let mut cpu_compound_contacts: Vec<Contact3D> = Vec::new();
        let mut pair_thread_count: u32 = 0;
        let mut pair_count: u32 = 0;

        if requires_cpu_pair_stage {
            let t_readback = Instant::now();
            let cpu_aabbs = self.aabbs.download(&self.ctx);
            broadphase.readback_ms += t_readback.elapsed().as_secs_f32() * 1000.0;
            let overlap_pairs =
                self.broadphase_pairs_3d_with_breakdown(num_bodies, &cpu_aabbs, &mut broadphase);
            if !overlap_pairs.is_empty() {
                let props = &self.body_props_cpu[..num_bodies as usize];
                let states = if has_compounds {
                    self.body_states.current_mut().set_len(num_bodies);
                    let t_state_readback = Instant::now();
                    let states = self.body_states.download(&self.ctx);
                    broadphase.readback_ms += t_state_readback.elapsed().as_secs_f32() * 1000.0;
                    Some(states)
                } else {
                    None
                };
                let t_build = Instant::now();
                let mut non_compound_pairs: Vec<GpuPair> = Vec::with_capacity(overlap_pairs.len());

                for p in &overlap_pairs {
                    let a = p[0];
                    let b = p[1];
                    if has_compounds {
                        let st_a = props[a as usize].shape_type;
                        let st_b = props[b as usize].shape_type;

                        let a_is_compound = st_a == rubble_math::SHAPE_COMPOUND;
                        let b_is_compound = st_b == rubble_math::SHAPE_COMPOUND;

                        if !a_is_compound && !b_is_compound {
                            non_compound_pairs.push(GpuPair { a, b });
                        } else {
                            self.generate_compound_contacts_cpu(
                                a,
                                b,
                                props,
                                states.as_ref().expect("compound path requires states"),
                                &mut cpu_compound_contacts,
                            );
                        }
                    } else {
                        non_compound_pairs.push(GpuPair { a, b });
                    }
                }

                if !non_compound_pairs.is_empty() {
                    non_compound_pairs.sort_unstable_by_key(|pair| {
                        let st_a = props[pair.a as usize].shape_type;
                        let st_b = props[pair.b as usize].shape_type;
                        let (lo, hi) = if st_a <= st_b {
                            (st_a, st_b)
                        } else {
                            (st_b, st_a)
                        };
                        (lo << 16) | hi
                    });
                    let count = non_compound_pairs.len() as u32;
                    self.pairs.upload(&self.ctx, &non_compound_pairs);
                    pair_count = count;
                }
                broadphase.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;
            }
        } else {
            pair_thread_count = self.gpu_lbvh.query_on_device_raw_with_breakdown(
                &self.ctx,
                self.aabbs.buffer(),
                num_bodies,
                &mut broadphase,
            );
        }
        timings.set_broadphase_breakdown(broadphase);
        debug_assert!(timings.broadphase_ms <= t1.elapsed().as_secs_f32() * 1000.0 + 0.5);

        let t2 = Instant::now();
        if requires_cpu_pair_stage {
            self.dispatch_narrowphase(num_bodies, pair_count);
        } else if pair_thread_count > 0 {
            let pair_buffer = self.gpu_lbvh.pair_buffer().clone();
            self.dispatch_narrowphase_indirect(&pair_buffer, pair_thread_count);
        }

        let contact_count_val = self.contact_count.read(&self.ctx);
        let capacity = self.contacts.capacity();
        if contact_count_val > capacity {
            let new_cap = (contact_count_val as usize) * 2;
            self.contacts.grow_if_needed(&self.ctx, new_cap);
            self.contact_count.reset(&self.ctx);
            if requires_cpu_pair_stage {
                self.dispatch_narrowphase(num_bodies, pair_count);
            } else if pair_thread_count > 0 {
                let pair_buffer = self.gpu_lbvh.pair_buffer().clone();
                self.dispatch_narrowphase_indirect(&pair_buffer, pair_thread_count);
            }
        }
        timings.narrowphase_ms = t2.elapsed().as_secs_f32() * 1000.0;

        cpu_compound_contacts
    }

    fn broadphase_pairs_3d(&mut self, num_bodies: u32, cpu_aabbs: &[Aabb3D]) -> Vec<[u32; 2]> {
        let mut breakdown = BroadphaseBreakdownMs::default();
        self.broadphase_pairs_3d_with_breakdown(num_bodies, cpu_aabbs, &mut breakdown)
    }

    fn broadphase_pairs_3d_with_breakdown(
        &mut self,
        num_bodies: u32,
        cpu_aabbs: &[Aabb3D],
        breakdown: &mut BroadphaseBreakdownMs,
    ) -> Vec<[u32; 2]> {
        #[cfg(target_arch = "wasm32")]
        use rubble_gpu::web_time::Instant;
        #[cfg(not(target_arch = "wasm32"))]
        use std::time::Instant;

        let t_build = Instant::now();
        let props = &self.body_props_cpu[..num_bodies as usize];
        let active_bodies: Vec<u32> = props
            .iter()
            .enumerate()
            .filter_map(|(idx, prop)| {
                (prop.shape_type != rubble_math::SHAPE_PLANE).then_some(idx as u32)
            })
            .collect();
        let plane_bodies: Vec<u32> = props
            .iter()
            .enumerate()
            .filter_map(|(idx, prop)| {
                (prop.shape_type == rubble_math::SHAPE_PLANE).then_some(idx as u32)
            })
            .collect();
        breakdown.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;

        let mut pairs = if active_bodies.len() >= 2 {
            if plane_bodies.is_empty() && active_bodies.len() == num_bodies as usize {
                self.gpu_lbvh.build_and_query_raw_with_breakdown(
                    &self.ctx,
                    self.aabbs.buffer(),
                    num_bodies,
                    breakdown,
                )
            } else {
                let t_build = Instant::now();
                let subset_aabbs: Vec<Aabb3D> = active_bodies
                    .iter()
                    .map(|&body_idx| cpu_aabbs[body_idx as usize])
                    .collect();
                self.lbvh_subset_aabbs.upload(&self.ctx, &subset_aabbs);
                breakdown.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;
                let lbvh_pairs = self.gpu_lbvh.build_and_query_raw_with_breakdown(
                    &self.ctx,
                    self.lbvh_subset_aabbs.buffer(),
                    subset_aabbs.len() as u32,
                    breakdown,
                );
                let t_build = Instant::now();
                let pairs = lbvh_pairs
                    .into_iter()
                    .map(|[a, b]| [active_bodies[a as usize], active_bodies[b as usize]])
                    .collect();
                breakdown.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;
                pairs
            }
        } else {
            Vec::new()
        };

        let t_build = Instant::now();
        for &plane_idx in &plane_bodies {
            for &body_idx in &active_bodies {
                pairs.push([plane_idx, body_idx]);
            }
        }

        pairs.sort_unstable();
        pairs.dedup();
        breakdown.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;
        pairs
    }

    #[cfg(target_arch = "wasm32")]
    async fn broadphase_pairs_3d_async_with_breakdown(
        &mut self,
        num_bodies: u32,
        cpu_aabbs: &[Aabb3D],
        breakdown: &mut BroadphaseBreakdownMs,
    ) -> Vec<[u32; 2]> {
        use rubble_gpu::web_time::Instant;

        let t_build = Instant::now();
        let props = &self.body_props_cpu[..num_bodies as usize];
        let active_bodies: Vec<u32> = props
            .iter()
            .enumerate()
            .filter_map(|(idx, prop)| {
                (prop.shape_type != rubble_math::SHAPE_PLANE).then_some(idx as u32)
            })
            .collect();
        let plane_bodies: Vec<u32> = props
            .iter()
            .enumerate()
            .filter_map(|(idx, prop)| {
                (prop.shape_type == rubble_math::SHAPE_PLANE).then_some(idx as u32)
            })
            .collect();
        breakdown.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;

        let mut pairs = if active_bodies.len() >= 2 {
            if plane_bodies.is_empty() && active_bodies.len() == num_bodies as usize {
                self.gpu_lbvh
                    .build_and_query_raw_async_with_breakdown(
                        &self.ctx,
                        self.aabbs.buffer(),
                        num_bodies,
                        breakdown,
                    )
                    .await
            } else {
                let t_build = Instant::now();
                let subset_aabbs: Vec<Aabb3D> = active_bodies
                    .iter()
                    .map(|&body_idx| cpu_aabbs[body_idx as usize])
                    .collect();
                self.lbvh_subset_aabbs.upload(&self.ctx, &subset_aabbs);
                breakdown.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;
                let lbvh_pairs = self
                    .gpu_lbvh
                    .build_and_query_raw_async_with_breakdown(
                        &self.ctx,
                        self.lbvh_subset_aabbs.buffer(),
                        subset_aabbs.len() as u32,
                        breakdown,
                    )
                    .await;
                let t_build = Instant::now();
                let pairs = lbvh_pairs
                    .into_iter()
                    .map(|[a, b]| [active_bodies[a as usize], active_bodies[b as usize]])
                    .collect();
                breakdown.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;
                pairs
            }
        } else {
            Vec::new()
        };

        let t_build = Instant::now();
        for &plane_idx in &plane_bodies {
            for &body_idx in &active_bodies {
                pairs.push([plane_idx, body_idx]);
            }
        }

        pairs.sort_unstable();
        pairs.dedup();
        breakdown.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;
        pairs
    }

    /// Reference to the GPU context.
    pub fn context(&self) -> &GpuContext {
        &self.ctx
    }

    /// Read the current contact count from GPU.
    pub fn read_contact_count(&self) -> u32 {
        self.contact_count.read(&self.ctx)
    }

    /// Download contacts from GPU.
    pub fn download_contacts(&mut self) -> Vec<Contact3D> {
        let count = self.contact_count.read(&self.ctx) as usize;
        if count == 0 {
            return Vec::new();
        }
        self.download_contacts_exact(count)
    }

    fn download_contacts_exact(&mut self, count: usize) -> Vec<Contact3D> {
        self.contacts.set_len(count as u32);
        let all = self.contacts.download(&self.ctx);
        all.into_iter().take(count).collect()
    }

    /// Async version of `run_detection` for WASM/WebGPU.
    #[cfg(target_arch = "wasm32")]
    async fn run_detection_async(
        &mut self,
        num_bodies: u32,
        timings: &mut rubble_gpu::StepTimingsMs,
    ) -> Vec<Contact3D> {
        use rubble_gpu::web_time::Instant;

        self.contact_count.reset(&self.ctx);

        let t0 = Instant::now();
        self.dispatch_predict_and_bounds(num_bodies);
        timings.predict_aabb_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t1 = Instant::now();
        let mut broadphase = BroadphaseBreakdownMs::default();
        let has_compounds = self.body_props_cpu[..num_bodies as usize]
            .iter()
            .any(|prop| prop.shape_type == rubble_math::SHAPE_COMPOUND);
        // Planes go through GPU LBVH broadphase (their large AABB handles pairing).
        // Only compound shapes still require CPU pair expansion.
        let requires_cpu_pair_stage = has_compounds;
        self.aabbs.set_len(num_bodies);

        let mut cpu_compound_contacts: Vec<Contact3D> = Vec::new();
        let mut pair_thread_count: u32 = 0;
        let mut pair_count: u32 = 0;

        if requires_cpu_pair_stage {
            let t_readback = Instant::now();
            let gpu_aabbs = self.aabbs.download_async(&self.ctx).await;
            broadphase.readback_ms += t_readback.elapsed().as_secs_f32() * 1000.0;
            let overlap_pairs = self
                .broadphase_pairs_3d_async_with_breakdown(num_bodies, &gpu_aabbs, &mut broadphase)
                .await;
            if !overlap_pairs.is_empty() {
                let props = &self.body_props_cpu[..num_bodies as usize];
                let states = if has_compounds {
                    self.body_states.set_len(num_bodies);
                    let t_state_readback = Instant::now();
                    let states = self.body_states.download_async(&self.ctx).await;
                    broadphase.readback_ms += t_state_readback.elapsed().as_secs_f32() * 1000.0;
                    Some(states)
                } else {
                    None
                };
                let t_build = Instant::now();
                let mut non_compound_pairs: Vec<GpuPair> = Vec::with_capacity(overlap_pairs.len());

                for p in &overlap_pairs {
                    let a = p[0];
                    let b = p[1];
                    if has_compounds {
                        let st_a = props[a as usize].shape_type;
                        let st_b = props[b as usize].shape_type;

                        let a_is_compound = st_a == rubble_math::SHAPE_COMPOUND;
                        let b_is_compound = st_b == rubble_math::SHAPE_COMPOUND;

                        if !a_is_compound && !b_is_compound {
                            non_compound_pairs.push(GpuPair { a, b });
                        } else {
                            self.generate_compound_contacts_cpu(
                                a,
                                b,
                                props,
                                states.as_ref().expect("compound path requires states"),
                                &mut cpu_compound_contacts,
                            );
                        }
                    } else {
                        non_compound_pairs.push(GpuPair { a, b });
                    }
                }

                if !non_compound_pairs.is_empty() {
                    non_compound_pairs.sort_unstable_by_key(|pair| {
                        let st_a = props[pair.a as usize].shape_type;
                        let st_b = props[pair.b as usize].shape_type;
                        let (lo, hi) = if st_a <= st_b {
                            (st_a, st_b)
                        } else {
                            (st_b, st_a)
                        };
                        (lo << 16) | hi
                    });
                    let count = non_compound_pairs.len() as u32;
                    self.pairs.upload(&self.ctx, &non_compound_pairs);
                    pair_count = count;
                }
                broadphase.build_ms += t_build.elapsed().as_secs_f32() * 1000.0;
            }
        } else {
            pair_thread_count = self
                .gpu_lbvh
                .query_on_device_raw_async_with_breakdown(
                    &self.ctx,
                    self.aabbs.buffer(),
                    num_bodies,
                    &mut broadphase,
                )
                .await;
        }
        timings.set_broadphase_breakdown(broadphase);
        debug_assert!(timings.broadphase_ms <= t1.elapsed().as_secs_f32() * 1000.0 + 0.5);

        let t2 = Instant::now();
        if requires_cpu_pair_stage {
            self.dispatch_narrowphase(num_bodies, pair_count);
        } else if pair_thread_count > 0 {
            // No pair-count await: args kernel + uniform patch + indirect
            // dispatch — identical to the native path.
            let pair_buffer = self.gpu_lbvh.pair_buffer().clone();
            self.dispatch_narrowphase_indirect(&pair_buffer, pair_thread_count);
        }
        // Contact-overflow healing happens at the frame-end count read (the
        // emission clamp keeps this frame's buffer contents valid); no
        // blocking mid-frame readback on web.
        timings.narrowphase_ms = t2.elapsed().as_secs_f32() * 1000.0;

        cpu_compound_contacts
    }

    /// Async version of `step_with_contacts` for WASM/WebGPU.
    ///
    /// Warmstarting runs on GPU. When `warm_contacts` is provided, those contacts
    /// are uploaded to `prev_contacts`. When `None`, the internal buffer is used.
    #[cfg(target_arch = "wasm32")]
    pub async fn step_with_contacts_async(
        &mut self,
        num_bodies: u32,
        solver_iterations: u32,
        warm_contacts: Option<&[Contact3D]>,
        want_contacts: bool,
        timings: &mut rubble_gpu::StepTimingsMs,
    ) -> (Vec<RigidBodyState3D>, Vec<Contact3D>) {
        use rubble_gpu::web_time::Instant;

        if num_bodies == 0 {
            return (Vec::new(), Vec::new());
        }

        // W5 zero-readback guard: snapshot the context's readback counter so we
        // can assert the resident (events-off, non-compound) hot path performs
        // no blocking GPU->CPU readback below.
        #[cfg(debug_assertions)]
        let readbacks_at_entry = self.ctx.readback_count();

        if let Some(warm) = warm_contacts {
            self.upload_warm_contacts(warm);
        }

        let compound_contacts = self.run_detection_async(num_bodies, timings).await;

        // Contact upper bound WITHOUT a pre-solve readback (W1): every
        // contact-sized stage clamps against the GPU counter in-shader, so the
        // dispatch bound only needs to be >= the live count. The narrowphase
        // emission clamp guarantees demand <= num_bodies*8, and that bound is
        // scene-sized — the raw buffer capacity is allocation-sized
        // (max_bodies*8 = 512k contacts at the wasm default), which made every
        // SwiftShader CI frame sort half a million sentinel slots and hang the
        // E2E lane. Compound scenes still read the count (their CPU tail
        // append must know where to write) — the only remaining pre-solve
        // await, and only when compounds exist.
        let t_cf = Instant::now();
        let total_upper = if compound_contacts.is_empty() {
            (num_bodies * 8).min(self.contacts.capacity())
        } else {
            let gpu_count = self.contact_count.read_async(&self.ctx).await;
            self.append_compound_contacts(gpu_count, &compound_contacts)
        };
        if total_upper > 0 {
            self.contacts.set_len(total_upper);
            // Deterministic order first (the sort key rides in lambda.w and the
            // warm-start pass overwrites lambda), then GPU warm-start matching.
            self.sort_contacts_gpu(total_upper);
            self.dispatch_gpu_warmstart(total_upper);
        }
        timings.contact_fetch_ms = t_cf.elapsed().as_secs_f32() * 1000.0;

        let t_solve = Instant::now();
        self.run_colored_solve(num_bodies, solver_iterations, total_upper);
        timings.solve_ms = t_solve.elapsed().as_secs_f32() * 1000.0;

        // QW-1: submit ALL remaining GPU work (extract + restitution) BEFORE any
        // readback await, so those passes drain concurrently with the
        // map-backed downloads below instead of after them. Restitution's guard
        // uses the dispatch bound (`total_upper > 0`), known without a readback
        // — the in-shader guards no-op against the true GPU count regardless.
        self.dispatch_extract(num_bodies);
        self.dispatch_restitution(num_bodies, total_upper > 0);

        // W5: NO count readback on the hot path. Warm-start persistence is
        // maintained entirely on-device — the swap copies the dispatch-bound
        // prefix of solved contacts into prev_contacts and publishes the true
        // GPU count into prev_contact_count_buf (which the next frame's insert
        // kernel clamps by). Buffer overflow is impossible: narrowphase emission
        // is clamped to `arrayLength(&contacts)` (capacity), so demand <=
        // capacity always — no grow needed. The contacts download only happens
        // on the opt-in events/debug path (W4a-2), which may block.
        let t_cf2 = Instant::now();
        let final_contacts = if total_upper > 0 {
            self.swap_contact_buffers(total_upper);
            if want_contacts {
                let demand = self.contact_count.read_async(&self.ctx).await;
                let total = demand.min(self.contacts.capacity());
                self.download_contacts_exact_async(total as usize).await
            } else {
                Vec::new()
            }
        } else {
            self.prev_contact_count = 0;
            Vec::new()
        };
        timings.contact_fetch_ms += t_cf2.elapsed().as_secs_f32() * 1000.0;

        // W4b: enqueue a NON-BLOCKING copy of the post-extract body states into
        // the snapshot ring and return the most recently completed snapshot
        // (~1 frame lag). This removes the blocking `body_states.download_async`
        // await (~4.8 ms of pure mapAsync latency on Chrome) from the hot path.
        // The first frame or two before any copy completes return empty, which
        // the caller treats as "state unchanged".
        let t_ext = Instant::now();
        self.body_states.set_len(num_bodies);
        let src = self.body_states.current().buffer();
        self.states_ring.enqueue(&self.ctx, src, num_bodies);
        let states = self
            .states_ring
            .latest()
            .map(|s| s.to_vec())
            .unwrap_or_default();
        timings.extract_ms = t_ext.elapsed().as_secs_f32() * 1000.0;

        // W5: on the resident path (events off, no compound CPU tail) the step
        // must not have blocked on any GPU->CPU readback.
        #[cfg(debug_assertions)]
        if !want_contacts && compound_contacts.is_empty() {
            debug_assert_eq!(
                self.ctx.readback_count(),
                readbacks_at_entry,
                "W5: {} unexpected hot-path GPU->CPU readback(s)",
                self.ctx.readback_count() - readbacks_at_entry
            );
        }

        (states, final_contacts)
    }

    /// Drain the snapshot ring to the newest enqueued frame and return exact,
    /// current body states (zero lag). For tests/queries that need synchronous
    /// state on the wasm path; not used by the hot render loop.
    #[cfg(target_arch = "wasm32")]
    pub async fn sync_states_async(&mut self) -> Vec<RigidBodyState3D> {
        self.states_ring.sync().await.to_vec()
    }

    #[cfg(target_arch = "wasm32")]
    async fn download_contacts_exact_async(&mut self, count: usize) -> Vec<Contact3D> {
        if count == 0 {
            return Vec::new();
        }
        self.contacts.set_len(count as u32);
        let mut all = self.contacts.download_async(&self.ctx).await;
        all.truncate(count);
        all
    }

    // -----------------------------------------------------------------------
    // Private dispatch helpers
    // -----------------------------------------------------------------------

    fn body_states_cache_key(&self) -> u64 {
        (self.body_states.current().byte_size() << 1) | self.body_states.current_index() as u64
    }

    fn ensure_solve_range_buffers(&mut self, required: usize) {
        while self.solve_range_buffers.len() < required {
            self.solve_range_buffers
                .push(self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("SolveRange uniform"),
                    size: std::mem::size_of::<SolveRangeGpu>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
        }
    }

    /// The per-color primal uniforms are static color indices now: the shader
    /// reads the actual (offset, count) from the GPU-written header in
    /// `body_order`, so `offset` carries the color index and `count` is
    /// unused. Written once, never per frame.
    fn ensure_color_index_uniforms(&mut self) {
        if self.color_uniforms_written {
            return;
        }
        self.ensure_solve_range_buffers(COLORING_MAX_ITERATIONS as usize);
        for (idx, buf) in self.solve_range_buffers.iter().enumerate() {
            let range = SolveRangeGpu {
                offset: idx as u32,
                count: 0,
            };
            self.ctx
                .queue
                .write_buffer(buf, 0, bytemuck::bytes_of(&range));
        }
        self.color_uniforms_written = true;
    }

    fn predict_bind_group(&mut self) -> &wgpu::BindGroup {
        let key = [
            self.body_states_cache_key(),
            self.old_states.byte_size(),
            self.inertial_states.byte_size(),
            self.prev_step_states.byte_size(),
        ];
        if self.predict_bg_cache.key.as_ref() != Some(&key) {
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("predict"),
                    layout: self.predict_kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.body_states.current().buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.old_states.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.inertial_states.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.prev_step_states.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.params_uniform.as_entire_binding(),
                        },
                    ],
                });
            self.predict_bg_cache.key = Some(key);
            self.predict_bg_cache.bind_group = Some(bg);
        }
        self.predict_bg_cache.bind_group.as_ref().unwrap()
    }

    fn aabb_bind_group(&mut self) -> &wgpu::BindGroup {
        let key = [
            self.body_states_cache_key(),
            self.body_props.byte_size(),
            self.spheres.byte_size(),
            self.boxes.byte_size(),
            self.aabbs.byte_size(),
            self.convex_hulls.byte_size(),
            self.convex_vertices.byte_size(),
            self.capsules.byte_size(),
            self.planes.byte_size(),
        ];
        if self.aabb_bg_cache.key.as_ref() != Some(&key) {
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("aabb"),
                    layout: self.aabb_kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.body_states.current().buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.body_props.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.spheres.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.boxes.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.aabbs.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.params_uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self.convex_hulls.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: self.convex_vertices.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: self.capsules.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: self.planes.buffer().as_entire_binding(),
                        },
                    ],
                });
            self.aabb_bg_cache.key = Some(key);
            self.aabb_bg_cache.bind_group = Some(bg);
        }
        self.aabb_bg_cache.bind_group.as_ref().unwrap()
    }

    fn narrowphase_bind_group(&mut self) -> &wgpu::BindGroup {
        let key = [
            self.body_states_cache_key(),
            self.body_props.byte_size(),
            self.pairs.byte_size(),
            self.spheres.byte_size(),
            self.boxes.byte_size(),
            self.contacts.byte_size(),
            self.convex_hulls.byte_size(),
            self.convex_vertices.byte_size(),
            self.capsules.byte_size(),
        ];
        if self.narrowphase_bg_cache.key.as_ref() != Some(&key) {
            let bg = self.create_narrowphase_bind_group(self.pairs.buffer(), "narrowphase");
            self.narrowphase_bg_cache.key = Some(key);
            self.narrowphase_bg_cache.bind_group = Some(bg);
        }
        self.narrowphase_bg_cache.bind_group.as_ref().unwrap()
    }

    fn create_narrowphase_bind_group(
        &self,
        pairs_buffer: &wgpu::Buffer,
        label: &'static str,
    ) -> wgpu::BindGroup {
        self.ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: self.narrowphase_kernel.bind_group_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.body_states.current().buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.body_props.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pairs_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.spheres.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.boxes.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.contacts.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.contact_count.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.params_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: self.convex_hulls.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: self.convex_vertices.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: self.capsules.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: self.plane_params_uniform.as_entire_binding(),
                    },
                ],
            })
    }

    fn free_motion_bind_group(&mut self) -> &wgpu::BindGroup {
        let key = [
            self.body_states_cache_key(),
            self.inertial_states.byte_size(),
            self.active_body_flags.byte_size(),
            self.params_uniform.size(),
        ];
        if self.free_motion_bg_cache.key.as_ref() != Some(&key) {
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("free_motion"),
                    layout: self.free_motion_kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.body_states.current().buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.inertial_states.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.active_body_flags.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.params_uniform.as_entire_binding(),
                        },
                    ],
                });
            self.free_motion_bg_cache.key = Some(key);
            self.free_motion_bg_cache.bind_group = Some(bg);
        }
        self.free_motion_bg_cache.bind_group.as_ref().unwrap()
    }

    fn sync_primal_bind_groups(&mut self, range_count: usize) {
        let key = [
            self.body_states_cache_key(),
            self.inertial_states.byte_size(),
            self.body_props.byte_size(),
            self.contacts.byte_size(),
            self.body_order.byte_size(),
            self.body_contact_ranges.byte_size(),
            self.body_contact_indices.byte_size(),
            range_count as u64,
        ];
        if self.primal_bg_cache.key.as_ref() != Some(&key) {
            self.primal_bg_cache.key = Some(key);
            self.primal_bg_cache.bind_groups.clear();
        }

        while self.primal_bg_cache.bind_groups.len() < range_count {
            let idx = self.primal_bg_cache.bind_groups.len();
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("avbd_primal"),
                    layout: self.primal_kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.body_states.current().buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.inertial_states.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.body_props.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.contacts.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.body_order.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.params_uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self.body_contact_ranges.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: self.body_contact_indices.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: self.solve_range_buffers[idx].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: self.solve_start_states.buffer().as_entire_binding(),
                        },
                    ],
                });
            self.primal_bg_cache.bind_groups.push(bg);
        }
    }

    fn persistent_bind_group(&mut self) -> &wgpu::BindGroup {
        let key = [
            self.body_states_cache_key(),
            self.inertial_states.byte_size(),
            self.body_props.byte_size(),
            self.contacts.byte_size(),
            self.body_order.byte_size(),
            self.body_contact_ranges.byte_size(),
            self.body_contact_indices.byte_size(),
            self.solve_start_states.byte_size(),
            0,
        ];
        if self.persistent_bg_cache.key.as_ref() != Some(&key) {
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("avbd_persistent"),
                    layout: self.persistent_kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.body_states.current().buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.inertial_states.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.body_props.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.contacts.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.body_order.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.params_uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self.body_contact_ranges.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: self.body_contact_indices.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 9,
                            resource: self.solve_start_states.buffer().as_entire_binding(),
                        },
                    ],
                });
            self.persistent_bg_cache.key = Some(key);
            self.persistent_bg_cache.bind_group = Some(bg);
        }
        self.persistent_bg_cache.bind_group.as_ref().unwrap()
    }

    fn dual_bind_group(&mut self) -> &wgpu::BindGroup {
        let key = [
            self.body_states_cache_key(),
            self.body_props.byte_size(),
            self.contacts.byte_size(),
        ];
        if self.dual_bg_cache.key.as_ref() != Some(&key) {
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("avbd_dual"),
                    layout: self.dual_kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.body_states.current().buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.body_props.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.contacts.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.params_uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.contact_count.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.solve_start_states.buffer().as_entire_binding(),
                        },
                    ],
                });
            self.dual_bg_cache.key = Some(key);
            self.dual_bg_cache.bind_group = Some(bg);
        }
        self.dual_bg_cache.bind_group.as_ref().unwrap()
    }

    /// Record one contact-stream kernel dispatch into `encoder`.
    fn record_stream_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        kernel: &ComputeKernel,
        entries: &[wgpu::BindGroupEntry],
        threads: u32,
        label: &'static str,
    ) {
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: kernel.bind_group_layout(),
                entries,
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(kernel.pipeline());
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(round_up_workgroups(threads, WORKGROUP_SIZE), 1, 1);
    }

    /// Deterministic GPU contact sort (M2(d)): chained stable LSD radix passes
    /// over the composite key (body_a, body_b, feature_id, pos_hash) — most
    /// significant word LAST — then a gather through the final permutation.
    /// Reproduces the CPU total order without downloading contacts. The whole
    /// chain (iota, 4x extract+sort, gather, copy-back) records into ONE
    /// submit; pass boundaries provide the write→read hazard ordering.
    fn sort_contacts_gpu(&mut self, count: u32) {
        if count <= 1 {
            return;
        }
        let n = count as usize;
        self.contact_stream.sort_keys.grow_if_needed(&self.ctx, n);
        self.contact_stream.sort_vals.grow_if_needed(&self.ctx, n);
        self.contact_stream
            .contacts_scratch
            .grow_if_needed(&self.ctx, n);
        self.contact_stream.sort_keys.set_len(count);
        self.contact_stream.sort_vals.set_len(count);

        let count_params: [u32; 4] = [count, 0, 0, 0];
        self.ctx.queue.write_buffer(
            &self.contact_stream.params_bufs[0],
            0,
            bytemuck::cast_slice(&count_params),
        );
        for (slot, word) in [(1usize, 0u32), (2, 1), (3, 2), (4, 3)] {
            let params: [u32; 4] = [count, word, 0, 0];
            self.ctx.queue.write_buffer(
                &self.contact_stream.params_bufs[slot],
                0,
                bytemuck::cast_slice(&params),
            );
        }

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("contact_sort"),
            });
        self.record_stream_pass(
            &mut encoder,
            &self.contact_stream.iota_kernel,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.contact_stream.sort_vals.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.contact_stream.params_bufs[0].as_entire_binding(),
                },
            ],
            count,
            "sort_iota",
        );

        // Least-significant word first: pos_hash, feature, body_b, body_a.
        for slot in [1usize, 2, 3, 4] {
            self.record_stream_pass(
                &mut encoder,
                &self.contact_stream.key_extract_kernel,
                &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.contacts.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.contact_stream.sort_vals.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.contact_stream.sort_keys.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.contact_stream.params_bufs[slot].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.contact_count.buffer().as_entire_binding(),
                    },
                ],
                count,
                "sort_key_extract",
            );
            let ContactStreamState {
                radix,
                radix_temps,
                sort_keys,
                sort_vals,
                ..
            } = &mut self.contact_stream;
            radix.record_sort_key_value(
                &self.ctx,
                &mut encoder,
                sort_keys,
                sort_vals,
                count,
                radix_temps,
            );
        }

        // Gather contacts through the permutation, then copy back.
        self.record_stream_pass(
            &mut encoder,
            &self.contact_stream.gather_kernel,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.contacts.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.contact_stream.sort_vals.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self
                        .contact_stream
                        .contacts_scratch
                        .buffer()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.contact_stream.params_bufs[0].as_entire_binding(),
                },
            ],
            count,
            "sort_gather",
        );
        let byte_len = count as u64 * std::mem::size_of::<Contact3D>() as u64;
        encoder.copy_buffer_to_buffer(
            self.contact_stream.contacts_scratch.buffer(),
            0,
            self.contacts.buffer(),
            0,
            byte_len,
        );
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// GPU CSR adjacency + active flags (M2(d)): clear → count/mark-active →
    /// scan → ranges, plus a stable incidence sort that reproduces the CPU
    /// `body_contact_indices` (contact order within each body — the contacts
    /// are already globally sorted, so per-body order is deterministic).
    /// The whole stage records into ONE submit.
    fn build_adjacency_gpu(&mut self, num_bodies: u32, contact_count: u32) {
        let nb = num_bodies as usize;
        self.contact_stream.adj_counts.grow_if_needed(&self.ctx, nb);
        self.contact_stream
            .adj_offsets
            .grow_if_needed(&self.ctx, nb);
        self.active_body_flags.grow_if_needed(&self.ctx, nb);
        self.body_contact_ranges.grow_if_needed(&self.ctx, nb);
        self.contact_stream.adj_counts.set_len(num_bodies);
        self.contact_stream.adj_offsets.set_len(num_bodies);
        self.active_body_flags.set_len(num_bodies);
        self.body_contact_ranges.set_len(num_bodies);

        let body_params: [u32; 4] = [num_bodies, 0, 0, 0];
        self.ctx.queue.write_buffer(
            &self.contact_stream.params_bufs[5],
            0,
            bytemuck::cast_slice(&body_params),
        );
        let contact_params: [u32; 4] = [contact_count, 0, 0, 0];
        self.ctx.queue.write_buffer(
            &self.contact_stream.params_bufs[6],
            0,
            bytemuck::cast_slice(&contact_params),
        );

        let inc = (2 * contact_count) as usize;
        if contact_count > 0 {
            self.contact_stream
                .incidence_keys
                .grow_if_needed(&self.ctx, inc);
            self.contact_stream
                .incidence_vals
                .grow_if_needed(&self.ctx, inc);
            self.contact_stream
                .incidence_keys
                .set_len(2 * contact_count);
            self.contact_stream
                .incidence_vals
                .set_len(2 * contact_count);
            self.body_contact_indices.grow_if_needed(&self.ctx, inc);
            self.body_contact_indices.set_len(2 * contact_count);
        }

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("adjacency"),
            });
        self.record_stream_pass(
            &mut encoder,
            &self.contact_stream.adj_clear_kernel,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.contact_stream.adj_counts.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.active_body_flags.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.contact_stream.params_bufs[5].as_entire_binding(),
                },
            ],
            num_bodies,
            "adj_clear",
        );
        if contact_count == 0 {
            self.ctx.queue.submit(Some(encoder.finish()));
            return;
        }

        self.record_stream_pass(
            &mut encoder,
            &self.contact_stream.adj_count_kernel,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.contacts.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.contact_stream.adj_counts.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.active_body_flags.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.contact_stream.params_bufs[6].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.contact_count.buffer().as_entire_binding(),
                },
            ],
            contact_count,
            "adj_count",
        );

        // offsets = exclusive_scan(copy of counts)
        encoder.copy_buffer_to_buffer(
            self.contact_stream.adj_counts.buffer(),
            0,
            self.contact_stream.adj_offsets.buffer(),
            0,
            num_bodies as u64 * 4,
        );
        {
            let ContactStreamState {
                scan,
                scan_temps,
                adj_offsets,
                ..
            } = &self.contact_stream;
            scan.record_exclusive_scan(
                &self.ctx,
                &mut encoder,
                adj_offsets.buffer(),
                num_bodies,
                scan_temps,
            );
        }

        self.record_stream_pass(
            &mut encoder,
            &self.contact_stream.adj_ranges_kernel,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.contact_stream.adj_offsets.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.contact_stream.adj_counts.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.body_contact_ranges.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.contact_stream.params_bufs[5].as_entire_binding(),
                },
            ],
            num_bodies,
            "adj_ranges",
        );

        // Incidence list: (body, contact) pairs, stable-sorted by body.
        self.record_stream_pass(
            &mut encoder,
            &self.contact_stream.adj_incidence_kernel,
            &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.contacts.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self
                        .contact_stream
                        .incidence_keys
                        .buffer()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self
                        .contact_stream
                        .incidence_vals
                        .buffer()
                        .as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.contact_stream.params_bufs[6].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.contact_count.buffer().as_entire_binding(),
                },
            ],
            contact_count,
            "adj_incidence",
        );
        {
            let ContactStreamState {
                radix,
                radix_temps,
                incidence_keys,
                incidence_vals,
                ..
            } = &mut self.contact_stream;
            radix.record_sort_key_value(
                &self.ctx,
                &mut encoder,
                incidence_keys,
                incidence_vals,
                2 * contact_count,
                radix_temps,
            );
        }
        encoder.copy_buffer_to_buffer(
            self.contact_stream.incidence_vals.buffer(),
            0,
            self.body_contact_indices.buffer(),
            0,
            (2 * contact_count) as u64 * 4,
        );
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    fn contact_prep_bind_group(&mut self) -> &wgpu::BindGroup {
        let key = [
            self.solve_start_states.byte_size(),
            self.contacts.byte_size(),
        ];
        if self.contact_prep_bg_cache.key.as_ref() != Some(&key) {
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("contact_prep"),
                    layout: self.contact_prep_kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.solve_start_states.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.contacts.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.params_uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.contact_count.buffer().as_entire_binding(),
                        },
                    ],
                });
            self.contact_prep_bg_cache.key = Some(key);
            self.contact_prep_bg_cache.bind_group = Some(bg);
        }
        self.contact_prep_bg_cache.bind_group.as_ref().unwrap()
    }

    /// Freeze the Taylor cache for all uploaded contacts (run after warm-start /
    /// contact upload, before the first primal iteration).
    fn dispatch_contact_prep(&mut self, contact_count: u32) {
        if contact_count == 0 {
            return;
        }
        let _ = self.contact_prep_bind_group();
        let bg = self.contact_prep_bg_cache.bind_group.as_ref().unwrap();
        self.run_pass("contact_prep", &self.contact_prep_kernel, bg, contact_count);
    }

    fn restitution_bind_group(&mut self) -> &wgpu::BindGroup {
        let key = [
            self.body_states_cache_key(),
            self.body_props.byte_size(),
            self.contacts.byte_size(),
            self.body_contact_ranges.byte_size(),
            self.restitution_snapshot.byte_size(),
        ];
        if self.restitution_bg_cache.key.as_ref() != Some(&key) {
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("restitution"),
                    layout: self.restitution_kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.body_states.current().buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.body_props.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.contacts.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.params_uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.body_contact_ranges.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.body_contact_indices.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self.solve_start_states.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: self.restitution_snapshot.buffer().as_entire_binding(),
                        },
                    ],
                });
            self.restitution_bg_cache.key = Some(key);
            self.restitution_bg_cache.bind_group = Some(bg);
        }
        self.restitution_bg_cache.bind_group.as_ref().unwrap()
    }

    /// Post-extract restitution pass (Jacobi, per body over its contacts).
    /// No-op unless some body has restitution > 0 and contacts exist this step.
    fn dispatch_restitution(&mut self, num_bodies: u32, had_contacts: bool) {
        if !self.any_restitution || !had_contacts {
            return;
        }
        self.restitution_snapshot
            .grow_if_needed(&self.ctx, num_bodies as usize);
        self.restitution_snapshot.set_len(num_bodies);
        let byte_len = num_bodies as u64 * std::mem::size_of::<RigidBodyState3D>() as u64;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("restitution_snapshot_3d"),
            });
        encoder.copy_buffer_to_buffer(
            self.body_states.current().buffer(),
            0,
            self.restitution_snapshot.buffer(),
            0,
            byte_len,
        );
        self.ctx.queue.submit(Some(encoder.finish()));

        let _ = self.restitution_bind_group();
        let bg = self.restitution_bg_cache.bind_group.as_ref().unwrap();
        self.run_pass("restitution", &self.restitution_kernel, bg, num_bodies);
    }

    fn extract_bind_group(&mut self) -> &wgpu::BindGroup {
        let key = [
            self.body_states_cache_key(),
            self.old_states.byte_size(),
            self.active_body_flags.byte_size(),
        ];
        if self.extract_bg_cache.key.as_ref() != Some(&key) {
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("extract_velocity"),
                    layout: self.extract_kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.body_states.current().buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.old_states.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.active_body_flags.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.params_uniform.as_entire_binding(),
                        },
                    ],
                });
            self.extract_bg_cache.key = Some(key);
            self.extract_bg_cache.bind_group = Some(bg);
        }
        self.extract_bg_cache.bind_group.as_ref().unwrap()
    }

    /// Predict + solve-start/prev-step snapshots + AABB in ONE submit.
    /// These four operations are strictly producer→consumer and used to cost
    /// four queue submits per step.
    fn dispatch_predict_and_bounds(&mut self, num_bodies: u32) {
        self.prev_step_states
            .grow_if_needed(&self.ctx, num_bodies as usize);
        self.prev_step_states.set_len(num_bodies);
        self.solve_start_states
            .grow_if_needed(&self.ctx, num_bodies as usize);
        self.solve_start_states.set_len(num_bodies);

        let _ = self.predict_bind_group();
        let _ = self.aabb_bind_group();
        let byte_len = num_bodies as u64 * std::mem::size_of::<RigidBodyState3D>() as u64;

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("predict_and_bounds_3d"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("predict"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.predict_kernel.pipeline());
            pass.set_bind_group(0, self.predict_bg_cache.bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(round_up_workgroups(num_bodies, WORKGROUP_SIZE), 1, 1);
        }
        // prev_step_states <- old_states (pre-predict pose, for accel-weighted
        // warm start next frame); solve_start_states <- post-predict bodies
        // (the Taylor-cache frame).
        encoder.copy_buffer_to_buffer(
            self.old_states.buffer(),
            0,
            self.prev_step_states.buffer(),
            0,
            byte_len,
        );
        encoder.copy_buffer_to_buffer(
            self.body_states.current().buffer(),
            0,
            self.solve_start_states.buffer(),
            0,
            byte_len,
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aabb"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.aabb_kernel.pipeline());
            pass.set_bind_group(0, self.aabb_bg_cache.bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(round_up_workgroups(num_bodies, WORKGROUP_SIZE), 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    fn dispatch_narrowphase(&mut self, _num_bodies: u32, num_pairs: u32) {
        if num_pairs == 0 {
            return;
        }
        self.sim_params.counts[2] = num_pairs;
        self.ctx.queue.write_buffer(
            &self.params_uniform,
            0,
            bytemuck::bytes_of(&self.sim_params),
        );

        let _ = self.narrowphase_bind_group();
        let bg = self.narrowphase_bg_cache.bind_group.as_ref().unwrap();
        self.run_pass("narrowphase", &self.narrowphase_kernel, bg, num_pairs);
    }

    /// Narrowphase without a CPU pair-count readback (W1): a single-thread
    /// kernel clamps the GPU-written pair demand to capacity and emits the
    /// dispatch args; a 4-byte buffer copy patches the clamped count into
    /// `SimParams.counts.z` (offset 40) so the narrowphase kernel is unchanged.
    /// One encoder, one submit, identical on native and web.
    fn dispatch_narrowphase_indirect(&mut self, pairs_buffer: &wgpu::Buffer, max_pairs: u32) {
        let args_params: [u32; 4] = [max_pairs, 0, 0, 0];
        self.ctx.queue.write_buffer(
            &self.narrowphase_args_params,
            0,
            bytemuck::cast_slice(&args_params),
        );
        let args_bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("narrowphase_args"),
                layout: self.narrowphase_args_kernel.bind_group_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.gpu_lbvh.pair_counter_buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.narrowphase_dispatch_args.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.clamped_pair_count.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.narrowphase_args_params.as_entire_binding(),
                    },
                ],
            });
        let narrow_bg = self.create_narrowphase_bind_group(pairs_buffer, "narrowphase_indirect");

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("narrowphase_indirect"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("narrowphase_args"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.narrowphase_args_kernel.pipeline());
            pass.set_bind_group(0, &args_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // Patch the clamped pair count into SimParams.counts.z. The CPU shadow
        // of counts[2] goes stale here by design; every consumer of counts.z
        // (narrowphase only) runs after this copy within the same submit, and
        // the next frame's refresh rewrites the whole uniform anyway.
        let counts_z = (std::mem::offset_of!(SimParamsGpu, counts) + 8) as u64;
        encoder.copy_buffer_to_buffer(
            &self.clamped_pair_count,
            0,
            &self.params_uniform,
            counts_z,
            4,
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("narrowphase"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.narrowphase_kernel.pipeline());
            pass.set_bind_group(0, &narrow_bg, &[]);
            pass.dispatch_workgroups_indirect(&self.narrowphase_dispatch_args, 0);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    fn dispatch_extract(&mut self, num_bodies: u32) {
        let _ = self.extract_bind_group();
        let bg = self.extract_bg_cache.bind_group.as_ref().unwrap();
        self.run_pass("extract_vel", &self.extract_kernel, bg, num_bodies);
    }

    /// GPU Luby body coloring: color bodies so no two adjacent bodies share a color.
    /// Sync variant for native (uses blocking buffer download).
    /// Run the GPU coloring dispatches (reset + N Luby iterations).
    /// Does NOT download results — caller is responsible for downloading body_colors.
    fn dispatch_gpu_coloring(&mut self, num_bodies: u32, contact_count: u32) {
        let ctx = &self.ctx;
        let contacts_buf = self.contacts.buffer();
        let cs = &mut self.gpu_coloring;

        cs.frame_counter = cs.frame_counter.wrapping_add(1);
        cs.body_colors.grow_if_needed(ctx, num_bodies as usize);
        cs.body_priorities.grow_if_needed(ctx, num_bodies as usize);
        cs.winners.grow_if_needed(ctx, num_bodies as usize);

        // Phase 1: Reset — mark all bodies uncolored, assign random priorities
        {
            let params: [u32; 4] = [num_bodies, contact_count, cs.frame_counter, 0];
            ctx.queue
                .write_buffer(&cs.params_buf, 0, bytemuck::cast_slice(&params));
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("coloring_reset"),
                layout: cs.reset_kernel.bind_group_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: cs.body_colors.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cs.body_priorities.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: cs.params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("coloring_reset"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(cs.reset_kernel.pipeline());
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(round_up_workgroups(num_bodies, WORKGROUP_SIZE), 1, 1);
            }
            ctx.queue.submit(Some(encoder.finish()));
        }

        // Phase 2: Iterative Luby coloring — all rounds in ONE submit. Each
        // round has its own tiny uniform (the round index doubles as the color
        // being assigned), and the mark/apply bind groups are cached across
        // frames keyed on the backing buffer sizes.
        let bg_key = [
            cs.body_colors.byte_size(),
            cs.winners.byte_size(),
            contacts_buf.size(),
            self.body_contact_ranges.byte_size(),
            self.body_contact_indices.byte_size(),
        ];
        if cs.round_bg_key != Some(bg_key)
            || cs.round_step_bgs.len() != COLORING_MAX_ITERATIONS as usize
        {
            cs.round_step_bgs.clear();
            cs.round_apply_bgs.clear();
            for round in 0..COLORING_MAX_ITERATIONS as usize {
                cs.round_step_bgs
                    .push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("coloring_step"),
                        layout: cs.step_kernel.bind_group_layout(),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: cs.body_colors.buffer().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: cs.body_priorities.buffer().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: contacts_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: cs.round_params[round].as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: cs.winners.buffer().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: self.body_contact_ranges.buffer().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: self.body_contact_indices.buffer().as_entire_binding(),
                            },
                        ],
                    }));
                cs.round_apply_bgs
                    .push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("coloring_apply"),
                        layout: cs.apply_kernel.bind_group_layout(),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: cs.body_colors.buffer().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: cs.winners.buffer().as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: cs.round_params[round].as_entire_binding(),
                            },
                        ],
                    }));
            }
            cs.round_bg_key = Some(bg_key);
        }

        for current_color in 0..COLORING_MAX_ITERATIONS {
            let params: [u32; 4] = [num_bodies, contact_count, current_color, 0];
            ctx.queue.write_buffer(
                &cs.round_params[current_color as usize],
                0,
                bytemuck::cast_slice(&params),
            );
        }

        // The compacted body_order carries a 64-word (offset,count) header.
        self.body_order
            .grow_if_needed(ctx, 64 + num_bodies as usize);
        self.body_order.set_len(64 + num_bodies);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("coloring_rounds"),
            });
        for round in 0..COLORING_MAX_ITERATIONS as usize {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("coloring_step"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(cs.step_kernel.pipeline());
                pass.set_bind_group(0, &cs.round_step_bgs[round], &[]);
                pass.dispatch_workgroups(round_up_workgroups(num_bodies, WORKGROUP_SIZE), 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("coloring_apply"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(cs.apply_kernel.pipeline());
                pass.set_bind_group(0, &cs.round_apply_bgs[round], &[]);
                pass.dispatch_workgroups(round_up_workgroups(num_bodies, WORKGROUP_SIZE), 1, 1);
            }
        }

        // Compaction (M2(c)): per-color counts -> header/offsets/indirect args
        // -> scatter. All recorded into the same submit; replaces the CPU
        // colors+active downloads and group build entirely.
        encoder.clear_buffer(cs.color_counts.buffer(), 0, None);
        let count_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("color_count"),
            layout: cs.color_count_kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cs.body_colors.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.active_body_flags.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cs.color_counts.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cs.params_buf.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("color_count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(cs.color_count_kernel.pipeline());
            pass.set_bind_group(0, &count_bg, &[]);
            pass.dispatch_workgroups(round_up_workgroups(num_bodies, WORKGROUP_SIZE), 1, 1);
        }
        let scan_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("color_scan_args"),
            layout: cs.color_scan_args_kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cs.color_counts.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.body_order.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cs.dispatch_args.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("color_scan_args"),
                timestamp_writes: None,
            });
            pass.set_pipeline(cs.color_scan_args_kernel.pipeline());
            pass.set_bind_group(0, &scan_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let scatter_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("color_scatter"),
            layout: cs.color_scatter_kernel.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cs.body_colors.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.active_body_flags.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cs.color_counts.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.body_order.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cs.params_buf.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("color_scatter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(cs.color_scatter_kernel.pipeline());
            pass.set_bind_group(0, &scatter_bg, &[]);
            pass.dispatch_workgroups(round_up_workgroups(num_bodies, WORKGROUP_SIZE), 1, 1);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }

    /// Append CPU-generated compound contacts to the tail of the GPU contact
    /// buffer (small upload, no download) and publish the new total count.
    fn append_compound_contacts(&mut self, gpu_count: u32, compounds: &[Contact3D]) -> u32 {
        if compounds.is_empty() {
            return gpu_count;
        }
        let mut tail = compounds.to_vec();
        for c in &mut tail {
            // Same content-derived sort disambiguator the narrowphase emits.
            c.lambda.w = f32::from_bits(cpu_contact_pos_hash(c.point.truncate()));
        }
        let total = gpu_count + tail.len() as u32;
        self.contacts.grow_if_needed(&self.ctx, total as usize);
        let offset = gpu_count as u64 * std::mem::size_of::<Contact3D>() as u64;
        self.ctx
            .queue
            .write_buffer(self.contacts.buffer(), offset, bytemuck::cast_slice(&tail));
        self.contact_count.write(&self.ctx, total);
        total
    }

    /// Free-motion pass: advance contact-free bodies straight to their inertial
    /// targets. Active flags are produced on the GPU by the adjacency count
    /// kernel (build_adjacency_gpu must run first).
    fn dispatch_free_motion(&mut self, num_bodies: u32) {
        let _ = self.free_motion_bind_group();
        let bg = self.free_motion_bg_cache.bind_group.as_ref().unwrap();
        self.run_pass("free_motion", &self.free_motion_kernel, bg, num_bodies);
    }

    /// Dispatch GPU warmstart: match new contacts against prev-frame contacts on GPU.
    /// Upload externally-provided warm contacts into the GPU `prev_contacts`
    /// buffer so the GPU warmstart kernel can match against them.
    fn upload_warm_contacts(&mut self, warm: &[Contact3D]) {
        if warm.is_empty() {
            self.prev_contact_count = 0;
            // Invalidate cached warmstart bind group (buffer contents changing)
            self.warmstart_bg_cache.key = None;
            return;
        }
        self.prev_contacts.grow_if_needed(&self.ctx, warm.len());
        self.prev_contacts.upload(&self.ctx, warm);
        self.prev_contact_count = warm.len() as u32;
        // Publish the GPU-resident prev count too (the insert kernel clamps by
        // it, W5) — otherwise explicit warm contacts never enter the map.
        self.ctx.queue.write_buffer(
            self.prev_contact_count_buf.buffer(),
            0,
            bytemuck::bytes_of(&(warm.len() as u32)),
        );
        // Invalidate cached warmstart bind group (buffer may have been regrown)
        self.warmstart_bg_cache.key = None;
    }

    /// GPU warm-start (W4a-3): build an open-addressing pair map from the
    /// previous frame's contacts, then look up each new contact in O(1)
    /// expected — replacing the old O(new x prev) linear scan. Clear + insert +
    /// match record into ONE submit (pass boundaries provide write->read
    /// ordering between the three stages).
    fn dispatch_gpu_warmstart(&mut self, new_count: u32) {
        if self.prev_contact_count == 0 || new_count == 0 {
            return;
        }

        // Grow the map if either count would push the load factor past ~0.5.
        let want_cap =
            (2 * self.prev_contact_count.max(new_count).max(1)).next_power_of_two();
        if want_cap > self.ws_map_cap {
            self.ws_map_hash.grow_if_needed(&self.ctx, want_cap as usize);
            self.ws_map_entry.grow_if_needed(&self.ctx, want_cap as usize);
            self.ws_map_cap = want_cap;
            self.warmstart_bg_cache.key = None;
            self.warmstart_insert_bg_cache.key = None;
        }

        let params = WarmstartParamsGpu {
            prev_count: self.prev_contact_count,
            new_count,
            alpha: AVBD_WARMSTART_ALPHA,
            gamma: self.warmstart_decay,
            cap_mask: self.ws_map_cap - 1,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        self.ctx.queue.write_buffer(
            &self.warmstart_params_uniform,
            0,
            bytemuck::bytes_of(&params),
        );

        // Insert bind group: prev_contacts, params, map_hash, map_entry.
        let insert_key = [
            self.prev_contacts.byte_size(),
            self.ws_map_hash.byte_size(),
            self.ws_map_entry.byte_size(),
        ];
        if self.warmstart_insert_bg_cache.key != Some(insert_key) {
            let layout = self.warmstart_insert_kernel.bind_group_layout();
            let bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("warmstart_insert_bg"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.prev_contacts.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.warmstart_params_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.ws_map_hash.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.ws_map_entry.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.prev_contact_count_buf.buffer().as_entire_binding(),
                    },
                ],
            });
            self.warmstart_insert_bg_cache.key = Some(insert_key);
            self.warmstart_insert_bg_cache.bind_group = Some(bg);
        }

        // Match bind group: prev_contacts, new_contacts, params, count_buf,
        // map_hash, map_entry (8-storage-buffer WebGPU limit respected — 5
        // storage + 1 uniform).
        let match_key = [
            self.prev_contacts.byte_size(),
            self.contacts.byte_size(),
            self.ws_map_hash.byte_size(),
        ];
        if self.warmstart_bg_cache.key != Some(match_key) {
            let layout = self.warmstart_kernel.bind_group_layout();
            let bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("warmstart_match_bg"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.prev_contacts.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.contacts.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.warmstart_params_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.contact_count.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.ws_map_hash.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.ws_map_entry.buffer().as_entire_binding(),
                    },
                ],
            });
            self.warmstart_bg_cache.key = Some(match_key);
            self.warmstart_bg_cache.bind_group = Some(bg);
        }

        let insert_bg = self.warmstart_insert_bg_cache.bind_group.as_ref().unwrap();
        let match_bg = self.warmstart_bg_cache.bind_group.as_ref().unwrap();

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("warmstart"),
            });
        // Clear the map to empty (0). Only the live prefix is touched by the
        // insert/match dispatches, so clearing that prefix suffices.
        let hash_bytes = (self.ws_map_cap as u64) * std::mem::size_of::<u32>() as u64;
        encoder.clear_buffer(self.ws_map_hash.buffer(), 0, Some(hash_bytes));
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warmstart_insert"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.warmstart_insert_kernel.pipeline());
            pass.set_bind_group(0, insert_bg, &[]);
            pass.dispatch_workgroups(round_up_workgroups(self.prev_contact_count, 64), 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("warmstart_match"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.warmstart_kernel.pipeline());
            pass.set_bind_group(0, match_bg, &[]);
            pass.dispatch_workgroups(round_up_workgroups(new_count, 64), 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Swap contacts → prev_contacts for next frame's warmstarting.
    ///
    /// `contact_count` is an exact count (native) or an upper bound (web, W5).
    /// The copied prefix is what the next frame's warm-start insert reads,
    /// clamped by the GPU-resident true count in `prev_contact_count_buf`.
    fn swap_contact_buffers(&mut self, contact_count: u32) {
        // Copy current contacts to prev_contacts buffer
        if contact_count > 0 {
            self.prev_contacts
                .grow_if_needed(&self.ctx, contact_count as usize);
            let mut encoder = self
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let byte_count = (contact_count as u64) * std::mem::size_of::<Contact3D>() as u64;
            encoder.copy_buffer_to_buffer(
                self.contacts.buffer(),
                0,
                self.prev_contacts.buffer(),
                0,
                byte_count,
            );
            // Publish the GPU-side true count for the next frame's insert clamp.
            encoder.copy_buffer_to_buffer(
                self.contact_count.buffer(),
                0,
                self.prev_contact_count_buf.buffer(),
                0,
                4,
            );
            self.ctx.queue.submit(Some(encoder.finish()));
            self.prev_contact_count = contact_count;
            // Invalidate cached bind group since buffer contents changed
            self.warmstart_bg_cache.key = None;
        } else {
            self.prev_contact_count = 0;
        }
    }

    fn run_pass(
        &self,
        label: &str,
        kernel: &ComputeKernel,
        bind_group: &wgpu::BindGroup,
        thread_count: u32,
    ) {
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(kernel.pipeline());
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(round_up_workgroups(thread_count, WORKGROUP_SIZE), 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }
}

// ---------------------------------------------------------------------------
// Body contact adjacency (per-body contact index lists for primal solver)
// ---------------------------------------------------------------------------

/// CPU mirror of the WGSL `contact_pos_hash` (narrowphase sort disambiguator),
/// used when appending CPU-generated compound contacts to the GPU stream.
fn cpu_contact_pos_hash(p: Vec3) -> u32 {
    let q = |v: f32| (v * 1024.0).floor() as i32 as u32;
    let mut h = q(p.x).wrapping_mul(0x8da6_b343);
    h ^= q(p.y).wrapping_mul(0xd816_3841);
    h ^= q(p.z).wrapping_mul(0xcb1a_b31f);
    h.wrapping_mul(0x9e37_79b1)
}

// ---------------------------------------------------------------------------
// Contact persistence + collision events
// ---------------------------------------------------------------------------

/// Tracks contacts frame-to-frame for warm-starting and collision events.
pub struct ContactPersistence3D {
    prev_contacts: Vec<Contact3D>,
}

impl ContactPersistence3D {
    pub fn new() -> Self {
        Self {
            prev_contacts: Vec::new(),
        }
    }

    /// Update with new contacts. Returns collision events (Started / Ended).
    pub fn update(&mut self, new_contacts: &[Contact3D]) -> Vec<CollisionEvent> {
        let mut events = Vec::new();

        let prev_pairs: HashSet<(u32, u32)> = self
            .prev_contacts
            .iter()
            .map(|c| (c.body_a.min(c.body_b), c.body_a.max(c.body_b)))
            .collect();

        let new_pairs: HashSet<(u32, u32)> = new_contacts
            .iter()
            .map(|c| (c.body_a.min(c.body_b), c.body_a.max(c.body_b)))
            .collect();

        // Started: in new but not in prev
        for &(a, b) in &new_pairs {
            if !prev_pairs.contains(&(a, b)) {
                events.push(CollisionEvent::Started {
                    body_a: BodyHandle::new(a, 0),
                    body_b: BodyHandle::new(b, 0),
                });
            }
        }

        // Ended: in prev but not in new
        for &(a, b) in &prev_pairs {
            if !new_pairs.contains(&(a, b)) {
                events.push(CollisionEvent::Ended {
                    body_a: BodyHandle::new(a, 0),
                    body_b: BodyHandle::new(b, 0),
                });
            }
        }

        self.prev_contacts = new_contacts.to_vec();
        events
    }

    /// Get previous frame's contacts (for warm-starting).
    pub fn prev_contacts(&self) -> &[Contact3D] {
        &self.prev_contacts
    }
}

impl Default for ContactPersistence3D {
    fn default() -> Self {
        Self::new()
    }
}
