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
mod extract_velocity_wgsl;
mod narrowphase_wgsl;
mod predict_wgsl;
mod warmstart_wgsl;

pub use avbd_solve_wgsl::{AVBD_DUAL_WGSL, AVBD_PRIMAL_WGSL};
pub use coloring_wgsl::{COLORING_RESET_WGSL, COLORING_STEP_WGSL};
pub use extract_velocity_wgsl::EXTRACT_VELOCITY_WGSL;
pub use narrowphase_wgsl::NARROWPHASE_WGSL;
pub use predict_wgsl::PREDICT_WGSL;
pub use warmstart_wgsl::WARMSTART_MATCH_WGSL;

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};
use rubble_gpu::{
    round_up_workgroups, BroadphaseBreakdownMs, ComputeKernel, GpuAtomicCounter, GpuBuffer,
    GpuContext, PingPongBuffer,
};
use rubble_math::{
    Aabb3D, BodyHandle, CollisionEvent, Contact3D, RigidBodyProps3D, RigidBodyState3D,
};
use rubble_primitives::GpuLbvh;
use rubble_shapes3d::{
    BoxData, CapsuleData, CompoundChildGpu, CompoundShapeGpu, ConvexHullData, ConvexVertex3D,
    SphereData,
};
use std::collections::HashSet;
const WORKGROUP_SIZE: u32 = 64;
const AVBD_WARMSTART_ALPHA: f32 = 0.99;
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
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SimParamsGpu {
    pub gravity: [f32; 4],
    pub solver: [f32; 4], // (dt, beta, k_start, max_penalty)
    pub counts: [u32; 4], // (num_bodies, solver_iterations, pair_count, flags)
    /// Narrowphase/solver quality knobs:
    /// x = contact_offset (prediction distance for speculative contacts)
    /// y = restitution_threshold (velocity below which restitution = 0)
    /// z = penetration_slop (allowable penetration before correction)
    /// w = reserved
    pub quality: [f32; 4],
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
}

// ---------------------------------------------------------------------------
// Compile-time GPU layout validation
// ---------------------------------------------------------------------------

const _: () = assert!(std::mem::size_of::<SimParamsGpu>() == 64);
const _: () = assert!(std::mem::size_of::<GpuPair>() == 8);
const _: () = assert!(std::mem::size_of::<SolveRangeGpu>() == 8);
const _: () = assert!(std::mem::size_of::<WarmstartParamsGpu>() == 16);

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
    warmstart_kernel: ComputeKernel,

    // Storage buffers
    body_states: PingPongBuffer<RigidBodyState3D>,
    body_props: GpuBuffer<RigidBodyProps3D>,
    old_states: GpuBuffer<RigidBodyState3D>,
    prev_step_states: GpuBuffer<RigidBodyState3D>,
    inertial_states: GpuBuffer<RigidBodyState3D>,
    aabbs: GpuBuffer<Aabb3D>,
    contacts: GpuBuffer<Contact3D>,
    contact_count: GpuAtomicCounter,
    pairs: GpuBuffer<GpuPair>,
    pair_count: GpuAtomicCounter,
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

    // Persistent contact buffers for GPU-side warmstarting
    prev_contacts: GpuBuffer<Contact3D>,
    prev_contact_count: u32,
    warmstart_params_uniform: wgpu::Buffer,
    warmstart_bg_cache: CachedBindGroup<[u64; 3]>,

    // Cached body coloring for steady-state contact graphs.
    cached_body_graph: Vec<(u32, u32)>,
    cached_body_order: Vec<u32>,
    cached_color_groups: Vec<(u32, u32)>,
    cached_color_num_bodies: u32,

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
    /// Uniform buffer for plane data in narrowphase (avoids a storage binding).
    plane_params_uniform: wgpu::Buffer,

    // Cached bind groups (reused while backing buffers stay stable).
    predict_bg_cache: CachedBindGroup<[u64; 4]>,
    aabb_bg_cache: CachedBindGroup<[u64; 9]>,
    narrowphase_bg_cache: CachedBindGroup<[u64; 9]>,
    free_motion_bg_cache: CachedBindGroup<[u64; 4]>,
    primal_bg_cache: CachedBindGroupVec<[u64; 8]>,
    dual_bg_cache: CachedBindGroup<[u64; 3]>,
    extract_bg_cache: CachedBindGroup<[u64; 3]>,

    // GPU broadphase
    gpu_lbvh: GpuLbvh,
}

/// GPU coloring state for Luby-style parallel body graph coloring.
struct GpuColoringState {
    reset_kernel: ComputeKernel,
    step_kernel: ComputeKernel,
    body_colors: GpuBuffer<u32>,
    body_priorities: GpuBuffer<u32>,
    params_buf: wgpu::Buffer,
    frame_counter: u32,
}

impl GpuColoringState {
    fn new(ctx: &GpuContext, max_bodies: usize) -> Self {
        let reset_kernel =
            ComputeKernel::from_wgsl(ctx, coloring_wgsl::COLORING_RESET_WGSL, "main");
        let step_kernel = ComputeKernel::from_wgsl(ctx, coloring_wgsl::COLORING_STEP_WGSL, "main");
        let body_colors = GpuBuffer::new(ctx, max_bodies.max(1));
        let body_priorities = GpuBuffer::new(ctx, max_bodies.max(1));
        let params_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coloring params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            reset_kernel,
            step_kernel,
            body_colors,
            body_priorities,
            params_buf,
            frame_counter: 0,
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
        let free_motion_kernel = ComputeKernel::from_wgsl(&ctx, FREE_MOTION_WGSL, "main");
        let primal_kernel = ComputeKernel::from_wgsl(&ctx, AVBD_PRIMAL_WGSL, "main");
        let dual_kernel = ComputeKernel::from_wgsl(&ctx, AVBD_DUAL_WGSL, "main");
        let extract_kernel = ComputeKernel::from_wgsl(&ctx, EXTRACT_VELOCITY_WGSL, "main");
        let warmstart_kernel = ComputeKernel::from_wgsl(&ctx, WARMSTART_MATCH_WGSL, "main");

        let body_states = PingPongBuffer::new(&ctx, max_bodies);
        let body_props = GpuBuffer::new(&ctx, max_bodies);
        let old_states = GpuBuffer::new(&ctx, max_bodies);
        let prev_step_states = GpuBuffer::new(&ctx, max_bodies);
        let inertial_states = GpuBuffer::new(&ctx, max_bodies);
        let aabbs = GpuBuffer::new(&ctx, max_bodies);
        let contacts = GpuBuffer::new(&ctx, max_contacts);
        let contact_count = GpuAtomicCounter::new(&ctx);
        let pairs = GpuBuffer::new(&ctx, max_pairs);
        let pair_count = GpuAtomicCounter::new(&ctx);
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

        let plane_params_uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PlaneParams uniform"),
            size: std::mem::size_of::<PlaneParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let prev_contacts: GpuBuffer<Contact3D> = GpuBuffer::new(&ctx, max_contacts);
        let warmstart_params_uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WarmstartParams uniform"),
            size: std::mem::size_of::<WarmstartParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gpu_lbvh = GpuLbvh::new(&ctx, max_bodies);
        let gpu_coloring = GpuColoringState::new(&ctx, max_bodies);

        Self {
            ctx,
            predict_kernel,
            aabb_kernel,
            narrowphase_kernel,
            free_motion_kernel,
            primal_kernel,
            dual_kernel,
            extract_kernel,
            warmstart_kernel,
            body_states,
            body_props,
            old_states,
            prev_step_states,
            inertial_states,
            aabbs,
            contacts,
            contact_count,
            pairs,
            pair_count,
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
            prev_contacts,
            prev_contact_count: 0,
            warmstart_params_uniform,
            warmstart_bg_cache: CachedBindGroup::default(),
            cached_body_graph: Vec::new(),
            cached_body_order: Vec::new(),
            cached_color_groups: Vec::new(),
            cached_color_num_bodies: 0,
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
                quality: [0.02, 0.5, 0.005, 0.0], // contact_offset, restitution_threshold, penetration_slop, reserved
            },
            warmstart_decay: 0.95,
            params_uniform,
            solve_range_buffers: Vec::new(),
            plane_params_uniform,
            predict_bg_cache: CachedBindGroup::default(),
            aabb_bg_cache: CachedBindGroup::default(),
            narrowphase_bg_cache: CachedBindGroup::default(),
            free_motion_bg_cache: CachedBindGroup::default(),
            primal_bg_cache: CachedBindGroupVec::default(),
            dual_bg_cache: CachedBindGroup::default(),
            extract_bg_cache: CachedBindGroup::default(),
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

        self.warmstart_decay = warmstart_decay;
        self.sim_params = SimParamsGpu {
            gravity: [gravity.x, gravity.y, gravity.z, 0.0],
            solver: [dt, beta, k_start, MAX_CONTACT_PENALTY],
            counts: [states.len() as u32, solver_iterations, 0, 0],
            quality: self.sim_params.quality, // preserve quality params
        };
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
    ) {
        self.sim_params.quality = [contact_offset, restitution_threshold, penetration_slop, 0.0];
    }

    /// Run predict → AABB → broadphase → narrowphase (shared by both step variants).
    ///
    /// When a broadphase pair involves a compound shape (SHAPE_COMPOUND = 5),
    /// the pair is expanded on the CPU into individual child-vs-body pairs.
    /// This avoids adding compound-specific bindings/logic to the GPU narrowphase.
    fn run_detection(&mut self, num_bodies: u32) -> Vec<Contact3D> {
        self.contact_count.reset(&self.ctx);
        self.pair_count.reset(&self.ctx);

        self.dispatch_predict(num_bodies);
        self.snapshot_prev_step_states(num_bodies);
        self.dispatch_aabb(num_bodies);

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
                    self.pair_count.write(&self.ctx, count);
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
            let pair_count = self.gpu_lbvh.read_pair_count(&self.ctx, pair_thread_count);
            let pair_buffer = self.gpu_lbvh.pair_buffer().clone();
            self.dispatch_narrowphase_with_source(num_bodies, pair_count, &pair_buffer);
        }

        // Buffer overflow recovery: if contact count exceeded capacity, grow and retry
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
                let pair_count = self.gpu_lbvh.read_pair_count(&self.ctx, pair_thread_count);
                let pair_buffer = self.gpu_lbvh.pair_buffer().clone();
                self.dispatch_narrowphase_with_source(num_bodies, pair_count, &pair_buffer);
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
    fn run_colored_solve(
        &mut self,
        num_bodies: u32,
        solver_iterations: u32,
        contacts: &mut [Contact3D],
    ) {
        self.apply_free_motion(num_bodies, contacts);
        if contacts.is_empty() {
            return;
        }

        // GPU graph coloring: Luby-style parallel body coloring on GPU.
        let graph_key = body_graph_key_3d(contacts);
        let (body_order, color_groups) =
            if self.cached_color_num_bodies == num_bodies && self.cached_body_graph == graph_key {
                (
                    self.cached_body_order.clone(),
                    self.cached_color_groups.clone(),
                )
            } else {
                let result = self.gpu_color_bodies(num_bodies, contacts);
                self.cached_color_num_bodies = num_bodies;
                self.cached_body_graph = graph_key;
                self.cached_body_order = result.0.clone();
                self.cached_color_groups = result.1.clone();
                result
            };
        let adjacency = build_body_contact_adjacency(num_bodies, contacts);
        self.contacts.upload(&self.ctx, contacts);
        self.contact_count.write(&self.ctx, contacts.len() as u32);
        self.body_order.upload(&self.ctx, &body_order);
        self.body_contact_ranges
            .upload(&self.ctx, &adjacency.ranges);
        self.body_contact_indices
            .upload(&self.ctx, &adjacency.indices);
        self.write_solve_ranges(&color_groups);
        self.sync_primal_bind_groups(color_groups.len());
        let _ = self.dual_bind_group();
        let contact_count = contacts.len() as u32;
        let primal_bind_groups = &self.primal_bg_cache.bind_groups;
        let dual_bind_group = self.dual_bg_cache.bind_group.as_ref().unwrap();
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("avbd_solve_batch_3d"),
            });
        for _ in 0..solver_iterations {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("avbd_primal_3d"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(self.primal_kernel.pipeline());
                for (range_idx, &(_, count)) in color_groups.iter().enumerate() {
                    if count == 0 {
                        continue;
                    }
                    pass.set_bind_group(0, &primal_bind_groups[range_idx], &[]);
                    pass.dispatch_workgroups(round_up_workgroups(count, WORKGROUP_SIZE), 1, 1);
                }
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

        let gpu_count = self.contact_count.read(&self.ctx) as usize;
        let mut contacts = if gpu_count > 0 {
            self.contacts.set_len(gpu_count as u32);
            let mut c = self.contacts.download(&self.ctx);
            c.truncate(gpu_count);
            c
        } else {
            Vec::new()
        };

        // Merge CPU-generated compound contacts with GPU narrowphase contacts
        contacts.extend(compound_contacts);

        if !contacts.is_empty() {
            self.run_colored_solve(num_bodies, solver_iterations, &mut contacts);
        }

        self.dispatch_extract(num_bodies);
        let states = self.body_states.download(&self.ctx);
        self.body_states.swap();
        states
    }

    /// Run the full GPU physics step with warm-starting support.
    /// Returns (updated_states, new_contacts) so the caller can track persistence.
    ///
    /// Warmstarting is handled entirely on GPU using the internally maintained
    /// `prev_contacts` buffer from the previous frame. The `warm_contacts` parameter
    /// is accepted for API compatibility but ignored — GPU warmstart is always used.
    pub fn step_with_contacts(
        &mut self,
        num_bodies: u32,
        solver_iterations: u32,
        _warm_contacts: Option<&[Contact3D]>,
    ) -> (Vec<RigidBodyState3D>, Vec<Contact3D>) {
        if num_bodies == 0 {
            return (Vec::new(), Vec::new());
        }

        let compound_contacts = self.run_detection(num_bodies);

        let gpu_count = self.contact_count.read(&self.ctx) as usize;

        // GPU warmstart: match new contacts against prev-frame contacts on GPU
        if gpu_count > 0 {
            self.contacts.set_len(gpu_count as u32);
            self.dispatch_gpu_warmstart(gpu_count as u32);
        }

        let mut contacts = if gpu_count > 0 {
            self.contacts.set_len(gpu_count as u32);
            let mut c = self.contacts.download(&self.ctx);
            c.truncate(gpu_count);
            c
        } else {
            Vec::new()
        };

        // Merge CPU-generated compound contacts with GPU narrowphase contacts
        contacts.extend(compound_contacts);

        self.run_colored_solve(num_bodies, solver_iterations, &mut contacts);

        // Download contacts after solve (lambdas are updated by GPU)
        let final_contacts = if !contacts.is_empty() {
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
        let states = self.body_states.download(&self.ctx);
        self.body_states.swap();
        (states, final_contacts)
    }

    /// Timed version of `step_with_contacts` that populates per-phase timings.
    pub fn step_with_contacts_timed(
        &mut self,
        num_bodies: u32,
        solver_iterations: u32,
        _warm_contacts: Option<&[Contact3D]>,
        timings: &mut rubble_gpu::StepTimingsMs,
    ) -> (Vec<RigidBodyState3D>, Vec<Contact3D>) {
        use std::time::Instant;

        if num_bodies == 0 {
            return (Vec::new(), Vec::new());
        }

        let compound_contacts = self.run_detection_timed(num_bodies, timings);

        let t_cf = Instant::now();
        let gpu_count = self.contact_count.read(&self.ctx) as usize;

        // GPU warmstart before download
        if gpu_count > 0 {
            self.contacts.set_len(gpu_count as u32);
            self.dispatch_gpu_warmstart(gpu_count as u32);
        }

        let mut contacts = if gpu_count > 0 {
            self.contacts.set_len(gpu_count as u32);
            let mut c = self.contacts.download(&self.ctx);
            c.truncate(gpu_count);
            c
        } else {
            Vec::new()
        };
        contacts.extend(compound_contacts);
        timings.contact_fetch_ms = t_cf.elapsed().as_secs_f32() * 1000.0;

        let t_solve = Instant::now();
        self.run_colored_solve(num_bodies, solver_iterations, &mut contacts);

        let final_contacts = if !contacts.is_empty() {
            let fc = self.download_contacts();
            let solved_count = fc.len() as u32;
            self.swap_contact_buffers(solved_count);
            fc
        } else {
            self.prev_contact_count = 0;
            Vec::new()
        };
        timings.solve_ms = t_solve.elapsed().as_secs_f32() * 1000.0;

        let t_ext = Instant::now();
        self.dispatch_extract(num_bodies);
        let states = self.body_states.download(&self.ctx);
        self.body_states.swap();
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
        self.pair_count.reset(&self.ctx);

        let t0 = Instant::now();
        self.dispatch_predict(num_bodies);
        self.snapshot_prev_step_states(num_bodies);
        self.dispatch_aabb(num_bodies);
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
                    self.pair_count.write(&self.ctx, count);
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
        let external_pair_count = if !requires_cpu_pair_stage && pair_thread_count > 0 {
            let t_readback = Instant::now();
            let count = self.gpu_lbvh.read_pair_count(&self.ctx, pair_thread_count);
            broadphase.readback_ms += t_readback.elapsed().as_secs_f32() * 1000.0;
            count
        } else {
            0
        };
        timings.set_broadphase_breakdown(broadphase);
        debug_assert!(timings.broadphase_ms <= t1.elapsed().as_secs_f32() * 1000.0 + 0.5);

        let t2 = Instant::now();
        if requires_cpu_pair_stage {
            self.dispatch_narrowphase(num_bodies, pair_count);
        } else if external_pair_count > 0 {
            let pair_buffer = self.gpu_lbvh.pair_buffer().clone();
            self.dispatch_narrowphase_with_source(num_bodies, external_pair_count, &pair_buffer);
        }

        let contact_count_val = self.contact_count.read(&self.ctx);
        let capacity = self.contacts.capacity();
        if contact_count_val > capacity {
            let new_cap = (contact_count_val as usize) * 2;
            self.contacts.grow_if_needed(&self.ctx, new_cap);
            self.contact_count.reset(&self.ctx);
            if requires_cpu_pair_stage {
                self.dispatch_narrowphase(num_bodies, pair_count);
            } else if external_pair_count > 0 {
                let pair_buffer = self.gpu_lbvh.pair_buffer().clone();
                self.dispatch_narrowphase_with_source(
                    num_bodies,
                    external_pair_count,
                    &pair_buffer,
                );
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
        self.pair_count.reset(&self.ctx);

        let t0 = Instant::now();
        self.dispatch_predict(num_bodies);
        self.snapshot_prev_step_states(num_bodies);
        self.dispatch_aabb(num_bodies);
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
                    self.pair_count.write(&self.ctx, count);
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
        let external_pair_count = if !requires_cpu_pair_stage && pair_thread_count > 0 {
            let t_readback = Instant::now();
            let count = self
                .gpu_lbvh
                .read_pair_count_async(&self.ctx, pair_thread_count)
                .await;
            broadphase.readback_ms += t_readback.elapsed().as_secs_f32() * 1000.0;
            count
        } else {
            0
        };
        timings.set_broadphase_breakdown(broadphase);
        debug_assert!(timings.broadphase_ms <= t1.elapsed().as_secs_f32() * 1000.0 + 0.5);

        let t2 = Instant::now();
        if requires_cpu_pair_stage {
            self.dispatch_narrowphase(num_bodies, pair_count);
        } else if external_pair_count > 0 {
            let pair_buffer = self.gpu_lbvh.pair_buffer().clone();
            self.dispatch_narrowphase_with_source(num_bodies, external_pair_count, &pair_buffer);
        }

        let contact_count_val = self.contact_count.read_async(&self.ctx).await;
        let capacity = self.contacts.capacity();
        if contact_count_val > capacity {
            let new_cap = (contact_count_val as usize) * 2;
            self.contacts.grow_if_needed(&self.ctx, new_cap);
            self.contact_count.reset(&self.ctx);
            if requires_cpu_pair_stage {
                self.dispatch_narrowphase(num_bodies, pair_count);
            } else if external_pair_count > 0 {
                let pair_buffer = self.gpu_lbvh.pair_buffer().clone();
                self.dispatch_narrowphase_with_source(
                    num_bodies,
                    external_pair_count,
                    &pair_buffer,
                );
            }
        }
        timings.narrowphase_ms = t2.elapsed().as_secs_f32() * 1000.0;

        cpu_compound_contacts
    }

    /// Async version of `step_with_contacts` for WASM/WebGPU.
    ///
    /// Warmstarting is handled entirely on GPU using the internally maintained
    /// `prev_contacts` buffer. The `_warm_contacts` parameter is accepted for
    /// API compatibility but ignored.
    #[cfg(target_arch = "wasm32")]
    pub async fn step_with_contacts_async(
        &mut self,
        num_bodies: u32,
        solver_iterations: u32,
        _warm_contacts: Option<&[Contact3D]>,
        timings: &mut rubble_gpu::StepTimingsMs,
    ) -> (Vec<RigidBodyState3D>, Vec<Contact3D>) {
        use rubble_gpu::web_time::Instant;

        if num_bodies == 0 {
            return (Vec::new(), Vec::new());
        }

        let compound_contacts = self.run_detection_async(num_bodies, timings).await;

        let t_cf = Instant::now();
        let gpu_count = self.contact_count.read_async(&self.ctx).await as usize;

        // GPU warmstart before download
        if gpu_count > 0 {
            self.contacts.set_len(gpu_count as u32);
            self.dispatch_gpu_warmstart(gpu_count as u32);
        }

        let mut contacts = if gpu_count > 0 {
            self.download_contacts_exact_async(gpu_count).await
        } else {
            Vec::new()
        };

        contacts.extend(compound_contacts);
        timings.contact_fetch_ms = t_cf.elapsed().as_secs_f32() * 1000.0;

        let t_solve = Instant::now();
        self.run_colored_solve(num_bodies, solver_iterations, &mut contacts);

        let final_contacts = if !contacts.is_empty() {
            let fc = self.download_contacts_exact_async(contacts.len()).await;
            let solved_count = fc.len() as u32;
            self.swap_contact_buffers(solved_count);
            fc
        } else {
            self.prev_contact_count = 0;
            Vec::new()
        };
        timings.solve_ms = t_solve.elapsed().as_secs_f32() * 1000.0;

        let t_ext = Instant::now();
        self.dispatch_extract(num_bodies);
        let states = self.body_states.download_async(&self.ctx).await;
        timings.extract_ms = t_ext.elapsed().as_secs_f32() * 1000.0;

        (states, final_contacts)
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

    fn write_solve_ranges(&mut self, color_groups: &[(u32, u32)]) {
        if color_groups.is_empty() {
            return;
        }

        self.ensure_solve_range_buffers(color_groups.len());

        for (idx, &(offset, count)) in color_groups.iter().enumerate() {
            let range = SolveRangeGpu { offset, count };
            self.ctx.queue.write_buffer(
                &self.solve_range_buffers[idx],
                0,
                bytemuck::bytes_of(&range),
            );
        }
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
                    ],
                });
            self.primal_bg_cache.bind_groups.push(bg);
        }
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
                    ],
                });
            self.dual_bg_cache.key = Some(key);
            self.dual_bg_cache.bind_group = Some(bg);
        }
        self.dual_bg_cache.bind_group.as_ref().unwrap()
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

    fn snapshot_prev_step_states(&mut self, num_bodies: u32) {
        self.prev_step_states
            .grow_if_needed(&self.ctx, num_bodies as usize);
        self.prev_step_states.set_len(num_bodies);

        let byte_len = num_bodies as u64 * std::mem::size_of::<RigidBodyState3D>() as u64;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("snapshot_prev_step_3d"),
            });
        encoder.copy_buffer_to_buffer(
            self.old_states.buffer(),
            0,
            self.prev_step_states.buffer(),
            0,
            byte_len,
        );
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    fn dispatch_predict(&mut self, num_bodies: u32) {
        let _ = self.predict_bind_group();
        let bg = self.predict_bg_cache.bind_group.as_ref().unwrap();
        self.run_pass("predict", &self.predict_kernel, bg, num_bodies);
    }

    fn dispatch_aabb(&mut self, num_bodies: u32) {
        let _ = self.aabb_bind_group();
        let bg = self.aabb_bg_cache.bind_group.as_ref().unwrap();
        self.run_pass("aabb", &self.aabb_kernel, bg, num_bodies);
    }

    fn apply_free_motion(&mut self, num_bodies: u32, contacts: &[Contact3D]) {
        let mut active = vec![0u32; num_bodies as usize];
        for c in contacts {
            active[c.body_a as usize] = 1;
            active[c.body_b as usize] = 1;
        }
        self.active_body_flags.upload(&self.ctx, &active);
        let _ = self.free_motion_bind_group();
        let bg = self.free_motion_bg_cache.bind_group.as_ref().unwrap();
        self.run_pass("free_motion", &self.free_motion_kernel, bg, num_bodies);
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

    fn dispatch_narrowphase_with_source(
        &mut self,
        _num_bodies: u32,
        num_pairs: u32,
        pairs_buffer: &wgpu::Buffer,
    ) {
        if num_pairs == 0 {
            return;
        }
        self.sim_params.counts[2] = num_pairs;
        self.ctx.queue.write_buffer(
            &self.params_uniform,
            0,
            bytemuck::bytes_of(&self.sim_params),
        );

        let bg = self.create_narrowphase_bind_group(pairs_buffer, "narrowphase_external");
        self.run_pass("narrowphase", &self.narrowphase_kernel, &bg, num_pairs);
    }

    fn dispatch_extract(&mut self, num_bodies: u32) {
        let _ = self.extract_bind_group();
        let bg = self.extract_bg_cache.bind_group.as_ref().unwrap();
        self.run_pass("extract_vel", &self.extract_kernel, bg, num_bodies);
    }

    /// GPU Luby body coloring: color bodies so no two adjacent bodies share a color.
    fn gpu_color_bodies(
        &mut self,
        num_bodies: u32,
        contacts: &[Contact3D],
    ) -> (Vec<u32>, Vec<(u32, u32)>) {
        if contacts.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // Borrow fields individually to avoid conflicting mutable borrows on self.
        let ctx = &self.ctx;
        let contacts_buf = self.contacts.buffer();
        let cs = &mut self.gpu_coloring;

        cs.frame_counter = cs.frame_counter.wrapping_add(1);
        let contact_count = contacts.len() as u32;

        // Grow buffers if needed
        cs.body_colors
            .grow_if_needed(&self.ctx, num_bodies as usize);
        cs.body_priorities
            .grow_if_needed(&self.ctx, num_bodies as usize);

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

        // Phase 2: Iterative Luby coloring (fixed iteration count, no blocking reads)
        let max_iterations = 32u32;
        for current_color in 0..max_iterations {
            let params: [u32; 4] = [num_bodies, contact_count, current_color, 0];
            ctx.queue
                .write_buffer(&cs.params_buf, 0, bytemuck::cast_slice(&params));

            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                        resource: cs.params_buf.as_entire_binding(),
                    },
                ],
            });
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("coloring_step"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(cs.step_kernel.pipeline());
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(round_up_workgroups(num_bodies, WORKGROUP_SIZE), 1, 1);
            }
            ctx.queue.submit(Some(encoder.finish()));
        }

        // Phase 3: Download colors and build body_order / color_groups on CPU
        cs.body_colors.set_len(num_bodies);
        let colors = cs.body_colors.download(ctx);

        let mut active = vec![false; num_bodies as usize];
        for c in contacts {
            active[c.body_a as usize] = true;
            active[c.body_b as usize] = true;
        }

        let mut body_order = Vec::new();
        let mut groups = Vec::new();
        for color in 0..max_iterations {
            let offset = body_order.len() as u32;
            for (body_idx, &body_color) in colors.iter().enumerate() {
                if active[body_idx] && body_color == color {
                    body_order.push(body_idx as u32);
                }
            }
            let count = body_order.len() as u32 - offset;
            if count > 0 {
                groups.push((offset, count));
            }
        }

        (body_order, groups)
    }

    /// Dispatch GPU warmstart: match new contacts against prev-frame contacts on GPU.
    fn dispatch_gpu_warmstart(&mut self, new_count: u32) {
        if self.prev_contact_count == 0 || new_count == 0 {
            return;
        }

        // Write warmstart params
        let params = WarmstartParamsGpu {
            prev_count: self.prev_contact_count,
            new_count,
            alpha: AVBD_WARMSTART_ALPHA,
            gamma: self.warmstart_decay,
        };
        self.ctx.queue.write_buffer(
            &self.warmstart_params_uniform,
            0,
            bytemuck::bytes_of(&params),
        );

        // Build or reuse bind group (keyed on buffer sizes to detect regrowth)
        let key = [self.prev_contacts.byte_size(), self.contacts.byte_size(), 0];
        if self.warmstart_bg_cache.key != Some(key) {
            let layout = self.warmstart_kernel.pipeline().get_bind_group_layout(0);
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("warmstart_bg"),
                    layout: &layout,
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
                    ],
                });
            self.warmstart_bg_cache.key = Some(key);
            self.warmstart_bg_cache.bind_group = Some(bg);
        }

        let bg = self.warmstart_bg_cache.bind_group.as_ref().unwrap();
        self.run_pass("warmstart_match", &self.warmstart_kernel, bg, new_count);
    }

    /// Swap contacts → prev_contacts for next frame's warmstarting.
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

struct BodyContactAdjacency3D {
    ranges: Vec<[u32; 2]>,
    indices: Vec<u32>,
}

fn build_body_contact_adjacency(num_bodies: u32, contacts: &[Contact3D]) -> BodyContactAdjacency3D {
    let mut counts = vec![0u32; num_bodies as usize];
    for contact in contacts {
        counts[contact.body_a as usize] += 1;
        counts[contact.body_b as usize] += 1;
    }

    let mut ranges = vec![[0u32; 2]; num_bodies as usize];
    let mut offset = 0u32;
    for (body_idx, &count) in counts.iter().enumerate() {
        ranges[body_idx] = [offset, count];
        offset += count;
    }

    let mut write_heads: Vec<u32> = ranges.iter().map(|range| range[0]).collect();
    let mut indices = vec![0u32; offset as usize];
    for (contact_idx, contact) in contacts.iter().enumerate() {
        let contact_idx = contact_idx as u32;

        let head_a = &mut write_heads[contact.body_a as usize];
        indices[*head_a as usize] = contact_idx;
        *head_a += 1;

        let head_b = &mut write_heads[contact.body_b as usize];
        indices[*head_b as usize] = contact_idx;
        *head_b += 1;
    }

    BodyContactAdjacency3D { ranges, indices }
}

fn body_graph_key_3d(contacts: &[Contact3D]) -> Vec<(u32, u32)> {
    let mut pairs: Vec<(u32, u32)> = contacts
        .iter()
        .map(|c| (c.body_a.min(c.body_b), c.body_a.max(c.body_b)))
        .collect();
    pairs.sort_unstable();
    pairs.dedup();
    pairs
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
