//! GPU compute pipeline for 2D physics simulation using AVBD solver.
//!
//! Orchestrates the full simulation step on the GPU using compute shaders:
//! predict -> AABB compute -> broadphase -> narrowphase -> AVBD solver -> velocity extraction.
//!
//! Supports WGSL (default) and rust-gpu/SPIR-V shader backends. See `rubble3d::gpu`
//! module docs for details on multi-GPU target support via the `spirv` feature.

mod avbd_solve_wgsl;
mod extract_velocity_wgsl;
mod narrowphase_wgsl;
mod predict_wgsl;

pub use avbd_solve_wgsl::{AVBD_DUAL_2D_WGSL, AVBD_PRIMAL_2D_WGSL};
pub use extract_velocity_wgsl::EXTRACT_VELOCITY_2D_WGSL;
pub use narrowphase_wgsl::NARROWPHASE_2D_WGSL;
pub use predict_wgsl::PREDICT_2D_WGSL;

use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use rubble_gpu::{
    round_up_workgroups, BroadphaseBreakdownMs, ComputeKernel, GpuAtomicCounter, GpuBuffer,
    GpuContext, PingPongBuffer,
};
use rubble_math::{
    greedy_coloring, Aabb2D, BodyHandle, CollisionEvent, Contact2D, RigidBodyState2D,
};
use rubble_primitives::GpuLbvh;
use rubble_shapes2d::{CapsuleData2D, CircleData, ConvexPolygonData, ConvexVertex2D, RectData};
use std::collections::{HashMap, HashSet};
const WORKGROUP_SIZE: u32 = 64;
const AVBD_WARMSTART_ALPHA: f32 = 0.95;
const ENABLE_POST_STABILIZATION_2D: bool = true;
const MAX_CONTACT_PENALTY: f32 = 1.0e6;
const SIM_FLAG_POST_STABILIZE: u32 = 1 << 0;

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

/// GPU simulation parameters for 2D. Must match the WGSL `SimParams2D` layout exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SimParams2DGpu {
    pub gravity: [f32; 4],
    pub solver: [f32; 4], // (dt, beta, k_start, max_penalty)
    pub counts: [u32; 4], // (num_bodies, solver_iterations, pair_count, flags)
}

/// Per-body shape information (type + index into shape-specific array).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ShapeInfo {
    pub shape_type: u32,
    pub shape_index: u32,
}

/// Solve range for graph-colored dispatch: (offset, count) into the sorted contact buffer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SolveRangeGpu {
    pub offset: u32,
    pub count: u32,
}

// ---------------------------------------------------------------------------
// Compile-time GPU layout validation
// ---------------------------------------------------------------------------

const _: () = assert!(std::mem::size_of::<SimParams2DGpu>() == 48);
const _: () = assert!(std::mem::size_of::<SolveRangeGpu>() == 8);

// ---------------------------------------------------------------------------
// AABB compute shader (inline)
// ---------------------------------------------------------------------------

const AABB_COMPUTE_2D_WGSL: &str = r#"
struct Body2D {
    position_inv_mass: vec4<f32>, // (x, y, angle, 1/m)
    lin_vel:           vec4<f32>,
    _pad0:             vec4<f32>,
    _pad1:             vec4<f32>,
};

struct ShapeInfo {
    shape_type:  u32,
    shape_index: u32,
};

struct CircleData {
    radius: f32,
    _pad0:  f32,
    _pad1:  f32,
    _pad2:  f32,
};

struct RectDataGpu {
    half_extents: vec4<f32>,
};

struct Aabb2D {
    min_pt: vec4<f32>,
    max_pt: vec4<f32>,
};

struct SimParams2D {
    gravity: vec4<f32>,
    solver:  vec4<f32>,
    counts:  vec4<u32>,
};

const SHAPE_CIRCLE:         u32 = 0u;
const SHAPE_RECT:           u32 = 1u;
const SHAPE_CONVEX_POLYGON: u32 = 2u;
const SHAPE_CAPSULE:        u32 = 3u;

struct ConvexPolyInfo {
    vertex_offset: u32,
    vertex_count:  u32,
    _pad0:         u32,
    _pad1:         u32,
};

struct ConvexVert2D {
    x: f32,
    y: f32,
    _pad0: f32,
    _pad1: f32,
};

struct CapsuleData2DGpu {
    half_height: f32,
    radius:      f32,
    _pad0:       f32,
    _pad1:       f32,
};

@group(0) @binding(0) var<storage, read>       bodies:        array<Body2D>;
@group(0) @binding(1) var<storage, read>       shape_infos:   array<ShapeInfo>;
@group(0) @binding(2) var<storage, read>       circles:       array<CircleData>;
@group(0) @binding(3) var<storage, read>       rects:         array<RectDataGpu>;
@group(0) @binding(4) var<storage, read_write> aabbs:         array<Aabb2D>;
@group(0) @binding(5) var<uniform>             params:        SimParams2D;
@group(0) @binding(6) var<storage, read>       convex_polys:  array<ConvexPolyInfo>;
@group(0) @binding(7) var<storage, read>       convex_verts:  array<ConvexVert2D>;
@group(0) @binding(8) var<storage, read>       capsules_2d:   array<CapsuleData2DGpu>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.counts.x {
        return;
    }

    let pos = bodies[idx].position_inv_mass.xy;
    let angle = bodies[idx].position_inv_mass.z;
    let st = shape_infos[idx].shape_type;
    let si = shape_infos[idx].shape_index;

    var aabb_min: vec2<f32>;
    var aabb_max: vec2<f32>;

    if st == SHAPE_CIRCLE {
        let r = circles[si].radius;
        aabb_min = pos - vec2<f32>(r, r);
        aabb_max = pos + vec2<f32>(r, r);
    } else if st == SHAPE_RECT {
        let he = rects[si].half_extents.xy;
        // Compute rotated AABB from half extents and angle
        let ca = cos(angle);
        let sa = sin(angle);
        let ex = abs(ca * he.x) + abs(sa * he.y);
        let ey = abs(sa * he.x) + abs(ca * he.y);
        aabb_min = pos - vec2<f32>(ex, ey);
        aabb_max = pos + vec2<f32>(ex, ey);
    } else if st == SHAPE_CONVEX_POLYGON {
        let poly = convex_polys[si];
        let ca = cos(angle);
        let sa = sin(angle);
        var mn = vec2<f32>(1e30, 1e30);
        var mx = vec2<f32>(-1e30, -1e30);
        for (var vi = 0u; vi < poly.vertex_count; vi = vi + 1u) {
            let cv = convex_verts[poly.vertex_offset + vi];
            let local_v = vec2<f32>(cv.x, cv.y);
            let world_v = pos + vec2<f32>(ca * local_v.x - sa * local_v.y, sa * local_v.x + ca * local_v.y);
            mn = min(mn, world_v);
            mx = max(mx, world_v);
        }
        aabb_min = mn;
        aabb_max = mx;
    } else if st == SHAPE_CAPSULE {
        let cap = capsules_2d[si];
        let hh = cap.half_height;
        let r  = cap.radius;
        let ca = cos(angle);
        let sa = sin(angle);
        // Local axis is Y; rotated endpoint offsets
        let ax = vec2<f32>(-sa * hh, ca * hh);
        let ep_a = pos + ax;
        let ep_b = pos - ax;
        aabb_min = min(ep_a, ep_b) - vec2<f32>(r, r);
        aabb_max = max(ep_a, ep_b) + vec2<f32>(r, r);
    } else {
        // Unknown shape: generous default AABB
        aabb_min = pos - vec2<f32>(2.0, 2.0);
        aabb_max = pos + vec2<f32>(2.0, 2.0);
    }

    aabbs[idx].min_pt = vec4<f32>(aabb_min, 0.0, 0.0);
    aabbs[idx].max_pt = vec4<f32>(aabb_max, 0.0, 0.0);
}
"#;

const FREE_MOTION_2D_WGSL: &str = r#"
struct Body2D {
    position_inv_mass: vec4<f32>,
    lin_vel:           vec4<f32>,
    _pad0:             vec4<f32>,
    _pad1:             vec4<f32>,
};

struct SimParams2D {
    gravity: vec4<f32>,
    solver:  vec4<f32>,
    counts:  vec4<u32>,
};

@group(0) @binding(0) var<storage, read_write> bodies:          array<Body2D>;
@group(0) @binding(1) var<storage, read>       inertial_states: array<Body2D>;
@group(0) @binding(2) var<storage, read>       active_bodies:   array<u32>;
@group(0) @binding(3) var<uniform>             params:          SimParams2D;

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
// GpuPipeline2D
// ---------------------------------------------------------------------------

/// Orchestrates GPU compute dispatches for the full 2D AVBD physics step.
///
/// The pipeline uses velocity-based AVBD (Averaged Velocity-Based Dynamics):
/// 1. Predict positions from velocities + gravity
/// 2. Compute AABBs from predicted positions
/// 3. Broadphase: hybrid LBVH broadphase (GPU kernels with current CPU staging/readback)
/// 4. Narrowphase: generate contacts from overlapping pairs
/// 5. AVBD solve: apply velocity impulses using averaged velocities
/// 6. Extract velocities from position changes
pub struct GpuPipeline2D {
    ctx: GpuContext,

    // Kernels
    predict_kernel: ComputeKernel,
    aabb_kernel: ComputeKernel,
    narrowphase_kernel: ComputeKernel,
    free_motion_kernel: ComputeKernel,
    primal_kernel: ComputeKernel,
    dual_kernel: ComputeKernel,
    extract_kernel: ComputeKernel,

    // Storage buffers
    body_states: PingPongBuffer<RigidBodyState2D>,
    old_states: GpuBuffer<RigidBodyState2D>,
    prev_step_states: GpuBuffer<RigidBodyState2D>,
    inertial_states: GpuBuffer<RigidBodyState2D>,
    aabbs: GpuBuffer<Aabb2D>,
    contacts: GpuBuffer<Contact2D>,
    contact_count: GpuAtomicCounter,
    circles: GpuBuffer<CircleData>,
    rects: GpuBuffer<RectData>,
    convex_polys: GpuBuffer<ConvexPolygonData>,
    convex_verts: GpuBuffer<ConvexVertex2D>,
    capsules: GpuBuffer<CapsuleData2D>,
    shape_infos: GpuBuffer<ShapeInfo>,
    body_order: GpuBuffer<u32>,
    active_body_flags: GpuBuffer<u32>,
    shape_info_cpu: Vec<ShapeInfo>,

    // Cached body coloring for steady-state contact graphs.
    cached_body_graph: Vec<(u32, u32)>,
    cached_body_order: Vec<u32>,
    cached_color_groups: Vec<(u32, u32)>,
    cached_color_num_bodies: u32,

    // Uniform buffers
    sim_params: SimParams2DGpu,
    warmstart_decay: f32,
    params_uniform: wgpu::Buffer,
    solve_range_buffers: Vec<wgpu::Buffer>,

    // Cached bind groups (reused while backing buffers stay stable).
    predict_bg_cache: CachedBindGroup<[u64; 4]>,
    aabb_bg_cache: CachedBindGroup<[u64; 8]>,
    free_motion_bg_cache: CachedBindGroup<[u64; 4]>,
    primal_bg_cache: CachedBindGroupVec<[u64; 6]>,
    dual_bg_cache: CachedBindGroup<[u64; 2]>,
    extract_bg_cache: CachedBindGroup<[u64; 2]>,

    // GPU broadphase
    gpu_lbvh: GpuLbvh,
}

impl GpuPipeline2D {
    /// Create a new 2D GPU pipeline. Compiles all shaders and allocates buffers.
    pub fn new(ctx: GpuContext, max_bodies: usize) -> Self {
        let max_contacts = max_bodies * 8;

        let predict_kernel = ComputeKernel::from_wgsl(&ctx, PREDICT_2D_WGSL, "main");
        let aabb_kernel = ComputeKernel::from_wgsl(&ctx, AABB_COMPUTE_2D_WGSL, "main");
        let narrowphase_kernel = ComputeKernel::from_wgsl(&ctx, NARROWPHASE_2D_WGSL, "main");
        let free_motion_kernel = ComputeKernel::from_wgsl(&ctx, FREE_MOTION_2D_WGSL, "main");
        let primal_kernel = ComputeKernel::from_wgsl(&ctx, AVBD_PRIMAL_2D_WGSL, "main");
        let dual_kernel = ComputeKernel::from_wgsl(&ctx, AVBD_DUAL_2D_WGSL, "main");
        let extract_kernel = ComputeKernel::from_wgsl(&ctx, EXTRACT_VELOCITY_2D_WGSL, "main");

        let body_states = PingPongBuffer::new(&ctx, max_bodies);
        let old_states = GpuBuffer::new(&ctx, max_bodies);
        let prev_step_states = GpuBuffer::new(&ctx, max_bodies);
        let inertial_states = GpuBuffer::new(&ctx, max_bodies);
        let aabbs = GpuBuffer::new(&ctx, max_bodies);
        let contacts = GpuBuffer::new(&ctx, max_contacts);
        let contact_count = GpuAtomicCounter::new(&ctx);
        let circles = GpuBuffer::new(&ctx, max_bodies.max(1));
        let rects = GpuBuffer::new(&ctx, max_bodies.max(1));
        let convex_polys = GpuBuffer::new(&ctx, max_bodies.max(1));
        let convex_verts = GpuBuffer::new(&ctx, (max_bodies * 8).max(1));
        let capsules = GpuBuffer::new(&ctx, max_bodies.max(1));
        let shape_infos = GpuBuffer::new(&ctx, max_bodies);
        let body_order = GpuBuffer::new(&ctx, max_bodies);
        let active_body_flags = GpuBuffer::new(&ctx, max_bodies.max(1));

        let params_uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SimParams2D uniform"),
            size: std::mem::size_of::<SimParams2DGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gpu_lbvh = GpuLbvh::new(&ctx, max_bodies);

        Self {
            ctx,
            predict_kernel,
            aabb_kernel,
            narrowphase_kernel,
            free_motion_kernel,
            primal_kernel,
            dual_kernel,
            extract_kernel,
            body_states,
            old_states,
            prev_step_states,
            inertial_states,
            aabbs,
            contacts,
            contact_count,
            circles,
            rects,
            convex_polys,
            convex_verts,
            capsules,
            shape_infos,
            body_order,
            active_body_flags,
            shape_info_cpu: Vec::new(),
            cached_body_graph: Vec::new(),
            cached_body_order: Vec::new(),
            cached_color_groups: Vec::new(),
            cached_color_num_bodies: 0,
            sim_params: SimParams2DGpu {
                gravity: [0.0; 4],
                solver: [0.0, 10.0, 1.0e4, MAX_CONTACT_PENALTY],
                counts: [0, 0, 0, SIM_FLAG_POST_STABILIZE],
            },
            warmstart_decay: 0.99,
            params_uniform,
            solve_range_buffers: Vec::new(),
            predict_bg_cache: CachedBindGroup::default(),
            aabb_bg_cache: CachedBindGroup::default(),
            free_motion_bg_cache: CachedBindGroup::default(),
            primal_bg_cache: CachedBindGroupVec::default(),
            dual_bg_cache: CachedBindGroup::default(),
            extract_bg_cache: CachedBindGroup::default(),
            gpu_lbvh,
        }
    }

    /// Try to create a 2D GPU pipeline (async). Returns None if no GPU adapter is available.
    pub async fn try_new_async(max_bodies: usize) -> Option<Self> {
        let ctx = GpuContext::new().await.ok()?;
        Some(Self::new(ctx, max_bodies))
    }

    /// Try to create a 2D GPU pipeline. Returns None if no GPU adapter is available.
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
        states: &[RigidBodyState2D],
        prev_step_states: &[RigidBodyState2D],
        shape_info_data: &[ShapeInfo],
        circle_data: &[CircleData],
        rect_data: &[RectData],
        poly_data: &[ConvexPolygonData],
        poly_verts: &[ConvexVertex2D],
        capsule_data: &[CapsuleData2D],
        gravity: Vec2,
        dt: f32,
        solver_iterations: u32,
        beta: f32,
        k_start: f32,
        warmstart_decay: f32,
    ) {
        self.shape_info_cpu = shape_info_data.to_vec();
        self.body_states.upload(&self.ctx, states);
        self.old_states.upload(&self.ctx, states);
        self.prev_step_states.upload(&self.ctx, prev_step_states);
        self.inertial_states.upload(&self.ctx, states);
        self.shape_infos.upload(&self.ctx, shape_info_data);

        if !circle_data.is_empty() {
            self.circles.upload(&self.ctx, circle_data);
        }
        if !rect_data.is_empty() {
            self.rects.upload(&self.ctx, rect_data);
        }
        if !poly_data.is_empty() {
            self.convex_polys.upload(&self.ctx, poly_data);
        }
        if !poly_verts.is_empty() {
            self.convex_verts.upload(&self.ctx, poly_verts);
        }
        if !capsule_data.is_empty() {
            self.capsules.upload(&self.ctx, capsule_data);
        }

        self.aabbs.grow_if_needed(&self.ctx, states.len());

        self.warmstart_decay = warmstart_decay;
        self.sim_params = SimParams2DGpu {
            gravity: [gravity.x, gravity.y, 0.0, 0.0],
            solver: [dt, beta, k_start, MAX_CONTACT_PENALTY],
            counts: [
                states.len() as u32,
                solver_iterations,
                0,
                if ENABLE_POST_STABILIZATION_2D {
                    SIM_FLAG_POST_STABILIZE
                } else {
                    0
                },
            ],
        };
        self.ctx.queue.write_buffer(
            &self.params_uniform,
            0,
            bytemuck::bytes_of(&self.sim_params),
        );
    }

    /// Run predict → AABB → broadphase → narrowphase (shared by both step variants).
    fn run_detection(&mut self, num_bodies: u32) {
        self.contact_count.reset(&self.ctx);

        self.dispatch_predict(num_bodies);
        self.snapshot_prev_step_states(num_bodies);
        self.dispatch_aabb(num_bodies);

        // GPU LBVH broadphase: scene bounds, Morton codes, radix sort, and pair finding
        // run on device. The current tree topology build/refit still executes on CPU.
        self.aabbs.set_len(num_bodies);
        let pair_thread_count =
            self.gpu_lbvh
                .query_on_device_raw(&self.ctx, self.aabbs.buffer(), num_bodies);
        if pair_thread_count > 0 {
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
            if pair_thread_count > 0 {
                let pair_count = self.gpu_lbvh.read_pair_count(&self.ctx, pair_thread_count);
                let pair_buffer = self.gpu_lbvh.pair_buffer().clone();
                self.dispatch_narrowphase_with_source(num_bodies, pair_count, &pair_buffer);
            }
        }
    }

    /// Body-colored AVBD solve in position space.
    fn run_colored_solve(
        &mut self,
        num_bodies: u32,
        solver_iterations: u32,
        contacts: &mut [Contact2D],
    ) {
        self.apply_free_motion(num_bodies, contacts);
        if contacts.is_empty() {
            return;
        }

        let graph_key = body_graph_key_2d(contacts);
        let (body_order, color_groups) =
            if self.cached_color_num_bodies == num_bodies && self.cached_body_graph == graph_key {
                (
                    self.cached_body_order.clone(),
                    self.cached_color_groups.clone(),
                )
            } else {
                let (body_order, color_groups) = color_bodies_2d(num_bodies, contacts);
                self.cached_color_num_bodies = num_bodies;
                self.cached_body_graph = graph_key;
                self.cached_body_order = body_order.clone();
                self.cached_color_groups = color_groups.clone();
                (body_order, color_groups)
            };
        self.contacts.upload(&self.ctx, contacts);
        self.contact_count.write(&self.ctx, contacts.len() as u32);
        self.body_order.upload(&self.ctx, &body_order);
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
                label: Some("avbd_solve_batch_2d"),
            });
        for _ in 0..solver_iterations {
            dispatch_primal_color_groups(
                &mut encoder,
                &self.primal_kernel,
                primal_bind_groups,
                &color_groups,
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("avbd_dual_2d"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.dual_kernel.pipeline());
            pass.set_bind_group(0, dual_bind_group, &[]);
            pass.dispatch_workgroups(round_up_workgroups(contact_count, WORKGROUP_SIZE), 1, 1);
        }
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    fn run_post_stabilization_pass(&mut self, contacts: &mut [Contact2D]) {
        if contacts.is_empty() {
            return;
        }

        let graph_key = body_graph_key_2d(contacts);
        let (body_order, color_groups) = color_bodies_2d(self.sim_params.counts[0], contacts);
        self.cached_body_graph = graph_key;
        self.cached_body_order = body_order.clone();
        self.cached_color_groups = color_groups.clone();
        self.cached_color_num_bodies = self.sim_params.counts[0];

        self.contacts.upload(&self.ctx, contacts);
        self.contact_count.write(&self.ctx, contacts.len() as u32);
        self.body_order.upload(&self.ctx, &body_order);
        self.write_solve_ranges(&color_groups);
        self.sync_primal_bind_groups(color_groups.len());

        let primal_bind_groups = &self.primal_bg_cache.bind_groups;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("avbd_post_stabilize_2d"),
            });
        dispatch_primal_color_groups(
            &mut encoder,
            &self.primal_kernel,
            primal_bind_groups,
            &color_groups,
        );
        self.ctx.queue.submit(Some(encoder.finish()));
    }

    /// Run the full GPU 2D physics step and download updated states.
    pub fn step(&mut self, num_bodies: u32, solver_iterations: u32) -> Vec<RigidBodyState2D> {
        if num_bodies == 0 {
            return Vec::new();
        }

        self.run_detection(num_bodies);

        let mut contacts = Vec::new();
        let count = self.contact_count.read(&self.ctx) as usize;
        if count > 0 {
            self.contacts.set_len(count as u32);
            contacts = self.contacts.download(&self.ctx);
            contacts.truncate(count);
            self.run_colored_solve(num_bodies, solver_iterations, &mut contacts);
        }

        self.dispatch_extract(num_bodies);
        if ENABLE_POST_STABILIZATION_2D {
            self.run_post_stabilization_pass(&mut contacts);
        }

        let states = self.body_states.download(&self.ctx);
        self.body_states.swap();
        states
    }

    /// Run the full GPU 2D physics step with warm-starting support.
    /// Returns (updated_states, new_contacts) for persistence tracking.
    pub fn step_with_contacts(
        &mut self,
        num_bodies: u32,
        solver_iterations: u32,
        warm_contacts: Option<&[Contact2D]>,
    ) -> (Vec<RigidBodyState2D>, Vec<Contact2D>) {
        if num_bodies == 0 {
            return (Vec::new(), Vec::new());
        }

        self.run_detection(num_bodies);

        let count = self.contact_count.read(&self.ctx) as usize;
        let mut contacts = if count > 0 {
            self.contacts.set_len(count as u32);
            let mut c = self.contacts.download(&self.ctx);
            c.truncate(count);
            // Warm-start: apply cached lambdas from previous frame
            if let Some(prev) = warm_contacts {
                warm_start_contacts_2d(
                    &mut c,
                    prev,
                    self.warmstart_decay,
                    AVBD_WARMSTART_ALPHA,
                    ENABLE_POST_STABILIZATION_2D,
                );
            }
            c
        } else {
            Vec::new()
        };

        self.run_colored_solve(num_bodies, solver_iterations, &mut contacts);

        // Download contacts after solve (lambdas are updated by GPU)
        let final_contacts = if !contacts.is_empty() {
            self.download_contacts()
        } else {
            Vec::new()
        };

        self.dispatch_extract(num_bodies);
        if ENABLE_POST_STABILIZATION_2D {
            self.run_post_stabilization_pass(&mut contacts);
        }
        let states = self.body_states.download(&self.ctx);
        self.body_states.swap();
        (states, final_contacts)
    }

    /// Timed version of `step_with_contacts` that populates per-phase timings.
    pub fn step_with_contacts_timed(
        &mut self,
        num_bodies: u32,
        solver_iterations: u32,
        warm_contacts: Option<&[Contact2D]>,
        timings: &mut rubble_gpu::StepTimingsMs,
    ) -> (Vec<RigidBodyState2D>, Vec<Contact2D>) {
        use std::time::Instant;

        if num_bodies == 0 {
            return (Vec::new(), Vec::new());
        }

        self.run_detection_timed(num_bodies, timings);

        let t_cf = Instant::now();
        let count = self.contact_count.read(&self.ctx) as usize;
        let mut contacts = if count > 0 {
            self.contacts.set_len(count as u32);
            let mut c = self.contacts.download(&self.ctx);
            c.truncate(count);
            if let Some(prev) = warm_contacts {
                warm_start_contacts_2d(
                    &mut c,
                    prev,
                    self.warmstart_decay,
                    AVBD_WARMSTART_ALPHA,
                    ENABLE_POST_STABILIZATION_2D,
                );
            }
            c
        } else {
            Vec::new()
        };
        timings.contact_fetch_ms = t_cf.elapsed().as_secs_f32() * 1000.0;

        let t_solve = Instant::now();
        self.run_colored_solve(num_bodies, solver_iterations, &mut contacts);
        timings.solve_ms = t_solve.elapsed().as_secs_f32() * 1000.0;

        let final_contacts = if !contacts.is_empty() {
            self.download_contacts()
        } else {
            Vec::new()
        };

        let t_ext = Instant::now();
        self.dispatch_extract(num_bodies);
        if ENABLE_POST_STABILIZATION_2D {
            self.run_post_stabilization_pass(&mut contacts);
        }
        let states = self.body_states.download(&self.ctx);
        self.body_states.swap();
        timings.extract_ms = t_ext.elapsed().as_secs_f32() * 1000.0;

        (states, final_contacts)
    }

    fn run_detection_timed(&mut self, num_bodies: u32, timings: &mut rubble_gpu::StepTimingsMs) {
        use std::time::Instant;

        self.contact_count.reset(&self.ctx);

        let t0 = Instant::now();
        self.dispatch_predict(num_bodies);
        self.snapshot_prev_step_states(num_bodies);
        self.dispatch_aabb(num_bodies);
        timings.predict_aabb_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t1 = Instant::now();
        let mut broadphase = BroadphaseBreakdownMs::default();
        self.aabbs.set_len(num_bodies);
        let pair_thread_count = self.gpu_lbvh.query_on_device_raw_with_breakdown(
            &self.ctx,
            self.aabbs.buffer(),
            num_bodies,
            &mut broadphase,
        );
        let pair_count = if pair_thread_count > 0 {
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
        if pair_count > 0 {
            let pair_buffer = self.gpu_lbvh.pair_buffer().clone();
            self.dispatch_narrowphase_with_source(num_bodies, pair_count, &pair_buffer);
        }

        let contact_count_val = self.contact_count.read(&self.ctx);
        let capacity = self.contacts.capacity();
        if contact_count_val > capacity {
            let new_cap = (contact_count_val as usize) * 2;
            self.contacts.grow_if_needed(&self.ctx, new_cap);
            self.contact_count.reset(&self.ctx);
            if pair_count > 0 {
                let pair_buffer = self.gpu_lbvh.pair_buffer().clone();
                self.dispatch_narrowphase_with_source(num_bodies, pair_count, &pair_buffer);
            }
        }
        timings.narrowphase_ms = t2.elapsed().as_secs_f32() * 1000.0;
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
    pub fn download_contacts(&mut self) -> Vec<Contact2D> {
        let count = self.contact_count.read(&self.ctx) as usize;
        if count == 0 {
            return Vec::new();
        }
        self.download_contacts_exact(count)
    }

    fn download_contacts_exact(&mut self, count: usize) -> Vec<Contact2D> {
        self.contacts.set_len(count as u32);
        let all = self.contacts.download(&self.ctx);
        all.into_iter().take(count).collect()
    }

    // -----------------------------------------------------------------------
    // Async variants for WASM/WebGPU (buffer mapping requires async)
    // -----------------------------------------------------------------------

    #[cfg(target_arch = "wasm32")]
    async fn run_detection_async(
        &mut self,
        num_bodies: u32,
        timings: &mut rubble_gpu::StepTimingsMs,
    ) {
        use rubble_gpu::web_time::Instant;

        self.contact_count.reset(&self.ctx);

        let t0 = Instant::now();
        self.dispatch_predict(num_bodies);
        self.snapshot_prev_step_states(num_bodies);
        self.dispatch_aabb(num_bodies);
        timings.predict_aabb_ms = t0.elapsed().as_secs_f32() * 1000.0;

        let t1 = Instant::now();
        let mut broadphase = BroadphaseBreakdownMs::default();
        self.aabbs.set_len(num_bodies);
        let pair_thread_count = self
            .gpu_lbvh
            .query_on_device_raw_async_with_breakdown(
                &self.ctx,
                self.aabbs.buffer(),
                num_bodies,
                &mut broadphase,
            )
            .await;
        let pair_count = if pair_thread_count > 0 {
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
        if pair_count > 0 {
            let pair_buffer = self.gpu_lbvh.pair_buffer().clone();
            self.dispatch_narrowphase_with_source(num_bodies, pair_count, &pair_buffer);
        }

        let contact_count_val = self.contact_count.read_async(&self.ctx).await;
        let capacity = self.contacts.capacity();
        if contact_count_val > capacity {
            let new_cap = (contact_count_val as usize) * 2;
            self.contacts.grow_if_needed(&self.ctx, new_cap);
            self.contact_count.reset(&self.ctx);
            if pair_count > 0 {
                let pair_buffer = self.gpu_lbvh.pair_buffer().clone();
                self.dispatch_narrowphase_with_source(num_bodies, pair_count, &pair_buffer);
            }
        }
        timings.narrowphase_ms = t2.elapsed().as_secs_f32() * 1000.0;
    }

    /// Async version of `step_with_contacts` for WASM/WebGPU.
    #[cfg(target_arch = "wasm32")]
    pub async fn step_with_contacts_async(
        &mut self,
        num_bodies: u32,
        solver_iterations: u32,
        warm_contacts: Option<&[Contact2D]>,
        timings: &mut rubble_gpu::StepTimingsMs,
    ) -> (Vec<RigidBodyState2D>, Vec<Contact2D>) {
        use rubble_gpu::web_time::Instant;

        if num_bodies == 0 {
            return (Vec::new(), Vec::new());
        }

        self.run_detection_async(num_bodies, timings).await;

        let t_cf = Instant::now();
        let count = self.contact_count.read_async(&self.ctx).await as usize;
        let mut contacts = if count > 0 {
            let mut c = self.download_contacts_exact_async(count).await;
            if let Some(prev) = warm_contacts {
                warm_start_contacts_2d(
                    &mut c,
                    prev,
                    self.warmstart_decay,
                    AVBD_WARMSTART_ALPHA,
                    ENABLE_POST_STABILIZATION_2D,
                );
            }
            c
        } else {
            Vec::new()
        };
        timings.contact_fetch_ms = t_cf.elapsed().as_secs_f32() * 1000.0;

        let t_solve = Instant::now();
        self.run_colored_solve(num_bodies, solver_iterations, &mut contacts);

        let final_contacts = if !contacts.is_empty() {
            self.download_contacts_exact_async(contacts.len()).await
        } else {
            Vec::new()
        };
        timings.solve_ms = t_solve.elapsed().as_secs_f32() * 1000.0;

        let t_ext = Instant::now();
        self.dispatch_extract(num_bodies);
        if ENABLE_POST_STABILIZATION_2D {
            self.run_post_stabilization_pass(&mut contacts);
        }
        let states = self.body_states.download_async(&self.ctx).await;
        timings.extract_ms = t_ext.elapsed().as_secs_f32() * 1000.0;

        (states, final_contacts)
    }

    #[cfg(target_arch = "wasm32")]
    async fn download_contacts_exact_async(&mut self, count: usize) -> Vec<Contact2D> {
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
                    label: Some("SolveRange2D uniform"),
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
                    label: Some("predict_2d"),
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
            self.shape_infos.byte_size(),
            self.circles.byte_size(),
            self.rects.byte_size(),
            self.aabbs.byte_size(),
            self.convex_polys.byte_size(),
            self.convex_verts.byte_size(),
            self.capsules.byte_size(),
        ];
        if self.aabb_bg_cache.key.as_ref() != Some(&key) {
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("aabb_2d"),
                    layout: self.aabb_kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.body_states.current().buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.shape_infos.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.circles.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.rects.buffer().as_entire_binding(),
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
                            resource: self.convex_polys.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: self.convex_verts.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: self.capsules.buffer().as_entire_binding(),
                        },
                    ],
                });
            self.aabb_bg_cache.key = Some(key);
            self.aabb_bg_cache.bind_group = Some(bg);
        }
        self.aabb_bg_cache.bind_group.as_ref().unwrap()
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
                        resource: self.old_states.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.shape_infos.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: pairs_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.circles.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.rects.buffer().as_entire_binding(),
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
                        resource: self.convex_polys.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: self.convex_verts.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: self.capsules.buffer().as_entire_binding(),
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
                    label: Some("free_motion_2d"),
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
            self.contacts.byte_size(),
            self.body_order.byte_size(),
            range_count as u64,
            self.contact_count.buffer().size(),
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
                    label: Some("avbd_primal_2d"),
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
                            resource: self.contacts.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.body_order.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.params_uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.contact_count.buffer().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: self.solve_range_buffers[idx].as_entire_binding(),
                        },
                    ],
                });
            self.primal_bg_cache.bind_groups.push(bg);
        }
    }

    fn dual_bind_group(&mut self) -> &wgpu::BindGroup {
        let key = [self.body_states_cache_key(), self.contacts.byte_size()];
        if self.dual_bg_cache.key.as_ref() != Some(&key) {
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("avbd_dual_2d"),
                    layout: self.dual_kernel.bind_group_layout(),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.body_states.current().buffer().as_entire_binding(),
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
            self.dual_bg_cache.key = Some(key);
            self.dual_bg_cache.bind_group = Some(bg);
        }
        self.dual_bg_cache.bind_group.as_ref().unwrap()
    }

    fn extract_bind_group(&mut self) -> &wgpu::BindGroup {
        let key = [self.body_states_cache_key(), self.old_states.byte_size()];
        if self.extract_bg_cache.key.as_ref() != Some(&key) {
            let bg = self
                .ctx
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("extract_velocity_2d"),
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

        let byte_len = num_bodies as u64 * std::mem::size_of::<RigidBodyState2D>() as u64;
        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("snapshot_prev_step_2d"),
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
        self.run_pass("predict_2d", &self.predict_kernel, bg, num_bodies);
    }

    fn dispatch_aabb(&mut self, num_bodies: u32) {
        let _ = self.aabb_bind_group();
        let bg = self.aabb_bg_cache.bind_group.as_ref().unwrap();
        self.run_pass("aabb_2d", &self.aabb_kernel, bg, num_bodies);
    }

    fn apply_free_motion(&mut self, num_bodies: u32, contacts: &[Contact2D]) {
        let mut active = vec![0u32; num_bodies as usize];
        for c in contacts {
            active[c.body_a as usize] = 1;
            active[c.body_b as usize] = 1;
        }
        self.active_body_flags.upload(&self.ctx, &active);
        let _ = self.free_motion_bind_group();
        let bg = self.free_motion_bg_cache.bind_group.as_ref().unwrap();
        self.run_pass("free_motion_2d", &self.free_motion_kernel, bg, num_bodies);
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

        let bg = self.create_narrowphase_bind_group(pairs_buffer, "narrowphase_2d_ext");
        self.run_pass("narrowphase_2d", &self.narrowphase_kernel, &bg, num_pairs);
    }

    fn dispatch_extract(&mut self, num_bodies: u32) {
        let _ = self.extract_bind_group();
        let bg = self.extract_bg_cache.bind_group.as_ref().unwrap();
        self.run_pass("extract_vel_2d", &self.extract_kernel, bg, num_bodies);
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

fn dispatch_primal_color_groups(
    encoder: &mut wgpu::CommandEncoder,
    primal_kernel: &ComputeKernel,
    primal_bind_groups: &[wgpu::BindGroup],
    color_groups: &[(u32, u32)],
) {
    for (range_idx, &(_, count)) in color_groups.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("avbd_primal_2d"),
            timestamp_writes: None,
        });
        pass.set_pipeline(primal_kernel.pipeline());
        pass.set_bind_group(0, &primal_bind_groups[range_idx], &[]);
        pass.dispatch_workgroups(round_up_workgroups(count, WORKGROUP_SIZE), 1, 1);
    }
}

// ---------------------------------------------------------------------------
// Body graph coloring
// ---------------------------------------------------------------------------

/// Color active bodies so no two bodies in the same color share a contact.
fn color_bodies_2d(num_bodies: u32, contacts: &[Contact2D]) -> (Vec<u32>, Vec<(u32, u32)>) {
    if contacts.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let contact_pairs: Vec<(u32, u32)> = contacts.iter().map(|c| (c.body_a, c.body_b)).collect();
    let (colors, num_colors) = greedy_coloring(num_bodies as usize, &contact_pairs);

    let mut active = vec![false; num_bodies as usize];
    for c in contacts {
        active[c.body_a as usize] = true;
        active[c.body_b as usize] = true;
    }

    let mut body_order = Vec::new();
    let mut groups = Vec::new();
    for color in 0..num_colors {
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

fn body_graph_key_2d(contacts: &[Contact2D]) -> Vec<(u32, u32)> {
    let mut pairs: Vec<(u32, u32)> = contacts
        .iter()
        .map(|c| (c.body_a.min(c.body_b), c.body_a.max(c.body_b)))
        .collect();
    pairs.sort_unstable();
    pairs.dedup();
    pairs
}

// ---------------------------------------------------------------------------
// Warm-starting helpers
// ---------------------------------------------------------------------------

/// Match new 2D contacts against previous-frame contacts and copy cached dual state.
fn warm_start_contacts_2d(
    new_contacts: &mut [Contact2D],
    prev_contacts: &[Contact2D],
    gamma: f32,
    alpha: f32,
    post_stabilize: bool,
) {
    let prev_by_key: HashMap<(u32, u32, u32), &Contact2D> = prev_contacts
        .iter()
        .map(|c| {
            (
                (c.body_a.min(c.body_b), c.body_a.max(c.body_b), c.feature_id),
                c,
            )
        })
        .collect();

    for nc in new_contacts.iter_mut() {
        let key = (
            nc.body_a.min(nc.body_b),
            nc.body_a.max(nc.body_b),
            nc.feature_id,
        );
        if let Some(prev) = prev_by_key.get(&key) {
            let lambda_scale = if post_stabilize { 1.0 } else { alpha * gamma };
            nc.lambda_penalty.x = prev.lambda_penalty.x * lambda_scale;
            nc.lambda_penalty.y = prev.lambda_penalty.y * lambda_scale;
            nc.lambda_penalty.z = prev.lambda_penalty.z * gamma;
            nc.lambda_penalty.w = prev.lambda_penalty.w * gamma;
            nc.flags = prev.flags;

            if prev.flags & rubble_math::CONTACT_FLAG_STICKING != 0 {
                nc.local_anchors = prev.local_anchors;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Contact persistence + collision events
// ---------------------------------------------------------------------------

/// Tracks 2D contacts frame-to-frame for warm-starting and collision events.
pub struct ContactPersistence2D {
    prev_contacts: Vec<Contact2D>,
}

impl ContactPersistence2D {
    pub fn new() -> Self {
        Self {
            prev_contacts: Vec::new(),
        }
    }

    /// Update with new contacts. Returns collision events (Started / Ended).
    pub fn update(&mut self, new_contacts: &[Contact2D]) -> Vec<CollisionEvent> {
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

        for &(a, b) in &new_pairs {
            if !prev_pairs.contains(&(a, b)) {
                events.push(CollisionEvent::Started {
                    body_a: BodyHandle::new(a, 0),
                    body_b: BodyHandle::new(b, 0),
                });
            }
        }

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

    pub fn prev_contacts(&self) -> &[Contact2D] {
        &self.prev_contacts
    }
}

impl Default for ContactPersistence2D {
    fn default() -> Self {
        Self::new()
    }
}
