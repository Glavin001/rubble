//! GPU compute pipeline for 2D physics simulation using AVBD solver.
//!
//! Orchestrates the full simulation step on the GPU using WGSL compute shaders:
//! predict -> AABB compute -> broadphase -> narrowphase -> AVBD solver -> velocity extraction.

mod avbd_solve_wgsl;
mod broadphase_pairs_wgsl;
mod extract_velocity_wgsl;
pub mod lbvh;
mod narrowphase_wgsl;
mod predict_wgsl;

pub use avbd_solve_wgsl::AVBD_SOLVE_2D_WGSL;
pub use broadphase_pairs_wgsl::BROADPHASE_PAIRS_2D_WGSL;
pub use extract_velocity_wgsl::EXTRACT_VELOCITY_2D_WGSL;
pub use narrowphase_wgsl::NARROWPHASE_2D_WGSL;
pub use predict_wgsl::PREDICT_2D_WGSL;

use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use rubble_gpu::{
    round_up_workgroups, ComputeKernel, GpuAtomicCounter, GpuBuffer, GpuContext, PingPongBuffer,
};
use rubble_math::{Aabb2D, BodyHandle, CollisionEvent, Contact2D, RigidBodyState2D};
use rubble_shapes2d::{CapsuleData2D, CircleData, ConvexPolygonData, ConvexVertex2D, RectData};
use std::collections::{HashMap, HashSet};

const WORKGROUP_SIZE: u32 = 64;

// ---------------------------------------------------------------------------
// GPU-side uniform structs
// ---------------------------------------------------------------------------

/// GPU simulation parameters for 2D. Must match the WGSL `SimParams2D` layout exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SimParams2DGpu {
    pub gravity: [f32; 4],
    pub dt: f32,
    pub num_bodies: u32,
    pub solver_iterations: u32,
    pub _pad: u32,
}

/// Broadphase pair (two body indices).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuPair {
    pub a: u32,
    pub b: u32,
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

const _: () = assert!(std::mem::size_of::<SimParams2DGpu>() == 32);
const _: () = assert!(std::mem::size_of::<GpuPair>() == 8);
const _: () = assert!(std::mem::size_of::<SolveRangeGpu>() == 8);

// ---------------------------------------------------------------------------
// GPU 2D raycast types
// ---------------------------------------------------------------------------

/// A 2D ray to be tested on the GPU. 32 bytes.
/// origin: (x, y, max_t, 0), direction: (dx, dy, 0, 0)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuRay2D {
    /// (x, y, max_t, 0)
    pub origin: [f32; 4],
    /// (dx, dy, 0, 0)
    pub direction: [f32; 4],
}

/// Result of a 2D GPU raycast. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuRayHit2D {
    pub t: f32,
    pub normal_x: f32,
    pub normal_y: f32,
    pub body_index: u32,
    pub hit: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
}

const _: () = assert!(std::mem::size_of::<GpuRay2D>() == 32);
const _: () = assert!(std::mem::size_of::<GpuRayHit2D>() == 32);

// ---------------------------------------------------------------------------
// 2D Raycast compute shader
// ---------------------------------------------------------------------------

const RAYCAST_2D_WGSL: &str = r#"
struct Body2D {
    position_inv_mass: vec4<f32>,
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

struct Ray2D {
    origin: vec4<f32>,    // (x, y, max_t, 0)
    direction: vec4<f32>, // (dx, dy, 0, 0)
};

struct RayHit2D {
    t: f32,
    normal_x: f32,
    normal_y: f32,
    body_index: u32,
    hit: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> bodies: array<Body2D>;
@group(0) @binding(1) var<storage, read> shape_infos: array<ShapeInfo>;
@group(0) @binding(2) var<storage, read> circles: array<CircleData>;
@group(0) @binding(3) var<storage, read> rects: array<RectDataGpu>;
@group(0) @binding(4) var<storage, read> rays: array<Ray2D>;
@group(0) @binding(5) var<storage, read_write> hits: array<RayHit2D>;
@group(0) @binding(6) var<uniform> params: vec4<u32>; // x = num_rays, y = num_bodies

fn ray_circle(origin: vec2<f32>, dir: vec2<f32>, center: vec2<f32>, radius: f32) -> vec2<f32> {
    let oc = origin - center;
    let b = dot(oc, dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 { return vec2<f32>(1e30, 0.0); }
    let t = -b - sqrt(disc);
    if t < 0.0 { return vec2<f32>(1e30, 0.0); }
    return vec2<f32>(t, 1.0);
}

fn ray_rect(origin: vec2<f32>, dir: vec2<f32>, pos: vec2<f32>, angle: f32, he: vec2<f32>) -> vec3<f32> {
    // Transform ray into local rect space
    let ca = cos(-angle);
    let sa = sin(-angle);
    let d = origin - pos;
    let local_o = vec2<f32>(ca * d.x - sa * d.y, sa * d.x + ca * d.y);
    let local_d = vec2<f32>(ca * dir.x - sa * dir.y, sa * dir.x + ca * dir.y);

    var tmin = -1e30;
    var tmax = 1e30;
    var best_axis = 0u;
    var best_sign = 1.0;

    // X axis
    if abs(local_d.x) < 1e-12 {
        if local_o.x < -he.x || local_o.x > he.x { return vec3<f32>(1e30, 0.0, 0.0); }
    } else {
        let t1 = (-he.x - local_o.x) / local_d.x;
        let t2 = (he.x - local_o.x) / local_d.x;
        let t_near = min(t1, t2);
        let t_far = max(t1, t2);
        if t_near > tmin {
            tmin = t_near;
            best_axis = 0u;
            best_sign = select(1.0, -1.0, local_d.x > 0.0);
        }
        tmax = min(tmax, t_far);
        if tmin > tmax { return vec3<f32>(1e30, 0.0, 0.0); }
    }

    // Y axis
    if abs(local_d.y) < 1e-12 {
        if local_o.y < -he.y || local_o.y > he.y { return vec3<f32>(1e30, 0.0, 0.0); }
    } else {
        let t1 = (-he.y - local_o.y) / local_d.y;
        let t2 = (he.y - local_o.y) / local_d.y;
        let t_near = min(t1, t2);
        let t_far = max(t1, t2);
        if t_near > tmin {
            tmin = t_near;
            best_axis = 1u;
            best_sign = select(1.0, -1.0, local_d.y > 0.0);
        }
        tmax = min(tmax, t_far);
        if tmin > tmax { return vec3<f32>(1e30, 0.0, 0.0); }
    }

    if tmin < 0.0 { return vec3<f32>(1e30, 0.0, 0.0); }
    return vec3<f32>(tmin, f32(best_axis), best_sign);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ray_idx = gid.x;
    let num_rays = params.x;
    let num_bodies = params.y;
    if ray_idx >= num_rays { return; }

    let ray = rays[ray_idx];
    let origin = ray.origin.xy;
    let dir = ray.direction.xy;
    let max_t = ray.origin.z;

    var best_t = max_t;
    var best_body = 0u;
    var best_normal = vec2<f32>(0.0, 1.0);
    var found_hit = false;

    for (var bi = 0u; bi < num_bodies; bi = bi + 1u) {
        let pos = bodies[bi].position_inv_mass.xy;
        let angle = bodies[bi].position_inv_mass.z;
        let st = shape_infos[bi].shape_type;
        let si = shape_infos[bi].shape_index;

        if st == 0u { // CIRCLE
            let r = circles[si].radius;
            let result = ray_circle(origin, dir, pos, r);
            if result.y > 0.5 && result.x < best_t && result.x >= 0.0 {
                best_t = result.x;
                best_body = bi;
                let hit_pt = origin + dir * result.x;
                best_normal = normalize(hit_pt - pos);
                found_hit = true;
            }
        } else if st == 1u { // RECT
            let he = rects[si].half_extents.xy;
            let result = ray_rect(origin, dir, pos, angle, he);
            if result.x < best_t && result.x >= 0.0 {
                best_t = result.x;
                best_body = bi;
                // Reconstruct world normal from local axis
                let ca = cos(angle);
                let sa = sin(angle);
                var local_n = vec2<f32>(0.0);
                let axis = u32(result.y);
                if axis == 0u { local_n.x = result.z; }
                else { local_n.y = result.z; }
                best_normal = vec2<f32>(ca * local_n.x - sa * local_n.y, sa * local_n.x + ca * local_n.y);
                found_hit = true;
            }
        }
    }

    hits[ray_idx].t = best_t;
    hits[ray_idx].normal_x = best_normal.x;
    hits[ray_idx].normal_y = best_normal.y;
    hits[ray_idx].body_index = best_body;
    hits[ray_idx].hit = select(0u, 1u, found_hit);
    hits[ray_idx]._pad0 = 0u;
    hits[ray_idx]._pad1 = 0u;
    hits[ray_idx]._pad2 = 0u;
}
"#;

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
    gravity:           vec4<f32>,
    dt:                f32,
    num_bodies:        u32,
    solver_iterations: u32,
    _pad:              u32,
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
    if idx >= params.num_bodies {
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

// ---------------------------------------------------------------------------
// GpuPipeline2D
// ---------------------------------------------------------------------------

/// Orchestrates GPU compute dispatches for the full 2D AVBD physics step.
///
/// The pipeline uses velocity-based AVBD (Averaged Velocity-Based Dynamics):
/// 1. Predict positions from velocities + gravity
/// 2. Compute AABBs from predicted positions
/// 3. Broadphase: find overlapping AABB pairs (O(N^2) GPU kernel)
/// 4. Narrowphase: generate contacts from overlapping pairs
/// 5. AVBD solve: apply velocity impulses using averaged velocities
/// 6. Extract velocities from position changes
pub struct GpuPipeline2D {
    ctx: GpuContext,

    // Kernels
    predict_kernel: ComputeKernel,
    aabb_kernel: ComputeKernel,
    narrowphase_kernel: ComputeKernel,
    solve_kernel: ComputeKernel,
    extract_kernel: ComputeKernel,

    // Storage buffers
    body_states: PingPongBuffer<RigidBodyState2D>,
    old_states: GpuBuffer<RigidBodyState2D>,
    aabbs: GpuBuffer<Aabb2D>,
    contacts: GpuBuffer<Contact2D>,
    contact_count: GpuAtomicCounter,
    pairs: GpuBuffer<GpuPair>,
    pair_count: GpuAtomicCounter,
    circles: GpuBuffer<CircleData>,
    rects: GpuBuffer<RectData>,
    convex_polys: GpuBuffer<ConvexPolygonData>,
    convex_verts: GpuBuffer<ConvexVertex2D>,
    capsules: GpuBuffer<CapsuleData2D>,
    shape_infos: GpuBuffer<ShapeInfo>,

    // Uniform buffers
    params_uniform: wgpu::Buffer,
    solve_range_uniform: wgpu::Buffer,

    // Raycast infrastructure
    raycast_kernel: ComputeKernel,
    rays_buffer: GpuBuffer<GpuRay2D>,
    hits_buffer: GpuBuffer<GpuRayHit2D>,
    raycast_params: wgpu::Buffer,
}

impl GpuPipeline2D {
    /// Create a new 2D GPU pipeline. Compiles all shaders and allocates buffers.
    pub fn new(ctx: GpuContext, max_bodies: usize) -> Self {
        let max_pairs = max_bodies * 8;
        let max_contacts = max_bodies * 8;

        let predict_kernel = ComputeKernel::from_wgsl(&ctx, PREDICT_2D_WGSL, "main");
        let aabb_kernel = ComputeKernel::from_wgsl(&ctx, AABB_COMPUTE_2D_WGSL, "main");
        let narrowphase_kernel = ComputeKernel::from_wgsl(&ctx, NARROWPHASE_2D_WGSL, "main");
        let solve_kernel = ComputeKernel::from_wgsl(&ctx, AVBD_SOLVE_2D_WGSL, "main");
        let extract_kernel = ComputeKernel::from_wgsl(&ctx, EXTRACT_VELOCITY_2D_WGSL, "main");

        let body_states = PingPongBuffer::new(&ctx, max_bodies);
        let old_states = GpuBuffer::new(&ctx, max_bodies);
        let aabbs = GpuBuffer::new(&ctx, max_bodies);
        let contacts = GpuBuffer::new(&ctx, max_contacts);
        let contact_count = GpuAtomicCounter::new(&ctx);
        let pairs = GpuBuffer::new(&ctx, max_pairs);
        let pair_count = GpuAtomicCounter::new(&ctx);
        let circles = GpuBuffer::new(&ctx, max_bodies.max(1));
        let rects = GpuBuffer::new(&ctx, max_bodies.max(1));
        let convex_polys = GpuBuffer::new(&ctx, max_bodies.max(1));
        let convex_verts = GpuBuffer::new(&ctx, (max_bodies * 8).max(1));
        let capsules = GpuBuffer::new(&ctx, max_bodies.max(1));
        let shape_infos = GpuBuffer::new(&ctx, max_bodies);

        let params_uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SimParams2D uniform"),
            size: std::mem::size_of::<SimParams2DGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let solve_range_uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SolveRange2D uniform"),
            size: std::mem::size_of::<SolveRangeGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let raycast_kernel = ComputeKernel::from_wgsl(&ctx, RAYCAST_2D_WGSL, "main");
        let rays_buffer = GpuBuffer::new(&ctx, 1024);
        let hits_buffer = GpuBuffer::new(&ctx, 1024);
        let raycast_params = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raycast 2d params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            ctx,
            predict_kernel,
            aabb_kernel,
            narrowphase_kernel,
            solve_kernel,
            extract_kernel,
            body_states,
            old_states,
            aabbs,
            contacts,
            contact_count,
            pairs,
            pair_count,
            circles,
            rects,
            convex_polys,
            convex_verts,
            capsules,
            shape_infos,
            params_uniform,
            solve_range_uniform,
            raycast_kernel,
            rays_buffer,
            hits_buffer,
            raycast_params,
        }
    }

    /// Try to create a 2D GPU pipeline. Returns None if no GPU adapter is available.
    pub fn try_new(max_bodies: usize) -> Option<Self> {
        let ctx = pollster::block_on(GpuContext::new()).ok()?;
        Some(Self::new(ctx, max_bodies))
    }

    /// Upload body data from CPU arrays to GPU buffers and set simulation params.
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        &mut self,
        states: &[RigidBodyState2D],
        shape_info_data: &[ShapeInfo],
        circle_data: &[CircleData],
        rect_data: &[RectData],
        poly_data: &[ConvexPolygonData],
        poly_verts: &[ConvexVertex2D],
        capsule_data: &[CapsuleData2D],
        gravity: Vec2,
        dt: f32,
        solver_iterations: u32,
    ) {
        self.body_states.upload(&self.ctx, states);
        self.old_states.upload(&self.ctx, states);
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

        let params = SimParams2DGpu {
            gravity: [gravity.x, gravity.y, 0.0, 0.0],
            dt,
            num_bodies: states.len() as u32,
            solver_iterations,
            _pad: 0,
        };
        self.ctx
            .queue
            .write_buffer(&self.params_uniform, 0, bytemuck::bytes_of(&params));
    }

    /// Run predict → AABB → broadphase → narrowphase (shared by both step variants).
    fn run_detection(&mut self, num_bodies: u32) {
        self.contact_count.reset(&self.ctx);
        self.pair_count.reset(&self.ctx);

        self.dispatch_predict(num_bodies);
        self.dispatch_aabb(num_bodies);

        // LBVH broadphase: download AABBs, build tree on CPU, upload pairs
        self.aabbs.set_len(num_bodies);
        let gpu_aabbs = self.aabbs.download(&self.ctx);
        let bvh = lbvh::Lbvh2D::build(&gpu_aabbs);
        let overlap_pairs = bvh.find_overlapping_pairs(&gpu_aabbs);

        if !overlap_pairs.is_empty() {
            // Download shape info for sorting pairs by shape type
            self.shape_infos.set_len(num_bodies);
            let shape_info = self.shape_infos.download(&self.ctx);

            let mut gpu_pairs: Vec<GpuPair> = overlap_pairs
                .iter()
                .map(|p| GpuPair { a: p[0], b: p[1] })
                .collect();

            // Sort pairs by (shape_type_a << 16 | shape_type_b) for SIMD-friendly
            // narrowphase dispatch. Bodies with the same shape-pair type are grouped
            // together, improving GPU warp/wavefront coherence.
            gpu_pairs.sort_unstable_by_key(|pair| {
                let st_a = shape_info[pair.a as usize].shape_type;
                let st_b = shape_info[pair.b as usize].shape_type;
                let (lo, hi) = if st_a <= st_b {
                    (st_a, st_b)
                } else {
                    (st_b, st_a)
                };
                (lo << 16) | hi
            });

            self.pairs.upload(&self.ctx, &gpu_pairs);
            self.pair_count.write(&self.ctx, gpu_pairs.len() as u32);
        }

        self.dispatch_narrowphase(num_bodies);

        // Buffer overflow recovery: if contact count exceeded capacity, grow and retry
        let contact_count_val = self.contact_count.read(&self.ctx);
        let capacity = self.contacts.capacity();
        if contact_count_val > capacity {
            // Grow contacts buffer to 2x the needed size
            let new_cap = (contact_count_val as usize) * 2;
            self.contacts.grow_if_needed(&self.ctx, new_cap);
            // Reset counter and re-run narrowphase
            self.contact_count.reset(&self.ctx);
            self.dispatch_narrowphase(num_bodies);
        }
    }

    /// Graph-colored AVBD solve: download contacts, color them so no two
    /// same-color contacts share a body, sort by color, re-upload, then
    /// dispatch one GPU pass per color group per iteration.
    fn run_colored_solve(&mut self, solver_iterations: u32, contacts: &mut Vec<Contact2D>) {
        if contacts.is_empty() {
            return;
        }

        let color_groups = color_contacts_2d(contacts);

        // Re-upload sorted contacts and update count
        self.contacts.upload(&self.ctx, contacts);
        self.contact_count.write(&self.ctx, contacts.len() as u32);

        // For each iteration, dispatch each color group sequentially
        for _ in 0..solver_iterations {
            for &(offset, count) in &color_groups {
                self.dispatch_solve_range(offset, count);
            }
        }
    }

    /// Run the full GPU 2D physics step and download updated states.
    pub fn step(&mut self, num_bodies: u32, solver_iterations: u32) -> Vec<RigidBodyState2D> {
        if num_bodies == 0 {
            return Vec::new();
        }

        self.run_detection(num_bodies);

        let count = self.contact_count.read(&self.ctx) as usize;
        if count > 0 {
            self.contacts.set_len(count as u32);
            let mut contacts = self.contacts.download(&self.ctx);
            contacts.truncate(count);
            self.run_colored_solve(solver_iterations, &mut contacts);
        }

        self.dispatch_extract(num_bodies);
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
                warm_start_contacts_2d(&mut c, prev);
            }
            c
        } else {
            Vec::new()
        };

        self.run_colored_solve(solver_iterations, &mut contacts);

        // Download contacts after solve (lambdas are updated by GPU)
        let final_contacts = if !contacts.is_empty() {
            self.download_contacts()
        } else {
            Vec::new()
        };

        self.dispatch_extract(num_bodies);
        let states = self.body_states.download(&self.ctx);
        self.body_states.swap();
        (states, final_contacts)
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
        self.contacts.set_len(count as u32);
        let all = self.contacts.download(&self.ctx);
        all.into_iter().take(count).collect()
    }

    /// Batch raycast on GPU. Returns hits for each ray.
    ///
    /// Each ray is tested against all bodies (circle and rect shapes only).
    /// Results are downloaded synchronously.
    pub fn raycast_batch_gpu(
        &mut self,
        rays: &[GpuRay2D],
        num_bodies: u32,
    ) -> Vec<GpuRayHit2D> {
        if rays.is_empty() || num_bodies == 0 {
            return Vec::new();
        }
        let num_rays = rays.len() as u32;
        self.rays_buffer.upload(&self.ctx, rays);
        self.hits_buffer.grow_if_needed(&self.ctx, rays.len());
        self.hits_buffer.set_len(num_rays);
        // Upload zeros to hits buffer
        let zeros = vec![GpuRayHit2D {
            t: 0.0, normal_x: 0.0, normal_y: 0.0,
            body_index: 0, hit: 0, _pad0: 0, _pad1: 0, _pad2: 0,
        }; rays.len()];
        self.hits_buffer.upload(&self.ctx, &zeros);

        let params_data: [u32; 4] = [num_rays, num_bodies, 0, 0];
        self.ctx.queue.write_buffer(
            &self.raycast_params,
            0,
            bytemuck::cast_slice(&params_data),
        );

        let bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raycast_2d"),
            layout: self.raycast_kernel.bind_group_layout(),
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
                    resource: self.rays_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.hits_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.raycast_params.as_entire_binding(),
                },
            ],
        });
        self.run_pass("raycast_2d", &self.raycast_kernel, &bg, num_rays);

        self.hits_buffer.download(&self.ctx)
    }

    // -----------------------------------------------------------------------
    // Private dispatch helpers
    // -----------------------------------------------------------------------

    fn dispatch_predict(&self, num_bodies: u32) {
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
                        resource: self.params_uniform.as_entire_binding(),
                    },
                ],
            });
        self.run_pass("predict_2d", &self.predict_kernel, &bg, num_bodies);
    }

    fn dispatch_aabb(&self, num_bodies: u32) {
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
        self.run_pass("aabb_2d", &self.aabb_kernel, &bg, num_bodies);
    }

    fn dispatch_narrowphase(&self, _num_bodies: u32) {
        let num_pairs = self.pair_count.read(&self.ctx);
        if num_pairs == 0 {
            return;
        }
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("narrowphase_2d"),
                layout: self.narrowphase_kernel.bind_group_layout(),
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
                        resource: self.pairs.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.pair_count.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.circles.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.rects.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.contacts.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.contact_count.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: self.params_uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: self.convex_polys.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: self.convex_verts.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: self.capsules.buffer().as_entire_binding(),
                    },
                ],
            });
        self.run_pass("narrowphase_2d", &self.narrowphase_kernel, &bg, num_pairs);
    }

    fn dispatch_solve_range(&self, offset: u32, count: u32) {
        if count == 0 {
            return;
        }
        // Write solve range uniform
        let range = SolveRangeGpu { offset, count };
        self.ctx
            .queue
            .write_buffer(&self.solve_range_uniform, 0, bytemuck::bytes_of(&range));

        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("avbd_solve_2d"),
                layout: self.solve_kernel.bind_group_layout(),
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
                        resource: self.solve_range_uniform.as_entire_binding(),
                    },
                ],
            });
        self.run_pass("avbd_solve_2d", &self.solve_kernel, &bg, count);
    }

    fn dispatch_extract(&self, num_bodies: u32) {
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
        self.run_pass("extract_vel_2d", &self.extract_kernel, &bg, num_bodies);
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
// Contact graph coloring
// ---------------------------------------------------------------------------

/// Color 2D contacts so no two same-color contacts share a body.
/// Sorts `contacts` in-place by color and returns (offset, count) for each color group.
fn color_contacts_2d(contacts: &mut Vec<Contact2D>) -> Vec<(u32, u32)> {
    let n = contacts.len();
    if n == 0 {
        return Vec::new();
    }

    // Build body → contact index list
    let mut body_contacts: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, c) in contacts.iter().enumerate() {
        body_contacts.entry(c.body_a).or_default().push(i);
        body_contacts.entry(c.body_b).or_default().push(i);
    }

    // Build contact adjacency: two contacts conflict if they share a body
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for indices in body_contacts.values() {
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                adj[indices[i]].push(indices[j]);
                adj[indices[j]].push(indices[i]);
            }
        }
    }

    // Greedy coloring
    let mut colors: Vec<u32> = vec![u32::MAX; n];
    let mut num_colors: u32 = 0;
    for ci in 0..n {
        let mut used: Vec<u32> = adj[ci]
            .iter()
            .filter_map(|&nb| {
                if colors[nb] != u32::MAX {
                    Some(colors[nb])
                } else {
                    None
                }
            })
            .collect();
        used.sort_unstable();
        used.dedup();
        let mut c = 0u32;
        for &u in &used {
            if c == u {
                c += 1;
            } else {
                break;
            }
        }
        colors[ci] = c;
        num_colors = num_colors.max(c + 1);
    }

    // Sort contacts by color (stable sort preserves order within each color)
    let mut indexed: Vec<(u32, Contact2D)> =
        colors.iter().copied().zip(contacts.drain(..)).collect();
    indexed.sort_by_key(|(color, _)| *color);

    // Rebuild contacts in sorted order and compute group boundaries
    let mut groups = Vec::with_capacity(num_colors as usize);
    contacts.reserve(n);
    let mut cur_offset = 0u32;
    for color in 0..num_colors {
        let count = indexed[cur_offset as usize..]
            .iter()
            .take_while(|(c, _)| *c == color)
            .count() as u32;
        if count > 0 {
            groups.push((cur_offset, count));
            cur_offset += count;
        }
    }
    *contacts = indexed.into_iter().map(|(_, c)| c).collect();

    groups
}

// ---------------------------------------------------------------------------
// Warm-starting helpers
// ---------------------------------------------------------------------------

/// Match new 2D contacts against previous-frame contacts and copy cached lambdas.
fn warm_start_contacts_2d(new_contacts: &mut [Contact2D], prev_contacts: &[Contact2D]) {
    let dist_thresh_sq: f32 = 0.01 * 0.01;
    let decay: f32 = 0.95;

    for nc in new_contacts.iter_mut() {
        let np = glam::Vec2::new(nc.point.x, nc.point.y);
        let mut best_dist_sq = f32::MAX;
        let mut best_idx: Option<usize> = None;

        for (i, pc) in prev_contacts.iter().enumerate() {
            let same_pair = (pc.body_a == nc.body_a && pc.body_b == nc.body_b)
                || (pc.body_a == nc.body_b && pc.body_b == nc.body_a);
            if !same_pair {
                continue;
            }
            let pp = glam::Vec2::new(pc.point.x, pc.point.y);
            let d2 = (pp - np).length_squared();
            if d2 < best_dist_sq {
                best_dist_sq = d2;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            if best_dist_sq < dist_thresh_sq {
                nc.lambda_n = prev_contacts[idx].lambda_n * decay;
                nc.lambda_t = prev_contacts[idx].lambda_t * decay;
                nc.penalty_k = prev_contacts[idx].penalty_k;
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
