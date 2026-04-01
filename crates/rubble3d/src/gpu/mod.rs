//! GPU compute pipeline for 3D physics simulation using AVBD solver.
//!
//! Orchestrates the full simulation step on the GPU using WGSL compute shaders:
//! predict → AABB compute → broadphase → narrowphase → AVBD solver → velocity extraction.

mod avbd_solve_wgsl;
mod broadphase_pairs_wgsl;
mod extract_velocity_wgsl;
pub mod lbvh;
mod narrowphase_wgsl;
mod predict_wgsl;

pub use avbd_solve_wgsl::AVBD_SOLVE_WGSL;
pub use broadphase_pairs_wgsl::BROADPHASE_PAIRS_WGSL;
pub use extract_velocity_wgsl::EXTRACT_VELOCITY_WGSL;
pub use narrowphase_wgsl::NARROWPHASE_WGSL;
pub use predict_wgsl::PREDICT_WGSL;

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};
use rubble_gpu::{
    round_up_workgroups, ComputeKernel, GpuAtomicCounter, GpuBuffer, GpuContext, PingPongBuffer,
};
use rubble_math::{
    Aabb3D, BodyHandle, CollisionEvent, Contact3D, RigidBodyProps3D, RigidBodyState3D,
};
use rubble_shapes3d::{
    BoxData, CapsuleData, CompoundChildGpu, CompoundShapeGpu, ConvexHullData, ConvexVertex3D,
    GaussMapEntry, SphereData,
};
use std::collections::{HashMap, HashSet};

const WORKGROUP_SIZE: u32 = 64;

// ---------------------------------------------------------------------------
// GPU-side uniform structs
// ---------------------------------------------------------------------------

/// GPU simulation parameters. Must match the WGSL `SimParams` layout exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SimParamsGpu {
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

const _: () = assert!(std::mem::size_of::<SimParamsGpu>() == 32);
const _: () = assert!(std::mem::size_of::<GpuPair>() == 8);
const _: () = assert!(std::mem::size_of::<SolveRangeGpu>() == 8);

// ---------------------------------------------------------------------------
// GPU raycast types
// ---------------------------------------------------------------------------

/// A ray to be tested on the GPU. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuRay {
    /// xyz = origin, w = max_t
    pub origin: [f32; 4],
    /// xyz = direction (normalized), w = 0
    pub direction: [f32; 4],
}

/// Result of a GPU raycast. 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuRayHit {
    pub t: f32,
    pub normal_x: f32,
    pub normal_y: f32,
    pub normal_z: f32,
    pub body_index: u32,
    pub hit: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

const _: () = assert!(std::mem::size_of::<GpuRay>() == 32);
const _: () = assert!(std::mem::size_of::<GpuRayHit>() == 32);

// ---------------------------------------------------------------------------
// Raycast compute shader
// ---------------------------------------------------------------------------

const RAYCAST_WGSL: &str = r#"
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
    _pad0: f32, _pad1: f32, _pad2: f32,
};

struct BoxDataGpu {
    half_extents: vec4<f32>,
};

struct Ray {
    origin: vec4<f32>,
    direction: vec4<f32>,
};

struct RayHit {
    t: f32,
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    body_index: u32,
    hit: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> bodies: array<Body>;
@group(0) @binding(1) var<storage, read> props: array<BodyProps>;
@group(0) @binding(2) var<storage, read> spheres: array<SphereData>;
@group(0) @binding(3) var<storage, read> boxes_data: array<BoxDataGpu>;
@group(0) @binding(4) var<storage, read> rays: array<Ray>;
@group(0) @binding(5) var<storage, read_write> hits: array<RayHit>;
@group(0) @binding(6) var<uniform> params: vec4<u32>;

fn quat_rotate_rc(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s * s - dot(u, u)) * v + 2.0 * s * cross(u, v);
}

fn quat_conjugate(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

fn ray_sphere(origin: vec3<f32>, dir: vec3<f32>, center: vec3<f32>, radius: f32) -> vec2<f32> {
    let oc = origin - center;
    let b = dot(oc, dir);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 { return vec2<f32>(1e30, 0.0); }
    let t = -b - sqrt(disc);
    if t < 0.0 { return vec2<f32>(1e30, 0.0); }
    return vec2<f32>(t, 1.0);
}

fn ray_box(origin: vec3<f32>, dir: vec3<f32>, pos: vec3<f32>, rot: vec4<f32>, he: vec3<f32>) -> vec3<f32> {
    let inv_rot = quat_conjugate(rot);
    let local_o = quat_rotate_rc(inv_rot, origin - pos);
    let local_d = quat_rotate_rc(inv_rot, dir);
    var tmin = -1e30;
    var tmax = 1e30;
    var best_axis = 0u;
    var best_sign = 1.0;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let o_i = select(select(local_o.z, local_o.y, i == 1u), local_o.x, i == 0u);
        let d_i = select(select(local_d.z, local_d.y, i == 1u), local_d.x, i == 0u);
        let h_i = select(select(he.z, he.y, i == 1u), he.x, i == 0u);
        if abs(d_i) < 1e-12 {
            if o_i < -h_i || o_i > h_i { return vec3<f32>(1e30, 0.0, 0.0); }
            continue;
        }
        let t1 = (-h_i - o_i) / d_i;
        let t2 = (h_i - o_i) / d_i;
        let t_near = min(t1, t2);
        let t_far = max(t1, t2);
        if t_near > tmin {
            tmin = t_near;
            best_axis = i;
            best_sign = select(1.0, -1.0, d_i > 0.0);
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
    let origin = ray.origin.xyz;
    let dir = ray.direction.xyz;
    let max_t = ray.origin.w;

    var best_t = max_t;
    var best_body = 0u;
    var best_normal = vec3<f32>(0.0, 1.0, 0.0);
    var found_hit = false;

    for (var bi = 0u; bi < num_bodies; bi = bi + 1u) {
        let pos = bodies[bi].position_inv_mass.xyz;
        let rot = bodies[bi].orientation;
        let st = props[bi].shape_type;
        let si = props[bi].shape_index;

        if st == 0u {
            let r = spheres[si].radius;
            let result = ray_sphere(origin, dir, pos, r);
            if result.y > 0.5 && result.x < best_t && result.x >= 0.0 {
                best_t = result.x;
                best_body = bi;
                let hit_pt = origin + dir * result.x;
                best_normal = normalize(hit_pt - pos);
                found_hit = true;
            }
        } else if st == 1u {
            let he = boxes_data[si].half_extents.xyz;
            let result = ray_box(origin, dir, pos, rot, he);
            if result.x < best_t && result.x >= 0.0 {
                best_t = result.x;
                best_body = bi;
                var local_n = vec3<f32>(0.0);
                let axis = u32(result.y);
                if axis == 0u { local_n.x = result.z; }
                else if axis == 1u { local_n.y = result.z; }
                else { local_n.z = result.z; }
                best_normal = quat_rotate_rc(rot, local_n);
                found_hit = true;
            }
        }
    }

    hits[ray_idx].t = best_t;
    hits[ray_idx].normal_x = best_normal.x;
    hits[ray_idx].normal_y = best_normal.y;
    hits[ray_idx].normal_z = best_normal.z;
    hits[ray_idx].body_index = best_body;
    hits[ray_idx].hit = select(0u, 1u, found_hit);
    hits[ray_idx]._pad0 = 0u;
    hits[ray_idx]._pad1 = 0u;
}
"#;

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
    gravity:           vec4<f32>,
    dt:                f32,
    num_bodies:        u32,
    solver_iterations: u32,
    _pad:              u32,
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
    gauss_map_offset: u32,
    gauss_map_count:  u32,
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
    if idx >= params.num_bodies {
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

// ---------------------------------------------------------------------------
// GpuPipeline
// ---------------------------------------------------------------------------

/// Orchestrates GPU compute dispatches for the full 3D AVBD physics step.
///
/// The pipeline uses velocity-based AVBD (Averaged Velocity-Based Dynamics):
/// 1. Predict positions from velocities + gravity
/// 2. Compute AABBs from predicted positions
/// 3. Broadphase: find overlapping AABB pairs (O(N^2) GPU kernel)
/// 4. Narrowphase: generate contacts from overlapping pairs
/// 5. AVBD solve: apply velocity impulses using averaged velocities
/// 6. Extract velocities from position changes
pub struct GpuPipeline {
    ctx: GpuContext,

    // Kernels
    predict_kernel: ComputeKernel,
    aabb_kernel: ComputeKernel,
    narrowphase_kernel: ComputeKernel,
    solve_kernel: ComputeKernel,
    extract_kernel: ComputeKernel,

    // Storage buffers
    body_states: PingPongBuffer<RigidBodyState3D>,
    body_props: GpuBuffer<RigidBodyProps3D>,
    old_states: GpuBuffer<RigidBodyState3D>,
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
    gauss_map: GpuBuffer<GaussMapEntry>,

    // Compound shape data (for CPU-side pair expansion)
    compound_shapes_data: Vec<CompoundShapeGpu>,
    compound_children_data: Vec<CompoundChildGpu>,
    /// CPU-side compound shapes with BVH data for broadphase culling.
    compound_shapes_cpu: Vec<rubble_shapes3d::CompoundShape>,

    // Uniform buffers
    params_uniform: wgpu::Buffer,
    solve_range_uniform: wgpu::Buffer,

    // Raycast infrastructure
    raycast_kernel: ComputeKernel,
    rays_buffer: GpuBuffer<GpuRay>,
    hits_buffer: GpuBuffer<GpuRayHit>,
    raycast_params: wgpu::Buffer,
}

impl GpuPipeline {
    /// Create a new GPU pipeline. Compiles all shaders and allocates buffers.
    pub fn new(ctx: GpuContext, max_bodies: usize) -> Self {
        let max_pairs = max_bodies * 8;
        let max_contacts = max_bodies * 8;

        let predict_kernel = ComputeKernel::from_wgsl(&ctx, PREDICT_WGSL, "main");
        let aabb_kernel = ComputeKernel::from_wgsl(&ctx, AABB_COMPUTE_WGSL, "main");
        let narrowphase_kernel = ComputeKernel::from_wgsl(&ctx, NARROWPHASE_WGSL, "main");
        let solve_kernel = ComputeKernel::from_wgsl(&ctx, AVBD_SOLVE_WGSL, "main");
        let extract_kernel = ComputeKernel::from_wgsl(&ctx, EXTRACT_VELOCITY_WGSL, "main");

        let body_states = PingPongBuffer::new(&ctx, max_bodies);
        let body_props = GpuBuffer::new(&ctx, max_bodies);
        let old_states = GpuBuffer::new(&ctx, max_bodies);
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
        let gauss_map = GpuBuffer::new(&ctx, (max_bodies * 64).max(1));

        let params_uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SimParams uniform"),
            size: std::mem::size_of::<SimParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let solve_range_uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SolveRange uniform"),
            size: std::mem::size_of::<SolveRangeGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let raycast_kernel = ComputeKernel::from_wgsl(&ctx, RAYCAST_WGSL, "main");
        let rays_buffer = GpuBuffer::new(&ctx, 1024);
        let hits_buffer = GpuBuffer::new(&ctx, 1024);
        let raycast_params = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raycast params"),
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
            body_props,
            old_states,
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
            gauss_map,
            compound_shapes_data: Vec::new(),
            compound_children_data: Vec::new(),
            compound_shapes_cpu: Vec::new(),
            params_uniform,
            solve_range_uniform,
            raycast_kernel,
            rays_buffer,
            hits_buffer,
            raycast_params,
        }
    }

    /// Try to create a GPU pipeline. Returns None if no GPU adapter is available.
    pub fn try_new(max_bodies: usize) -> Option<Self> {
        let ctx = pollster::block_on(GpuContext::new()).ok()?;
        Some(Self::new(ctx, max_bodies))
    }

    /// Upload body data from CPU arrays to GPU buffers and set simulation params.
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        &mut self,
        states: &[RigidBodyState3D],
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
        gauss_map_data: &[GaussMapEntry],
        gravity: Vec3,
        dt: f32,
        solver_iterations: u32,
    ) {
        // Store compound data for CPU-side pair expansion
        self.compound_shapes_data = compound_shapes.to_vec();
        self.compound_children_data = compound_children.to_vec();
        self.compound_shapes_cpu = compound_shapes_cpu.to_vec();
        self.body_states.upload(&self.ctx, states);
        self.old_states.upload(&self.ctx, states);
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
        if !gauss_map_data.is_empty() {
            self.gauss_map.upload(&self.ctx, gauss_map_data);
        }

        self.aabbs.grow_if_needed(&self.ctx, states.len());

        let params = SimParamsGpu {
            gravity: [gravity.x, gravity.y, gravity.z, 0.0],
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
    ///
    /// When a broadphase pair involves a compound shape (SHAPE_COMPOUND = 5),
    /// the pair is expanded on the CPU into individual child-vs-body pairs.
    /// This avoids adding compound-specific bindings/logic to the GPU narrowphase.
    fn run_detection(&mut self, num_bodies: u32) -> Vec<Contact3D> {
        self.contact_count.reset(&self.ctx);
        self.pair_count.reset(&self.ctx);

        self.dispatch_predict(num_bodies);
        self.dispatch_aabb(num_bodies);

        // LBVH broadphase: download AABBs, build tree on CPU, upload pairs
        self.aabbs.set_len(num_bodies);
        let gpu_aabbs = self.aabbs.download(&self.ctx);
        let bvh = lbvh::Lbvh::build(&gpu_aabbs);
        let overlap_pairs = bvh.find_overlapping_pairs(&gpu_aabbs);

        let mut cpu_compound_contacts: Vec<Contact3D> = Vec::new();

        if !overlap_pairs.is_empty() {
            // Download props to identify compound shapes
            self.body_props.set_len(num_bodies);
            let props = self.body_props.download(&self.ctx);
            // Also download states for compound child world position computation
            self.body_states.current_mut().set_len(num_bodies);
            let states = self.body_states.download(&self.ctx);

            let mut non_compound_pairs: Vec<GpuPair> = Vec::with_capacity(overlap_pairs.len());

            for p in &overlap_pairs {
                let a = p[0];
                let b = p[1];
                let st_a = props[a as usize].shape_type;
                let st_b = props[b as usize].shape_type;

                let a_is_compound = st_a == rubble_math::SHAPE_COMPOUND;
                let b_is_compound = st_b == rubble_math::SHAPE_COMPOUND;

                if !a_is_compound && !b_is_compound {
                    // Neither is compound, pass through to GPU narrowphase
                    non_compound_pairs.push(GpuPair { a, b });
                } else {
                    // At least one is compound: generate contacts on CPU.
                    self.generate_compound_contacts_cpu(
                        a,
                        b,
                        &props,
                        &states,
                        &mut cpu_compound_contacts,
                    );
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
                self.pairs.upload(&self.ctx, &non_compound_pairs);
                self.pair_count
                    .write(&self.ctx, non_compound_pairs.len() as u32);
            }
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
                    let local_min = Self::transform_aabb_to_local(world_min_a, world_max_a, pos_b, inv_rot_b);
                    let local_max = Self::transform_aabb_to_local_max(world_min_a, world_max_a, pos_b, inv_rot_b);
                    for idx_b in Self::traverse_compound_bvh(&cs_b.bvh_nodes, local_min, local_max) {
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
                    let local_min = Self::transform_aabb_to_local(world_min_b, world_max_b, pos_a, inv_rot_a);
                    let local_max = Self::transform_aabb_to_local_max(world_min_b, world_max_b, pos_a, inv_rot_a);
                    for idx_a in Self::traverse_compound_bvh(&cs_a.bvh_nodes, local_min, local_max) {
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

            out.push(Contact3D {
                point: Vec4::new(point.x, point.y, point.z, depth),
                normal: Vec4::new(normal.x, normal.y, normal.z, 0.0),
                body_a,
                body_b,
                feature_id: 0,
                _pad: 0,
                lambda_n: 0.0,
                lambda_t1: 0.0,
                lambda_t2: 0.0,
                penalty_k: 1e4,
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
        world_min: Vec3, world_max: Vec3, parent_pos: Vec3, inv_parent_rot: glam::Quat,
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
        world_min: Vec3, world_max: Vec3, parent_pos: Vec3, inv_parent_rot: glam::Quat,
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

    /// Get a rough bounding radius for a child shape based on its type.
    fn child_extent(&self, shape_type: u32, _shape_index: u32) -> f32 {
        // Conservative estimate per shape type. Actual dimensions could be
        // looked up from CPU-side shape buffers for higher accuracy.
        match shape_type {
            0 => 1.0, // SHAPE_SPHERE
            1 => 1.5, // SHAPE_BOX
            2 => 1.5, // SHAPE_CAPSULE
            3 => 2.0, // SHAPE_CONVEX_HULL
            4 => 1e4, // SHAPE_PLANE
            _ => 2.0,
        }
    }

    /// Graph-colored AVBD solve: download contacts, color them so no two
    /// same-color contacts share a body, sort by color, re-upload, then
    /// dispatch one GPU pass per color group per iteration.
    fn run_colored_solve(&mut self, solver_iterations: u32, contacts: &mut Vec<Contact3D>) {
        if contacts.is_empty() {
            return;
        }

        // Color contacts: two contacts conflict if they share a body
        let color_groups = color_contacts(contacts);

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
            self.run_colored_solve(solver_iterations, &mut contacts);
        }

        self.dispatch_extract(num_bodies);
        let states = self.body_states.download(&self.ctx);
        self.body_states.swap();
        states
    }

    /// Run the full GPU physics step with warm-starting support.
    /// Returns (updated_states, new_contacts) so the caller can track persistence.
    pub fn step_with_contacts(
        &mut self,
        num_bodies: u32,
        solver_iterations: u32,
        warm_contacts: Option<&[Contact3D]>,
    ) -> (Vec<RigidBodyState3D>, Vec<Contact3D>) {
        if num_bodies == 0 {
            return (Vec::new(), Vec::new());
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

        // Warm-start: apply cached lambdas from previous frame
        if let Some(prev) = warm_contacts {
            warm_start_contacts_3d(&mut contacts, prev);
        }

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
    pub fn download_contacts(&mut self) -> Vec<Contact3D> {
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
    /// Each ray is tested against all bodies (sphere and box shapes only).
    /// Results are downloaded synchronously.
    pub fn raycast_batch_gpu(
        &mut self,
        rays: &[GpuRay],
        num_bodies: u32,
    ) -> Vec<GpuRayHit> {
        if rays.is_empty() || num_bodies == 0 {
            return Vec::new();
        }
        let num_rays = rays.len() as u32;
        self.rays_buffer.upload(&self.ctx, rays);
        self.hits_buffer.grow_if_needed(&self.ctx, rays.len());
        self.hits_buffer.set_len(num_rays);
        // Upload zeros to hits buffer
        let zeros = vec![GpuRayHit {
            t: 0.0, normal_x: 0.0, normal_y: 0.0, normal_z: 0.0,
            body_index: 0, hit: 0, _pad0: 0, _pad1: 0,
        }; rays.len()];
        self.hits_buffer.upload(&self.ctx, &zeros);

        let params_data: [u32; 4] = [num_rays, num_bodies, 0, 0];
        self.ctx.queue.write_buffer(
            &self.raycast_params,
            0,
            bytemuck::cast_slice(&params_data),
        );

        let bg = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raycast"),
            layout: self.raycast_kernel.bind_group_layout(),
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
        self.run_pass("raycast", &self.raycast_kernel, &bg, num_rays);

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
                        resource: self.params_uniform.as_entire_binding(),
                    },
                ],
            });
        self.run_pass("predict", &self.predict_kernel, &bg, num_bodies);
    }

    fn dispatch_aabb(&self, num_bodies: u32) {
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
        self.run_pass("aabb", &self.aabb_kernel, &bg, num_bodies);
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
                label: Some("narrowphase"),
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
                        resource: self.pairs.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.pair_count.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.spheres.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.boxes.buffer().as_entire_binding(),
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
                        resource: self.convex_hulls.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: self.convex_vertices.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: self.capsules.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: self.planes.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: self.gauss_map.buffer().as_entire_binding(),
                    },
                ],
            });
        self.run_pass("narrowphase", &self.narrowphase_kernel, &bg, num_pairs);
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
                label: Some("avbd_solve"),
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
                        resource: self.body_props.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.contacts.buffer().as_entire_binding(),
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
                        resource: self.solve_range_uniform.as_entire_binding(),
                    },
                ],
            });
        self.run_pass("avbd_solve", &self.solve_kernel, &bg, count);
    }

    fn dispatch_extract(&self, num_bodies: u32) {
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
                        resource: self.params_uniform.as_entire_binding(),
                    },
                ],
            });
        self.run_pass("extract_vel", &self.extract_kernel, &bg, num_bodies);
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

/// Color contacts so no two same-color contacts share a body.
/// Sorts `contacts` in-place by color and returns (offset, count) for each color group.
fn color_contacts(contacts: &mut Vec<Contact3D>) -> Vec<(u32, u32)> {
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
    let mut indexed: Vec<(u32, Contact3D)> =
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

/// Match new contacts against previous-frame contacts and copy cached lambdas.
fn warm_start_contacts_3d(new_contacts: &mut [Contact3D], prev_contacts: &[Contact3D]) {
    let dist_thresh_sq: f32 = 0.01 * 0.01; // 1cm matching threshold
    let decay: f32 = 0.95;

    for nc in new_contacts.iter_mut() {
        let np = nc.contact_point();
        let mut best_dist_sq = f32::MAX;
        let mut best_idx: Option<usize> = None;

        for (i, pc) in prev_contacts.iter().enumerate() {
            // Match by body pair
            let same_pair = (pc.body_a == nc.body_a && pc.body_b == nc.body_b)
                || (pc.body_a == nc.body_b && pc.body_b == nc.body_a);
            if !same_pair {
                continue;
            }
            let d2 = (pc.contact_point() - np).length_squared();
            if d2 < best_dist_sq {
                best_dist_sq = d2;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            if best_dist_sq < dist_thresh_sq {
                nc.lambda_n = prev_contacts[idx].lambda_n * decay;
                nc.lambda_t1 = prev_contacts[idx].lambda_t1 * decay;
                nc.lambda_t2 = prev_contacts[idx].lambda_t2 * decay;
                nc.penalty_k = prev_contacts[idx].penalty_k;
            }
        }
    }
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
