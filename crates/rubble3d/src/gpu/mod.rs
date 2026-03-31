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
use rubble_gpu::{round_up_workgroups, ComputeKernel, GpuAtomicCounter, GpuBuffer, GpuContext};
use rubble_math::{
    Aabb3D, BodyHandle, CollisionEvent, Contact3D, RigidBodyProps3D, RigidBodyState3D,
};
use rubble_shapes3d::{BoxData, CapsuleData, ConvexHullData, ConvexVertex3D, SphereData};
use std::collections::HashSet;

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
        aabb_min = pos - vec3<f32>(2.0, 2.0, 2.0);
        aabb_max = pos + vec3<f32>(2.0, 2.0, 2.0);
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
    body_states: GpuBuffer<RigidBodyState3D>,
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

    // Uniform buffer (shaders expect `var<uniform>`)
    params_uniform: wgpu::Buffer,
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

        let body_states = GpuBuffer::new(&ctx, max_bodies);
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

        let params_uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SimParams uniform"),
            size: std::mem::size_of::<SimParamsGpu>() as u64,
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
            params_uniform,
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
        gravity: Vec3,
        dt: f32,
        solver_iterations: u32,
    ) {
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

    /// Run the full GPU physics step and download updated states.
    pub fn step(&mut self, num_bodies: u32, solver_iterations: u32) -> Vec<RigidBodyState3D> {
        if num_bodies == 0 {
            return Vec::new();
        }

        self.contact_count.reset(&self.ctx);
        self.pair_count.reset(&self.ctx);

        self.dispatch_predict(num_bodies);
        self.dispatch_aabb(num_bodies);

        // LBVH broadphase: download AABBs, build tree on CPU, upload pairs
        self.aabbs.set_len(num_bodies);
        let gpu_aabbs = self.aabbs.download(&self.ctx);
        let bvh = lbvh::Lbvh::build(&gpu_aabbs);
        let overlap_pairs = bvh.find_overlapping_pairs(&gpu_aabbs);

        if !overlap_pairs.is_empty() {
            let gpu_pairs: Vec<GpuPair> = overlap_pairs
                .iter()
                .map(|p| GpuPair { a: p[0], b: p[1] })
                .collect();
            self.pairs.upload(&self.ctx, &gpu_pairs);
            self.pair_count.write(&self.ctx, gpu_pairs.len() as u32);
        }

        self.dispatch_narrowphase(num_bodies);

        for _ in 0..solver_iterations {
            self.dispatch_solve(num_bodies);
        }

        self.dispatch_extract(num_bodies);
        self.body_states.download(&self.ctx)
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

        self.contact_count.reset(&self.ctx);
        self.pair_count.reset(&self.ctx);

        self.dispatch_predict(num_bodies);
        self.dispatch_aabb(num_bodies);

        // LBVH broadphase
        self.aabbs.set_len(num_bodies);
        let gpu_aabbs = self.aabbs.download(&self.ctx);
        let bvh = lbvh::Lbvh::build(&gpu_aabbs);
        let overlap_pairs = bvh.find_overlapping_pairs(&gpu_aabbs);

        if !overlap_pairs.is_empty() {
            let gpu_pairs: Vec<GpuPair> = overlap_pairs
                .iter()
                .map(|p| GpuPair { a: p[0], b: p[1] })
                .collect();
            self.pairs.upload(&self.ctx, &gpu_pairs);
            self.pair_count.write(&self.ctx, gpu_pairs.len() as u32);
        }

        self.dispatch_narrowphase(num_bodies);

        // Warm-start: download contacts, apply cached lambdas, re-upload
        if let Some(prev) = warm_contacts {
            let count = self.contact_count.read(&self.ctx) as usize;
            if count > 0 {
                let mut contacts = self.contacts.download(&self.ctx);
                contacts.truncate(count);
                warm_start_contacts_3d(&mut contacts, prev);
                self.contacts.upload(&self.ctx, &contacts);
            }
        }

        for _ in 0..solver_iterations {
            self.dispatch_solve(num_bodies);
        }

        // Download contacts after solve (lambdas are updated)
        let final_contacts = self.download_contacts();

        self.dispatch_extract(num_bodies);
        let states = self.body_states.download(&self.ctx);
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
    pub fn download_contacts(&self) -> Vec<Contact3D> {
        let count = self.contact_count.read(&self.ctx) as usize;
        if count == 0 {
            return Vec::new();
        }
        let all = self.contacts.download(&self.ctx);
        all.into_iter().take(count).collect()
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
                        resource: self.body_states.buffer().as_entire_binding(),
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
                        resource: self.body_states.buffer().as_entire_binding(),
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
                        resource: self.body_states.buffer().as_entire_binding(),
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
                ],
            });
        self.run_pass("narrowphase", &self.narrowphase_kernel, &bg, num_pairs);
    }

    fn dispatch_solve(&self, _num_bodies: u32) {
        let num_contacts = self.contact_count.read(&self.ctx);
        if num_contacts == 0 {
            return;
        }
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("avbd_solve"),
                layout: self.solve_kernel.bind_group_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.body_states.buffer().as_entire_binding(),
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
                ],
            });
        self.run_pass("avbd_solve", &self.solve_kernel, &bg, num_contacts);
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
                        resource: self.body_states.buffer().as_entire_binding(),
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
