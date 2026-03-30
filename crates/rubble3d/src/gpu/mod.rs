//! GPU compute pipeline for 3D physics simulation using AVBD solver.
//!
//! Orchestrates the full simulation step on the GPU using WGSL compute shaders:
//! predict → AABB compute → broadphase → narrowphase → AVBD solver → velocity extraction.

mod avbd_solve_wgsl;
mod broadphase_pairs_wgsl;
mod extract_velocity_wgsl;
mod morton_codes_wgsl;
mod narrowphase_wgsl;
mod predict_wgsl;

pub use avbd_solve_wgsl::AVBD_SOLVE_WGSL;
pub use broadphase_pairs_wgsl::BROADPHASE_PAIRS_WGSL;
pub use extract_velocity_wgsl::EXTRACT_VELOCITY_WGSL;
pub use morton_codes_wgsl::MORTON_CODES_WGSL;
pub use narrowphase_wgsl::NARROWPHASE_WGSL;
pub use predict_wgsl::PREDICT_WGSL;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use rubble_gpu::{round_up_workgroups, ComputeKernel, GpuAtomicCounter, GpuBuffer, GpuContext};
use rubble_math::{Aabb3D, Contact3D, RigidBodyProps3D, RigidBodyState3D};
use rubble_shapes3d::{BoxData, SphereData};

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

const SHAPE_SPHERE: u32 = 0u;
const SHAPE_BOX:    u32 = 1u;

@group(0) @binding(0) var<storage, read>       bodies:  array<Body>;
@group(0) @binding(1) var<storage, read>       props:   array<BodyProps>;
@group(0) @binding(2) var<storage, read>       spheres: array<SphereData>;
@group(0) @binding(3) var<storage, read>       boxes:   array<BoxDataGpu>;
@group(0) @binding(4) var<storage, read_write> aabbs:   array<Aabb>;
@group(0) @binding(5) var<uniform>             params:  SimParams;

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
    } else {
        // Capsule/other: generous default AABB
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
    pairs_kernel: ComputeKernel,
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
        let pairs_kernel = ComputeKernel::from_wgsl(&ctx, BROADPHASE_PAIRS_WGSL, "main");
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
            pairs_kernel,
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
        self.dispatch_broadphase(num_bodies);
        self.dispatch_narrowphase(num_bodies);

        for _ in 0..solver_iterations {
            self.dispatch_solve(num_bodies);
        }

        self.dispatch_extract(num_bodies);
        self.body_states.download(&self.ctx)
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
                ],
            });
        self.run_pass("aabb", &self.aabb_kernel, &bg, num_bodies);
    }

    fn dispatch_broadphase(&self, num_bodies: u32) {
        let total_pairs = num_bodies * num_bodies.saturating_sub(1) / 2;
        if total_pairs == 0 {
            return;
        }
        let bg = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("broadphase"),
                layout: self.pairs_kernel.bind_group_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.aabbs.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.pairs.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.pair_count.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.params_uniform.as_entire_binding(),
                    },
                ],
            });
        self.run_pass("broadphase", &self.pairs_kernel, &bg, total_pairs);
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
