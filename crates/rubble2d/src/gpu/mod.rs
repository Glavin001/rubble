//! GPU compute pipeline for 2D physics simulation using AVBD solver.
//!
//! Orchestrates the full simulation step on the GPU using WGSL compute shaders:
//! predict -> AABB compute -> broadphase -> narrowphase -> AVBD solver -> velocity extraction.

mod avbd_solve_wgsl;
mod broadphase_pairs_wgsl;
mod extract_velocity_wgsl;
mod narrowphase_wgsl;
mod predict_wgsl;

pub use avbd_solve_wgsl::AVBD_SOLVE_2D_WGSL;
pub use broadphase_pairs_wgsl::BROADPHASE_PAIRS_2D_WGSL;
pub use extract_velocity_wgsl::EXTRACT_VELOCITY_2D_WGSL;
pub use narrowphase_wgsl::NARROWPHASE_2D_WGSL;
pub use predict_wgsl::PREDICT_2D_WGSL;

use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use rubble_gpu::{round_up_workgroups, ComputeKernel, GpuAtomicCounter, GpuBuffer, GpuContext};
use rubble_math::{Aabb2D, Contact2D, RigidBodyState2D};
use rubble_shapes2d::{CircleData, ConvexPolygonData, ConvexVertex2D, RectData};

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

@group(0) @binding(0) var<storage, read>       bodies:        array<Body2D>;
@group(0) @binding(1) var<storage, read>       shape_infos:   array<ShapeInfo>;
@group(0) @binding(2) var<storage, read>       circles:       array<CircleData>;
@group(0) @binding(3) var<storage, read>       rects:         array<RectDataGpu>;
@group(0) @binding(4) var<storage, read_write> aabbs:         array<Aabb2D>;
@group(0) @binding(5) var<uniform>             params:        SimParams2D;
@group(0) @binding(6) var<storage, read>       convex_polys:  array<ConvexPolyInfo>;
@group(0) @binding(7) var<storage, read>       convex_verts:  array<ConvexVert2D>;

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
    pairs_kernel: ComputeKernel,
    narrowphase_kernel: ComputeKernel,
    solve_kernel: ComputeKernel,
    extract_kernel: ComputeKernel,

    // Storage buffers
    body_states: GpuBuffer<RigidBodyState2D>,
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
    shape_infos: GpuBuffer<ShapeInfo>,

    // Uniform buffer (shaders expect `var<uniform>`)
    params_uniform: wgpu::Buffer,
}

impl GpuPipeline2D {
    /// Create a new 2D GPU pipeline. Compiles all shaders and allocates buffers.
    pub fn new(ctx: GpuContext, max_bodies: usize) -> Self {
        let max_pairs = max_bodies * 8;
        let max_contacts = max_bodies * 8;

        let predict_kernel = ComputeKernel::from_wgsl(&ctx, PREDICT_2D_WGSL, "main");
        let aabb_kernel = ComputeKernel::from_wgsl(&ctx, AABB_COMPUTE_2D_WGSL, "main");
        let pairs_kernel = ComputeKernel::from_wgsl(&ctx, BROADPHASE_PAIRS_2D_WGSL, "main");
        let narrowphase_kernel = ComputeKernel::from_wgsl(&ctx, NARROWPHASE_2D_WGSL, "main");
        let solve_kernel = ComputeKernel::from_wgsl(&ctx, AVBD_SOLVE_2D_WGSL, "main");
        let extract_kernel = ComputeKernel::from_wgsl(&ctx, EXTRACT_VELOCITY_2D_WGSL, "main");

        let body_states = GpuBuffer::new(&ctx, max_bodies);
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
        let shape_infos = GpuBuffer::new(&ctx, max_bodies);

        let params_uniform = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SimParams2D uniform"),
            size: std::mem::size_of::<SimParams2DGpu>() as u64,
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
            shape_infos,
            params_uniform,
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

    /// Run the full GPU 2D physics step and download updated states.
    pub fn step(&mut self, num_bodies: u32, solver_iterations: u32) -> Vec<RigidBodyState2D> {
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
    pub fn download_contacts(&self) -> Vec<Contact2D> {
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
                label: Some("predict_2d"),
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
                        resource: self.body_states.buffer().as_entire_binding(),
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
                ],
            });
        self.run_pass("aabb_2d", &self.aabb_kernel, &bg, num_bodies);
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
                label: Some("broadphase_2d"),
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
        self.run_pass("broadphase_2d", &self.pairs_kernel, &bg, total_pairs);
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
                        resource: self.body_states.buffer().as_entire_binding(),
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
                ],
            });
        self.run_pass("narrowphase_2d", &self.narrowphase_kernel, &bg, num_pairs);
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
                label: Some("avbd_solve_2d"),
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
        self.run_pass("avbd_solve_2d", &self.solve_kernel, &bg, num_contacts);
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
