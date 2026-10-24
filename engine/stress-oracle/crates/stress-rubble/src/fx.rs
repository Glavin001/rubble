//! Fragment effects for [`Renderer`](crate::render::Renderer): dust puffs and chips where a
//! chunk's velocity jumps (an impact, a break, a landing), read from the poses the renderer
//! draws. Render only: nothing here reads or changes the simulation.
//!
//! Each frame, one compute pass compares every chunk's centre with the last frame's: the
//! change in its velocity over the simulated time the frame advanced picks how much it
//! sheds. A second pass ages and moves the particles (dust drifts and spreads, chips fall
//! and skid on the ground). They draw as camera-facing quads in the renderer's pass.
//! A frame that advances no simulated time (paused, or a slow-motion frame between
//! physics frames) freezes them.
//!
//! Deterministic: each object owns a fixed range of particle slots (a ring of its own, its
//! head kept with its last pose), so which slot a particle takes, its random numbers and
//! the draw order follow the object's index and history, never the order GPU threads run
//! in. Two recordings of a take give the same frames.

use stress_gpu::gpu::Gpu;

/// Particles alive at once, at least: each object gets an equal share of the slots (a ring
/// of its own: its newest replace its oldest), and never fewer than [`MIN_SLOTS`].
pub const PARTICLES: u32 = 1 << 16;
/// Slots per object at least (the most one frame sheds).
pub const MIN_SLOTS: u32 = 6;

const COMMON: &str = r#"
// render.rs `CameraUniform`.
struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    light: vec4<f32>,
    horizon: vec4<f32>,
    zenith: vec4<f32>,
    inverse_view_proj: mat4x4<f32>,
};
struct Object {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    color: vec4<f32>,
};
// p: position and life left (s); v: velocity and age (s); c: colour and kind (0 dust, 1 chip).
struct Particle {
    p: vec4<f32>,
    v: vec4<f32>,
    c: vec4<f32>,
};
// A chunk's centre (w: 1 once seen) and velocity at the last frame; v.w holds (as bits)
// the next of its particle slots to use.
struct Prev {
    p: vec4<f32>,
    v: vec4<f32>,
};
struct Params {
    dt: f32,
    objects: u32,
    seed: u32,
    // Particle slots per object: object i owns [i * slots, (i + 1) * slots).
    slots: u32,
    // 1: something was set moving between frames (a launch), so this frame's velocity
    // jumps are not impacts: they update the comparison and shed nothing.
    quiet: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
@group(0) @binding(0) var<uniform> camera: Camera;
const N: u32 = PARTICLES_N;
"#;

const COMPUTE: &str = r#"
@group(0) @binding(1) var<storage, read> objects: array<Object>;
// Each object's mesh centre (object frame) and radius; radius 0: sheds nothing.
@group(0) @binding(2) var<storage, read> bounds: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> prev: array<Prev>;
@group(0) @binding(4) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(6) var<uniform> params: Params;

fn hash(a: u32, b: u32) -> f32 {
    var x = a * 747796405u + b * 2891336453u + params.seed * 1664525u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return f32((x >> 22u) ^ x) / 4294967295.0;
}
fn hash_dir(a: u32, b: u32) -> vec3<f32> {
    let z = hash(a, b) * 2.0 - 1.0;
    let ph = hash(a, b + 7u) * 6.28318;
    let r = sqrt(max(1.0 - z * z, 0.0));
    return vec3<f32>(r * cos(ph), r * sin(ph), z);
}

@compute @workgroup_size(64)
fn spawn(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.objects) { return; }
    let b = bounds[i];
    let o = objects[i];
    // A negative radius: an object that sheds chips but no dust (a vehicle's sheet steel and
    // wheels, glass, metal).
    let radius = abs(b.w);
    let chips_only = b.w < 0.0;
    if (radius <= 0.0 || params.dt <= 0.0) { return; }
    if (dot(o.r0.xyz, o.r0.xyz) == 0.0) {
        // Not drawn (removed, or not yet placed).
        prev[i].p.w = 0.0;
        return;
    }
    let q = vec4<f32>(b.xyz, 1.0);
    let c = vec3<f32>(dot(o.r0, q), dot(o.r1, q), dot(o.r2, q));
    let last = prev[i];
    let ring = bitcast<u32>(last.v.w);
    if (last.p.w < 0.5) {
        // First seen: no velocity to compare with yet.
        prev[i] = Prev(vec4<f32>(c, 1.0), vec4<f32>(0.0, 0.0, 0.0, bitcast<f32>(ring)));
        return;
    }
    let v = (c - last.p.xyz) / params.dt;
    let dv = length(v - last.v.xyz);
    // A jump in velocity sheds; a teleport (a reload, a reset) does not.
    var count = 0u;
    if (dv >= 2.0 && dv <= 150.0 && params.quiet == 0u) {
        let size = clamp(radius, 0.1, 3.0);
        count = min(u32(clamp(dv * 0.2 * size, 1.0, 6.0)), params.slots);
        // Sheet steel, glass and metal: a few low chips, not a spray.
        if (chips_only) { count = max(1u, count / 2u); }
    }
    prev[i] = Prev(vec4<f32>(c, 1.0), vec4<f32>(v, bitcast<f32>(ring + count)));
    let dust = mix(o.color.rgb, vec3<f32>(0.66, 0.63, 0.58), 0.65);
    for (var j = 0u; j < count; j++) {
        // This object's own slots, in turn.
        let k = i * params.slots + (ring + j) % params.slots;
        let d = hash_dir(i, j * 13u);
        let chip = chips_only || hash(i, j * 13u + 3u) < 0.35;
        var part: Particle;
        let at = c + d * radius * 0.6 * hash(i, j * 13u + 5u);
        if (chip) {
            let spread = select(1.0 + 0.25 * dv, 0.5 + 0.1 * dv, chips_only);
            let vel = v * 0.6 + d * spread + vec3<f32>(0.0, 0.0, select(1.5, 0.8, chips_only));
            part = Particle(vec4<f32>(at, 1.5 + 1.5 * hash(i, j * 13u + 9u)), vec4<f32>(vel, 0.0), vec4<f32>(o.color.rgb * 0.75, 1.0));
        } else {
            let vel = v * 0.35 + d * (0.4 + 0.08 * dv) + vec3<f32>(0.0, 0.0, 0.4);
            part = Particle(vec4<f32>(at, 1.5 + 1.5 * hash(i, j * 13u + 11u)), vec4<f32>(vel, 0.0), vec4<f32>(dust, 0.0));
        }
        particles[k] = part;
    }
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= N || params.dt <= 0.0) { return; }
    var part = particles[i];
    if (part.p.w <= 0.0) { return; }
    let dt = params.dt;
    var v = part.v.xyz;
    var p = part.p.xyz;
    if (part.c.w < 0.5) {
        // Dust: drag, a slight rise.
        v = v * exp(-1.6 * dt) + vec3<f32>(0.0, 0.0, 0.25 * dt);
        p = p + v * dt;
    } else {
        // A chip: falls, bounces low and skids on the ground.
        v.z = v.z - 9.81 * dt;
        p = p + v * dt;
        if (p.z < 0.03) {
            p.z = 0.03;
            v = vec3<f32>(v.xy * 0.55, abs(v.z) * 0.25);
        }
    }
    part.p = vec4<f32>(p, part.p.w - dt);
    part.v = vec4<f32>(v, part.v.w + dt);
    particles[i] = part;
}
"#;

// The draw reads the particles: WebGPU allows no writable storage binding in a vertex
// stage, so it has a read-only binding (and layout) of its own.
const DRAW: &str = r#"
@group(0) @binding(4) var<storage, read> particles: array<Particle>;

struct FxOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn fx_vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FxOut {
    var out: FxOut;
    let part = particles[ii];
    if (part.p.w <= 0.0) {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        return out;
    }
    let corners = array<vec2<f32>, 6>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
    let uv = corners[vi];
    let age = part.v.w;
    let chip = part.c.w > 0.5;
    let size = select(0.25 + 1.0 * age / (age + 1.2), 0.06, chip);
    let to_eye = normalize(camera.eye.xyz - part.p.xyz);
    var right = cross(vec3<f32>(0.0, 0.0, 1.0), to_eye);
    if (dot(right, right) < 1e-6) { right = vec3<f32>(1.0, 0.0, 0.0); }
    right = normalize(right);
    let up = cross(to_eye, right);
    let world = part.p.xyz + (uv.x * right + uv.y * up) * size;
    out.clip = camera.view_proj * vec4<f32>(world, 1.0);
    out.uv = uv;
    // Dust fades in over 0.15 s and out over its last second; chips stay solid.
    let alpha = select(0.14 * clamp(age / 0.15, 0.0, 1.0) * clamp(part.p.w, 0.0, 1.0), 1.0, chip);
    // Colours are sRGB (the objects'); lit as render.rs lights a surface half turned to
    // the sun, through the same tone curve and haze, so dust and chips match the pieces.
    let lit = pow(max(part.c.rgb, vec3<f32>(0.0)), vec3<f32>(2.2)) * select(1.2, 1.05, chip);
    let toned = clamp((lit * (2.51 * lit + 0.03)) / (lit * (2.43 * lit + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
    let haze = 1.0 - exp(-max(length(camera.eye.xyz - part.p.xyz) - 40.0, 0.0) / (8.0 * camera.light.w));
    out.color = vec4<f32>(mix(toned, camera.horizon.rgb, haze), alpha);
    return out;
}

@fragment
fn fx_fs(in: FxOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    if (in.color.a >= 1.0) {
        return vec4<f32>(in.color.rgb, 1.0);
    }
    if (r2 > 1.0) { discard; }
    let a = in.color.a * (1.0 - r2) * (1.0 - r2);
    return vec4<f32>(in.color.rgb * a, a);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    dt: f32,
    objects: u32,
    seed: u32,
    slots: u32,
    quiet: u32,
    _pad: [u32; 3],
}

/// The particle buffers and pipelines (see the module docs).
pub(crate) struct Fx {
    spawn: wgpu::ComputePipeline,
    update: wgpu::ComputePipeline,
    draw: wgpu::RenderPipeline,
    compute_bind: wgpu::BindGroup,
    draw_bind: wgpu::BindGroup,
    params: wgpu::Buffer,
    objects: u32,
    /// Slots per object, and all of them (`objects * slots`).
    slots: u32,
    particles: u32,
    seed: std::cell::Cell<u32>,
}

impl Fx {
    /// Effects over `objects` (the renderer's objects buffer) with each object's
    /// `bounds` (mesh centre in the object's frame, and radius; radius 0 for objects that
    /// shed nothing), drawn into `format` with `samples` and a Depth32Float depth.
    pub(crate) fn new(
        gpu: &Gpu,
        camera: &wgpu::Buffer,
        objects: &wgpu::Buffer,
        bounds: &[[f32; 4]],
        format: wgpu::TextureFormat,
        samples: u32,
    ) -> Fx {
        use wgpu::util::DeviceExt as _;
        let device = &gpu.device;
        let storage = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let bounds_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fx bounds"),
            contents: bytemuck::cast_slice(bounds),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let prev = storage("fx prev", (bounds.len().max(1) * 32) as u64);
        let objects_n = (bounds.len() as u32).max(1);
        let slots = (PARTICLES / objects_n).max(MIN_SLOTS);
        let particle_count = (objects_n * slots).div_ceil(64) * 64;
        let particles = storage("fx particles", particle_count as u64 * 48);
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fx params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fx"),
            source: wgpu::ShaderSource::Wgsl(source(COMPUTE, particle_count).into()),
        });
        let all =
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
        let entry = |binding: u32, ty: wgpu::BufferBindingType, visibility: wgpu::ShaderStages| {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let compute = wgpu::ShaderStages::COMPUTE;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform, all),
                entry(1, ro, compute),
                entry(2, ro, compute),
                entry(3, rw, compute),
                entry(4, rw, compute),
                entry(6, wgpu::BufferBindingType::Uniform, compute),
            ],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: objects.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bounds_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: prev.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&compute_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (spawn, update) = (pipeline("spawn"), pipeline("update"));
        let draw_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fx draw"),
            source: wgpu::ShaderSource::Wgsl(source(DRAW, particle_count).into()),
        });
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fx draw"),
            entries: &[
                entry(
                    0,
                    wgpu::BufferBindingType::Uniform,
                    wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ),
                entry(4, ro, wgpu::ShaderStages::VERTEX),
            ],
        });
        let draw_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fx draw"),
            layout: &draw_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: particles.as_entire_binding(),
                },
            ],
        });
        let draw_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fx draw"),
            bind_group_layouts: &[Some(&draw_layout)],
            immediate_size: 0,
        });
        // Premultiplied alpha over the geometry; depth-tested, not written.
        let over = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let draw = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fx draw"),
            layout: Some(&draw_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &draw_module,
                entry_point: Some("fx_vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: samples,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &draw_module,
                entry_point: Some("fx_fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: over,
                        alpha: over,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Fx {
            spawn,
            update,
            draw,
            compute_bind: bind,
            draw_bind,
            params,
            objects: bounds.len() as u32,
            slots,
            particles: particle_count,
            seed: std::cell::Cell::new(1),
        }
    }

    /// Shed from what moved over `dt` s of simulated time, and age the particles by it.
    /// After the objects buffer holds this frame's poses; nothing when `dt` is 0.
    /// `quiet`: something was set moving between frames (a launch); this frame's velocity
    /// jumps are not impacts.
    pub(crate) fn encode(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, dt: f32, quiet: bool) {
        if dt <= 0.0 {
            return;
        }
        let seed = self.seed.get().wrapping_add(1);
        self.seed.set(seed);
        gpu.queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                dt,
                objects: self.objects,
                seed,
                slots: self.slots,
                quiet: quiet as u32,
                _pad: [0; 3],
            }),
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fx"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.compute_bind, &[]);
        pass.set_pipeline(&self.spawn);
        pass.dispatch_workgroups(self.objects.div_ceil(64).max(1), 1, 1);
        pass.set_pipeline(&self.update);
        pass.dispatch_workgroups(self.particles / 64, 1, 1);
    }

    /// Draw the live particles into the renderer's pass (after the opaque geometry).
    pub(crate) fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.draw);
        pass.set_bind_group(0, &self.draw_bind, &[]);
        pass.draw(0..6, 0..self.particles);
    }
}

/// A shader stage's source: the shared structs, then `body`.
fn source(body: &str, particles: u32) -> String {
    format!("{COMMON}{body}").replace("PARTICLES_N", &format!("{particles}u"))
}
