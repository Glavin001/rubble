//! Offscreen video renderer for the GPU world coupled to rubble (`stress_gpu::world::GpuWorld`
//! with [`crate::RubbleEngine::with_device`] on the solver's device), or any `ReferenceSolver`
//! state: every chunk drawn as its own convex hull (or box) at the pose the solver gives it
//! (cluster rotation, hidden rotation, chunk rotation; `World::chunk_box`), the impactors,
//! and a ground plane.
//! Lambert shading under a key light and a sky/ground ambient, 4x MSAA, written in WGSL
//! directly (no Slang). Frames are read back as RGBA (sRGB) for an overlay drawn on the
//! host (`Overlay`) and piped into ffmpeg (`FfmpegSink`).
//!
//! Colours: a chunk takes its material's colour while its cluster still stands on a
//! support; once its cluster is free (debris, a toppled piece) the colour is mixed with a
//! palette colour hashed from the cluster id, so each fragment reads as one piece.
//!
//! Nothing here touches the simulation: it reads the host mirror after a frame.

use std::io::Write as _;
use std::process::{Child, Command, Stdio};

use stress_gpu::gpu::Gpu;
use stress_ref::math::{Mat3, Quat, Vec3};
use stress_ref::scene::ImpactorShape;
use stress_ref::solver::ReferenceSolver;
use stress_ref::world::Impactor;

use wgpu::util::DeviceExt as _;

const SHADER: &str = r#"
struct Camera {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    light: vec4<f32>,
};
struct Object {
    r0: vec4<f32>,
    r1: vec4<f32>,
    r2: vec4<f32>,
    color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> objects: array<Object>;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) object: u32) -> VsOut {
    let o = objects[object];
    let p = vec4<f32>(pos, 1.0);
    let world = vec3<f32>(dot(o.r0, p), dot(o.r1, p), dot(o.r2, p));
    let n = vec3<f32>(dot(o.r0.xyz, normal), dot(o.r1.xyz, normal), dot(o.r2.xyz, normal));
    var out: VsOut;
    out.clip = camera.view_proj * vec4<f32>(world, 1.0);
    out.world = world;
    out.normal = n;
    out.color = o.color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    var n = normalize(in.normal);
    let to_eye = normalize(camera.eye.xyz - in.world);
    if (dot(n, to_eye) < 0.0) {
        n = -n;
    }
    var base = in.color.rgb;
    let dist = length(camera.eye.xyz - in.world);
    if (in.color.a > 1.5) {
        // Ground: a 1 m grid with 5 m major lines, faded with distance.
        let g = in.world.xy;
        let f1 = abs(fract(g + 0.5) - 0.5) / fwidth(g);
        let l1 = 1.0 - clamp(min(f1.x, f1.y), 0.0, 1.0);
        let f5 = abs(fract(g / 5.0 + 0.5) - 0.5) / fwidth(g / 5.0);
        let l5 = 1.0 - clamp(min(f5.x, f5.y), 0.0, 1.0);
        let fade = clamp(1.0 - dist / camera.light.w, 0.0, 1.0);
        base = mix(base, base * 0.78, max(0.5 * l1, l5) * fade);
    }
    let l = normalize(camera.light.xyz);
    let diffuse = max(dot(n, l), 0.0);
    let sky = mix(vec3<f32>(0.33, 0.31, 0.29), vec3<f32>(0.52, 0.58, 0.68), 0.5 + 0.5 * n.z);
    var c = base * (sky + vec3<f32>(1.0, 0.96, 0.88) * 0.85 * diffuse);
    // Aerial perspective toward the sky colour.
    let haze = clamp(dist / (2.5 * camera.light.w), 0.0, 0.6);
    c = mix(c, vec3<f32>(0.70, 0.76, 0.84), haze * haze);
    return vec4<f32>(c, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
    object: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Object {
    rows: [[f32; 4]; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    /// Key light direction (towards the light) and, in `w`, the fade distance (m).
    light: [f32; 4],
}

/// A camera orbiting `target` (Z up): `yaw` (rad, from +x), `pitch` above the horizon,
/// `distance` (m), vertical field of view `fov` (rad).
#[derive(Clone, Copy, Debug)]
pub struct CameraPose {
    pub target: Vec3,
    pub distance: f64,
    pub yaw: f64,
    pub pitch: f64,
    pub fov: f64,
}

impl CameraPose {
    pub fn eye(&self) -> Vec3 {
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        self.target + Vec3::new(self.yaw.cos() * cp, self.yaw.sin() * cp, sp) * self.distance
    }

    fn view_proj(&self, aspect: f64) -> [[f32; 4]; 4] {
        let eye = self.eye();
        let f = (self.target - eye).normalized();
        let s = f.cross(Vec3::new(0.0, 0.0, 1.0)).normalized();
        let u = s.cross(f);
        // Column-major view (right-handed, looking down -z) and projection (depth 0..1).
        let view = [
            [s.x, u.x, -f.x, 0.0],
            [s.y, u.y, -f.y, 0.0],
            [s.z, u.z, -f.z, 0.0],
            [-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0],
        ];
        let (near, far) = (0.05 * self.distance.min(20.0), 50.0 * self.distance);
        let t = 1.0 / (0.5 * self.fov).tan();
        let proj = [[t / aspect, 0.0, 0.0, 0.0], [0.0, t, 0.0, 0.0], [0.0, 0.0, far / (near - far), -1.0], [0.0, 0.0, near * far / (near - far), 0.0]];
        std::array::from_fn(|c| std::array::from_fn(|r| (0..4).map(|k| proj[k][r] * view[c][k]).sum::<f64>() as f32))
    }
}

/// The colour of a calibrated material (stress-ref material names).
pub fn material_color(name: &str) -> [f32; 3] {
    match name {
        "brick" => [0.66, 0.33, 0.24],
        "stone" => [0.74, 0.70, 0.62],
        "glass" => [0.58, 0.74, 0.84],
        "steel" => [0.50, 0.53, 0.58],
        "timber" => [0.66, 0.50, 0.32],
        "mortar" => [0.80, 0.78, 0.74],
        _ => [0.80, 0.79, 0.76],
    }
}

/// A palette colour for a fragment (cluster id).
fn fragment_color(id: u64) -> [f32; 3] {
    const PALETTE: [[f32; 3]; 10] = [
        [0.90, 0.45, 0.20],
        [0.25, 0.55, 0.85],
        [0.35, 0.72, 0.38],
        [0.85, 0.30, 0.45],
        [0.95, 0.75, 0.25],
        [0.55, 0.40, 0.80],
        [0.20, 0.70, 0.70],
        [0.80, 0.55, 0.35],
        [0.60, 0.80, 0.30],
        [0.90, 0.40, 0.70],
    ];
    let mut h = id.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h ^= h >> 29;
    PALETTE[(h % PALETTE.len() as u64) as usize]
}

/// Flat-shaded triangles of a convex hull (faces as vertex index loops in any order).
fn hull_triangles(vertices: &[Vec3], faces: &[(Vec<usize>, Vec3)], object: u32, out: &mut Vec<Vertex>) {
    for (idx, normal) in faces {
        if idx.len() < 3 {
            continue;
        }
        let centroid = idx.iter().fold(Vec3::ZERO, |a, &i| a + vertices[i]) * (1.0 / idx.len() as f64);
        let e1 = (vertices[idx[0]] - centroid).normalized();
        let e2 = normal.cross(e1);
        let mut ring: Vec<(f64, usize)> = idx
            .iter()
            .map(|&i| {
                let d = vertices[i] - centroid;
                (d.dot(e2).atan2(d.dot(e1)), i)
            })
            .collect();
        ring.sort_by(|a, b| a.0.total_cmp(&b.0));
        let n = [normal.x as f32, normal.y as f32, normal.z as f32];
        let v = |i: usize| {
            let p = vertices[i];
            Vertex { pos: [p.x as f32, p.y as f32, p.z as f32], normal: n, object }
        };
        for k in 1..ring.len() - 1 {
            out.push(v(ring[0].1));
            out.push(v(ring[k].1));
            out.push(v(ring[k + 1].1));
        }
    }
}

/// One chunk's flat-shaded triangles in its own frame (centred, chunk axes): its convex
/// hull, or its box when it has none.
fn chunk_mesh(ch: &stress_ref::structure::ChunkData, object: u32, out: &mut Vec<Vertex>) {
    match &ch.hull {
        Some(h) => {
            let faces: Vec<(Vec<usize>, Vec3)> = h.faces.iter().map(|f| (f.vertices.clone(), f.normal)).collect();
            hull_triangles(&h.vertices, &faces, object, out);
        }
        None => {
            let (v, f) = box_faces(ch.half_extents);
            hull_triangles(&v, &f, object, out);
        }
    }
}

/// [`chunk_mesh`] as (position, face normal) per vertex, three per triangle, for other
/// renderers (`destruction-view` draws these at the GPU solver's chunk poses).
pub fn chunk_triangles(ch: &stress_ref::structure::ChunkData) -> Vec<([f32; 3], [f32; 3])> {
    let mut verts = Vec::new();
    chunk_mesh(ch, 0, &mut verts);
    verts.into_iter().map(|v| (v.pos, v.normal)).collect()
}

fn box_faces(h: Vec3) -> (Vec<Vec3>, Vec<(Vec<usize>, Vec3)>) {
    let verts: Vec<Vec3> = (0..8).map(|k| Vec3::new(if k & 1 == 0 { -h.x } else { h.x }, if k & 2 == 0 { -h.y } else { h.y }, if k & 4 == 0 { -h.z } else { h.z })).collect();
    let mut faces = Vec::new();
    for axis in 0..3 {
        for sign in [-1.0f64, 1.0] {
            let bit = 1usize << axis;
            let idx: Vec<usize> = (0..8).filter(|&k| ((k & bit) != 0) == (sign > 0.0)).collect();
            let mut n = Vec3::ZERO;
            n[axis] = sign;
            faces.push((idx, n));
        }
    }
    (verts, faces)
}

fn sphere_triangles(radius: f64, object: u32, out: &mut Vec<Vertex>) {
    let (rings, segs) = (12usize, 20usize);
    let point = |i: usize, j: usize| {
        let th = std::f64::consts::PI * i as f64 / rings as f64;
        let ph = 2.0 * std::f64::consts::PI * j as f64 / segs as f64;
        Vec3::new(th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos())
    };
    let v = |n: Vec3| Vertex { pos: [(n.x * radius) as f32, (n.y * radius) as f32, (n.z * radius) as f32], normal: [n.x as f32, n.y as f32, n.z as f32], object };
    for i in 0..rings {
        for j in 0..segs {
            let (a, b, c, d) = (point(i, j), point(i + 1, j), point(i + 1, j + 1), point(i, j + 1));
            out.extend([v(a), v(b), v(c), v(a), v(c), v(d)]);
        }
    }
}

fn rows(r: Mat3, t: Vec3) -> [[f32; 4]; 3] {
    std::array::from_fn(|i| [r.m[i][0] as f32, r.m[i][1] as f32, r.m[i][2] as f32, t[i] as f32])
}

/// Renders the mirror of a solver (and its impactors) offscreen at a fixed size.
pub struct Renderer {
    pub width: u32,
    pub height: u32,
    pipeline: wgpu::RenderPipeline,
    bind: wgpu::BindGroup,
    camera: wgpu::Buffer,
    objects: wgpu::Buffer,
    vertices: wgpu::Buffer,
    vertex_count: u32,
    msaa: wgpu::TextureView,
    resolve: wgpu::Texture,
    depth: wgpu::TextureView,
    readback: wgpu::Buffer,
    row: u32,
    /// Object index of each chunk, by structure.
    chunk_object: Vec<Vec<u32>>,
    impactor_base: u32,
    object_count: usize,
    /// Distance (m) over which the ground grid fades.
    pub fade: f64,
    pub background: [f64; 3],
    /// Draw the ground plane (off for a scene without ground).
    pub ground: bool,
}

const SAMPLES: u32 = 4;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

impl Renderer {
    /// Meshes for every chunk of `solver` (their hulls, or boxes) and every impactor.
    pub fn new(gpu: &Gpu, solver: &ReferenceSolver, impactors: &[Impactor], width: u32, height: u32) -> Renderer {
        let device = &gpu.device;
        let mut verts: Vec<Vertex> = Vec::new();
        let mut chunk_object = Vec::new();
        let mut object = 1u32; // 0 is the ground
        // Ground: a large quad at z = 0 (the scene packs' grade).
        let g = 2000.0f32;
        let up = [0.0, 0.0, 1.0];
        for p in [[-g, -g], [g, -g], [g, g], [-g, -g], [g, g], [-g, g]] {
            verts.push(Vertex { pos: [p[0], p[1], 0.0], normal: up, object: 0 });
        }
        for st in &solver.structures {
            let mut ids = Vec::with_capacity(st.chunks.len());
            for ch in &st.chunks {
                chunk_mesh(ch, object, &mut verts);
                ids.push(object);
                object += 1;
            }
            chunk_object.push(ids);
        }
        let impactor_base = object;
        for imp in impactors {
            match imp.shape {
                ImpactorShape::Sphere { radius } => sphere_triangles(radius, object, &mut verts),
                ImpactorShape::Box { half_extents } => {
                    let (v, f) = box_faces(Vec3::from_array(half_extents));
                    hull_triangles(&v, &f, object, &mut verts);
                }
            }
            object += 1;
        }
        let object_count = object as usize;
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render vertices"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let objects = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render objects"),
            size: (object_count * std::mem::size_of::<Object>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("render"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: objects.as_entire_binding() }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("render"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Uint32];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout { array_stride: std::mem::size_of::<Vertex>() as u64, step_mode: wgpu::VertexStepMode::Vertex, attributes: &attributes })],
            },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: SAMPLES, ..Default::default() },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[Some(FORMAT.into())] }),
            multiview_mask: None,
            cache: None,
        });
        let texture = |label: &str, format: wgpu::TextureFormat, samples: u32, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let msaa = texture("render msaa", FORMAT, SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default());
        let resolve = texture("render resolve", FORMAT, 1, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let depth = texture("render depth", wgpu::TextureFormat::Depth32Float, SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default());
        let row = (width * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render readback"),
            size: (row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Renderer {
            width,
            height,
            pipeline,
            bind,
            camera,
            objects,
            vertices,
            vertex_count: verts.len() as u32,
            msaa,
            resolve,
            depth,
            readback,
            row,
            chunk_object,
            impactor_base,
            object_count,
            fade: 120.0,
            background: [0.70, 0.76, 0.84],
            ground: true,
        }
    }

    /// Triangles drawn per frame.
    pub fn triangles(&self) -> usize {
        self.vertex_count as usize / 3
    }

    /// Render the solver's current state from `cam`; returns tightly packed RGBA (sRGB).
    pub fn render(&self, gpu: &Gpu, solver: &ReferenceSolver, impactors: &[Impactor], cam: &CameraPose) -> Vec<u8> {
        let mut objs = vec![Object { rows: rows(Mat3::IDENTITY, Vec3::ZERO), color: [0.56, 0.58, 0.55, 2.0] }; self.object_count];
        if !self.ground {
            objs[0] = Object { rows: [[0.0; 4]; 3], color: [0.0; 4] };
        }
        for (s, st) in solver.structures.iter().enumerate() {
            for (c, ch) in st.chunks.iter().enumerate() {
                let state = &solver.chunks[s][c];
                let k = self.chunk_object[s][c] as usize;
                if !state.active || state.removed {
                    objs[k] = Object { rows: [[0.0; 4]; 3], color: [0.0; 4] };
                    continue;
                }
                let cl = &solver.clusters[state.cluster];
                let hidden = Quat::from_axis_angle(state.th, state.th.norm()).to_mat3();
                let r = cl.rotation() * hidden * ch.rotation;
                let base = material_color(&ch.material);
                let color = if cl.anchored {
                    base
                } else {
                    let f = fragment_color(cl.id);
                    std::array::from_fn(|i| 0.45 * base[i] + 0.55 * f[i])
                };
                objs[k] = Object { rows: rows(r, solver.chunk_position(s, c)), color: [color[0], color[1], color[2], 1.0] };
            }
        }
        for (i, imp) in impactors.iter().enumerate() {
            let k = self.impactor_base as usize + i;
            if k < self.object_count {
                objs[k] = Object { rows: rows(imp.pose.rotation.to_mat3(), imp.pose.position), color: [0.18, 0.19, 0.22, 1.0] };
            }
        }
        gpu.queue.write_buffer(&self.objects, 0, bytemuck::cast_slice(&objs));
        let eye = cam.eye();
        let light = Vec3::new(-0.45, -0.55, 0.70).normalized();
        let uniform = CameraUniform {
            view_proj: cam.view_proj(self.width as f64 / self.height as f64),
            eye: [eye.x as f32, eye.y as f32, eye.z as f32, 1.0],
            light: [light.x as f32, light.y as f32, light.z as f32, self.fade as f32],
        };
        gpu.queue.write_buffer(&self.camera, 0, bytemuck::bytes_of(&uniform));

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let view = self.resolve.create_view(&Default::default());
            let bg = self.background;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa,
                    depth_slice: None,
                    resolve_target: Some(&view),
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: bg[0], g: bg[1], b: bg[2], a: 1.0 }), store: wgpu::StoreOp::Discard },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.set_vertex_buffer(0, self.vertices.slice(..));
            pass.draw(0..self.vertex_count, 0..1);
        }
        encoder.copy_texture_to_buffer(
            self.resolve.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &self.readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.row), rows_per_image: Some(self.height) } },
            wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
        );
        gpu.queue.submit([encoder.finish()]);
        self.readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("map the frame"));
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let mut out = vec![0u8; (self.width * self.height * 4) as usize];
        {
            let data = self.readback.get_mapped_range(..).expect("mapped frame");
            let w = (self.width * 4) as usize;
            for y in 0..self.height as usize {
                out[y * w..(y + 1) * w].copy_from_slice(&data[y * self.row as usize..][..w]);
            }
        }
        self.readback.unmap();
        out
    }
}

/// Text and bars drawn into an RGBA frame (a 5x7 bitmap font, upper case).
pub struct Overlay<'a> {
    pub rgba: &'a mut [u8],
    pub width: u32,
    pub height: u32,
}

fn glyph(c: char) -> [u8; 7] {
    match c.to_ascii_uppercase() {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x0C, 0x04, 0x08],
        ':' => [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00],
        '/' => [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x00],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '+' => [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
        '=' => [0x00, 0x00, 0x1F, 0x00, 0x1F, 0x00, 0x00],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        '%' => [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03],
        '>' => [0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08],
        '<' => [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02],
        '|' => [0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        _ => [0; 7],
    }
}

impl Overlay<'_> {
    /// Blend `color` (RGB, sRGB bytes) with opacity `alpha` over the rectangle.
    pub fn rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: [u8; 3], alpha: f32) {
        let (x0, y0) = (x.max(0), y.max(0));
        let (x1, y1) = ((x + w).min(self.width as i32), (y + h).min(self.height as i32));
        for yy in y0..y1 {
            for xx in x0..x1 {
                let p = &mut self.rgba[((yy as u32 * self.width + xx as u32) * 4) as usize..][..3];
                for i in 0..3 {
                    p[i] = (p[i] as f32 * (1.0 - alpha) + color[i] as f32 * alpha).round() as u8;
                }
            }
        }
    }

    /// Width in pixels of `text` at `scale`.
    pub fn text_width(text: &str, scale: i32) -> i32 {
        text.chars().count() as i32 * 6 * scale - scale
    }

    /// Draw `text` with its top-left corner at (x, y), each font pixel `scale` pixels.
    pub fn text(&mut self, x: i32, y: i32, scale: i32, color: [u8; 3], text: &str) {
        for (i, ch) in text.chars().enumerate() {
            let g = glyph(ch);
            for (row, bits) in g.iter().enumerate() {
                for col in 0..5 {
                    if bits & (0x10 >> col) != 0 {
                        self.rect(x + (i as i32 * 6 + col) * scale, y + row as i32 * scale, scale, scale, color, 1.0);
                    }
                }
            }
        }
    }
}

/// ffmpeg reading raw RGBA frames on stdin, writing H.264 (yuv420p).
pub struct FfmpegSink {
    child: Child,
}

impl FfmpegSink {
    pub fn new(path: &std::path::Path, width: u32, height: u32, fps: u32) -> std::io::Result<FfmpegSink> {
        let ffmpeg = std::env::var("FFMPEG").unwrap_or_else(|_| if std::path::Path::new("/opt/homebrew/bin/ffmpeg").exists() { "/opt/homebrew/bin/ffmpeg".into() } else { "ffmpeg".into() });
        let child = Command::new(ffmpeg)
            .args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", &format!("{width}x{height}"), "-r", &fps.to_string(), "-i", "-"])
            .args(["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            .arg(path)
            .stdin(Stdio::piped())
            .spawn()?;
        Ok(FfmpegSink { child })
    }

    pub fn write(&mut self, rgba: &[u8]) -> std::io::Result<()> {
        self.child.stdin.as_mut().expect("ffmpeg stdin").write_all(rgba)
    }

    /// Close the input and wait for the encoder.
    pub fn finish(mut self) -> std::io::Result<()> {
        drop(self.child.stdin.take());
        let status = self.child.wait()?;
        if status.success() {
            Ok(())
        } else {
            Err(std::io::Error::other(format!("ffmpeg exited with {status}")))
        }
    }
}
