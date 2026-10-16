//! GPU rasterizer (wgpu: Metal, Vulkan, DX12) with the semantics of the software
//! [`Raster`](crate::raster::Raster).
//!
//! A [`GpuPanel`] is a [`Target`]: the shared front end clips, projects and splits the
//! scene into screen-space triangles exactly as for the software raster, and the panel
//! records them as an ordered display list of segments (opaque runs, overlay runs, the
//! outline). [`GpuFrame::finish`] replays every panel of a frame in one submission and
//! reads the resolved panels back into the canvas:
//!
//! * opaque: reversed-Z depth test and write (`NEAR / z`), colour, object id and the
//!   linear view depth into separate targets;
//! * overlay ([`Depth::Test`] / [`Depth::Always`]): drawn where the fragment's view depth
//!   is within its bias of the stored linear depth (`Always` uses an infinite bias);
//! * outline: a full-screen pass darkening samples next to an id change;
//! * resolve: the `ss x ss` box average into an RGBA8 target that is copied out.
//!
//! Differences to the software raster are confined to edge samples (the GPU's top-left
//! fill rule) and coplanar overlapping opaque faces (the nearer by float noise wins
//! instead of the first drawn).

use std::sync::{mpsc, Arc};

use bytemuck::{Pod, Zeroable};

use crate::camera::{Projection, NEAR};
use crate::canvas::Canvas;
use crate::colormap::Rgb;
use crate::raster::{Depth, OutlineStyle, Target};

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const LINEAR_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const OUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Linear depth of samples no opaque geometry covers (the software raster's infinity).
const FAR: f32 = 1e30;

const SHADER: &str = r#"
const NEAR: f32 = __NEAR__;
const FAR: f32 = __FAR__;

struct Panel {
    size: vec2f,
    ss: u32,
    pad: u32,
    outline: vec4f,
    sky_top: vec4f,
    sky_bottom: vec4f,
};

@group(0) @binding(0) var<uniform> panel: Panel;
@group(0) @binding(1) var tex_f: texture_2d<f32>;
@group(0) @binding(2) var tex_u: texture_2d<u32>;

struct TriIn {
    @location(0) pos: vec3f,
    @location(1) color: vec3f,
    @location(2) id: u32,
    @location(3) bias: f32,
};

struct TriOut {
    @builtin(position) clip: vec4f,
    @location(0) color: vec3f,
    @location(1) @interpolate(flat) id: u32,
    @location(2) z: f32,
    @location(3) @interpolate(flat) bias: f32,
};

struct Opaque {
    @location(0) color: vec4f,
    @location(1) id: u32,
    @location(2) z: f32,
};

// Sample coordinates (x right, y down, view depth z) to clip space: w = z makes every
// varying perspective-correct and the depth NEAR / z linear on screen (reversed Z).
@vertex
fn vs_tri(v: TriIn) -> TriOut {
    let ndc = vec2f(v.pos.x / panel.size.x * 2.0 - 1.0, 1.0 - v.pos.y / panel.size.y * 2.0);
    var o: TriOut;
    o.clip = vec4f(ndc * v.pos.z, NEAR, v.pos.z);
    o.color = v.color;
    o.id = v.id;
    o.z = v.pos.z;
    o.bias = v.bias;
    return o;
}

@fragment
fn fs_opaque(i: TriOut) -> Opaque {
    return Opaque(vec4f(i.color, 1.0), i.id, i.z);
}

@fragment
fn fs_overlay(i: TriOut) -> @location(0) vec4f {
    let stored = textureLoad(tex_f, vec2i(i.clip.xy), 0).x;
    if i.z > stored + i.bias {
        discard;
    }
    return vec4f(i.color, 1.0);
}

@vertex
fn vs_full(@builtin(vertex_index) k: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((k << 1u) & 2u), f32(k & 2u));
    return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_background(@builtin(position) p: vec4f) -> Opaque {
    let t = (p.y - 0.5) / (max(panel.size.y, 2.0) - 1.0);
    return Opaque(vec4f(mix(panel.sky_top.rgb, panel.sky_bottom.rgb, t), 1.0), 0u, FAR);
}

fn strength(a: u32, b: u32) -> f32 {
    if a == b {
        return 0.0;
    }
    if a == 0u || b == 0u {
        return panel.outline.x;
    }
    if (a >> 3u) != (b >> 3u) {
        return panel.outline.y;
    }
    return panel.outline.z;
}

@fragment
fn fs_outline(@builtin(position) p: vec4f) -> @location(0) vec4f {
    let q = vec2i(p.xy);
    let hi = vec2i(panel.size) - vec2i(1, 1);
    let id = textureLoad(tex_u, q, 0).x;
    var d = strength(id, textureLoad(tex_u, min(q + vec2i(1, 0), hi), 0).x);
    d = max(d, strength(id, textureLoad(tex_u, min(q + vec2i(0, 1), hi), 0).x));
    d = max(d, strength(id, textureLoad(tex_u, max(q - vec2i(1, 0), vec2i(0, 0)), 0).x));
    d = max(d, strength(id, textureLoad(tex_u, max(q - vec2i(0, 1), vec2i(0, 0)), 0).x));
    let c = textureLoad(tex_f, q, 0).rgb;
    if d > 0.0 {
        return vec4f(clamp(c * (1.0 - d), vec3f(0.0), vec3f(1.0)), 1.0);
    }
    return vec4f(c, 1.0);
}

@fragment
fn fs_resolve(@builtin(position) p: vec4f) -> @location(0) vec4f {
    let ss = i32(panel.ss);
    let o = vec2i(p.xy) * ss;
    var acc = vec3f(0.0);
    for (var y = 0; y < ss; y++) {
        for (var x = 0; x < ss; x++) {
            acc += textureLoad(tex_f, o + vec2i(x, y), 0).rgb;
        }
    }
    return vec4f(acc / f32(ss * ss), 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
    id: u32,
    bias: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PanelUniform {
    size: [f32; 2],
    ss: u32,
    pad: u32,
    outline: [f32; 4],
    sky_top: [f32; 4],
    sky_bottom: [f32; 4],
}

/// Device, queue and pipelines, shared by every panel and render thread.
pub struct GpuContext {
    #[cfg_attr(not(feature = "viewer"), allow(dead_code))]
    pub(crate) adapter: wgpu::Adapter,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    /// Adapter name and backend, for the log.
    pub description: String,
    /// The adapter is a software implementation (llvmpipe, WARP).
    pub software: bool,
    uniform_layout: wgpu::BindGroupLayout,
    texture_layout: wgpu::BindGroupLayout,
    background: wgpu::RenderPipeline,
    opaque: wgpu::RenderPipeline,
    overlay: wgpu::RenderPipeline,
    outline: wgpu::RenderPipeline,
    resolve: wgpu::RenderPipeline,
}

impl GpuContext {
    /// Opens the highest-performance adapter (`WGPU_BACKEND` / `WGPU_ADAPTER_NAME`
    /// select one explicitly).
    pub fn new() -> Result<Arc<GpuContext>, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        GpuContext::open(&instance, None)
    }

    /// A context that can present to a window: the surface of `target` and the context
    /// on an adapter compatible with it.
    #[cfg(feature = "viewer")]
    pub fn for_window(target: impl Into<wgpu::SurfaceTarget<'static>>) -> Result<(Arc<GpuContext>, wgpu::Surface<'static>), String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(target).map_err(|e| format!("window surface: {e}"))?;
        let ctx = GpuContext::open(&instance, Some(&surface))?;
        Ok((ctx, surface))
    }

    fn open(instance: &wgpu::Instance, surface: Option<&wgpu::Surface>) -> Result<Arc<GpuContext>, String> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU adapter: {e}"))?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("stress-viz"),
            ..Default::default()
        }))
        .map_err(|e| format!("GPU device on {}: {e}", info.name))?;
        let description = format!("{} ({:?})", info.name, info.backend);
        let software = info.device_type == wgpu::DeviceType::Cpu;

        let source = SHADER.replace("__NEAR__", &format!("{NEAR:e}")).replace("__FAR__", &format!("{FAR:e}"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stress-viz raster"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let texture_entry = |binding, sample_type| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("panel uniform"), entries: &[uniform_entry] });
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("panel textures"),
            entries: &[
                uniform_entry,
                texture_entry(1, wgpu::TextureSampleType::Float { filterable: false }),
                texture_entry(2, wgpu::TextureSampleType::Uint),
            ],
        });
        let pipeline_layout = |layout: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(layout)], immediate_size: 0 })
        };
        let (uniform_pl, texture_pl) = (pipeline_layout(&uniform_layout), pipeline_layout(&texture_layout));

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Uint32, 3 => Float32],
        };
        let color_target = |format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL });
        let opaque_targets = [color_target(COLOR_FORMAT), color_target(ID_FORMAT), color_target(LINEAR_DEPTH_FORMAT)];
        let color_only = [color_target(COLOR_FORMAT)];
        let out_only = [color_target(OUT_FORMAT)];
        let depth = |write: bool, compare| {
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(write),
                depth_compare: Some(compare),
                stencil: Default::default(),
                bias: Default::default(),
            })
        };
        let pipeline = |label: &str,
                        layout: &wgpu::PipelineLayout,
                        vs: &str,
                        fs: &str,
                        buffers: &[Option<wgpu::VertexBufferLayout>],
                        targets: &[Option<wgpu::ColorTargetState>],
                        depth_stencil: Option<wgpu::DepthStencilState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState { module: &module, entry_point: Some(vs), compilation_options: Default::default(), buffers },
                primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
                depth_stencil,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some(fs), compilation_options: Default::default(), targets }),
                multiview_mask: None,
                cache: None,
            })
        };
        let tri = &[Some(vertex_layout)];
        let background =
            pipeline("background", &uniform_pl, "vs_full", "fs_background", &[], &opaque_targets, depth(false, wgpu::CompareFunction::Always));
        let opaque = pipeline("opaque", &uniform_pl, "vs_tri", "fs_opaque", tri, &opaque_targets, depth(true, wgpu::CompareFunction::Greater));
        let overlay = pipeline("overlay", &texture_pl, "vs_tri", "fs_overlay", tri, &color_only, None);
        let outline = pipeline("outline", &texture_pl, "vs_full", "fs_outline", &[], &color_only, None);
        let resolve = pipeline("resolve", &texture_pl, "vs_full", "fs_resolve", &[], &out_only, None);
        Ok(Arc::new(GpuContext {
            adapter,
            device,
            queue,
            description,
            software,
            uniform_layout,
            texture_layout,
            background,
            opaque,
            overlay,
            outline,
            resolve,
        }))
    }
}

fn attach(view: &wgpu::TextureView, load: wgpu::LoadOp<wgpu::Color>) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment { view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load, store: wgpu::StoreOp::Store } })
}

/// A pass drawing into `view` only, keeping its contents.
fn color_pass<'e>(enc: &'e mut wgpu::CommandEncoder, label: &str, view: &wgpu::TextureView) -> wgpu::RenderPass<'e> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[attach(view, wgpu::LoadOp::Load)],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Segment {
    Opaque(u32, u32),
    Overlay(u32, u32),
    Outline,
}

/// One panel: a display list recorded through [`Target`] and the textures it is
/// replayed into.
pub struct GpuPanel {
    ctx: Arc<GpuContext>,
    proj: Projection,
    /// Sample resolution.
    width: usize,
    height: usize,
    ss: usize,
    verts: Vec<Vertex>,
    segments: Vec<Segment>,
    uniform: PanelUniform,
    uniform_buf: wgpu::Buffer,
    vertex_buf: wgpu::Buffer,
    color: [wgpu::TextureView; 2],
    id: wgpu::TextureView,
    linear_depth: wgpu::TextureView,
    depth: wgpu::TextureView,
    out: wgpu::Texture,
    uniform_group: wgpu::BindGroup,
    overlay_group: wgpu::BindGroup,
    /// Reads colour texture k (outline from k, resolve of k).
    color_groups: [wgpu::BindGroup; 2],
}

fn texture(ctx: &GpuContext, label: &str, w: usize, h: usize, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> wgpu::Texture {
    ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn vertex_buffer(ctx: &GpuContext, verts: usize) -> wgpu::Buffer {
    ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("panel vertices"),
        size: (verts.max(1024) * std::mem::size_of::<Vertex>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl GpuPanel {
    /// A panel of `width x height` samples (`ss` samples per output pixel and axis).
    pub fn new(ctx: &Arc<GpuContext>, width: usize, height: usize, ss: usize, proj: Projection) -> GpuPanel {
        use wgpu::TextureUsages as U;
        let target = U::RENDER_ATTACHMENT | U::TEXTURE_BINDING;
        let view = |t: wgpu::Texture| t.create_view(&Default::default());
        let color = [
            view(texture(ctx, "colour a", width, height, COLOR_FORMAT, target)),
            view(texture(ctx, "colour b", width, height, COLOR_FORMAT, target)),
        ];
        let id = view(texture(ctx, "object id", width, height, ID_FORMAT, target));
        let linear_depth = view(texture(ctx, "linear depth", width, height, LINEAR_DEPTH_FORMAT, target));
        let depth = view(texture(ctx, "depth", width, height, DEPTH_FORMAT, U::RENDER_ATTACHMENT));
        let out = texture(ctx, "resolved", width / ss, height / ss, OUT_FORMAT, U::RENDER_ATTACHMENT | U::COPY_SRC | U::TEXTURE_BINDING);
        let uniform_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("panel uniform"),
            size: std::mem::size_of::<PanelUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ctx.uniform_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() }],
        });
        let textures = |f: &wgpu::TextureView| {
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &ctx.texture_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(f) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&id) },
                ],
            })
        };
        let overlay_group = textures(&linear_depth);
        let color_groups = [textures(&color[0]), textures(&color[1])];
        GpuPanel {
            ctx: ctx.clone(),
            proj,
            width,
            height,
            ss,
            verts: Vec::new(),
            segments: Vec::new(),
            uniform: PanelUniform {
                size: [width as f32, height as f32],
                ss: ss as u32,
                pad: 0,
                outline: [0.0; 4],
                sky_top: [1.0; 4],
                sky_bottom: [1.0; 4],
            },
            vertex_buf: vertex_buffer(ctx, 0),
            uniform_buf,
            color,
            id,
            linear_depth,
            depth,
            out,
            uniform_group,
            overlay_group,
            color_groups,
        }
    }

    /// The resolved image (RGBA8, output size) the last [`GpuPanel::encode`] wrote.
    #[cfg(feature = "viewer")]
    pub fn output(&self) -> &wgpu::Texture {
        &self.out
    }

    /// Sets the projection of view space to this panel's samples.
    #[cfg(feature = "viewer")]
    pub fn set_proj(&mut self, proj: Projection) {
        self.proj = proj;
    }

    /// Output (resolved) size in pixels.
    pub fn out_size(&self) -> (usize, usize) {
        (self.width / self.ss, self.height / self.ss)
    }

    /// Uploads the display list and encodes its replay and the resolve.
    pub fn encode(&mut self, enc: &mut wgpu::CommandEncoder) {
        let ctx = &*self.ctx;
        let bytes: &[u8] = bytemuck::cast_slice(&self.verts);
        if bytes.len() as u64 > self.vertex_buf.size() {
            self.vertex_buf = vertex_buffer(ctx, self.verts.len().next_power_of_two());
        }
        if !bytes.is_empty() {
            ctx.queue.write_buffer(&self.vertex_buf, 0, bytes);
        }
        ctx.queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&self.uniform));

        // Current colour texture (the outline ping-pongs between the two).
        let mut cur = 0;
        {
            let mut pass = self.opaque_pass(enc, cur, true);
            pass.set_pipeline(&ctx.background);
            pass.set_bind_group(0, &self.uniform_group, &[]);
            pass.draw(0..3, 0..1);
        }
        for seg in &self.segments {
            match *seg {
                Segment::Opaque(a, b) => {
                    let mut pass = self.opaque_pass(enc, cur, false);
                    pass.set_pipeline(&ctx.opaque);
                    pass.set_bind_group(0, &self.uniform_group, &[]);
                    pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
                    pass.draw(a..b, 0..1);
                }
                Segment::Overlay(a, b) => {
                    let mut pass = color_pass(enc, "overlay", &self.color[cur]);
                    pass.set_pipeline(&ctx.overlay);
                    pass.set_bind_group(0, &self.overlay_group, &[]);
                    pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
                    pass.draw(a..b, 0..1);
                }
                Segment::Outline => {
                    let mut pass = color_pass(enc, "outline", &self.color[1 - cur]);
                    pass.set_pipeline(&ctx.outline);
                    pass.set_bind_group(0, &self.color_groups[cur], &[]);
                    pass.draw(0..3, 0..1);
                    drop(pass);
                    cur = 1 - cur;
                }
            }
        }
        let out_view = self.out.create_view(&Default::default());
        let mut pass = color_pass(enc, "resolve", &out_view);
        pass.set_pipeline(&ctx.resolve);
        pass.set_bind_group(0, &self.color_groups[cur], &[]);
        pass.draw(0..3, 0..1);
    }

    /// A pass drawing into colour texture `cur`, the id, linear depth and depth targets
    /// (cleared, with `clear`).
    fn opaque_pass<'e>(&self, enc: &'e mut wgpu::CommandEncoder, cur: usize, clear: bool) -> wgpu::RenderPass<'e> {
        let (load, depth_load) =
            if clear { (wgpu::LoadOp::Clear(wgpu::Color::BLACK), wgpu::LoadOp::Clear(0.0)) } else { (wgpu::LoadOp::Load, wgpu::LoadOp::Load) };
        enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("opaque"),
            color_attachments: &[attach(&self.color[cur], load), attach(&self.id, load), attach(&self.linear_depth, load)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth,
                depth_ops: Some(wgpu::Operations { load: depth_load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn push(&mut self, overlay: bool, p: [[f32; 3]; 3], color: Rgb, id: u32, bias: f32) {
        let start = self.verts.len() as u32;
        for v in p {
            self.verts.push(Vertex { pos: v, color, id, bias });
        }
        let end = start + 3;
        // Vertices are appended in draw order, so a run of the same kind just extends.
        match self.segments.last_mut() {
            Some(Segment::Opaque(_, e)) if !overlay => *e = end,
            Some(Segment::Overlay(_, e)) if overlay => *e = end,
            _ => self.segments.push(if overlay { Segment::Overlay(start, end) } else { Segment::Opaque(start, end) }),
        }
    }
}

impl Target for GpuPanel {
    fn proj(&self) -> &Projection {
        &self.proj
    }

    fn clear(&mut self, top: Rgb, bottom: Rgb) {
        self.verts.clear();
        self.segments.clear();
        self.uniform.sky_top = [top[0], top[1], top[2], 1.0];
        self.uniform.sky_bottom = [bottom[0], bottom[1], bottom[2], 1.0];
    }

    fn triangle(&mut self, p: [[f32; 3]; 3], color: Rgb, mode: Depth) {
        // Same degenerate-triangle rejection as the software raster.
        let area = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]);
        if !area.is_finite() || area.abs() <= 1e-8 {
            return;
        }
        match mode {
            Depth::Write { id } => self.push(false, p, color, id, 0.0),
            Depth::Test { bias } => self.push(true, p, color, 0, bias),
            Depth::Always => self.push(true, p, color, 0, FAR),
        }
    }

    fn outline(&mut self, style: OutlineStyle) {
        self.uniform.outline = [style.silhouette, style.object, style.crease, 0.0];
        self.segments.push(Segment::Outline);
    }
}

/// The panels of one output frame, replayed in one submission and read back together.
pub struct GpuFrame {
    ctx: Arc<GpuContext>,
    pub panels: Vec<GpuPanel>,
    readback: wgpu::Buffer,
    /// Byte offset and padded row pitch of each panel in `readback`.
    layout: Vec<(u64, u32)>,
}

impl GpuFrame {
    pub fn new(ctx: &Arc<GpuContext>, panels: usize, width: usize, height: usize, ss: usize, proj: Projection) -> GpuFrame {
        let panels: Vec<GpuPanel> = (0..panels).map(|_| GpuPanel::new(ctx, width, height, ss, proj)).collect();
        let mut layout = Vec::with_capacity(panels.len());
        let mut size = 0u64;
        for p in &panels {
            let (w, h) = p.out_size();
            let pitch = (4 * w as u32).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
            layout.push((size, pitch));
            size += pitch as u64 * h as u64;
        }
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame readback"),
            size: size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        GpuFrame { ctx: ctx.clone(), panels, readback, layout }
    }

    /// Renders every panel and writes it into `canvas` with its top-left at `origin(i)`.
    pub fn finish(&mut self, canvas: &mut Canvas, origin: impl Fn(usize) -> (usize, usize)) -> Result<(), String> {
        let ctx = self.ctx.clone();
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        for (p, &(offset, pitch)) in self.panels.iter_mut().zip(&self.layout) {
            p.encode(&mut enc);
            let (w, h) = p.out_size();
            enc.copy_texture_to_buffer(
                p.out.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.readback,
                    layout: wgpu::TexelCopyBufferLayout { offset, bytes_per_row: Some(pitch), rows_per_image: Some(h as u32) },
                },
                wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
            );
        }
        let submission = ctx.queue.submit([enc.finish()]);
        let slice = self.readback.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.device
            .poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: None })
            .map_err(|e| format!("GPU: {e}"))?;
        rx.recv().map_err(|e| format!("GPU readback: {e}"))?.map_err(|e| format!("GPU readback: {e}"))?;
        {
            let data = slice.get_mapped_range().map_err(|e| format!("GPU readback: {e:?}"))?;
            for (i, (p, &(offset, pitch))) in self.panels.iter().zip(&self.layout).enumerate() {
                let (w, h) = p.out_size();
                let (x0, y0) = origin(i);
                canvas.blit_rgba(x0, y0, w, h, &data[offset as usize..], pitch as usize);
            }
        }
        self.readback.unmap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::{object_id, Raster};

    /// A GPU context, or `None` (test skipped) on machines without an adapter.
    fn context() -> Option<Arc<GpuContext>> {
        match GpuContext::new() {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("skipping GPU test: {e}");
                None
            }
        }
    }

    /// Draws the same primitives into the software raster and a GPU panel and returns
    /// both resolved images.
    fn both(ctx: &Arc<GpuContext>, w: usize, h: usize, ss: usize, draw: impl Fn(&mut dyn Target)) -> (Canvas, Canvas) {
        let proj = Projection { focal: (w * ss) as f32, cx: (w * ss) as f32 * 0.5, cy: (h * ss) as f32 * 0.5 };
        let mut cpu = Raster::new(w * ss, h * ss, proj);
        draw(&mut cpu);
        let mut a = Canvas::new(w, h, [0.0; 3]);
        cpu.resolve(&mut a, 0, 0, ss);
        let mut frame = GpuFrame::new(ctx, 1, w * ss, h * ss, ss, proj);
        draw(&mut frame.panels[0]);
        let mut b = Canvas::new(w, h, [0.0; 3]);
        frame.finish(&mut b, |_| (0, 0)).unwrap();
        (a, b)
    }

    /// Fraction of pixels differing by more than `tol` (0..255) in any channel.
    fn mismatch(a: &Canvas, b: &Canvas, tol: u8) -> f64 {
        let bad = a.pixels.chunks(3).zip(b.pixels.chunks(3)).filter(|(p, q)| p.iter().zip(*q).any(|(x, y)| x.abs_diff(*y) > tol)).count();
        bad as f64 / (a.width * a.height) as f64
    }

    #[test]
    fn gpu_matches_the_software_raster() {
        let Some(ctx) = context() else { return };
        let red = [0.9, 0.2, 0.1];
        let blue = [0.1, 0.3, 0.9];
        let (a, b) = both(&ctx, 160, 90, 2, |t| {
            t.clear([1.0; 3], [0.8, 0.85, 0.9]);
            // Ground grid line before the solids (overlay under later opaque geometry).
            t.polygon(&[[-3.0, -0.4, 2.0], [3.0, -0.4, 2.0], [3.0, -0.4, 6.0], [-3.0, -0.4, 6.0]], [0.85; 3], Depth::Write { id: object_id(100, 0) });
            t.line([-3.0, -0.4, 3.0], [3.0, -0.4, 3.0], 2.0, [0.6; 3], Depth::Test { bias: 0.05 });
            // Two boxes' front faces, one tilted in depth and behind the other in part.
            t.polygon(&[[-0.8, -0.3, 3.0], [0.1, -0.3, 3.0], [0.1, 0.4, 3.0], [-0.8, 0.4, 3.0]], red, Depth::Write { id: object_id(1, 0) });
            t.polygon(&[[-0.2, -0.2, 2.5], [0.7, -0.2, 3.5], [0.7, 0.5, 3.5], [-0.2, 0.5, 2.5]], blue, Depth::Write { id: object_id(2, 1) });
            t.outline(OutlineStyle { silhouette: 0.55, object: 0.38, crease: 0.22 });
            // A crack on the red face (shown) and one behind it (hidden), an arrow on top.
            t.line([-0.6, 0.0, 3.0], [-0.3, 0.2, 3.0], 2.5, [1.0; 3], Depth::Test { bias: 0.05 });
            t.line([-0.6, -0.2, 4.0], [-0.3, -0.1, 4.0], 2.5, [0.0; 3], Depth::Test { bias: 0.05 });
            t.line([0.0, 0.0, 2.0], [0.5, 0.3, 2.0], 3.0, [0.1; 3], Depth::Always);
        });
        let m = mismatch(&a, &b, 3);
        assert!(m < 0.01, "{:.2}% of pixels differ", 100.0 * m);
    }
}
