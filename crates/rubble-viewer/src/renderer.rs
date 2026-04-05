use crate::mesh::{self, Vertex};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
    pub light_dir: [f32; 3],
    pub _pad0: f32,
    pub camera_pos: [f32; 3],
    pub _pad1: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct InstanceData {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
}

impl InstanceData {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceData>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 16,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 32,
                shader_location: 4,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 48,
                shader_location: 5,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 64,
                shader_location: 6,
            },
        ],
    };
}

pub struct GpuMesh {
    pub vertex_buf: wgpu::Buffer,
    pub index_buf: wgpu::Buffer,
    pub index_count: u32,
}

impl GpuMesh {
    pub fn from_mesh(device: &wgpu::Device, m: &mesh::Mesh) -> Self {
        use wgpu::util::DeviceExt;
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&m.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&m.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertex_buf,
            index_buf,
            index_count: m.indices.len() as u32,
        }
    }
}

pub struct Renderer {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub depth_view: wgpu::TextureView,
    pub pipeline: wgpu::RenderPipeline,
    pub grid_pipeline: wgpu::RenderPipeline,
    pub uniform_buf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub sphere_mesh: GpuMesh,
    pub cube_mesh: GpuMesh,
    pub capsule_mesh: GpuMesh,
    pub circle_mesh: GpuMesh,
    pub quad_mesh: GpuMesh,
    pub egui_renderer: egui_wgpu::Renderer,
}

impl Renderer {
    pub async fn new(window: Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: Default::default(),
        });
        let surface = instance.create_surface(window).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapter found");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .expect("Failed to request device");

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let depth_view = Self::create_depth_view(&device, config.width, config.height);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("basic.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/basic.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::LAYOUT, InstanceData::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let grid_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("grid"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_grid"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_grid"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let sphere_mesh = GpuMesh::from_mesh(&device, &mesh::icosphere(2));
        let cube_mesh = GpuMesh::from_mesh(&device, &mesh::unit_cube());
        let capsule_mesh = GpuMesh::from_mesh(&device, &mesh::unit_capsule(8, 16));
        let circle_mesh = GpuMesh::from_mesh(&device, &mesh::circle_2d(32));
        let quad_mesh = GpuMesh::from_mesh(&device, &mesh::quad_2d());

        let egui_renderer =
            egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());

        Self {
            device,
            queue,
            surface,
            config,
            depth_view,
            pipeline,
            grid_pipeline,
            uniform_buf,
            bind_group,
            sphere_mesh,
            cube_mesh,
            capsule_mesh,
            circle_mesh,
            quad_mesh,
            egui_renderer,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = Self::create_depth_view(&self.device, width, height);
    }

    pub fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height as f32
    }

    fn create_depth_view(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("depth"),
                size: wgpu::Extent3d {
                    width: w.max(1),
                    height: h.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_3d(
        &mut self,
        view_proj: Mat4,
        camera_pos: Vec3,
        instances: &DrawList,
        draw_grid: bool,
        egui_primitives: &[egui::ClippedPrimitive],
        egui_textures_delta: &egui::TexturesDelta,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
    ) {
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            light_dir: Vec3::new(0.5, 1.0, 0.3).normalize().to_array(),
            _pad0: 0.0,
            camera_pos: camera_pos.to_array(),
            _pad1: 0.0,
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        for (id, delta) in &egui_textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut self.device.create_command_encoder(&Default::default()),
            egui_primitives,
            screen_descriptor,
        );

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(tex)
            | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
            _ => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.08,
                            g: 0.08,
                            b: 0.10,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);

            self.draw_batch(&mut pass, &self.sphere_mesh, &instances.spheres);
            self.draw_batch(&mut pass, &self.cube_mesh, &instances.cubes);
            self.draw_batch(&mut pass, &self.capsule_mesh, &instances.capsules);

            if draw_grid {
                // Draw the transparent grid after opaque meshes without writing depth,
                // so it cannot occlude bodies resting on the ground plane.
                pass.set_pipeline(&self.grid_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.draw(0..6, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        let mut egui_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("egui"),
                });
        {
            let pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            let mut pass = pass.forget_lifetime();
            self.egui_renderer
                .render(&mut pass, egui_primitives, screen_descriptor);
        }
        self.queue.submit(std::iter::once(egui_encoder.finish()));
        frame.present();

        for &id in &egui_textures_delta.free {
            self.egui_renderer.free_texture(&id);
        }
    }

    pub fn render_2d(
        &mut self,
        view_proj: Mat4,
        instances: &DrawList,
        egui_primitives: &[egui::ClippedPrimitive],
        egui_textures_delta: &egui::TexturesDelta,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
    ) {
        let uniforms = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            light_dir: [0.0, 0.0, 1.0],
            _pad0: 0.0,
            camera_pos: [0.0, 0.0, 1.0],
            _pad1: 0.0,
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        for (id, delta) in &egui_textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut self.device.create_command_encoder(&Default::default()),
            egui_primitives,
            screen_descriptor,
        );

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(tex)
            | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
            _ => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.08,
                            g: 0.08,
                            b: 0.10,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);

            self.draw_batch(&mut pass, &self.circle_mesh, &instances.spheres);
            self.draw_batch(&mut pass, &self.quad_mesh, &instances.cubes);
            self.draw_batch(&mut pass, &self.capsule_mesh, &instances.capsules);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        let mut egui_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("egui"),
                });
        {
            let pass = egui_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            let mut pass = pass.forget_lifetime();
            self.egui_renderer
                .render(&mut pass, egui_primitives, screen_descriptor);
        }
        self.queue.submit(std::iter::once(egui_encoder.finish()));
        frame.present();

        for &id in &egui_textures_delta.free {
            self.egui_renderer.free_texture(&id);
        }
    }

    fn draw_batch<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        mesh: &'a GpuMesh,
        instances: &[InstanceData],
    ) {
        if instances.is_empty() {
            return;
        }
        use wgpu::util::DeviceExt;
        let instance_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(instances),
                usage: wgpu::BufferUsages::VERTEX,
            });
        pass.set_vertex_buffer(0, mesh.vertex_buf.slice(..));
        pass.set_vertex_buffer(1, instance_buf.slice(..));
        pass.set_index_buffer(mesh.index_buf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.index_count, 0, 0..instances.len() as u32);
    }
}

/// Collected instance data ready for a single frame.
#[derive(Default)]
pub struct DrawList {
    pub spheres: Vec<InstanceData>,
    pub cubes: Vec<InstanceData>,
    pub capsules: Vec<InstanceData>,
}

impl DrawList {
    pub fn clear(&mut self) {
        self.spheres.clear();
        self.cubes.clear();
        self.capsules.clear();
    }
}

/// Build a model matrix from position, rotation quaternion, and non-uniform scale.
pub fn model_matrix(pos: Vec3, rot: glam::Quat, scale: Vec3) -> Mat4 {
    Mat4::from_scale_rotation_translation(scale, rot, pos)
}

/// A fixed palette of body colors.
const PALETTE: &[[f32; 4]] = &[
    [0.90, 0.35, 0.15, 1.0], // orange
    [0.20, 0.65, 0.90, 1.0], // blue
    [0.35, 0.85, 0.45, 1.0], // green
    [0.90, 0.80, 0.20, 1.0], // yellow
    [0.70, 0.30, 0.85, 1.0], // purple
    [0.95, 0.55, 0.65, 1.0], // pink
    [0.40, 0.80, 0.80, 1.0], // teal
    [0.95, 0.60, 0.30, 1.0], // amber
];

pub fn palette_color(index: usize) -> [f32; 4] {
    PALETTE[index % PALETTE.len()]
}

/// Static body color (dark gray).
pub fn static_color() -> [f32; 4] {
    [0.35, 0.35, 0.38, 1.0]
}
