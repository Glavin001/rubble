//! Metal draw of the fracture scene (port of `fracture/render.mjs`).
//!
//! Vertices are body-local; the Slang vertex shader reads live poses straight from the
//! physics body buffer, so there is no per-frame CPU transform readback.

use super::kernels::{DRAW_FS, DRAW_VS};
use engine_fracture::FractureScene;
use glam::{Mat4, Vec3};
use metal::*;

pub const FOV_Y: f32 = 0.7;

/// Orbit camera with the browser demo's defaults and limits.
#[derive(Clone, Copy, Debug)]
pub struct Orbit {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
    pub extent: f32,
}

impl Orbit {
    pub fn new(scene: &FractureScene) -> Self {
        Self {
            yaw: 0.25,
            pitch: 0.22,
            distance: scene.extent * 1.9,
            target: Vec3::from(scene.center),
            extent: scene.extent,
        }
    }

    pub fn eye(&self, aspect: f32) -> Vec3 {
        let fitted = self.distance / aspect.min(1.0);
        self.target
            + Vec3::new(
                self.yaw.sin() * self.pitch.cos(),
                self.pitch.sin(),
                self.yaw.cos() * self.pitch.cos(),
            ) * fitted
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let far = (self.extent * 50.0).max(500.0);
        Mat4::perspective_rh(FOV_Y, aspect, 0.02, far)
            * Mat4::look_at_rh(self.eye(aspect), self.target, Vec3::Y)
    }

    pub fn drag(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.006;
        self.pitch = (self.pitch + dy * 0.006).clamp(-0.1, 1.3);
    }

    pub fn zoom(&mut self, delta_y: f32) {
        self.distance =
            (self.distance * (delta_y * 0.001).exp()).clamp(self.extent * 0.35, self.extent * 8.0);
    }

    /// Ray through a window point (normalized 0..1 from top-left).
    pub fn aim(&self, u: f32, v: f32, aspect: f32, radius: f32) -> (Vec3, Vec3) {
        let eye = self.eye(aspect);
        let forward = (self.target - eye).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward);
        let t = (FOV_Y / 2.0).tan();
        let (x, y) = ((2.0 * u - 1.0) * aspect * t, (1.0 - 2.0 * v) * t);
        let dir = (forward + right * x + up * y).normalize();
        (eye + dir * radius * 2.0, dir)
    }
}

pub struct Renderer {
    solid: RenderPipelineState,
    ground: RenderPipelineState,
    wire: RenderPipelineState,
    depth_less: DepthStencilState,
    depth_always: DepthStencilState,
    mesh: Buffer,
    lines: Buffer,
    floor: Buffer,
    projectiles: Buffer,
    counts: [u64; 4],
    projectile_per_slot: u64,
    depth: Option<Texture>,
    size: (u64, u64),
}

fn buffer(device: &Device, data: &[f32]) -> Buffer {
    let bytes = (data.len() * 4).max(4) as u64;
    if data.is_empty() {
        return device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    }
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        bytes,
        MTLResourceOptions::StorageModeShared,
    )
}

impl Renderer {
    pub fn new(
        device: &Device,
        format: MTLPixelFormat,
        scene: &FractureScene,
    ) -> anyhow::Result<Self> {
        let opts = CompileOptions::new();
        let vs_lib = device
            .new_library_with_source(DRAW_VS, &opts)
            .map_err(|e| anyhow::anyhow!("draw vs: {e}"))?;
        let vs = vs_lib
            .get_function("vs", None)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let pipeline = |entry: &str| -> anyhow::Result<RenderPipelineState> {
            let src = DRAW_FS.iter().find(|(n, _)| *n == entry).unwrap().1;
            let lib = device
                .new_library_with_source(src, &opts)
                .map_err(|e| anyhow::anyhow!("draw {entry}: {e}"))?;
            let fs = lib
                .get_function(entry, None)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let desc = RenderPipelineDescriptor::new();
            desc.set_vertex_function(Some(&vs));
            desc.set_fragment_function(Some(&fs));
            desc.color_attachments()
                .object_at(0)
                .unwrap()
                .set_pixel_format(format);
            desc.set_depth_attachment_pixel_format(MTLPixelFormat::Depth32Float);
            device
                .new_render_pipeline_state(&desc)
                .map_err(|e| anyhow::anyhow!("pipeline {entry}: {e}"))
        };
        let depth_state = |compare, write| {
            let d = DepthStencilDescriptor::new();
            d.set_depth_compare_function(compare);
            d.set_depth_write_enabled(write);
            device.new_depth_stencil_state(&d)
        };
        let size = (scene.extent * 8.0).max(20.0);
        let mut g = Vec::new();
        for p in [
            [-size, 0.0, -size],
            [-size, 0.0, size],
            [size, 0.0, size],
            [-size, 0.0, -size],
            [size, 0.0, size],
            [size, 0.0, -size],
        ] {
            g.extend(p);
            g.extend([0.0, 1.0, 0.0, 0.1, 0.15, 0.18, 0.0]);
        }
        Ok(Self {
            solid: pipeline("fs")?,
            ground: pipeline("ground")?,
            wire: pipeline("wire")?,
            depth_less: depth_state(MTLCompareFunction::Less, true),
            depth_always: depth_state(MTLCompareFunction::Always, false),
            mesh: buffer(device, &scene.vertices),
            lines: buffer(device, &scene.lines),
            floor: buffer(device, &g),
            projectiles: buffer(device, &scene.projectile_vertices),
            counts: [
                6,
                scene.vertices.len() as u64 / 10,
                scene.lines.len() as u64 / 10,
                0,
            ],
            projectile_per_slot: scene.projectile_vertices.len() as u64
                / 10
                / scene.projectile_count as u64,
            depth: None,
            size: (0, 0),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        device: &Device,
        queue: &CommandQueue,
        drawable: &MetalDrawableRef,
        bodies: &Buffer,
        view_proj: Mat4,
        shots: u32,
        show_bonds: bool,
    ) {
        let tex = drawable.texture();
        let (w, h) = (tex.width(), tex.height());
        if self.size != (w, h) || self.depth.is_none() {
            let d = TextureDescriptor::new();
            d.set_pixel_format(MTLPixelFormat::Depth32Float);
            d.set_width(w);
            d.set_height(h);
            d.set_storage_mode(MTLStorageMode::Private);
            d.set_usage(MTLTextureUsage::RenderTarget);
            self.depth = Some(device.new_texture(&d));
            self.size = (w, h);
        }
        let pass = RenderPassDescriptor::new();
        let color = pass.color_attachments().object_at(0).unwrap();
        color.set_texture(Some(tex));
        color.set_load_action(MTLLoadAction::Clear);
        color.set_clear_color(MTLClearColor::new(0.035, 0.055, 0.075, 1.0));
        color.set_store_action(MTLStoreAction::Store);
        let depth = pass.depth_attachment().unwrap();
        depth.set_texture(self.depth.as_deref());
        depth.set_load_action(MTLLoadAction::Clear);
        depth.set_clear_depth(1.0);
        depth.set_store_action(MTLStoreAction::DontCare);

        let cb = queue.new_command_buffer();
        cb.set_label("fracture draw");
        let e = cb.new_render_command_encoder(pass);
        let camera = view_proj.to_cols_array();
        e.set_vertex_buffer(0, Some(bodies), 0);
        e.set_vertex_bytes(2, 64, camera.as_ptr() as *const _);
        e.set_cull_mode(MTLCullMode::None);
        let draw = |pipe: &RenderPipelineState,
                    depth: &DepthStencilState,
                    verts: &Buffer,
                    prim,
                    count: u64| {
            if count == 0 {
                return;
            }
            e.set_render_pipeline_state(pipe);
            e.set_depth_stencil_state(depth);
            e.set_vertex_buffer(1, Some(verts), 0);
            e.draw_primitives(prim, 0, count);
        };
        draw(
            &self.ground,
            &self.depth_less,
            &self.floor,
            MTLPrimitiveType::Triangle,
            self.counts[0],
        );
        draw(
            &self.solid,
            &self.depth_less,
            &self.mesh,
            MTLPrimitiveType::Triangle,
            self.counts[1],
        );
        draw(
            &self.solid,
            &self.depth_less,
            &self.projectiles,
            MTLPrimitiveType::Triangle,
            shots as u64 * self.projectile_per_slot,
        );
        if show_bonds {
            draw(
                &self.wire,
                &self.depth_always,
                &self.lines,
                MTLPrimitiveType::Line,
                self.counts[2],
            );
        }
        e.end_encoding();
        cb.present_drawable(drawable);
        cb.commit();
    }
}
