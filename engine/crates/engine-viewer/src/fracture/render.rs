//! Metal draw of the fracture lab (the browser's `fracture/render.mjs` on
//! Metal): the shared Slang draw shader reads live poses from the physics body
//! buffer (no CPU transform readback); house copies draw instanced; camera,
//! floor and the indexed mesh come from `engine_fracture::view`.

use super::kernels::{DRAW_FS, DRAW_VS};
use engine_fracture::view::{floor_vertices, index_mesh};
use engine_fracture::Scene;
use metal::*;

pub use engine_fracture::view::Orbit;

/// Draw uniform: column-major view-projection, then (instance stride, colour mode, joints
/// per copy).
#[repr(C)]
struct View {
    vp: [f32; 16],
    mode: [u32; 4],
}

pub struct Renderer {
    solid: RenderPipelineState,
    ground: RenderPipelineState,
    wire: RenderPipelineState,
    depth_less: DepthStencilState,
    depth_decal: DepthStencilState,
    depth_always: DepthStencilState,
    mesh: Buffer,
    index: Buffer,
    index_count: u64,
    lines: Buffer,
    floor: Buffer,
    roads: Buffer,
    projectiles: Buffer,
    counts: [u64; 3],
    projectile_per_slot: u64,
    copies: u64,
    stride: u32,
    /// Joints per house copy; 0 when the scene has none (bond lines then hide once their
    /// pieces part, `fracture_draw.slang`).
    joints_per_copy: u32,
    depth: Option<Texture>,
    size: (u64, u64),
    /// 0: materials; 1: sleep state (awake orange, asleep blue, static grey); 2: bodies
    /// (one colour per body).
    pub color_mode: u32,
}

fn buffer<T>(device: &Device, data: &[T]) -> Buffer {
    let bytes = std::mem::size_of_val(data).max(4) as u64;
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
    pub fn new(device: &Device, format: MTLPixelFormat, scene: &Scene) -> anyhow::Result<Self> {
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
        let (unique, index) = index_mesh(&scene.vertices);
        Ok(Self {
            solid: pipeline("fs")?,
            ground: pipeline("ground")?,
            wire: pipeline("wire")?,
            depth_less: depth_state(MTLCompareFunction::Less, true),
            depth_decal: depth_state(MTLCompareFunction::LessEqual, true),
            depth_always: depth_state(MTLCompareFunction::Always, false),
            mesh: buffer(device, &unique),
            index: buffer(device, &index),
            index_count: index.len() as u64,
            lines: buffer(device, &scene.lines),
            floor: buffer(device, &floor_vertices(scene.extent)),
            roads: buffer(device, &scene.road_vertices),
            projectiles: buffer(device, &scene.projectile_vertices),
            counts: [
                6,
                scene.lines.len() as u64 / 10,
                scene.road_vertices.len() as u64 / 10,
            ],
            projectile_per_slot: scene.projectile_vertices.len() as u64
                / 10
                / scene.projectile_count.max(1) as u64,
            copies: scene.copies.max(1) as u64,
            stride: scene.instance_stride,
            joints_per_copy: scene.joints.len() as u32 / scene.copies.max(1),
            depth: None,
            size: (0, 0),
            color_mode: 0,
        })
    }

    /// Draw into the window's next drawable and present it.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        device: &Device,
        queue: &CommandQueue,
        drawable: &MetalDrawableRef,
        bodies: &Buffer,
        joints: &Buffer,
        view_proj: [f32; 16],
        shots: u32,
        show_bonds: bool,
    ) {
        let cb = queue.new_command_buffer();
        cb.set_label("fracture draw");
        self.encode(
            device,
            cb,
            drawable.texture(),
            bodies,
            joints,
            view_proj,
            shots,
            show_bonds,
        );
        cb.present_drawable(drawable);
        cb.commit();
    }

    /// Encode one frame into `target` (a drawable or an offscreen texture).
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        &mut self,
        device: &Device,
        cb: &CommandBufferRef,
        tex: &TextureRef,
        bodies: &Buffer,
        joints: &Buffer,
        view_proj: [f32; 16],
        shots: u32,
        show_bonds: bool,
    ) {
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

        let e = cb.new_render_command_encoder(pass);
        let view = View {
            vp: view_proj,
            mode: [self.stride, self.color_mode, self.joints_per_copy, 0],
        };
        e.set_vertex_buffer(0, Some(bodies), 0);
        e.set_vertex_buffer(3, Some(joints), 0);
        e.set_vertex_bytes(
            2,
            std::mem::size_of::<View>() as u64,
            &view as *const View as *const _,
        );
        e.set_cull_mode(MTLCullMode::None);
        let bind = |pipe: &RenderPipelineState, depth: &DepthStencilState, verts: &Buffer| {
            e.set_render_pipeline_state(pipe);
            e.set_depth_stencil_state(depth);
            e.set_vertex_buffer(1, Some(verts), 0);
        };
        bind(&self.ground, &self.depth_less, &self.floor);
        e.draw_primitives(MTLPrimitiveType::Triangle, 0, self.counts[0]);
        if self.counts[2] > 0 {
            // Road decals lie millimetres above the ground: a depth bias keeps
            // them in front of it at any distance.
            bind(&self.solid, &self.depth_decal, &self.roads);
            e.set_depth_bias(-16.0, -2.0, 0.0);
            e.draw_primitives(MTLPrimitiveType::Triangle, 0, self.counts[2]);
            e.set_depth_bias(0.0, 0.0, 0.0);
        }
        bind(&self.solid, &self.depth_less, &self.mesh);
        e.draw_indexed_primitives_instanced(
            MTLPrimitiveType::Triangle,
            self.index_count,
            MTLIndexType::UInt32,
            &self.index,
            0,
            self.copies,
        );
        if shots > 0 {
            bind(&self.solid, &self.depth_less, &self.projectiles);
            e.draw_primitives(
                MTLPrimitiveType::Triangle,
                0,
                shots as u64 * self.projectile_per_slot,
            );
        }
        if show_bonds && self.counts[1] > 0 {
            bind(&self.wire, &self.depth_always, &self.lines);
            e.draw_primitives_instanced(MTLPrimitiveType::Line, 0, self.counts[1], self.copies);
        }
        e.end_encoding();
    }
}
