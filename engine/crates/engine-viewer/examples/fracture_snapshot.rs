//! Render a Scene-menu entry offscreen on Metal after a shot and write a BMP,
//! to check the native draw without a window.
//!
//! `cargo run -p engine-viewer --release --example fracture_snapshot -- suburb 180 out.bmp`
use engine_fracture::samples::{base_scene, initial_level, layout_scene, sample};
use engine_viewer::fracture::{FractureSim, Orbit, Renderer};
use metal::*;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let entry = sample(args.get(1).map_or("house", String::as_str))?;
    let ticks: u32 = args.get(2).map_or(Ok(120), |s| s.parse())?;
    let out = args.get(3).map_or("fracture.bmp", String::as_str);
    let asset = entry
        .file
        .map(|f| {
            engine_fracture::parse_fracture_glb(&std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../engine-webgpu/fracture/assets")
                    .join(f),
            )?)
        })
        .transpose()?;
    let level = asset
        .as_ref()
        .and_then(|a| initial_level(a, entry.level))
        .unwrap_or(0);
    let scene = layout_scene(
        base_scene(asset.as_ref(), level, entry.max_hull_vertices)?.without_joints(),
        entry.layout,
        asset.is_none(),
    );
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();
    let mut sim = FractureSim::new(&device, &queue, &scene)?;
    let (w, h) = (1280u64, 800u64);
    let orbit = Orbit::new(&scene);
    if let Some(shot) = scene
        .row_shot
        .or_else(|| Some(orbit.aim(0.5, 0.5, w as f64 / h as f64, scene.projectile_radius)))
    {
        sim.launch(shot.origin, shot.direction)?;
    }
    sim.step(ticks)?;
    let mut renderer = Renderer::new(&device, MTLPixelFormat::BGRA8Unorm, &scene)?;
    renderer.color_mode = 1;
    let d = TextureDescriptor::new();
    d.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    d.set_width(w);
    d.set_height(h);
    d.set_storage_mode(MTLStorageMode::Shared);
    d.set_usage(MTLTextureUsage::RenderTarget);
    let tex = device.new_texture(&d);
    let cb = queue.new_command_buffer();
    renderer.encode(
        &device,
        cb,
        &tex,
        sim.body_buffer(),
        sim.joint_buffer(),
        orbit.view_proj(w as f64 / h as f64),
        sim.shots(),
        false,
    );
    cb.commit();
    cb.wait_until_completed();
    let mut bgra = vec![0u8; (w * h * 4) as usize];
    tex.get_bytes(
        bgra.as_mut_ptr() as *mut _,
        w * 4,
        MTLRegion::new_2d(0, 0, w, h),
        0,
    );
    // 32-bit top-down BMP.
    let mut bmp = Vec::with_capacity(bgra.len() + 54);
    let u32le = |v: u32| v.to_le_bytes();
    bmp.extend(b"BM");
    bmp.extend(u32le(54 + bgra.len() as u32));
    bmp.extend(u32le(0));
    bmp.extend(u32le(54));
    bmp.extend(u32le(40));
    bmp.extend(u32le(w as u32));
    bmp.extend((-(h as i32)).to_le_bytes());
    bmp.extend([1, 0, 32, 0]);
    bmp.extend([0u8; 24]);
    bmp.extend(&bgra);
    std::fs::write(out, bmp)?;
    let c = sim.last;
    println!(
        "{}: {} ticks · {} awake · {} contacts → {out}",
        entry.key,
        sim.steps(),
        c.awake,
        c.contact
    );
    Ok(())
}
