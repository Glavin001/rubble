//! Offscreen video of the towers volley (`fracture_viewer towers --volley`'s
//! scene and shots), rendered tick by tick to an MP4 through ffmpeg: the same
//! camera and shots for any setting, whatever is on screen. Not a timing run
//! (it waits for every frame); the viewer's title and log carry real-time rates.
//!
//! ```bash
//! cargo run -p engine-viewer --release --bin towers_capture -- --out on.mp4
//! cargo run -p engine-viewer --release --bin towers_capture -- --out off.mp4 --no-ccd
//! ```

use engine_viewer::fracture::volley::{towers_scene, volley, Trails};
use engine_viewer::fracture::{FractureSim, Orbit, Renderer};
use metal::*;
use std::io::Write;
use std::process::{Command, Stdio};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let (mut out, mut ccd, mut grid, mut seconds, mut speed, mut strength) = (
        "towers.mp4".to_string(),
        true,
        (2u32, 2u32),
        18.0f64,
        16.0f64,
        0.2f64,
    );
    let (mut ball, mut size, mut yaw, mut pitch, mut zoom) =
        ((1.0f64, 30000.0f64), (1600u64, 1000u64), None, None, 1.0f64);
    while let Some(a) = args.next() {
        let mut next = || args.next().unwrap_or_default();
        match a.as_str() {
            "--out" => out = next(),
            "--no-ccd" => ccd = false,
            "--grid" => {
                let g = next();
                let (c, r) = g.split_once('x').expect("--grid CxR");
                grid = (c.parse()?, r.parse()?);
            }
            "--seconds" => seconds = next().parse()?,
            "--speed" => speed = next().parse()?,
            "--bond-strength" => strength = next().parse()?,
            "--ball" => {
                let b = next();
                let (r, m) = b.split_once(',').expect("--ball R,M");
                ball = (r.parse()?, m.parse()?);
            }
            "--size" => {
                let s = next();
                let (w, h) = s.split_once('x').expect("--size WxH");
                size = (w.parse()?, h.parse()?);
            }
            "--yaw" => yaw = Some(next().parse()?),
            "--pitch" => pitch = Some(next().parse()?),
            "--zoom" => zoom = next().parse()?,
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("no Metal device"))?;
    let queue = device.new_command_queue();
    let scene = towers_scene(grid.0, grid.1, 6.0, true, Some(ball), strength)?;
    let mut sim = FractureSim::new(&device, &queue, &scene)?;
    sim.plan.set_projectile_ccd(ccd);
    sim.plan.set_projectile_speed_scale(speed)?;
    let shots = volley(&scene, 8, 16, 45);
    let mut renderer = Renderer::new(&device, MTLPixelFormat::BGRA8Unorm, &scene)?;
    let mut orbit = Orbit::new(&scene);
    orbit.yaw = yaw.unwrap_or(orbit.yaw);
    orbit.pitch = pitch.unwrap_or(orbit.pitch);
    orbit.distance *= zoom;
    let (w, h) = size;
    let desc = TextureDescriptor::new();
    desc.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    desc.set_width(w);
    desc.set_height(h);
    desc.set_storage_mode(MTLStorageMode::Shared);
    desc.set_usage(MTLTextureUsage::RenderTarget);
    let target = device.new_texture(&desc);
    let hz = 1.0 / scene.params.dt as f64;
    let ticks = (seconds * hz) as u32;
    // 60 fps video of a 120 Hz simulation: a frame every second tick.
    let every = (hz / 60.0).round().max(1.0) as u32;
    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgra",
            "-s",
            &format!("{w}x{h}"),
            "-r",
            "60",
            "-i",
            "-",
        ])
        .args([
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", &out,
        ])
        .stdin(Stdio::piped())
        .spawn()?;
    let mut pipe = ffmpeg.stdin.take().unwrap();
    let mut trails = Trails::new(&scene);
    let mut next = 0;
    let mut pixels = vec![0u8; (w * h * 4) as usize];
    println!(
        "{}×{} towers, {} pieces, CCD {ccd}, {:.0} m/s, {ticks} ticks → {out}",
        grid.0,
        grid.1,
        scene.projectile_start - 1,
        sim.plan.shot_speed() * speed
    );
    for t in 0..ticks {
        while next < shots.len() && shots[next].step <= sim.steps() {
            let id = sim.launch(shots[next].origin, shots[next].dir)?;
            trails.launched((id - scene.projectile_start) as usize, ccd);
            next += 1;
        }
        sim.step(1)?;
        trails.update(&sim, &scene);
        if t % every == 0 {
            renderer.set_trails(&device, &trails.vertices());
            let cb = queue.new_command_buffer();
            renderer.encode(
                &device,
                cb,
                &target,
                sim.body_buffer(),
                sim.joint_buffer(),
                orbit.view_proj(w as f64 / h as f64),
                sim.shots(),
                false,
            );
            cb.commit();
            cb.wait_until_completed();
            target.get_bytes(
                pixels.as_mut_ptr() as *mut _,
                w * 4,
                MTLRegion::new_2d(0, 0, w, h),
                0,
            );
            pipe.write_all(&pixels)?;
        }
    }
    drop(pipe);
    ffmpeg.wait()?;
    println!("wrote {out}");
    Ok(())
}
