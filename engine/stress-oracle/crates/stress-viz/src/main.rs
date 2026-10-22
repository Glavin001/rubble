//! `stress-viz`: renders reference-solver runs as debug-visualisation videos.
//!
//! ```text
//! stress-viz render  <scene> [options]                    one run, a grid of view modes
//! stress-viz compare <scene> --variant "label:item;item" --variant ... [options]
//!                                                          variants side by side, same camera and scales
//! stress-viz still   <scene> --time T [options]           one PNG at simulated time T
//! stress-viz view    <scene> [options]                    interactive window: live run,
//!                                                          replay, free camera, projectiles
//! stress-viz list                                          catalogue scene names
//! ```
//!
//! `<scene>` is a scene JSON file (`stress-ref gen-scenes <dir>`) or a catalogue name
//! (`builders::catalog()`, `showcases::catalog()`); `--name NAME` / `--scene PATH` work too.
//! A variant item is `dotted.path=json` (a scene override, see `Scene::with_override`),
//! `feature NAME=on|off` (a model switch, see `Scene::with_feature`) or `engine rubble`
//! (rubble's AVBD GPU engine as the rigid-body engine, `World::with_engine`; needs the
//! `rubble` cargo feature). `--engine rubble` does the same for `render`, `still` and `view`.
//!
//! The scene is stepped one frame at a time, every rendered frame is snapshotted, then
//! frames are rendered in parallel and piped to ffmpeg. The 3-D panels are rasterized
//! on the GPU through wgpu (Metal, Vulkan, DX12; cargo feature `gpu`, on by default)
//! when an adapter is available, else by the software rasterizer (`--renderer`).

mod camera;
mod canvas;
mod colormap;
mod font;
#[cfg(feature = "gpu")]
mod gpu;
mod layout;
mod plot;
mod png;
mod raster;
mod record;
mod scene3d;
mod video;
#[cfg(feature = "viewer")]
mod viewer;
mod views;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use stress_ref::math::Vec3;
use stress_ref::scene::Scene;
use stress_ref::{builders, showcases};

use camera::CameraSpec;
use canvas::Canvas;
use layout::{build_plot, columns_for, Composer, GpuHandle, PanelSpec};
use record::{RecordOptions, Recording};
use scene3d::{Clip, DrawOptions};
use views::{ScaleOverrides, Scales, ViewMode};

const USAGE: &str = "\
usage: stress-viz render  <scene> [options]
       stress-viz compare <scene> --variant \"label:item;item\" [--variant ...] [options]
       stress-viz still   <scene> --time T [options]
       stress-viz view    <scene> [options]   (window: drag orbit, click fires, H help)
       stress-viz list

scene: a scene JSON file or a catalogue name (also --name NAME / --scene PATH)
variant item: dotted.path=json (scene override) | feature NAME=on|off | engine rubble

options:
  -o, --out PATH          output .mp4 (render/compare) or .png (still)
  --views LIST            utilization,stress,von_mises,damage,fragments,velocity,deformation
                          (default utilization,stress,damage,fragments)
  --cols N                panel grid columns (default: one row up to 3 panels, else
                          near-square; compare: one column per variant)
  --set PATH=JSON         scene override, repeatable (e.g. sim.stiffness_scale=0.01)
  --feature NAME=on|off   model switch, repeatable
  --engine NAME           rigid-body engine: standalone (default, the oracle's own) |
                          rubble (rubble's AVBD GPU engine; cargo feature `rubble`) |
                          learned (rubble's engine, each routed island's 1/60 s frame from
                          the learned model, closed loop; cargo feature `learned`) |
                          ceiling (the same handoff with the oracle as the model) |
                          a_learned (harness A: the standalone oracle, each routed island's
                          frame from the learned model) | qs (CPU blast_parity, standalone)
  --checkpoint PATH       (engine learned) the model checkpoint (runs/<job>/best.pt)
  --model-device NAME     (engine learned) cpu (default) | mps
  --model-python DIR      (engine learned) the model's Python package with its .venv
                          (default: this checkout's learned-impact/python)
  --duration S            simulated time (default: the scene's sim.duration)
  --frame-dt S            simulation frame (default: the scene's, refined to --min-frames)
  --every K               render every K-th simulated frame (default: fit --max-frames)
  --min-frames N          refine frame_dt so at least N frames are rendered (default 150)
  --max-frames N          default --every keeps at most N rendered frames (default 450)
  --fps N                 video frame rate (default 30)
  --hold S                seconds the last frame is held (default 1.5)
  --size WxH              output size (default 1920x1080)
  --ss N                  supersampling per axis (default 2)
  --jobs N                render threads (default: half the cores)
  --sim-threads N         (view) simulation threads (default 4; the solver gains little
                          beyond that)
  --solver-mode NAME      (view) explicit (default) | soft | implicit | mass-scaling |
                          adaptive; M switches in the window
  --renderer auto|gpu|cpu 3-D rasterizer: GPU through wgpu (Metal, Vulkan, DX12) or the
                          software one (default auto: GPU when a hardware adapter exists)
  --camera NAME           front | side | top | iso | 3/4 (default 3/4)
  --yaw DEG --pitch DEG --distance M --target X,Y,Z --fov DEG
  --clip AXIS>V|AXIS<V    hide chunks beyond a plane (section), e.g. x>0
  --clip-y V              same as --clip y>V
  --no-cracks             no crack lines in the heat-map views
  --crack-color white|black  (default white; dark in the stress view)
  --arrows                fragment velocity arrows (fragments view)
  --plot LIST             energy (default) | broken | <probe name> | none
  --stress-scale PA       stress colour range (default: 99th percentile)
  --exaggerate F          deformation display factor (default: auto)
  --png DIR               also write every frame as PNG
  --title TEXT            header title (default: scene name)
  --time T                (still) simulated time of the image";

struct Args {
    cmd: String,
    scene: Option<String>,
    out: Option<PathBuf>,
    views: Vec<ViewMode>,
    cols: Option<usize>,
    overrides: Vec<String>,
    features: Vec<String>,
    variants: Vec<String>,
    /// Rigid-body engine of a single run (`render`, `still`).
    engine: record::Engine,
    /// `engine learned`: the model checkpoint and its device.
    checkpoint: Option<PathBuf>,
    model_device: String,
    model_python: Option<PathBuf>,
    duration: Option<f64>,
    frame_dt: Option<f64>,
    every: Option<usize>,
    min_frames: usize,
    max_frames: usize,
    fps: f64,
    hold: f64,
    size: (usize, usize),
    ss: usize,
    /// `--ss` was given (the viewer otherwise picks it from the display density).
    ss_given: bool,
    jobs: usize,
    sim_threads: usize,
    /// (view) solver mode at start, a name from `viewer::MODES`.
    solver_mode: String,
    renderer: RendererChoice,
    camera: CameraSpec,
    clip: Option<Clip>,
    cracks: bool,
    crack_color: Option<colormap::Rgb>,
    arrows: bool,
    plot: Vec<String>,
    scale_overrides: ScaleOverrides,
    png_dir: Option<PathBuf>,
    title: Option<String>,
    time: Option<f64>,
}

/// Which rasterizer draws the 3-D panels.
#[derive(Clone, Copy, Debug, PartialEq)]
enum RendererChoice {
    /// The GPU when a hardware adapter exists, else the software rasterizer.
    Auto,
    Gpu,
    Cpu,
}

/// Opens the GPU for `choice` (`None`: the software rasterizer) and logs the choice.
fn open_gpu(choice: RendererChoice) -> Result<Option<GpuHandle>, String> {
    if choice == RendererChoice::Cpu {
        eprintln!("renderer: cpu");
        return Ok(None);
    }
    #[cfg(feature = "gpu")]
    match gpu::GpuContext::new() {
        Ok(ctx) if ctx.software && choice == RendererChoice::Auto => {
            eprintln!("renderer: cpu (GPU adapter {} is a software implementation)", ctx.description);
            Ok(None)
        }
        Ok(ctx) => {
            eprintln!("renderer: gpu, {}", ctx.description);
            Ok(Some(ctx))
        }
        Err(e) if choice == RendererChoice::Auto => {
            eprintln!("renderer: cpu ({e})");
            Ok(None)
        }
        Err(e) => Err(e),
    }
    #[cfg(not(feature = "gpu"))]
    match choice {
        RendererChoice::Gpu => Err("--renderer gpu: stress-viz was built without the gpu feature".into()),
        _ => {
            eprintln!("renderer: cpu (built without the gpu feature)");
            Ok(None)
        }
    }
}

fn parse_num<T: std::str::FromStr>(flag: &str, v: &str) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    v.parse().map_err(|e| format!("{flag} {v}: {e}"))
}

fn parse_args(raw: Vec<String>) -> Result<Args, String> {
    let mut it = raw.into_iter();
    let cmd = it.next().ok_or("missing command")?;
    let cores = std::thread::available_parallelism().map_or(2, |n| n.get());
    let mut a = Args {
        cmd,
        scene: None,
        out: None,
        views: vec![ViewMode::Utilization, ViewMode::Stress, ViewMode::Damage, ViewMode::Fragments],
        cols: None,
        overrides: Vec::new(),
        features: Vec::new(),
        variants: Vec::new(),
        engine: record::Engine::Standalone,
        checkpoint: None,
        model_device: "cpu".into(),
        model_python: None,
        duration: None,
        frame_dt: None,
        every: None,
        min_frames: 150,
        max_frames: 450,
        fps: 30.0,
        hold: 1.5,
        size: (1920, 1080),
        ss: 2,
        ss_given: false,
        jobs: (cores / 2).max(1),
        sim_threads: 4,
        solver_mode: "explicit".into(),
        renderer: RendererChoice::Auto,
        camera: CameraSpec::default(),
        clip: None,
        cracks: true,
        crack_color: None,
        arrows: false,
        plot: vec!["energy".into()],
        scale_overrides: ScaleOverrides::default(),
        png_dir: None,
        title: None,
        time: None,
    };
    let (mut yaw, mut pitch) = (None, None);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "-o" | "--out" => a.out = Some(PathBuf::from(value()?)),
            "--name" | "--scene" => a.scene = Some(value()?),
            "--views" => a.views = value()?.split(',').map(str::parse).collect::<Result<_, _>>()?,
            "--cols" => a.cols = Some(parse_num::<usize>(&flag, &value()?)?.max(1)),
            "--set" => a.overrides.push(value()?),
            "--feature" => a.features.push(value()?),
            "--variant" => a.variants.push(value()?),
            "--engine" => a.engine = value()?.parse()?,
            "--checkpoint" => a.checkpoint = Some(PathBuf::from(value()?)),
            "--model-device" => a.model_device = value()?,
            "--model-python" => a.model_python = Some(PathBuf::from(value()?)),
            "--duration" => a.duration = Some(parse_num(&flag, &value()?)?),
            "--frame-dt" => a.frame_dt = Some(parse_num(&flag, &value()?)?),
            "--every" => a.every = Some(parse_num::<usize>(&flag, &value()?)?.max(1)),
            "--min-frames" => a.min_frames = parse_num(&flag, &value()?)?,
            "--max-frames" => a.max_frames = parse_num::<usize>(&flag, &value()?)?.max(2),
            "--fps" => a.fps = parse_num(&flag, &value()?)?,
            "--hold" => a.hold = parse_num(&flag, &value()?)?,
            "--size" => {
                let v = value()?;
                let (w, h) = v.split_once('x').ok_or_else(|| format!("--size {v}: expected WxH"))?;
                // yuv420p needs even dimensions.
                a.size = (parse_num::<usize>(&flag, w)? & !1, parse_num::<usize>(&flag, h)? & !1);
                if a.size.0 < 320 || a.size.1 < 240 {
                    return Err(format!("--size {v}: at least 320x240"));
                }
            }
            "--ss" => {
                a.ss = parse_num::<usize>(&flag, &value()?)?.clamp(1, 4);
                a.ss_given = true;
            }
            "--sim-threads" => a.sim_threads = parse_num::<usize>(&flag, &value()?)?.max(1),
            "--solver-mode" => a.solver_mode = value()?,
            "--jobs" => a.jobs = parse_num::<usize>(&flag, &value()?)?.max(1),
            "--renderer" => {
                a.renderer = match value()?.as_str() {
                    "auto" => RendererChoice::Auto,
                    "gpu" => RendererChoice::Gpu,
                    "cpu" => RendererChoice::Cpu,
                    v => return Err(format!("--renderer {v}: auto, gpu or cpu")),
                }
            }
            "--camera" => {
                let v = value()?;
                let (y, p) = CameraSpec::preset(&v).ok_or_else(|| format!("--camera {v}: front, side, top, iso or 3/4"))?;
                a.camera.yaw_deg = y;
                a.camera.pitch_deg = p;
            }
            "--yaw" => yaw = Some(parse_num(&flag, &value()?)?),
            "--pitch" => pitch = Some(parse_num(&flag, &value()?)?),
            "--distance" => a.camera.distance = Some(parse_num(&flag, &value()?)?),
            "--fov" => a.camera.fov_deg = parse_num(&flag, &value()?)?,
            "--target" => {
                let v = value()?;
                let c: Vec<f64> = v.split(',').map(|x| parse_num(&flag, x)).collect::<Result<_, _>>()?;
                if c.len() != 3 {
                    return Err(format!("--target {v}: expected X,Y,Z"));
                }
                a.camera.target = Some(Vec3::new(c[0], c[1], c[2]));
            }
            "--clip" => a.clip = Some(Clip::parse(&value()?)?),
            "--clip-y" => a.clip = Some(Clip { axis: 1, hide_above: true, value: parse_num(&flag, &value()?)? }),
            "--no-cracks" => a.cracks = false,
            "--crack-color" => {
                a.crack_color = match value()?.as_str() {
                    "white" => Some([1.0; 3]),
                    "black" => Some(scene3d::CRACK_DARK),
                    v => return Err(format!("--crack-color {v}: white or black")),
                }
            }
            "--arrows" => a.arrows = true,
            "--plot" => a.plot = value()?.split(',').map(String::from).collect(),
            "--stress-scale" => a.scale_overrides.stress = Some(parse_num(&flag, &value()?)?),
            "--exaggerate" => a.scale_overrides.exaggeration = Some(parse_num(&flag, &value()?)?),
            "--png" => a.png_dir = Some(PathBuf::from(value()?)),
            "--title" => a.title = Some(value()?),
            "--time" => a.time = Some(parse_num(&flag, &value()?)?),
            f if f.starts_with('-') => return Err(format!("unknown option {f}")),
            _ if a.scene.is_none() => a.scene = Some(flag),
            _ => return Err(format!("unexpected argument {flag}")),
        }
    }
    if let Some(y) = yaw {
        a.camera.yaw_deg = y;
    }
    if let Some(p) = pitch {
        a.camera.pitch_deg = p;
    }
    Ok(a)
}

/// A scene from a JSON file or by catalogue name.
fn load_scene(spec: &str) -> Result<Scene, String> {
    let path = std::path::Path::new(spec);
    if path.is_file() {
        return Scene::load(path);
    }
    // The M5 acceptance scenes in P0's pinned real-time configuration (stress_rubble::scenes),
    // as `learned_p0` and the GPU benchmarks run them (the ram strikes in frame 2).
    #[cfg(feature = "rubble")]
    {
        use stress_rubble::scenes as rs;
        match spec {
            "m5_free_wall" => return Ok(rs::realtime(rs::wall_ram_late(), 1.0)),
            "m5_wall_row" => return Ok(rs::realtime(rs::wall_row_late(2), 1.0)),
            _ => {}
        }
    }
    builders::catalog()
        .into_iter()
        .chain(showcases::catalog())
        .find(|s| s.name == spec)
        .ok_or_else(|| format!("'{spec}' is neither a scene file nor a catalogue name (see `stress-viz list`)"))
}

/// Applies one override item: `path=json` or `feature NAME=on|off`.
fn apply_item(scene: &Scene, item: &str) -> Result<Scene, String> {
    let item = item.trim();
    if item.starts_with("engine ") || item.starts_with("engine:") {
        // Not a scene change: see `variant_engines`.
        return Ok(scene.clone());
    }
    if let Some(f) = item.strip_prefix("feature ").or_else(|| item.strip_prefix("feature:")) {
        let (name, state) = f.trim().split_once('=').ok_or_else(|| format!("'{item}': expected feature NAME=on|off"))?;
        let on = match state.trim() {
            "on" | "true" => true,
            "off" | "false" => false,
            s => return Err(format!("'{item}': '{s}' is not on or off")),
        };
        return scene.with_feature(name.trim(), on);
    }
    let (path, value) = item.split_once('=').ok_or_else(|| format!("'{item}': expected path=json"))?;
    scene.with_override(path.trim(), value.trim())
}

/// Timing of a run: simulated duration, frame step, frames kept.
struct Timing {
    duration: f64,
    frame_dt: f64,
    every: usize,
}

fn plan_timing(scene: &Scene, a: &Args, duration: f64) -> Timing {
    let mut frame_dt = a.frame_dt.unwrap_or(scene.sim.frame_dt);
    let frames = (duration / frame_dt).round().max(1.0) as usize;
    if a.frame_dt.is_none() && a.every.is_none() && frames < a.min_frames {
        frame_dt = duration / a.min_frames as f64;
        eprintln!(
            "note: frame_dt {:.3e} s gives {frames} frames; using {frame_dt:.3e} s for {} frames (set --frame-dt to override)",
            scene.sim.frame_dt, a.min_frames
        );
    }
    let frames = (duration / frame_dt).round().max(1.0) as usize;
    let every = a.every.unwrap_or_else(|| frames.div_ceil(a.max_frames).max(1));
    Timing { duration, frame_dt, every }
}

/// The labelled scenes to run: the base scene, or one per `--variant`.
fn build_scenes(a: &Args) -> Result<Vec<(String, Scene)>, String> {
    let spec = a.scene.as_deref().ok_or("missing scene (path or catalogue name)")?;
    let mut base = load_scene(spec)?;
    for o in &a.overrides {
        base = apply_item(&base, o)?;
    }
    for f in &a.features {
        base = apply_item(&base, &format!("feature {f}"))?;
    }
    if a.cmd != "compare" {
        return Ok(vec![(base.name.clone(), base)]);
    }
    if a.variants.len() < 2 {
        return Err("compare needs at least two --variant \"label:item;item\"".into());
    }
    a.variants
        .iter()
        .map(|v| {
            let (label, items) = v.split_once(':').unwrap_or((v.as_str(), ""));
            let mut s = base.clone();
            for item in items.split(';').filter(|i| !i.trim().is_empty()) {
                s = apply_item(&s, item)?;
            }
            Ok((label.trim().to_string(), s))
        })
        .collect()
}

/// The rigid-body engine of each run: `--engine`, or per `--variant` its `engine` item.
fn variant_engines(a: &Args) -> Result<Vec<record::Engine>, String> {
    if a.cmd != "compare" {
        return Ok(vec![a.engine]);
    }
    a.variants
        .iter()
        .map(|v| {
            let items = v.split_once(':').map_or("", |(_, i)| i);
            let mut engine = a.engine;
            for item in items.split(';').map(str::trim) {
                if let Some(e) = item.strip_prefix("engine ").or_else(|| item.strip_prefix("engine:")) {
                    engine = e.trim().parse()?;
                }
            }
            Ok(engine)
        })
        .collect()
}

/// Records every scene (in parallel, at most `jobs` at a time).
fn record_all(scenes: &[(String, Scene)], a: &Args, until: Option<f64>) -> Result<Vec<Recording>, String> {
    let engines = variant_engines(a)?;
    let base = &scenes[0].1;
    let mut timing = plan_timing(base, a, until.or(a.duration).unwrap_or(base.sim.duration));
    if engines.iter().any(|e| e.model_coupled()) {
        // The model is called once per engine frame at the frame it was trained on: keep the
        // scene's frame (1/60 s) for every variant, so the panels stay in step.
        if a.frame_dt.is_some_and(|d| (d - base.sim.frame_dt).abs() > 1e-12) {
            return Err(format!("engine learned/ceiling run at the scene's frame_dt ({} s); drop --frame-dt", base.sim.frame_dt));
        }
        timing.frame_dt = base.sim.frame_dt;
        timing.every = a.every.unwrap_or(1);
        if engines.iter().any(|e| matches!(e, record::Engine::Learned | record::Engine::ALearned)) && a.checkpoint.is_none() {
            return Err("engine learned needs --checkpoint PATH".into());
        }
    }
    let mut out: Vec<Option<Recording>> = (0..scenes.len()).map(|_| None).collect();
    for ((batch_scenes, batch_out), batch_engines) in scenes.chunks(a.jobs).zip(out.chunks_mut(a.jobs)).zip(engines.chunks(a.jobs)) {
        std::thread::scope(|s| {
            for (((label, scene), slot), &engine) in batch_scenes.iter().zip(batch_out.iter_mut()).zip(batch_engines) {
                let timing = &timing;
                s.spawn(move || {
                    let mut scene = scene.clone();
                    scene.sim.frame_dt = timing.frame_dt;
                    let opts = RecordOptions { every: timing.every, duration: timing.duration, label: label.clone(), engine,
                                               checkpoint: a.checkpoint.clone(), model_device: a.model_device.clone(), model_python: a.model_python.clone() };
                    *slot = Some(record::record(&scene, &opts));
                });
            }
        });
    }
    Ok(out.into_iter().map(|r| r.expect("recorded")).collect())
}

fn header_text(a: &Args, scenes: &[(String, Scene)]) -> (String, String) {
    let scene = &scenes[0].1;
    let title = a.title.clone().unwrap_or_else(|| {
        if scenes.len() > 1 {
            let labels: Vec<&str> = scenes.iter().map(|(l, _)| l.as_str()).collect();
            format!("{} — {}", scene.name, labels.join(" vs "))
        } else {
            scene.name.clone()
        }
    });
    let mut subtitle = scene.description.clone();
    let extra: Vec<&str> = a.overrides.iter().chain(&a.features).map(String::as_str).collect();
    if !extra.is_empty() {
        subtitle = format!("{subtitle}  [{}]", extra.join(", "));
    }
    (title, subtitle)
}

fn composer<'a>(a: &Args, scenes: &[(String, Scene)], recs: &'a [Recording]) -> Result<Composer<'a>, String> {
    let refs: Vec<&Recording> = recs.iter().collect();
    let scales = Scales::fit(&refs, a.scale_overrides);
    let labels: Vec<String> = scenes.iter().map(|(l, _)| l.clone()).collect();
    let mut panels = Vec::new();
    for &mode in &a.views {
        for (k, label) in labels.iter().enumerate() {
            let title = if recs.len() > 1 { format!("{label} · {}", mode.title(&scales)) } else { mode.title(&scales) };
            panels.push(PanelSpec { title, mode, rec: k });
        }
    }
    let cols = a.cols.unwrap_or(if recs.len() > 1 { recs.len() } else { columns_for(panels.len()) }).min(panels.len());
    let plot = build_plot(&refs, &labels, &a.plot)?;
    let (title, subtitle) = header_text(a, scenes);
    let opts = DrawOptions { cracks: a.cracks, crack_color: a.crack_color, arrows: a.arrows, clip: a.clip, ss: a.ss };
    Ok(Composer::new(refs, panels, cols, a.size, &a.camera, scales, opts, plot, title, subtitle, a.fps))
}

/// Renders every frame (in parallel batches, written in order) to ffmpeg and/or PNGs.
fn render_video(a: &Args, comp: &Composer, default_out: &str) -> Result<(), String> {
    let out = match (&a.out, &a.png_dir) {
        (Some(o), _) => Some(o.clone()),
        (None, Some(_)) => None,
        (None, None) => Some(PathBuf::from(default_out)),
    };
    let mut sink = match &out {
        Some(p) => Some(video::Ffmpeg::start(p, a.size.0, a.size.1, a.fps)?),
        None => None,
    };
    let n = comp.frame_count();
    let start = Instant::now();
    let mut render_seconds = 0.0;
    let mut last: Option<Canvas> = None;
    let batch = 2 * a.jobs;
    let gpu = open_gpu(a.renderer)?;
    let mut rasters: Vec<_> = (0..a.jobs).map(|_| comp.new_rasterizer(gpu.as_ref())).collect();
    for first in (0..n).step_by(batch) {
        let idx: Vec<usize> = (first..(first + batch).min(n)).collect();
        let mut frames: Vec<Option<Result<(Canvas, f64), String>>> = (0..idx.len()).map(|_| None).collect();
        let chunk = idx.len().div_ceil(a.jobs);
        std::thread::scope(|s| {
            for ((ids, slots), raster) in idx.chunks(chunk).zip(frames.chunks_mut(chunk)).zip(rasters.iter_mut()) {
                s.spawn(move || {
                    for (&i, slot) in ids.iter().zip(slots.iter_mut()) {
                        let t = Instant::now();
                        let canvas = match comp.render(i, raster) {
                            Ok(c) => c,
                            Err(e) => {
                                *slot = Some(Err(e));
                                return;
                            }
                        };
                        let secs = t.elapsed().as_secs_f64();
                        if let Some(dir) = &a.png_dir {
                            if let Err(e) = video::write_png(&dir.join(format!("frame_{i:05}.png")), &canvas) {
                                eprintln!("{e}");
                            }
                        }
                        *slot = Some(Ok((canvas, secs)));
                    }
                });
            }
        });
        for frame in frames.into_iter().flatten() {
            let (canvas, secs) = frame?;
            render_seconds += secs;
            if let Some(s) = sink.as_mut() {
                s.write(&canvas)?;
            }
            last = Some(canvas);
        }
        eprint!("\rrendered {}/{n} frames", (first + batch).min(n));
    }
    eprintln!();
    if let (Some(s), Some(l)) = (sink.as_mut(), &last) {
        for _ in 0..(a.hold * a.fps).round() as usize {
            s.write(l)?;
        }
    }
    if let Some(s) = sink {
        let path = s.finish()?;
        eprintln!("wrote {}", path.display());
    }
    eprintln!(
        "{n} frames in {:.1} s wall ({:.0} ms per frame per thread, {} threads)",
        start.elapsed().as_secs_f64(),
        1e3 * render_seconds / n.max(1) as f64,
        a.jobs
    );
    Ok(())
}

fn run(raw: Vec<String>) -> Result<(), String> {
    let a = parse_args(raw)?;
    match a.cmd.as_str() {
        "list" => {
            for s in builders::catalog().into_iter().chain(showcases::catalog()) {
                println!("{:28} {}", s.name, s.description);
            }
            Ok(())
        }
        "render" | "compare" => {
            let scenes = build_scenes(&a)?;
            let recs = record_all(&scenes, &a, None)?;
            for r in &recs {
                eprintln!("{}: {} frames recorded, {:.1} s simulation", r.scene.name, r.frames.len(), r.wall_seconds);
            }
            let comp = composer(&a, &scenes, &recs)?;
            let suffix = if a.cmd == "compare" { "_compare" } else { "" };
            render_video(&a, &comp, &format!("{}{suffix}.mp4", scenes[0].1.name))
        }
        "still" => {
            let t = a.time.ok_or("still needs --time T")?;
            let scenes = build_scenes(&a)?;
            let recs = record_all(&scenes, &a, Some(t))?;
            let comp = composer(&a, &scenes, &recs)?;
            let gpu = open_gpu(a.renderer)?;
            let start = Instant::now();
            let canvas = comp.render(comp.frame_count() - 1, &mut comp.new_rasterizer(gpu.as_ref()))?;
            let render_ms = 1e3 * start.elapsed().as_secs_f64();
            let out = a.out.clone().unwrap_or_else(|| PathBuf::from(format!("{}_t{t}.png", scenes[0].1.name)));
            video::write_png(&out, &canvas)?;
            eprintln!("wrote {} (render {render_ms:.0} ms)", out.display());
            Ok(())
        }
        #[cfg(feature = "viewer")]
        "view" => {
            if a.engine.model_coupled() {
                return Err("view: engine learned/ceiling is offline only for now; use render or compare".into());
            }
            let scenes = build_scenes(&a)?;
            let mut scene = scenes.into_iter().next().map(|(_, s)| s).ok_or("missing scene")?;
            // A finer frame gives smoother live updates of slow scenes (it is also the
            // solver's frame: results can differ).
            if let Some(dt) = a.frame_dt {
                scene.sim.frame_dt = dt;
            }
            viewer::run(viewer::ViewOptions {
                scene,
                views: a.views.clone(),
                camera: a.camera.clone(),
                opts: DrawOptions { cracks: a.cracks, crack_color: a.crack_color, arrows: a.arrows, clip: a.clip, ss: a.ss },
                ss: a.ss_given.then_some(a.ss),
                scale_overrides: a.scale_overrides,
                sim_threads: a.sim_threads,
                engine: a.engine,
                mode: viewer::mode_index(&a.solver_mode).ok_or_else(|| {
                    let names: Vec<&str> = viewer::MODES.iter().map(|m| m.0).collect();
                    format!("--solver-mode {}: one of {}", a.solver_mode, names.join(", "))
                })?,
            })
        }
        #[cfg(not(feature = "viewer"))]
        "view" => Err("view: stress-viz was built without the viewer feature".into()),
        other => Err(format!("unknown command '{other}'")),
    }
}

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() || raw[0] == "-h" || raw[0] == "--help" {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    match run(raw) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("stress-viz: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Args {
        parse_args(s.split_whitespace().map(String::from).collect()).unwrap()
    }

    #[test]
    fn parses_options() {
        let a = args("render b5_wall_impact_v40 --views utilization,damage --camera front --yaw 10 --clip-y 0.25 --size 1281x720");
        assert_eq!(a.scene.as_deref(), Some("b5_wall_impact_v40"));
        assert_eq!(a.views, vec![ViewMode::Utilization, ViewMode::Damage]);
        assert_eq!((a.camera.yaw_deg, a.camera.pitch_deg), (10.0, 6.0));
        assert_eq!(a.clip, Some(Clip { axis: 1, hide_above: true, value: 0.25 }));
        assert_eq!(a.size, (1280, 720));
        assert!(parse_args(vec!["render".into(), "--bogus".into()]).is_err());
    }

    #[test]
    fn variants_apply_overrides_and_features() {
        let mut a = args("compare s_arch");
        a.variants = vec!["stiff:sim.stiffness_scale=0.5;sim.solve_mode=adaptive".into(), "plain:feature softening=off".into()];
        let scenes = build_scenes(&a).unwrap();
        assert_eq!(scenes[0].0, "stiff");
        assert_eq!(scenes[0].1.sim.stiffness_scale, 0.5);
        assert!(!scenes[1].1.sim.features.softening);
        assert!(apply_item(&scenes[0].1, "feature nonsense=on").is_err());
        assert!(apply_item(&scenes[0].1, "no_equals_sign").is_err());
    }

    #[test]
    fn timing_refines_short_scenes_and_thins_long_ones() {
        let a = args("render x");
        let mut scene = builders::catalog().into_iter().find(|s| s.name == "b5_wall_impact_v40").unwrap();
        let t = plan_timing(&scene, &a, scene.sim.duration);
        assert!((scene.sim.duration / t.frame_dt - 150.0).abs() < 1e-6 && t.every == 1);
        scene.sim.frame_dt = 1e-4;
        let t = plan_timing(&scene, &a, 1.0);
        assert_eq!((t.frame_dt, t.every), (1e-4, 23));
    }
}
