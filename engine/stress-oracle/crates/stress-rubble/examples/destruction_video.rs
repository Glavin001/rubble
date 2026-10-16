//! Destruction videos of the product path: the GPU stress solver coupled to rubble's AVBD
//! GPU engine on the solver's own device (`GpuWorld::with_engine` +
//! `RubbleEngine::with_device`, as `tests/integrated.rs`), stepped one 1/60 s frame at a
//! time at the realtime_peak settings (established methods, certified true step, contact
//! safety 0.35, stress safety 0.5, adaptive). Each frame's host mirror is rendered
//! offscreen (`stress_rubble::render`) and piped into ffmpeg. Ported from the PhysX fork's
//! `stress-physx` example of the same name (PhysX GPU there).
//!
//! Scenes (`--scene`):
//!   town   house-1story, brick-building, house-2story and villa-savoye (`packs/`) on one
//!          ground, each struck by a lobbed concrete ball at a staggered arrival time
//!          (offline: 1x playback, any compute)
//!   live   free concrete walls of 1/3 m chunks (realtime_envelope's wall, default six of
//!          3 x 1 x 3) in zero gravity, each struck in turn by a 200 kg steel ram at
//!          10 m/s; with `--live-readout` every frame carries its measured `step_frame`
//!          time and the last 3 s of them (`--walls`, `--wall NXxNYxNZ`, `--chunk`,
//!          `--ram-mass`, `--ram-half` vary it)
//!   live-gravity  the same walls standing on the ground with intact house-1story packs
//!          behind (`--houses`)
//!   <pack>[,<pack>...]  a subset of the town (same layout and schedule)
//!
//! The packs are imported as they are (`scene_pack::import`): every hull collider in
//! `packs/` has at most 42 points, inside the 64-corner limit stress-ref and rubble share
//! (`hull::MAX_HULL_VERTICES`, `engine_schema::convex::HULL_MAX_VERTICES`), and a pack
//! beyond it fails at import with that error. (The fork's town also used the version-1
//! Voronoi packs fractured-tower, -bridge and -wall, whose colliders it reduced to 64
//! corners at load time; they are not in `packs/` and that reduction is not ported.)
//!
//! Town impactors start at t = 0 and fly ballistic arcs under gravity. An arc reaching its
//! target at time T with a descent of at most 70 degrees arrives at no less than
//! 0.53 g T, so arrivals after about 2.6 s cannot stay at 12-15 m/s: those impactors arrive
//! as fast as the arc requires, with their mass lowered to carry the same momentum as a
//! 400 kg ball at 12 m/s (stated per impactor at start-up).
//!
//! The `step_frame` time is measured alone (rendering, read-back and encoding excluded);
//! with `--encode-after` the frames are kept in memory and encoded after the run, so the
//! encoder does not share the CPU with the simulation. A per-frame CSV (step time, broken
//! bonds, clusters, substeps) is written next to the video.
//!
//!   cargo build --release -p stress-rubble --features gpu --example destruction_video
//!   target/release/examples/destruction_video --scene town --seconds 6 --out videos/demo/town.mp4
//!     [--width 1920 --height 1080] [--encode-after] [--live-readout] [--no-render] [--info]
//!   ... --scene live --seconds 8 --encode-after --live-readout --out videos/demo/live.mp4

use std::path::{Path, PathBuf};
use std::time::Instant;

use stress_gpu::gpu::Gpu;
use stress_gpu::world::GpuWorld;
use stress_ref::math::Vec3;
use stress_ref::scene::{ImpactorDesc, ImpactorShape, Methods, Scene, SolveMode, Support};
use stress_rubble::render::{CameraPose, FfmpegSink, Overlay, Renderer};
use stress_rubble::RubbleEngine;

const G: f64 = 9.81;

/// The realtime_peak settings (examples/realtime_peak.rs `configure`).
fn configure(mut s: Scene, seconds: f64) -> Scene {
    s.sim.methods = Methods::established();
    s.sim.methods.true_step = true;
    s.sim.courant_safety = 0.35;
    s.sim.stress_courant_safety = Some(0.5);
    s.sim.solve_mode = SolveMode::Adaptive;
    s.sim.duration = seconds;
    s
}

/// rubble's scene packs.
fn packs_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs")
}

/// A pack from `packs/`, imported unchanged.
fn pack(name: &str) -> Scene {
    stress_ref::scene_pack::import(&packs_dir().join(format!("{name}.json")), name, name).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Bounds of a body's chunk boxes (centres +- half extents).
fn body_bounds(b: &stress_ref::scene::BodyDesc) -> (Vec3, Vec3) {
    let (mut lo, mut hi) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
    for c in &b.chunks {
        let (p, h) = (Vec3::from_array(c.center), Vec3::from_array(c.half_extents));
        lo = lo.component_min(p - h);
        hi = hi.component_max(p + h);
    }
    (lo, hi)
}

/// `part`'s bodies moved by `offset` into `into` (chunk centres and bond patches; hulls
/// are in the chunk frame), its materials merged (same names are the same presets).
fn place(into: &mut Scene, part: &Scene, offset: Vec3) {
    for (k, m) in &part.materials {
        into.materials.entry(k.clone()).or_insert_with(|| m.clone());
    }
    for b in &part.bodies {
        let mut body = b.clone();
        for c in &mut body.chunks {
            c.center = (Vec3::from_array(c.center) + offset).to_array();
        }
        for bd in &mut body.bonds {
            bd.centroid = (Vec3::from_array(bd.centroid) + offset).to_array();
        }
        into.bodies.push(body);
    }
    if into.ground.is_none() {
        into.ground = part.ground.clone();
    }
}

/// A lobbed impactor: a concrete ball reaching body `body`'s face on side `side`
/// (outward horizontal normal) at time `arrival`, at `height` of the body's height and
/// `lateral` along the face, launched from 1 m above the ground. Arrival speed `speed`
/// when the arc allows it (descent at most 70 degrees), else the slowest such arc; mass
/// `momentum / speed` up to `max_mass`, radius from concrete's density (2400 kg/m^3).
struct Lob {
    body: usize,
    arrival: f64,
    side: Vec3,
    height: f64,
    lateral: f64,
}

fn lob(scene: &Scene, l: &Lob, speed: f64, momentum: f64, max_mass: f64, material: &str) -> ImpactorDesc {
    let (lo, hi) = body_bounds(&scene.bodies[l.body]);
    let side = Vec3::new(l.side.x, l.side.y, 0.0).normalized();
    let along = Vec3::new(-side.y, side.x, 0.0);
    let mid = (lo + hi) * 0.5;
    let half = (hi - lo) * 0.5;
    // The face: the bound along `side`, at `lateral` along it.
    let reach = half.x * side.x.abs() + half.y * side.y.abs();
    let width = half.x * along.x.abs() + half.y * along.y.abs();
    let mut face = mid + side * reach + along * (l.lateral * width);
    face.z = lo.z.max(0.0) + l.height * (hi.z - lo.z.max(0.0));
    // Arrival speed and the arc.
    let t = l.arrival;
    let z0 = 1.0;
    let dz = face.z - z0;
    let vz_arrive = dz / t - 0.5 * G * t;
    let min_h = vz_arrive.abs() * (20f64).to_radians().tan();
    let vh = if speed * speed - vz_arrive * vz_arrive >= min_h * min_h { (speed * speed - vz_arrive * vz_arrive).sqrt() } else { min_h };
    let u = (vh * vh + vz_arrive * vz_arrive).sqrt();
    let mass = (momentum / u).min(max_mass);
    let radius = (3.0 * mass / (4.0 * std::f64::consts::PI * 2400.0)).cbrt();
    let target = face + side * (radius + 0.05);
    let start = Vec3::new(target.x + side.x * vh * t, target.y + side.y * vh * t, z0);
    let v0 = Vec3::new(-side.x * vh, -side.y * vh, dz / t + 0.5 * G * t);
    ImpactorDesc {
        name: format!("ball_{}", scene.bodies[l.body].name),
        shape: ImpactorShape::Sphere { radius },
        mass,
        position: start.to_array(),
        orientation: [1.0, 0.0, 0.0, 0.0],
        velocity: v0.to_array(),
        angular_velocity: [0.0; 3],
        material: material.into(),
        crush: None,
    }
}

/// The town's packs.
const TOWN: [&str; 4] = ["house-1story", "house-2story", "brick-building", "villa-savoye"];

/// The town's layout and schedule: (pack, row (0 front, 1 back), arrival time (s) of its
/// ball, height of the struck point as a fraction of the structure's height). Rows run
/// along x, 4 m apart within a row, the back row 5 m behind the front row; every ball
/// comes from the camera's side (-y), 20 degrees off square.
const LAYOUT: [(&str, usize, f64, f64); 4] = [("house-1story", 0, 0.8, 0.55), ("house-2story", 1, 2.0, 0.55), ("brick-building", 0, 3.2, 0.55), ("villa-savoye", 1, 4.4, 0.6)];
/// Front-row order along x (left to right as the camera sees it), then the back row's.
const ROWS: [[&str; 2]; 2] = [["house-1story", "brick-building"], ["house-2story", "villa-savoye"]];

struct Staged {
    scene: Scene,
    camera: CameraPose,
    orbit: f64,
    /// (impactor, body, arrival time (s), arrival speed (m/s))
    schedule: Vec<(String, String, f64, f64)>,
}

fn town(only: Option<&[String]>) -> Staged {
    let mut s = stress_ref::builders::new_scene("town", "Four authored structures struck in turn by lobbed concrete balls");
    let packs: Vec<(&str, Scene)> = TOWN.iter().map(|n| (*n, pack(n))).collect();
    let bounds = |name: &str| body_bounds(&packs.iter().find(|p| p.0 == name).unwrap().1.bodies[0]);
    let gap = 4.0;
    // Offsets of every pack (the full layout, also for subsets).
    let mut offsets: Vec<(&str, Vec3)> = Vec::new();
    let front_depth = ROWS[0].iter().filter(|n| !n.is_empty()).map(|n| bounds(n)).map(|(lo, hi)| hi.y - lo.y).fold(0.0, f64::max);
    for (row, names) in ROWS.iter().enumerate() {
        let names: Vec<&str> = names.iter().copied().filter(|n| !n.is_empty()).collect();
        let width: f64 = names.iter().map(|n| bounds(n)).map(|(lo, hi)| hi.x - lo.x).sum::<f64>() + gap * (names.len() - 1) as f64;
        let mut x = -0.5 * width;
        for n in names {
            let (lo, hi) = bounds(n);
            // Front row: footprints centred on y = 0; back row: fronts 5 m behind it.
            let y = if row == 0 { -(lo.y + hi.y) * 0.5 } else { 0.5 * front_depth + 5.0 - lo.y };
            offsets.push((n, Vec3::new(x - lo.x, y, 0.0)));
            x += hi.x - lo.x + gap;
        }
    }
    let mut arrivals = Vec::new();
    for (name, _row, arrival, height) in LAYOUT {
        if only.is_some_and(|o| !o.iter().any(|m| m == name)) {
            continue;
        }
        let p = &packs.iter().find(|p| p.0 == name).unwrap().1;
        let offset = offsets.iter().find(|o| o.0 == name).unwrap().1;
        place(&mut s, p, offset);
        arrivals.push((s.bodies.len() - 1, arrival, height));
    }
    let mut schedule = Vec::new();
    for (body, arrival, height) in arrivals {
        let l = Lob { body, arrival, side: Vec3::new(-0.36, -1.0, 0.0), height, lateral: 0.0 };
        let imp = lob(&s, &l, 12.0, 400.0 * 12.0, 400.0, "concrete");
        let u = Vec3::from_array(imp.velocity) + Vec3::new(0.0, 0.0, -G * arrival);
        schedule.push((imp.name.clone(), s.bodies[body].name.clone(), arrival, u.norm()));
        s.impactors.push(imp);
    }
    let (lo, hi) = scene_bounds(&s);
    let target = Vec3::new((lo.x + hi.x) * 0.5, (lo.y + hi.y) * 0.5, 0.25 * hi.z);
    let span = (hi.x - lo.x).max(0.8 * (hi.y - lo.y));
    let camera = CameraPose { target, distance: 0.95 * span + 6.0, yaw: -1.66, pitch: 0.46, fov: 40f64.to_radians() };
    Staged { scene: s, camera, orbit: 0.012, schedule }
}

/// A wall of `n` chunks of `size` (concrete as realtime_peak's wall); with `grounded`
/// its bottom course stands on the ground (fixed, as the packs' grade courses are),
/// otherwise it is free (realtime_peak's and realtime_envelope's wall).
fn coarse_wall(name: &str, origin: Vec3, n: [usize; 3], size: f64, grounded: bool) -> stress_ref::scene::BodyDesc {
    use stress_ref::builders::*;
    let mut chunks = grid(origin, n, Vec3::splat(size), "concrete");
    if grounded {
        for c in chunks.iter_mut().filter(|c| c.center[2] < origin.z + size) {
            c.support = Support::Fixed;
        }
    }
    let bonds = auto_bonds(&chunks, |_, _| "concrete".into());
    body(name, chunks, bonds)
}

/// The live scene: `walls` concrete walls of `wall` chunks of 1/3 m in a row, each struck
/// in turn (arrivals 0.8 s, then every 1.2 s) by a 200 kg steel ram at 10 m/s.
///
/// `gravity` false (the default): the realtime_envelope regime, free walls in zero
/// gravity and no ground; the rams fly straight, started 10 m/s x arrival time away.
/// `gravity` true: the walls stand on the ground (bottom course fixed), `houses` intact
/// house-1story packs behind them, the rams lobbed (`lob`). Debris then rests on the
/// ground in contact, which keeps it in impact islands at the contact step.
fn live(walls: usize, houses: usize, wall: [usize; 3], gravity: bool, size: f64, ram_mass: f64, ram_half: f64) -> Staged {
    use stress_ref::builders::*;
    use stress_ref::scene::GroundDesc;
    let mut s = new_scene("live", "Coarse concrete walls struck in turn by steel rams");
    s.materials.insert("concrete".into(), oracle_concrete(None));
    s.materials.insert("steel".into(), elastic_steel());
    if gravity {
        s.ground = Some(GroundDesc { height: 0.0, friction: 0.6, material: "concrete".into() });
    } else {
        s.gravity = [0.0; 3];
    }
    let width = wall[0] as f64 * size;
    let pitch = width + 2.0;
    for i in 0..walls {
        let x = i as f64 * pitch - 0.5 * (walls as f64 * pitch - 2.0);
        s.bodies.push(coarse_wall(&format!("wall{i}"), Vec3::new(x, -0.5 * wall[1] as f64 * size, 0.0), wall, size, gravity));
    }
    let row = walls as f64 * pitch;
    if gravity {
        for h in 0..houses {
            let p = pack("house-1story");
            let (lo, hi) = body_bounds(&p.bodies[0]);
            let hx = hi.x - lo.x;
            let x = -0.5 * (houses as f64 * (hx + 3.0) - 3.0) + h as f64 * (hx + 3.0) - lo.x;
            let mut part = p.clone();
            part.bodies[0].name = format!("house{h}");
            place(&mut s, &part, Vec3::new(x, 6.0 - lo.y, 0.0));
        }
    }
    let mut schedule = Vec::new();
    for i in 0..walls {
        let arrival = 0.8 + 1.2 * i as f64;
        let imp = if gravity {
            let l = Lob { body: i, arrival, side: Vec3::new(-0.3, -1.0, 0.0), height: 0.6, lateral: 0.0 };
            lob(&s, &l, 10.0, ram_mass * 10.0, ram_mass, "steel")
        } else {
            // realtime_envelope's ram (200 kg, 0.2 m steel box), at the wall's middle.
            let (lo, hi) = body_bounds(&s.bodies[i]);
            let speed = 10.0;
            let face = Vec3::new(0.5 * (lo.x + hi.x), lo.y - ram_half - 0.02, 0.5 * (lo.z + hi.z));
            ImpactorDesc {
                name: format!("ram_wall{i}"),
                shape: ImpactorShape::Box { half_extents: [ram_half; 3] },
                mass: ram_mass,
                position: (face - Vec3::new(0.0, speed * arrival, 0.0)).to_array(),
                orientation: [1.0, 0.0, 0.0, 0.0],
                velocity: [0.0, speed, 0.0],
                angular_velocity: [0.0; 3],
                material: "steel".into(),
                crush: None,
            }
        };
        let g = Vec3::from_array(s.gravity);
        let u = Vec3::from_array(imp.velocity) + g * arrival;
        schedule.push((imp.name.clone(), s.bodies[i].name.clone(), arrival, u.norm()));
        s.impactors.push(imp);
    }
    let (lo, hi) = scene_bounds(&s);
    let target = Vec3::new(0.0, 0.5 * (lo.y + hi.y) + if gravity { 0.0 } else { 1.0 }, if gravity { 0.25 * hi.z.max(2.0) } else { 0.5 * (lo.z + hi.z) });
    let camera = if gravity {
        CameraPose { target, distance: 0.8 * row.max(hi.y - lo.y) + 6.0, yaw: -1.85, pitch: 0.30, fov: 40f64.to_radians() }
    } else {
        CameraPose { target, distance: 0.5 * row + 2.5, yaw: -1.92, pitch: 0.30, fov: 40f64.to_radians() }
    };
    Staged { scene: s, camera, orbit: 0.01, schedule }
}

fn scene_bounds(s: &Scene) -> (Vec3, Vec3) {
    let (mut lo, mut hi) = (Vec3::splat(f64::INFINITY), Vec3::splat(f64::NEG_INFINITY));
    for b in &s.bodies {
        let (l, h) = body_bounds(b);
        lo = lo.component_min(l);
        hi = hi.component_max(h);
    }
    (lo, hi)
}

fn info(s: &Scene) {
    for b in &s.bodies {
        let (lo, hi) = body_bounds(b);
        let mut sizes: Vec<f64> = b.chunks.iter().map(|c| 2.0 * c.half_extents.iter().copied().fold(f64::INFINITY, f64::min)).collect();
        sizes.sort_by(f64::total_cmp);
        let fixed = b.chunks.iter().filter(|c| matches!(c.support, Support::Fixed)).count();
        let mut mats: Vec<String> = b.chunks.iter().map(|c| c.material.clone()).collect();
        mats.sort();
        mats.dedup();
        println!(
            "{:<18} {:6} chunks ({fixed} fixed), {:6} bonds; size {:.2} x {:.2} x {:.2} m at ({:.1}, {:.1}); chunk thickness min {:.3} median {:.3} m; {}",
            b.name,
            b.chunks.len(),
            b.bonds.len(),
            hi.x - lo.x,
            hi.y - lo.y,
            hi.z - lo.z,
            lo.x,
            lo.y,
            sizes[0],
            sizes[sizes.len() / 2],
            mats.join(",")
        );
    }
    for imp in &s.impactors {
        println!("{:<24} mass {:6.1} kg at {:?} v0 {:?}", imp.name, imp.mass, imp.position.map(|v| (v * 100.0).round() / 100.0), imp.velocity.map(|v| (v * 100.0).round() / 100.0));
    }
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn main() {
    let mut scene_arg = "town".to_string();
    let mut seconds = 10.0f64;
    let (mut width, mut height) = (1920u32, 1080u32);
    let mut out: Option<PathBuf> = None;
    let mut show_info = false;
    let mut encode_after = false;
    let mut readout = false;
    let mut no_render = false;
    let (mut walls, mut houses, mut wall) = (6usize, 1usize, [3usize, 1, 3]);
    let (mut chunk, mut ram_mass, mut ram_half) = (1.0 / 3.0, 200.0f64, 0.1f64);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut next = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--scene" => scene_arg = next(),
            "--seconds" => seconds = next().parse().expect("--seconds S"),
            "--width" => width = next().parse().expect("--width W"),
            "--height" => height = next().parse().expect("--height H"),
            "--out" => out = Some(PathBuf::from(next())),
            "--info" => show_info = true,
            "--encode-after" => encode_after = true,
            "--live-readout" => readout = true,
            "--no-render" => no_render = true,
            "--chunk" => chunk = next().parse().expect("--chunk M"),
            "--ram-mass" => ram_mass = next().parse().expect("--ram-mass KG"),
            "--ram-half" => ram_half = next().parse().expect("--ram-half M"),
            "--walls" => walls = next().parse().expect("--walls N"),
            "--houses" => houses = next().parse().expect("--houses N"),
            "--wall" => {
                let v: Vec<usize> = next().split('x').map(|t| t.parse().expect("--wall NXxNYxNZ")).collect();
                wall = [v[0], v[1], v[2]];
            }
            other => panic!("unknown argument {other}"),
        }
    }
    let staged = match scene_arg.as_str() {
        "town" => town(None),
        "live" => live(walls, houses, wall, false, chunk, ram_mass, ram_half),
        "live-gravity" => live(walls, houses, wall, true, chunk, ram_mass, ram_half),
        list => {
            let names: Vec<String> = list.split(',').map(str::to_string).collect();
            town(Some(&names))
        }
    };
    let scene = configure(staged.scene, seconds);
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let bonds: usize = scene.bodies.iter().map(|b| b.bonds.len()).sum();
    println!("scene {}: {} bodies, {chunks} chunks, {bonds} bonds, {} impactors", scene_arg, scene.bodies.len(), scene.impactors.len());
    for (imp, body, t, u) in &staged.schedule {
        let m = scene.impactors.iter().find(|i| &i.name == imp).map_or(0.0, |i| i.mass);
        println!("  {imp:<26} -> {body:<18} arrives at {t:4.1} s, {u:5.1} m/s, {m:5.0} kg");
    }
    if show_info {
        info(&scene);
        for name in TOWN {
            println!("{name}: {}", pack(name).description);
        }
        return;
    }
    let out = out.unwrap_or_else(|| PathBuf::from(format!("videos/demo/{scene_arg}.mp4")));
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).ok();
    }

    let gpu = Gpu::new().expect("GPU");
    let t_init = Instant::now();
    let engine = RubbleEngine::with_device(Vec3::from_array(scene.gravity), gpu.adapter_info.clone(), gpu.device.clone(), gpu.queue.clone()).expect("rubble on the solver's device");
    let mut w = GpuWorld::with_engine(&gpu, &scene, Box::new(engine)).expect("GPU coupled world");
    let init_ms = t_init.elapsed().as_secs_f64() * 1e3;
    let mut renderer = Renderer::new(&gpu, &w.solver.mirror, &w.solver.impactors, width, height);
    renderer.ground = scene.ground.is_some();
    let zero_g = scene.gravity.iter().all(|g| *g == 0.0);
    let adapter = gpu.adapter_info.name.to_uppercase();
    let title = match (readout, zero_g) {
        (true, true) => format!("GPU STRESS SOLVER + RUBBLE AVBD ({adapter}), ZERO GRAVITY, STEPPED LIVE AT 1/60 S"),
        (true, false) => format!("GPU STRESS SOLVER + RUBBLE AVBD ({adapter}), STEPPED LIVE AT 1/60 S"),
        (false, _) => "GPU STRESS SOLVER + RUBBLE AVBD, OFFLINE RENDER, 1X PLAYBACK".to_string(),
    };
    println!("world created in {init_ms:.0} ms; renderer {} triangles at {width}x{height}", renderer.triangles());

    let frames = (seconds / scene.sim.frame_dt).round() as usize;
    let mut sink = if encode_after || no_render { None } else { Some(FfmpegSink::new(&out, width, height, 60).expect("ffmpeg")) };
    let mut kept: Vec<Vec<u8>> = Vec::new();
    let mut times: Vec<f64> = Vec::with_capacity(frames);
    let mut csv = String::from("frame,time_s,step_ms,substeps,broken_bonds,clusters\n");
    let wall_start = Instant::now();
    for f in 0..frames {
        let s0 = w.solver.mirror.substeps;
        let t = Instant::now();
        w.step_frame(&gpu).expect("frame");
        let ms = t.elapsed().as_secs_f64() * 1e3;
        times.push(ms);
        let m = &w.solver.mirror;
        let broken = m.broken_bond_count();
        let clusters = m.clusters.len();
        let subs = m.substeps - s0;
        let sim_t = (f + 1) as f64 * scene.sim.frame_dt;
        csv.push_str(&format!("{f},{sim_t:.4},{ms:.3},{subs},{broken},{clusters}\n"));
        let mut cam = staged.camera;
        cam.yaw += staged.orbit * sim_t;
        if !no_render {
            let mut rgba = renderer.render(&gpu, m, &w.solver.impactors, &cam);
            draw_overlay(&mut rgba, width, height, readout, &title, sim_t, f, frames, ms, &times, chunks, bonds, broken, clusters);
            match sink.as_mut() {
                Some(s) => s.write(&rgba).expect("ffmpeg write"),
                None => kept.push(rgba),
            }
        }
        if f % 30 == 0 || ms > 1000.0 / 60.0 && readout {
            eprintln!("frame {f:4} t {sim_t:6.3} s: step {ms:8.1} ms, {subs:5} substeps, {broken:6} broken bonds, {clusters:5} clusters; {:.0} s elapsed", wall_start.elapsed().as_secs_f64());
        }
    }
    let compute = wall_start.elapsed().as_secs_f64();
    if let Some(s) = sink.take() {
        s.finish().expect("ffmpeg");
    } else if !no_render {
        let mut s = FfmpegSink::new(&out, width, height, 60).expect("ffmpeg");
        for f in &kept {
            s.write(f).expect("ffmpeg write");
        }
        s.finish().expect("ffmpeg");
    }
    std::fs::write(out.with_extension("csv"), csv).expect("csv");
    w.solver.sync(&gpu);
    let o = w.observation();
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    let over = |b: f64| times.iter().filter(|&&t| t > b).count();
    println!(
        "{}: {frames} frames ({seconds} s simulated) in {compute:.0} s wall clock (rendering and encoding included); world creation {init_ms:.0} ms",
        out.display()
    );
    println!(
        "step_frame ms: first {:.1}, median {:.1}, p95 {:.1}, p99 {:.1}, worst {:.1} (frame {}); over 8 ms: {}, over 16.67 ms: {} of {frames}; sum {:.1} s",
        times[0],
        percentile(&sorted, 0.5),
        percentile(&sorted, 0.95),
        percentile(&sorted, 0.99),
        sorted[sorted.len() - 1],
        times.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i),
        over(8.0),
        over(1000.0 / 60.0),
        times.iter().sum::<f64>() / 1e3
    );
    println!("outcome: {} fragments, {} broken bonds, {} clusters; {chunks} chunks, {bonds} bonds", o.values["fragments"], o.values["broken_bonds"], w.solver.mirror.clusters.len());
}

#[allow(clippy::too_many_arguments)]
fn draw_overlay(rgba: &mut [u8], width: u32, height: u32, live: bool, title: &str, t: f64, f: usize, frames: usize, ms: f64, history: &[f64], chunks: usize, bonds: usize, broken: usize, clusters: usize) {
    let mut o = Overlay { rgba, width, height };
    let s = (height as i32 / 360).max(2);
    let pad = 6 * s;
    let white = [245, 245, 245];
    let lines: Vec<String> = if live {
        vec![
            title.into(),
            format!("T {t:5.2} S   FRAME {:3}/{frames}   {chunks} CHUNKS  {bonds} BONDS", f + 1),
            format!("BROKEN BONDS {broken}   CLUSTERS {clusters}"),
        ]
    } else {
        vec![
            title.into(),
            format!("T {t:5.2} S   FRAME {:3}/{frames}   {chunks} CHUNKS  {bonds} BONDS", f + 1),
            format!("BROKEN BONDS {broken}   CLUSTERS {clusters}   COMPUTE {ms:.0} MS/FRAME (NOT REAL TIME)"),
        ]
    };
    let wmax = lines.iter().map(|l| Overlay::text_width(l, s)).max().unwrap_or(0);
    o.rect(pad - 3 * s, pad - 3 * s, wmax + 6 * s, lines.len() as i32 * 10 * s + 4 * s, [10, 12, 16], 0.55);
    for (i, l) in lines.iter().enumerate() {
        o.text(pad, pad + i as i32 * 10 * s, s, white, l);
    }
    if !live {
        return;
    }
    // The measured step time, large, and a history of the last 180 frames with the 8 ms
    // and 16.67 ms lines.
    let color = if ms <= 8.0 {
        [90, 220, 110]
    } else if ms <= 1000.0 / 60.0 {
        [240, 200, 60]
    } else {
        [240, 80, 70]
    };
    let big = 2 * s;
    let label = format!("STEP {ms:6.2} MS");
    let bw = Overlay::text_width(&label, big);
    let (gw, gh) = (180 * s, 40 * s);
    let gx = width as i32 - pad - gw;
    let panel = 7 * big + 4 * s + 7 * s + 4 * s + gh;
    let py = height as i32 - pad - panel;
    let gy = py + 7 * big + 4 * s + 7 * s + 4 * s;
    o.rect(gx - 3 * s, py - 3 * s, gw + 6 * s, panel + 6 * s, [10, 12, 16], 0.55);
    o.text(gx + gw - bw, py, big, color, &label);
    o.text(gx, py + 7 * big + 4 * s, s, white, "SIMULATION STEP ONLY, LAST 3 S");
    let scale_ms = 40.0;
    let y_of = |v: f64| gy + gh - ((v / scale_ms).min(1.0) * gh as f64) as i32;
    let n = 180usize;
    let start = history.len().saturating_sub(n);
    let bar = (gw / n as i32).max(1);
    for (k, &v) in history[start..].iter().enumerate() {
        let c = if v <= 8.0 {
            [90, 220, 110]
        } else if v <= 1000.0 / 60.0 {
            [240, 200, 60]
        } else {
            [240, 80, 70]
        };
        let y = y_of(v);
        o.rect(gx + k as i32 * bar, y, bar, gy + gh - y, c, 0.9);
    }
    for (v, name) in [(8.0, "8"), (1000.0 / 60.0, "16.7")] {
        let y = y_of(v);
        o.rect(gx, y, gw, (s / 2).max(1), [255, 255, 255], 0.7);
        o.text(gx + gw - Overlay::text_width(name, s), y - 8 * s, s, white, name);
    }
}
