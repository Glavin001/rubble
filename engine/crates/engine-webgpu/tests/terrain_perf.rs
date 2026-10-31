//! The browser demo's event (`impact.html`: an 8 m stony impactor at 20 km/s
//! into dry sand, the absorbing ground), run natively from the strike to the
//! bake with each frame's GPU time, then the baked crater's shape. Measures
//! the terrain's frame cost across the impact timeline and what each region
//! or profile setting does to the crater (`engine/docs/deformable-terrain.md`,
//! "Performance"). Run alone, release:
//!
//! `RUBBLE_TERRAIN_GPU_MS=1 cargo test --release -p engine-webgpu --test
//! terrain_perf -- --ignored --nocapture`
//!
//! Settings (environment): `CPR` cells per transient radius (12), `MARGIN`
//! lateral margin in R (1.5), `BED` bed depth in R (unset: as deep as the
//! margin reaches), `HZ` frames per second (120), `PPC` particles per cell
//! axis (2), `E` Young's modulus in Pa (50 MPa), `UNTIL` the longest run in s
//! (40), `COURANT`, `GROWTH` (`speed_growth`), `GAME=1` the game profile's
//! settings (`TerrainProfile::Game`) before the others.

use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::WorldCapacity;
use engine_terrain::zmodel::point_source_z;
use engine_terrain::{substeps_for, CraterTarget, ImpactSource, SandMaterial, TerrainConfig};
use engine_webgpu::WgpuSim;
use glam::{DVec3, Quat, Vec3};

const G: f32 = 9.81;

/// One run's settings: `K=V` pairs (comma-separated) over the environment.
struct Settings(Vec<(String, String)>);

impl Settings {
    fn get(&self, k: &str) -> Option<f64> {
        self.0
            .iter()
            .rev()
            .find(|(key, _)| key == k)
            .map(|(_, v)| v.clone())
            .or_else(|| std::env::var(k).ok())
            .and_then(|v| v.parse().ok())
    }
}

fn gpu_heights(s: &WgpuSim, collider: u32, field: &HeightField) -> Vec<f32> {
    let base = engine_schema::convex::convex_params_base(s.collider(collider).params);
    let (nx, nz) = field.samples();
    let words = s.read_words(18, u64::from(base + 2) * 4, (nx * nz) as usize);
    words.into_iter().map(f32::from_bits).collect()
}

struct Frame {
    t: f32,
    substeps: u32,
    ms: f64,
}

/// `RUNS` (runs separated by `+`, each `K=V,K=V`) runs several settings in
/// one process, one after another.
#[test]
#[ignore = "benchmark and crater shape (long): run alone, release"]
fn demo_event_frame_cost_and_crater() {
    let runs = std::env::var("RUNS").unwrap_or_default();
    for run in runs.split('+') {
        let pairs = run
            .split(',')
            .filter_map(|kv| kv.split_once('='))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .collect();
        demo_event(&Settings(pairs));
    }
}

fn demo_event(set: &Settings) {
    let env = |k: &str| set.get(k);
    let mut cfg = TerrainConfig {
        max_particles: 3 << 19,
        max_blocks: 8192,
        max_parcels: 1 << 18,
        max_substeps: 1024,
        max_columns: 1 << 17,
        max_field_samples: 1 << 19,
        max_coupled: 64,
        cells_per_radius: 12.0,
        bed_depth: None,
        ..TerrainConfig::default()
    };
    if std::env::var("PROFILE").is_ok_and(|v| v == "game") || env("GAME") == Some(1.0) {
        cfg = cfg.with_profile(engine_terrain::TerrainProfile::Game);
    }
    if let Some(g) = env("GROWTH") {
        cfg.speed_growth = g as f32;
    }
    let mut sand = SandMaterial::default();
    let mut margin = 1.5f64;
    if let Some(c) = env("CPR") {
        cfg.cells_per_radius = c as f32;
    }
    if let Some(m) = env("MARGIN") {
        margin = m;
    }
    if let Some(b) = env("BED") {
        cfg.bed_depth = Some(b as f32);
    }
    if let Some(p) = env("PPC") {
        cfg.particles_per_cell_axis = p as u32;
    }
    if let Some(c) = env("COURANT") {
        cfg.courant = c as f32;
    }
    if let Some(e) = env("E") {
        sand.youngs_modulus = e as f32;
    }
    let hz = env("HZ").unwrap_or(120.0) as f32;
    let until = env("UNTIL").unwrap_or(40.0) as f32;
    let frame = 1.0 / hz;
    let mut s = WgpuSim::new(WorldCapacity {
        max_bodies: 64,
        max_colliders: 64,
        max_contacts: 4096,
        max_joints: 16,
        max_candidates: 8192,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .unwrap();
    s.terrain_enable(&cfg).unwrap();
    s.terrain_set_material(Some(&sand)).unwrap();
    let src = ImpactSource {
        impactor_radius: 8.0,
        impactor_density: 3000.0,
        velocity: DVec3::new(0.0, -20000.0, 0.0),
        target: CraterTarget::sand(&sand).unwrap(),
    };
    let g = f64::from(G);
    let crater = src.crater(g).unwrap();
    let model = src.field(g, point_source_z(src.target.ejecta.mu)).unwrap();
    let plan = crater.region(&cfg, Vec3::ZERO, margin, 0, 0).unwrap();
    let r = crater.radius as f32;
    // The field: 3 R each side, at the region's cell.
    let n = 2 * (3.0 * r / plan.dx).ceil() as u32 + 1;
    let field = HeightField::new(n, n, plan.dx, plan.dx, vec![0.0; (n * n) as usize]).unwrap();
    let mut arena = HullArena::new();
    let (_, collider) = s
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    // WARM=1: keep the GPU busy for half a second right before the strike
    // (a second device stepping a sand block): a game renders every frame,
    // so its GPU is clocked up when a strike lands, while this harness's GPU
    // idles through the setup and would time the strike frame cold.
    if env("WARM") == Some(1.0) {
        warm_gpu(0.5);
    }
    let wall = std::time::Instant::now();
    s.terrain_activate(
        &plan,
        collider,
        &field,
        Vec3::ZERO,
        sand.density,
        sand.friction_angle.tan(),
    )
    .unwrap();
    let activate_ms = s.terrain_last_gpu_ms();
    s.terrain_apply_source(&model, Vec3::ZERO).unwrap();
    let source_ms = s.terrain_last_gpu_ms();
    let settle = 0.01 * (g * crater.radius).sqrt() as f32;
    let mut speed = engine_terrain::plan::source_peak_speed(&model);
    let (mut t, mut airborne, mut quiet) = (0.0f32, 1u32, 0u32);
    let mut frames = Vec::new();
    let mut particles = 0;
    // The demo's clock: speed_growth × the last speed, gravity over a frame of lag,
    // 20 m/s of headroom; settled after 30 quiet frames past 2 s.
    while t < until && quiet < 30 {
        let (k, dt) = substeps_for(
            &cfg,
            frame,
            plan.dx,
            sand.p_wave_speed(),
            cfg.speed_growth * speed + 20.0 + G * frame,
        )
        .unwrap();
        // GAP_MS: idle the GPU before each frame (how much its clock state
        // moves the numbers: an idle GPU clocks down).
        if let Some(gap) = env("GAP_MS") {
            std::thread::sleep(std::time::Duration::from_secs_f64(gap * 1e-3));
        }
        let w = std::time::Instant::now();
        let rep = s.terrain_step(k, dt, -G).unwrap();
        let ms = s
            .terrain_last_gpu_ms()
            .unwrap_or_else(|| w.elapsed().as_secs_f64() * 1e3);
        frames.push(Frame { t, substeps: k, ms });
        particles = rep.particles;
        speed = rep.max_speed;
        airborne = rep.airborne;
        t += frame;
        quiet = if t > 2.0 && speed < settle && airborne == 0 {
            quiet + 1
        } else {
            0
        };
    }
    let lowered = gpu_heights(&s, collider, &field);
    s.terrain_bake().unwrap();
    let baked = gpu_heights(&s, collider, &field);
    let wall = wall.elapsed().as_secs_f64();
    eprintln!(
        "cpr {} margin {margin} bed {:?} ppc {} courant {} growth {} E {:.1} MPa at {hz} Hz: dx {:.2} m, region {:?} blocks, {particles} particles; {} at t = {t:.2} s (top {speed:.3} m/s, tolerance {settle:.3}, airborne {airborne}); {wall:.0} s wall",
        cfg.cells_per_radius,
        cfg.bed_depth,
        cfg.particles_per_cell_axis,
        cfg.courant,
        cfg.speed_growth,
        sand.youngs_modulus * 1e-6,
        plan.dx,
        plan.blocks,
        if quiet >= 30 { "settled" } else { "not settled" }
    );
    let fmt = |v: Option<f64>| v.map_or("-".into(), |v| format!("{v:.2} ms"));
    eprintln!(
        "  activation {}, source {}",
        fmt(activate_ms),
        fmt(source_ms)
    );
    let spans = [
        (0.0, frame * 0.5),
        (frame * 0.5, 0.1),
        (0.1, 0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 6.0),
        (6.0, 12.0),
        (12.0, 1e9),
    ];
    let mut all: Vec<f64> = frames.iter().map(|f| f.ms).collect();
    for (a, b) in spans {
        let span: Vec<&Frame> = frames.iter().filter(|f| f.t >= a && f.t < b).collect();
        if span.is_empty() {
            continue;
        }
        let mean = span.iter().map(|f| f.ms).sum::<f64>() / span.len() as f64;
        let max = span.iter().map(|f| f.ms).fold(0.0, f64::max);
        let (k0, k1) = (
            span.iter().map(|f| f.substeps).min().unwrap(),
            span.iter().map(|f| f.substeps).max().unwrap(),
        );
        eprintln!(
            "  {a:6.2}-{:<6.2} s: {:4} frames, {k0}-{k1} substeps, mean {mean:6.2} ms, max {max:6.2} ms",
            b.min(t),
            span.len()
        );
    }
    all.sort_by(f64::total_cmp);
    let mean = all.iter().sum::<f64>() / all.len().max(1) as f64;
    eprintln!(
        "  all {} frames: mean {mean:.2} ms, median {:.2}, 99% {:.2}, worst {:.2} ms",
        all.len(),
        all[all.len() / 2],
        all[all.len() * 99 / 100],
        all[all.len() - 1]
    );
    // The crater: height change against the plain, by ring of one sample.
    let c = (n / 2) as i64;
    let rings = c as usize;
    let mut sum = vec![0.0f64; rings + 1];
    let mut cnt = vec![0usize; rings + 1];
    let (mut deepest, mut volume_in) = (0.0f32, Vec::new());
    for j in 0..n as i64 {
        for i in 0..n as i64 {
            let h = baked[(j * n as i64 + i) as usize];
            let d = (((i - c).pow(2) + (j - c).pow(2)) as f64).sqrt();
            let k = d.round() as usize;
            if k <= rings {
                sum[k] += f64::from(h);
                cnt[k] += 1;
            }
            deepest = deepest.min(h);
            volume_in.push((d, h));
        }
    }
    let mean: Vec<f64> = (0..=rings).map(|k| sum[k] / cnt[k].max(1) as f64).collect();
    let apparent = (1..=rings).find(|&k| mean[k] >= 0.0).unwrap_or(rings);
    let (rim_k, rim_h) = (apparent..=rings)
        .map(|k| (k, mean[k]))
        .fold((apparent, f64::MIN), |a, b| if b.1 > a.1 { b } else { a });
    let area = f64::from(plan.dx * plan.dx);
    let cavity: f64 = volume_in
        .iter()
        .filter(|(d, _)| *d < apparent as f64)
        .map(|(_, h)| -f64::from(*h) * area)
        .sum();
    let dx = f64::from(plan.dx);
    let rr = f64::from(r);
    eprintln!(
        "  crater: centre {:.1} m deep, deepest {:.1} m, apparent radius {:.0} m ({:.2} R), rim {:.1} m high at {:.0} m ({:.2} R), cavity {:.3e} m³ ({:.2} × π-scaling's {:.3e}); window floor {:.1} m",
        -baked[(c * n as i64 + c) as usize],
        -deepest,
        apparent as f64 * dx,
        apparent as f64 * dx / rr,
        rim_h,
        rim_k as f64 * dx,
        rim_k as f64 * dx / rr,
        cavity,
        cavity / crater.volume,
        crater.volume,
        -lowered[(c * n as i64 + c) as usize],
    );
    let prof: Vec<String> = (0..=rings.min(40))
        .step_by(2)
        .map(|k| format!("{:.1}", mean[k]))
        .collect();
    eprintln!(
        "  profile (every 2 samples of {:.1} m): {}",
        plan.dx,
        prof.join(" ")
    );
}

/// The showcase's barrage (`showcase.html`, scene `drive`): its 1 m stony
/// meteor at 20 km/s into dry sand, `STRIKES` (11) of them opened `EVERY` s
/// apart (1.6; 0 opens them all in one frame) in areas of their own `SPACING` m apart (150) on one field,
/// none baked, stepped at `HZ` (120) for `UNTIL` s (21.5) on the showcase's
/// clock; each frame's GPU time. `CPR` (6), `MARGIN` (1.0), `GAME=1` the game
/// profile's settings; `FORCE_K` a floor on each frame's substeps,
/// `STATE_HASH=1` a hash of the end state and the substep counts (to compare
/// kernels on the same run). With `RUBBLE_TERRAIN_TIMING=1` each submission prints
/// its kernels' times (each kernel timed in a pass of its own). Run alone:
///
/// `RUBBLE_TERRAIN_GPU_MS=1 cargo test --release -p engine-webgpu --test
/// terrain_perf barrage -- --ignored --nocapture`
#[test]
#[ignore = "benchmark: run alone, release"]
fn barrage_frame_cost() {
    use engine_terrain::plan::source_peak_speed;
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
    let mut cfg = TerrainConfig {
        max_regions: 16,
        max_particles: 1 << 20,
        max_blocks: 8192,
        max_parcels: 1 << 17,
        max_substeps: 1024,
        max_columns: 1 << 16,
        max_field_samples: 1 << 19,
        max_coupled: 256,
        ..TerrainConfig::default()
    };
    if env("GAME") == Some(1.0) {
        cfg = cfg.with_profile(engine_terrain::TerrainProfile::Game);
    }
    // The opt-in options (engine/docs/deformable-terrain.md, "Area clocks and
    // block sleep"): AREA_CLOCKS=1, SLEEP=<rest speed m/s>.
    cfg.area_clocks = env("AREA_CLOCKS") == Some(1.0);
    cfg.sleep_speed = env("SLEEP").unwrap_or(0.0) as f32;
    cfg.cells_per_radius = env("CPR").unwrap_or(6.0) as f32;
    let margin = env("MARGIN").unwrap_or(1.0);
    let strikes = env("STRIKES").unwrap_or(11.0) as u32;
    let every = env("EVERY").unwrap_or(1.6) as f32;
    let spacing = env("SPACING").unwrap_or(150.0) as f32;
    let hz = env("HZ").unwrap_or(120.0) as f32;
    let until = env("UNTIL").unwrap_or(21.5) as f32;
    let frame = 1.0 / hz;
    let sand = SandMaterial::default();
    let mut s = WgpuSim::new(WorldCapacity {
        max_bodies: 64,
        max_colliders: 64,
        max_contacts: 4096,
        max_joints: 16,
        max_candidates: 8192,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .unwrap();
    s.terrain_enable(&cfg).unwrap();
    s.terrain_set_material(Some(&sand)).unwrap();
    let src = ImpactSource {
        impactor_radius: 1.0,
        impactor_density: 3000.0,
        velocity: DVec3::new(0.0, -20000.0, 0.0),
        target: CraterTarget::sand(&sand).unwrap(),
    };
    let g = f64::from(G);
    let crater = src.crater(g).unwrap();
    let model = src.field(g, point_source_z(src.target.ejecta.mu)).unwrap();
    let r = crater.radius as f32;
    let dx = crater.asked_cell(&cfg);
    let span = (strikes.max(1) - 1) as f32 * spacing;
    let nx = 2 * ((0.5 * span + 3.0 * r) / dx).ceil() as u32 + 1;
    let nz = 2 * (3.0 * r / dx).ceil() as u32 + 1;
    let field = HeightField::new(nx, nz, dx, dx, vec![0.0; (nx * nz) as usize]).unwrap();
    let mut arena = HullArena::new();
    let (_, collider) = s
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    let peak = source_peak_speed(&model);
    let (mut t, mut next, mut opened) = (0.0f32, 0.0f32, 0u32);
    // Each area slot's speed bound basis (the World's clock per area).
    let mut speed = [0.0f32; 16];
    let mut frames: Vec<(f32, u32, f64, u32, bool)> = Vec::new();
    let wall = std::time::Instant::now();
    while t < until {
        let mut strike = false;
        // `while`: with EVERY=0 every strike opens in the first frame.
        while opened < strikes && t >= next {
            let at = Vec3::new(-0.5 * span + opened as f32 * spacing, 0.0, 0.0);
            let cell = s.terrain_plan_ref().unwrap().event_cell(dx).unwrap();
            let plan = crater.region_with_cell(&cfg, at, margin, cell).unwrap();
            let o = s
                .terrain_open_area(
                    &plan,
                    collider,
                    &field,
                    Vec3::ZERO,
                    sand.density,
                    sand.friction_angle.tan(),
                )
                .unwrap();
            s.terrain_apply_source_to(o.area, &model, at).unwrap();
            let k = o.area as usize;
            speed[k] = speed[k].max(peak);
            opened += 1;
            next += every;
            strike = true;
        }
        // Each open area's CFL count from its own bound; the frame takes the
        // finest (and on the area clocks each area its own).
        let open: Vec<usize> = s.terrain_areas().iter().map(|a| a.slot as usize).collect();
        let mut each = [0u32; 16];
        let mut k = 1;
        for &a in &open {
            let (n, _) = substeps_for(
                &cfg,
                frame,
                dx,
                sand.p_wave_speed(),
                cfg.speed_growth * speed[a] + 20.0 + G * frame,
            )
            .unwrap();
            each[a] = n;
            k = k.max(n);
        }
        // FORCE_K=n: at least n substeps every frame (each frame then costs
        // what a strike frame's substeps cost at that particle count).
        if let Some(f) = env("FORCE_K") {
            k = k.max(f as u32);
        }
        s.terrain_set_area_substeps(each).unwrap();
        let dt = frame / k as f32;
        let w = std::time::Instant::now();
        let rep = s.terrain_step(k, dt, -G).unwrap();
        let ms = s
            .terrain_last_gpu_ms()
            .unwrap_or_else(|| w.elapsed().as_secs_f64() * 1e3);
        frames.push((t, k, ms, rep.particles, strike));
        for &a in &open {
            speed[a] = rep.area_speed[a];
        }
        // Block sleep: how much of the sand is asleep, each 4 s.
        if (t / 4.0).floor() != ((t + frame) / 4.0).floor() {
            let ps = s.terrain_particles();
            let live = ps
                .iter()
                .filter(|p| p.f[2][3].to_bits() & engine_terrain::layout::MPM_PARTICLE_DEAD == 0);
            let (n, asleep) = live.fold((0, 0), |(n, a), p| {
                let z = p.f[2][3].to_bits() & engine_terrain::layout::MPM_PARTICLE_ASLEEP != 0;
                (n + 1, a + usize::from(z))
            });
            eprintln!(
                "  t {:5.2} s: {asleep} of {n} particles asleep, {} blocks",
                t + frame,
                rep.blocks
            );
        }
        t += frame;
    }
    // STATE_HASH=1: the run's end state (every particle's words) and each
    // frame's substep count hashed, to compare kernels on the same run.
    if env("STATE_HASH") == Some(1.0) {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |w: u32| {
            h ^= u64::from(w);
            h = h.wrapping_mul(0x0100_0000_01b3);
        };
        for p in s.terrain_particles() {
            bytemuck::cast_slice::<_, u32>(&[p])
                .iter()
                .for_each(|&w| mix(w));
        }
        frames.iter().for_each(|f| mix(f.1));
        eprintln!("  state hash {h:016x}");
    }
    let pct = |v: &mut Vec<f64>, p: f64| {
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * p) as usize]
    };
    let mut all: Vec<f64> = frames.iter().map(|f| f.2).collect();
    let mut steady: Vec<f64> = frames.iter().filter(|f| !f.4).map(|f| f.2).collect();
    let worst = frames.iter().fold((0.0, 0.0, 0, 0), |a, f| {
        if f.2 > a.1 {
            (f.0, f.2, f.1, f.3)
        } else {
            a
        }
    });
    eprintln!(
        "barrage: {strikes} strikes every {every} s, {} profile, area clocks {}, sleep {} m/s, cpr {}, margin {margin}, dx {dx:.2} m, {hz} Hz, {} frames in {:.0} s wall; {} particles at the end",
        if env("GAME") == Some(1.0) { "game" } else { "physical" },
        cfg.area_clocks,
        cfg.sleep_speed,
        cfg.cells_per_radius,
        frames.len(),
        wall.elapsed().as_secs_f64(),
        frames.last().map_or(0, |f| f.3),
    );
    eprintln!(
        "  all frames: median {:.2} ms, p95 {:.2}, worst {:.2} ms (t {:.2} s, {} substeps, {} particles); non-strike: median {:.2}, p95 {:.2}, worst {:.2}",
        pct(&mut all, 0.5),
        pct(&mut all, 0.95),
        worst.1,
        worst.0,
        worst.2,
        worst.3,
        pct(&mut steady, 0.5),
        pct(&mut steady, 0.95),
        pct(&mut steady, 1.0),
    );
    for f in frames.iter().filter(|f| f.4) {
        eprintln!(
            "  strike at {:5.2} s: {:3} substeps, {:6.2} ms, {} particles",
            f.0, f.1, f.2, f.3
        );
    }
    let mut sec = 0u32;
    while (sec as f32) < until {
        let mut v: Vec<f64> = frames
            .iter()
            .filter(|f| !f.4 && f.0 >= sec as f32 && f.0 < sec as f32 + 1.0)
            .map(|f| f.2)
            .collect();
        if !v.is_empty() {
            let p = frames
                .iter()
                .filter(|f| f.0 < sec as f32 + 1.0)
                .map(|f| f.3)
                .max()
                .unwrap_or(0);
            let k = frames
                .iter()
                .filter(|f| !f.4 && f.0 >= sec as f32 && f.0 < sec as f32 + 1.0)
                .map(|f| f.1)
                .max()
                .unwrap_or(0);
            eprintln!(
                "  {sec:2}-{:2} s: {p:7} particles, ≤{k} substeps, median {:.2} ms, max {:.2}",
                sec + 1,
                pct(&mut v, 0.5),
                pct(&mut v, 1.0)
            );
        }
        sec += 1;
    }
}

/// Steps a sand block on a device of its own for `seconds` of wall time.
fn warm_gpu(seconds: f64) {
    use engine_terrain::layout::MpmParticle;
    use engine_terrain::RegionPlan;
    let mut w = WgpuSim::new(WorldCapacity {
        max_bodies: 64,
        max_colliders: 64,
        max_contacts: 4096,
        max_joints: 16,
        max_candidates: 8192,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .unwrap();
    let cfg = TerrainConfig {
        max_particles: 1 << 17,
        max_blocks: 4096,
        ..TerrainConfig::default()
    };
    w.terrain_enable(&cfg).unwrap();
    let sand = SandMaterial::default();
    w.terrain_set_material(Some(&sand)).unwrap();
    let plan = RegionPlan::spanning(&cfg, Vec3::splat(-1.0), Vec3::splat(1.0), 0.05, 0, 0).unwrap();
    let h = 0.025f32;
    let mut ps = Vec::new();
    for k in 0..40 {
        for j in 0..40 {
            for i in 0..40 {
                let x = Vec3::new(-0.5, -0.5, -0.5)
                    + (Vec3::new(i as f32, j as f32, k as f32) + 0.5) * h;
                ps.push(MpmParticle::at_rest(x.to_array(), h * h * h, sand.density));
            }
        }
    }
    w.terrain_load(&plan, &ps, None, 0.5).unwrap();
    let t = std::time::Instant::now();
    while t.elapsed().as_secs_f64() < seconds {
        let _ = w.terrain_step(4, 1.0e-4, 0.0);
    }
}
