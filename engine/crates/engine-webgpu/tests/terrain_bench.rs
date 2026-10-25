//! Terrain MPM frame cost (run alone, release: `cargo test --release -p
//! engine-webgpu --test terrain_bench -- --ignored --nocapture`).

use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::WorldCapacity;
use engine_terrain::layout::MpmParticle;
use engine_terrain::{substeps_for, RegionPlan, SandMaterial, TerrainConfig};
use engine_webgpu::WgpuSim;
use glam::{Quat, Vec3};

#[test]
#[ignore = "benchmark"]
fn terrain_frame_cost() {
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
    let cfg = TerrainConfig {
        max_particles: 1 << 18,
        max_blocks: 1 << 14,
        ..TerrainConfig::default()
    };
    s.terrain_enable(&cfg).unwrap();
    let sand = SandMaterial::default();
    s.terrain_set_material(Some(&sand)).unwrap();
    // The column is 16 cells across and tall whatever dx (BENCH_DX, default 5 cm).
    let dx: f32 = std::env::var("BENCH_DX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.05);
    let scale = dx / 0.05;
    let field =
        HeightField::new(65, 65, 0.0625 * scale, 0.0625 * scale, vec![0.0; 65 * 65]).unwrap();
    let mut arena = HullArena::new();
    let (_, bed) = s
        .add_height_field(
            &mut arena,
            &field,
            Vec3::new(0.0, -0.6 * scale, 0.0),
            Quat::IDENTITY,
            0,
        )
        .unwrap();
    let plan = RegionPlan::spanning(
        &cfg,
        Vec3::new(-1.9, -0.7, -1.9) * scale,
        Vec3::new(1.9, 0.4, 1.9) * scale,
        dx,
        0,
        0,
    )
    .unwrap();
    let h = plan.dx / 2.0;
    let mut ps = Vec::new();
    let (w, tall) = (0.8f32 * scale, 0.8f32 * scale);
    for k in 0..(w / h) as i32 {
        for j in 0..(tall / h) as i32 {
            for i in 0..(w / h) as i32 {
                let x = Vec3::new(-w / 2.0, -0.6 * scale, -w / 2.0)
                    + (Vec3::new(i as f32, j as f32, k as f32) + 0.5) * h;
                ps.push(MpmParticle::at_rest(x.to_array(), h * h * h, sand.density));
            }
        }
    }
    if std::env::var("BENCH_EMPTY").is_ok() {
        ps.clear();
    }
    s.terrain_load(&plan, &ps, Some(bed), sand.friction_angle.tan())
        .unwrap();
    let (n, dt) = substeps_for(&cfg, 1.0 / 60.0, plan.dx, sand.p_wave_speed(), 5.0).unwrap();
    eprintln!(
        "{} particles, {n} substeps of {dt:.2e} s per frame",
        ps.len()
    );
    // Warm-up, then timed frames at the frame's substeps and at 1 (fixed cost).
    for _ in 0..5 {
        s.terrain_step(n, dt, -9.81).unwrap();
    }
    // State hash after the warm-up: optimisations must leave it unchanged.
    let bytes = bytemuck::cast_slice::<MpmParticle, u8>(&s.terrain_particles()).to_vec();
    let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    });
    eprintln!("state hash after warm-up: {hash:016x}");
    if std::env::var("WG_SKIP").is_ok() {
        let t = std::time::Instant::now();
        for _ in 0..5 {
            let _ = s.terrain_step(n, dt, -9.81);
        }
        eprintln!(
            "skip {:?}: {:.2} ms per frame",
            std::env::var("WG_SKIP").unwrap(),
            t.elapsed().as_secs_f64() * 1e3 / 5.0
        );
        return;
    }
    for subs in [n, n / 4, 1] {
        let t = std::time::Instant::now();
        let frames = 10;
        let mut blocks = 0;
        for _ in 0..frames {
            blocks = s.terrain_step(subs, dt, -9.81).unwrap().blocks;
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / frames as f64;
        eprintln!(
            "{subs:4} substeps: {ms:8.2} ms per call, {:.3} ms per substep, {blocks} blocks",
            ms / subs as f64
        );
    }
}

/// How much of the region is at rest: per block of the region's lattice (a
/// particle's base node's block), its particles' top speed; the share of
/// particles in blocks below each speed, alone and with the blocks one and
/// two blocks around them (the neighbours a sleeping block needs at rest).
fn rest_histogram(ps: &[MpmParticle], plan: &RegionPlan) {
    let b = plan.blocks;
    let (bx, by, bz) = (b.x as usize, b.y as usize, b.z as usize);
    let mut top = vec![-1.0f32; bx * by * bz];
    let mut count = vec![0usize; bx * by * bz];
    let block = plan.dx * 4.0;
    let at = |x: [f32; 3]| -> Option<usize> {
        let r = (Vec3::from(x) - plan.origin) / plan.dx - 0.5;
        let k = (r.floor() / 4.0).floor();
        let _ = block;
        (k.cmpge(Vec3::ZERO).all() && k.x < bx as f32 && k.y < by as f32 && k.z < bz as f32)
            .then(|| k.x as usize + bx * (k.y as usize + by * k.z as usize))
    };
    let mut live = 0usize;
    for p in ps {
        if p.f[2][3].to_bits() & 1 != 0 {
            continue;
        }
        live += 1;
        if let Some(c) = at([p.xm[0], p.xm[1], p.xm[2]]) {
            let v = Vec3::new(p.vj[0], p.vj[1], p.vj[2]).length();
            top[c] = top[c].max(v);
            count[c] += 1;
        }
    }
    let dilate = |t: &[f32]| -> Vec<f32> {
        let mut o = t.to_vec();
        for z in 0..bz {
            for y in 0..by {
                for x in 0..bx {
                    let mut m = t[x + bx * (y + by * z)];
                    for dz in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let (u, v, w) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                                if u >= 0
                                    && v >= 0
                                    && w >= 0
                                    && (u as usize) < bx
                                    && (v as usize) < by
                                    && (w as usize) < bz
                                {
                                    m = m.max(t[u as usize + bx * (v as usize + by * w as usize)]);
                                }
                            }
                        }
                    }
                    o[x + bx * (y + by * z)] = m;
                }
            }
        }
        o
    };
    let d1 = dilate(&top);
    let d2 = dilate(&d1);
    let share = |t: &[f32], v: f32| -> f64 {
        let n: usize = (0..t.len())
            .filter(|&c| count[c] > 0 && t[c] < v)
            .map(|c| count[c])
            .sum();
        100.0 * n as f64 / live.max(1) as f64
    };
    let mut line = String::new();
    for v in [1e-3f32, 3e-3, 1e-2, 3e-2, 0.1, 0.38] {
        line += &format!(
            " <{v}: {:.0}/{:.0}/{:.0}%",
            share(&top, v),
            share(&d1, v),
            share(&d2, v)
        );
    }
    eprintln!("    at rest (block / +1 / +2 blocks):{line}");
    if std::env::var("PROFILE_LAYERS").is_ok() {
        // Per block layer (y), the median and top block speeds, and per
        // horizontal ring (blocks from the centre column) below the top layer.
        for y in 0..by {
            let mut v: Vec<f32> = (0..bx * bz)
                .map(|c| (c % bx) + bx * (y + by * (c / bx)))
                .filter(|&c| count[c] > 0)
                .map(|c| top[c])
                .collect();
            if v.is_empty() {
                continue;
            }
            v.sort_by(f32::total_cmp);
            eprintln!(
                "      layer {y}: {} blocks, median {:.2e}, 90% {:.2e}, top {:.2e} m/s",
                v.len(),
                v[v.len() / 2],
                v[v.len() * 9 / 10],
                v[v.len() - 1]
            );
        }
    }
}

/// The browser demo's event (`impact.html`: an 8 m stony impactor at 20 km/s
/// into dry sand, margin 1.5 R at 12 cells/R, the absorbing ground), stepped
/// natively frame by frame as the demo's clock schedules it, with a per-kernel
/// GPU timing of the frames at `PROFILE_AT` (seconds, comma-separated). Run
/// alone, release, with timestamp queries:
/// `RUBBLE_GPU_TIMING=1 cargo test --release -p engine-webgpu --test
/// terrain_bench terrain_demo_profile -- --ignored --nocapture`.
/// `BENCH_MARGIN`, `BENCH_CPR`, `BENCH_BED` (bed depth in R), `BENCH_HZ`
/// (frames per second, 60) and `PROFILE_UNTIL` (s) change the event.
#[test]
#[ignore = "benchmark"]
fn terrain_demo_profile() {
    use engine_terrain::zmodel::point_source_z;
    use engine_terrain::{CraterTarget, ImpactSource};
    let env = |k: &str, d: f64| -> f64 {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let margin = env("BENCH_MARGIN", 1.5);
    let cpr = env("BENCH_CPR", 12.0) as f32;
    let until = env("PROFILE_UNTIL", 3.0) as f32;
    let at: Vec<f32> = std::env::var("PROFILE_AT")
        .unwrap_or_else(|_| "0.02,0.25,0.5,1,2,3".into())
        .split(',')
        .filter_map(|v| v.parse().ok())
        .collect();
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
    // The demo's limits (impact.html).
    let cfg = TerrainConfig {
        max_particles: 3 << 19,
        max_blocks: 8192,
        max_parcels: 1 << 18,
        max_substeps: 1024,
        max_columns: 1 << 17,
        max_field_samples: 1 << 19,
        max_coupled: 64,
        cells_per_radius: cpr,
        bed_depth: std::env::var("BENCH_BED").ok().and_then(|v| v.parse().ok()),
        ..TerrainConfig::default()
    };
    s.terrain_enable(&cfg).unwrap();
    let sand = SandMaterial::default();
    s.terrain_set_material(Some(&sand)).unwrap();
    let src = ImpactSource {
        impactor_radius: 8.0,
        impactor_density: 3000.0,
        velocity: glam::DVec3::new(0.0, -20000.0, 0.0),
        target: CraterTarget::sand(&sand).unwrap(),
    };
    let g = 9.81f64;
    let crater = src.crater(g).unwrap();
    let model = src.field(g, point_source_z(src.target.ejecta.mu)).unwrap();
    let plan = crater.region(&cfg, Vec3::ZERO, margin, 0, 0).unwrap();
    let n = 2 * (3.0 * crater.radius as f32 / plan.dx).ceil() as u32 + 1;
    let field = HeightField::new(n, n, plan.dx, plan.dx, vec![0.0; (n * n) as usize]).unwrap();
    let mut arena = HullArena::new();
    let (_, collider) = s
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    s.terrain_activate(
        &plan,
        collider,
        &field,
        Vec3::ZERO,
        sand.density,
        sand.friction_angle.tan(),
    )
    .unwrap();
    s.terrain_apply_source(&model, Vec3::ZERO).unwrap();
    let frame = 1.0 / env("BENCH_HZ", 60.0) as f32;
    let mut speed = engine_terrain::plan::source_peak_speed(&model);
    let mut t = 0.0f32;
    let mut next = 0usize;
    let timed = std::env::var("RUBBLE_GPU_TIMING").is_ok_and(|v| v == "1");
    eprintln!(
        "region {:?} blocks, dx {:.2} m, R {:.0} m",
        plan.blocks, plan.dx, crater.radius
    );
    let wall = std::time::Instant::now();
    while t < until {
        let (k, dt) = substeps_for(
            &cfg,
            frame,
            plan.dx,
            sand.p_wave_speed(),
            1.5 * speed + 20.0 + 9.81 * frame,
        )
        .unwrap();
        let profile = next < at.len() && t + 0.5 * frame >= at[next];
        if profile {
            next += 1;
        }
        let rep = if profile && timed {
            let (rep, times) = s.terrain_step_timed(k, dt, -9.81).unwrap();
            let total: f64 = times.iter().map(|(_, ms)| ms).sum();
            eprintln!(
                "t {t:.2} s: {k} substeps, {} blocks, {} particles, max speed {:.2} m/s: GPU {total:.2} ms ({:.3} per substep)",
                rep.blocks,
                rep.particles,
                rep.max_speed,
                total / f64::from(k)
            );
            for (name, ms) in &times {
                eprintln!(
                    "    {name:20} {:7.3} ms per substep ({:4.1}%)",
                    ms / f64::from(k),
                    100.0 * ms / total
                );
            }
            rep
        } else {
            let w = std::time::Instant::now();
            let rep = s.terrain_step(k, dt, -9.81).unwrap();
            if profile {
                eprintln!(
                    "t {t:.2} s: {k} substeps, {} blocks, {} particles, max speed {:.2} m/s: wall {:.2} ms",
                    rep.blocks,
                    rep.particles,
                    rep.max_speed,
                    w.elapsed().as_secs_f64() * 1e3
                );
            }
            rep
        };
        if profile && std::env::var("PROFILE_REST").is_ok() {
            rest_histogram(&s.terrain_particles(), &plan);
        }
        speed = rep.max_speed;
        t += frame;
    }
    eprintln!(
        "{until} s simulated in {:.1} s wall",
        wall.elapsed().as_secs_f64()
    );
}
