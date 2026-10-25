//! A km/s impact into dry sand (T4): the Z-model source on the GPU matches the
//! CPU field, and the crater the MPM then digs, baked into the heightfield, is
//! compared with gravity-regime π-scaling (Holsapple 1993). The crater's shape
//! is never prescribed: only the initial velocity comes from scaling.

use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::WorldCapacity;
use engine_terrain::zmodel::point_source_z;
use engine_terrain::{substeps_for, CraterTarget, ImpactSource, SandMaterial, TerrainConfig};
use engine_webgpu::WgpuSim;
use glam::{DVec3, Mat3, Quat, Vec3};

const G: f64 = 9.81;
const FRAME: f32 = 1.0 / 60.0;

fn sim() -> WgpuSim {
    WgpuSim::new(WorldCapacity {
        max_bodies: 64,
        max_colliders: 64,
        max_contacts: 4096,
        max_joints: 16,
        max_candidates: 8192,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    })
    .expect("wgpu adapter")
}

fn impact(sand: &SandMaterial) -> ImpactSource {
    ImpactSource {
        impactor_radius: 0.005,
        impactor_density: 2700.0,
        velocity: DVec3::new(0.0, -2000.0, 0.0),
        target: CraterTarget::sand(sand).unwrap(),
    }
}

fn gpu_heights(s: &WgpuSim, collider: u32, field: &HeightField) -> Vec<f32> {
    let base = engine_schema::convex::convex_params_base(s.collider(collider).params);
    let (nx, nz) = field.samples();
    let words = s.read_words(18, u64::from(base + 2) * 4, (nx * nz) as usize);
    words.into_iter().map(f32::from_bits).collect()
}

/// Diagnostics of the region's state (CRATER_PROBE=k, every k frames): kinetic
/// energy, the elastic volumetric energy ½K(ln J)²·V of the elastic F, the
/// most compressed particle, the volume compressed past J < 0.9 and 0.7, the
/// plastic compaction (log Jp < 0 or `c[0][3]` when a cap runs), and the
/// surface on the axis (the highest particle within one cell of it).
thread_local! {
    static COLUMNS: std::cell::RefCell<Option<std::collections::HashMap<(i32, i32), f64>>> =
        const { std::cell::RefCell::new(None) };
}

fn probe(s: &WgpuSim, sand: &SandMaterial, dx: f32, t: f32) {
    let k = f64::from(sand.lambda() + 2.0 * sand.mu() / 3.0);
    let (mut ke, mut pe, mut jmin, mut v09, mut v07, mut vol, mut cap) =
        (0.0f64, 0.0f64, f64::MAX, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut floor = f32::MIN;
    let (mut mom, mut ke_down) = (glam::DVec3::ZERO, 0.0f64);
    // Surface (highest particle) in rings of width 2dx at r = 0, 0.04, ... m,
    // and the separated volume Σ V0·(exp(log Jp) − 1) (gaps a particle gained
    // at the tension tip, which compression consumes first).
    let mut rings = [f32::MIN; 8];
    let (mut gap, mut gap_floor) = (0.0f64, 0.0f64);
    let mut jmin_at = Vec3::ZERO;
    // Virtual bake: rest volume per dx-wide column of live particles, against
    // the first probe's; the cavity is the summed deficit (material in flight
    // counts as gone), the uplift the summed excess.
    let mut cols: std::collections::HashMap<(i32, i32), f64> = std::collections::HashMap::new();
    for p in s.terrain_particles() {
        if p.xm[3] > 0.0 && p.f[2][3].to_bits() & 1 == 0 {
            let key = ((p.xm[0] / dx).floor() as i32, (p.xm[2] / dx).floor() as i32);
            *cols.entry(key).or_default() += f64::from(p.f[0][3]);
        }
    }
    COLUMNS.with(|c| {
        let mut c = c.borrow_mut();
        let base = c.get_or_insert_with(|| cols.clone());
        let (mut deficit, mut excess) = (0.0f64, 0.0f64);
        for (key, v0) in base.iter() {
            let d = cols.get(key).copied().unwrap_or(0.0) - v0;
            if d < 0.0 {
                deficit -= d;
            } else {
                excess += d;
            }
        }
        for (key, v) in cols.iter() {
            if !base.contains_key(key) {
                excess += v;
            }
        }
        eprintln!("probe t {t:.4}: virtual-bake cavity {deficit:.3e} m³, uplift {excess:.3e} m³");
    });
    for p in s.terrain_particles() {
        if p.xm[3] <= 0.0 || p.f[2][3].to_bits() & 1 != 0 {
            continue;
        }
        let v = Vec3::from_slice(&p.vj[..3]);
        ke += 0.5 * f64::from(p.xm[3]) * f64::from(v.length_squared());
        mom += v.as_dvec3() * f64::from(p.xm[3]);
        if v.y < 0.0 {
            ke_down += 0.5 * f64::from(p.xm[3]) * f64::from(v.length_squared());
        }
        let f = glam::DMat3::from_cols_array(&[
            f64::from(p.f[0][0]),
            f64::from(p.f[1][0]),
            f64::from(p.f[2][0]),
            f64::from(p.f[0][1]),
            f64::from(p.f[1][1]),
            f64::from(p.f[2][1]),
            f64::from(p.f[0][2]),
            f64::from(p.f[1][2]),
            f64::from(p.f[2][2]),
        ]);
        let j = f.determinant();
        let v0 = f64::from(p.f[0][3]);
        vol += v0;
        if j > 0.0 {
            pe += 0.5 * k * j.ln().powi(2) * v0;
        }
        if j < jmin {
            jmin = j;
            jmin_at = Vec3::from_slice(&p.xm[..3]);
        }
        if j < 0.9 {
            v09 += v0;
        }
        if j < 0.7 {
            v07 += v0;
        }
        cap += f64::from(p.c[0][3].min(0.0)) * v0;
        let x = Vec3::from_slice(&p.xm[..3]);
        if x.x * x.x + x.z * x.z < dx * dx {
            floor = floor.max(x.y);
        }
        let r = (x.x * x.x + x.z * x.z).sqrt();
        for (k, h) in rings.iter_mut().enumerate() {
            if (r - 0.04 * k as f32).abs() < dx {
                *h = h.max(x.y);
            }
        }
        let g = (f64::from(p.vj[3]).exp() - 1.0) * v0;
        gap += g;
        if r < 0.15 && x.y < -0.02 {
            gap_floor += g;
        }
    }
    if std::env::var("CRATER_COLUMN").is_ok() {
        // The axis column (r < 2dx) in 1 cm bins: particles, mean v_y, mean
        // separated strain log Jp, mean elastic J.
        let mut bins = [(0u32, 0.0f64, 0.0f64, 0.0f64); 30];
        for p in s.terrain_particles() {
            if p.xm[3] <= 0.0 || p.f[2][3].to_bits() & 1 != 0 {
                continue;
            }
            let x = Vec3::from_slice(&p.xm[..3]);
            if x.x * x.x + x.z * x.z >= 4.0 * dx * dx {
                continue;
            }
            let k = ((0.05 - x.y) / 0.01).floor();
            if !(0.0..30.0).contains(&k) {
                continue;
            }
            let f = Mat3::from_cols_array(&[
                p.f[0][0], p.f[1][0], p.f[2][0], p.f[0][1], p.f[1][1], p.f[2][1], p.f[0][2],
                p.f[1][2], p.f[2][2],
            ]);
            let b = &mut bins[k as usize];
            b.0 += 1;
            b.1 += f64::from(p.vj[1]);
            b.2 += f64::from(p.vj[3]);
            b.3 += f64::from(f.determinant());
        }
        for (k, b) in bins.iter().enumerate() {
            if b.0 > 0 {
                let n = f64::from(b.0);
                eprintln!(
                    "  column y {:+.2}: {:5} particles, v_y {:+.3} m/s, log Jp {:.4}, J {:.4}",
                    0.05 - 0.01 * (k as f64 + 0.5),
                    b.0,
                    b.1 / n,
                    b.2 / n,
                    b.3 / n
                );
            }
        }
    }
    eprintln!("probe t {t:.4}: momentum {mom:.3?} kg·m/s, KE moving down {ke_down:.2} J");
    eprintln!(
        "probe t {t:.4}: separated volume {gap:.3e} m³ ({gap_floor:.3e} under the cavity), surface at r = 0, 0.04, … {:?}",
        rings.map(|h| (h * 1e4).round() / 1e4)
    );
    eprintln!(
        "probe t {t:.4}: KE {ke:.2} J, elastic vol {pe:.2} J, min J {jmin:.3} at {jmin_at:?}, \
         V(J<0.9) {v09:.3e} V(J<0.7) {v07:.3e} of {vol:.3e} m³, compaction {cap:.3e} m³, axis surface {floor:.4} m"
    );
}

/// The field the GPU adds to the particles is the CPU Z-model's.
#[test]
fn the_source_kernel_applies_the_z_model_field() {
    let mut s = sim();
    let cfg = TerrainConfig {
        max_particles: 1 << 18,
        max_blocks: 1 << 14,
        min_cell_size: 0.002,
        ..TerrainConfig::default()
    };
    s.terrain_enable(&cfg).unwrap();
    let sand = SandMaterial::default();
    let src = impact(&sand);
    let crater = src.crater(G).unwrap();
    let z = point_source_z(src.target.ejecta.mu);
    let field_model = src.field(G, z).unwrap();
    let r = crater.radius as f32;
    let spacing = r / 16.0;
    let n = 129u32;
    let field = HeightField::new(n, n, spacing, spacing, vec![0.0; (n * n) as usize]).unwrap();
    let mut arena = HullArena::new();
    let (_, bed) = s
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    let plan = crater.region(&cfg, Vec3::ZERO, 0.6, 0, 0).unwrap();
    s.terrain_activate(
        &plan,
        bed,
        &field,
        Vec3::ZERO,
        sand.density,
        sand.friction_angle.tan(),
    )
    .unwrap();
    s.terrain_apply_source(&field_model, Vec3::ZERO).unwrap();
    let mut worst = 0.0f32;
    for p in s.terrain_particles() {
        let x = Vec3::from_slice(&p.xm[..3]);
        let want = field_model.velocity_f32(x);
        let got = Vec3::from_slice(&p.vj[..3]);
        worst = worst.max((got - want).length() / want.length().max(1e-3));
    }
    assert!(
        worst < 1e-4,
        "GPU field differs from the CPU's by {worst} (relative)"
    );
}

/// The crater dug by the MPM from the Z-model start, after it settles and is
/// baked, against the π-scaling prediction for the same impact.
#[test]
#[ignore = "validation run (long): cargo test --release ... -- --ignored"]
fn an_impact_crater_in_dry_sand_follows_pi_scaling() {
    let mut s = sim();
    let cfg = TerrainConfig {
        max_particles: 65535 * 64,
        max_blocks: 65535,
        max_substeps: 1 << 14,
        max_parcels: 1 << 20,
        min_cell_size: 0.002,
        cells_per_radius: std::env::var("CRATER_CPR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16.0),
        ..TerrainConfig::default()
    };
    s.terrain_enable(&cfg).unwrap();
    // CRATER_E: Young's modulus override (Pa), to measure what softer sand
    // would change (a lossy option, reported, not adopted).
    let mut sand = SandMaterial::default();
    if let Some(e) = std::env::var("CRATER_E").ok().and_then(|v| v.parse().ok()) {
        sand.youngs_modulus = e;
    }
    // Sensitivities (diagnostics, reported, not adopted): CRATER_PHI (degrees),
    // CRATER_Z (Maxwell's Z), CRATER_CORE (core radius over the flow-centre depth).
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
    if let Some(phi) = env("CRATER_PHI") {
        sand.friction_angle = (phi as f32).to_radians();
    }
    s.terrain_set_material(Some(&sand)).unwrap();
    let src = impact(&sand);
    let crater = src.crater(G).unwrap();
    let z = env("CRATER_Z").unwrap_or_else(|| point_source_z(src.target.ejecta.mu));
    let h = engine_terrain::source::EQUIVALENT_BURIAL * src.impactor_radius;
    let field_model = src
        .field_with(G, z, h, env("CRATER_CORE").unwrap_or(2.0) * h)
        .unwrap();
    eprintln!(
        "field: Z {z:.3}, core {:.4} m, KE {:.1} J of the impactor's {:.0} J",
        field_model.core_radius,
        field_model.kinetic_energy(f64::from(sand.density)),
        src.kinetic_energy()
    );
    let r = crater.radius as f32;
    let spacing = r / cfg.cells_per_radius;
    let n = 257u32;
    let field = HeightField::new(n, n, spacing, spacing, vec![0.0; (n * n) as usize]).unwrap();
    let mut arena = HullArena::new();
    let (_, bed) = s
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    let margin: f64 = std::env::var("CRATER_MARGIN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2.0);
    let mut plan = crater.region(&cfg, Vec3::ZERO, margin, 0, 0).unwrap();
    // CRATER_BED: the box's bed this many transient radii below the surface,
    // the sides as the margin sets them (diagnostics: the bed's reflection).
    if let Some(bed) = std::env::var("CRATER_BED")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
    {
        let reach = r * (1.0 + margin as f32);
        plan = engine_terrain::RegionPlan::spanning(
            &cfg,
            Vec3::new(-reach, -bed * r, -reach),
            Vec3::splat(reach),
            plan.dx,
            0,
            0,
        )
        .unwrap();
        eprintln!("box {:?}..{:?}", plan.origin, plan.end());
    }
    // The pre-impact surface.
    let before = gpu_heights(&s, bed, &field);
    s.terrain_activate(
        &plan,
        bed,
        &field,
        Vec3::ZERO,
        sand.density,
        sand.friction_angle.tan(),
    )
    .unwrap();
    if std::env::var("CRATER_NO_SOURCE").is_err() {
        s.terrain_apply_source(&field_model, Vec3::ZERO).unwrap();
    }
    if std::env::var("CRATER_PROBE").is_ok() {
        probe(&s, &sand, plan.dx, 0.0);
    }
    // Substeps per frame from the last frame's top speed (the GPU checks each
    // substep against the bound and fails the step if it is exceeded).
    let mut speed = field_model
        .velocity_f32(Vec3::new(
            0.0,
            -(field_model.flow_center_depth as f32),
            field_model.core_radius as f32,
        ))
        .length();
    speed = speed.max(field_model.surface_speed(field_model.core_radius) as f32);
    let t_end = 6.0 * crater.formation_time as f32;
    let t0 = std::time::Instant::now();
    let (mut t, mut frames) = (0.0f32, 0u32);
    let mut last = None;
    // CRATER_SPLIT=k: the first frame in k parts, probed after each (diagnostics).
    if let Some(k) = std::env::var("CRATER_SPLIT")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
    {
        let part = FRAME / k as f32;
        let mut tp = 0.0f32;
        let parts: u32 = std::env::var("CRATER_SPLIT_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(k);
        for _ in 0..parts {
            let bound = (1.5 * speed).max(2.0);
            let (n_sub, dt) =
                substeps_for(&cfg, part, plan.dx, sand.p_wave_speed(), bound).unwrap();
            let rep = s.terrain_step(n_sub, dt, -(G as f32)).unwrap();
            speed = rep.max_speed;
            tp += part;
            eprintln!(
                "split: {n_sub} substeps, max speed {:.1} m/s",
                rep.max_speed
            );
            probe(&s, &sand, plan.dx, tp);
        }
        t += FRAME;
        frames += 1;
    }
    while t < t_end {
        let bound = (1.5 * speed).max(2.0);
        let (n_sub, dt) = substeps_for(&cfg, FRAME, plan.dx, sand.p_wave_speed(), bound).unwrap();
        let rep = s.terrain_step(n_sub, dt, -(G as f32)).unwrap();
        speed = rep.max_speed;
        if std::env::var("CRATER_TRACE").is_ok() {
            eprintln!(
                "t {t:.3} s: {n_sub} substeps, max speed {:.3} m/s, {} parcels ({} airborne), {} blocks",
                rep.max_speed, rep.parcels, rep.airborne, rep.blocks
            );
        }
        t += FRAME;
        frames += 1;
        last = Some(rep);
        if let Some(every) = std::env::var("CRATER_PROBE")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
        {
            if frames % every.max(1) == 1 || every == 1 {
                probe(&s, &sand, plan.dx, t);
            }
        }
    }
    let rep = last.unwrap();
    if std::env::var("CRATER_DUMP").is_ok() {
        let mut ps: Vec<(f32, Vec3)> = s
            .terrain_particles()
            .iter()
            .filter(|p| p.xm[3] > 0.0)
            .map(|p| {
                (
                    Vec3::from_slice(&p.vj[..3]).length(),
                    Vec3::from_slice(&p.xm[..3]),
                )
            })
            .collect();
        ps.sort_by(|a, b| b.0.total_cmp(&a.0));
        let fast = ps.iter().filter(|p| p.0 > 0.05).count();
        eprintln!(
            "box {:?}..{:?}, {} live, {fast} above 0.05 m/s",
            plan.origin,
            plan.end(),
            ps.len()
        );
        for (v, x) in ps.iter().take(12) {
            eprintln!("  {v:.3} m/s at {x:?}");
        }
    }
    eprintln!(
        "{} particle slots, dx {:.4} m, region {:?} blocks",
        rep.particles, plan.dx, plan.blocks
    );
    s.terrain_bake().unwrap();
    let after = gpu_heights(&s, bed, &field);
    let dv: f64 = after
        .iter()
        .zip(&before)
        .map(|(a, b)| f64::from(a - b))
        .sum::<f64>()
        * f64::from(spacing * spacing);
    let quantum = f64::from(plan.dx / cfg.particles_per_cell_axis as f32).powi(3) / 64.0;
    eprintln!(
        "ledger: field change {dv:.6} m³, escaped {:.6} m³, airborne parcels {}",
        f64::from(rep.escaped) * quantum,
        rep.airborne
    );
    // Profile against radius: the mean height in rings one sample wide.
    let rings = (n / 2) as usize;
    let (mut sum, mut cnt) = (vec![0.0f64; rings + 1], vec![0u32; rings + 1]);
    for j in 0..n {
        for i in 0..n {
            let v = field.vertex(i, j);
            let k = ((v.x * v.x + v.z * v.z).sqrt() / spacing).round() as usize;
            if k <= rings {
                sum[k] += f64::from(after[(j * n + i) as usize] - before[(j * n + i) as usize]);
                cnt[k] += 1;
            }
        }
    }
    let profile: Vec<f64> = sum
        .iter()
        .zip(&cnt)
        .map(|(s, &c)| if c > 0 { s / f64::from(c) } else { 0.0 })
        .collect();
    let depth = -profile[0];
    let crossing = profile.iter().position(|&h| h >= 0.0).unwrap_or(rings);
    let apparent = crossing as f64 * f64::from(spacing);
    let (rim_k, rim_h) =
        profile
            .iter()
            .enumerate()
            .skip(crossing)
            .fold(
                (crossing, f64::MIN),
                |a, (k, &h)| if h > a.1 { (k, h) } else { a },
            );
    let rim = rim_k as f64 * f64::from(spacing);
    eprintln!(
        "π-scaling: transient radius {:.4} m, depth {:.4} m, rim {:.4} m, formation {:.3} s ({:?})",
        crater.radius, crater.depth, crater.final_rim_radius, crater.formation_time, crater.regime
    );
    eprintln!(
        "MPM: apparent radius {apparent:.4} m, depth {depth:.4} m, rim at {rim:.4} m ({rim_h:.4} m high); \
         {frames} frames in {:.1} s wall, {} parcels, {} escaped quanta, last speed {:.3} m/s",
        t0.elapsed().as_secs_f64(),
        rep.parcels,
        rep.escaped,
        rep.max_speed
    );
    eprintln!(
        "profile (m, per ring of {spacing:.4} m): {:?}",
        &profile[..rings.min(48)]
            .iter()
            .map(|h| (h * 1e4).round() / 1e4)
            .collect::<Vec<_>>()
    );

    // The ledger closes: what the field lost escaped or is still in flight.
    let in_flight = f64::from(rep.airborne) * 64.0 * quantum;
    assert!(
        (dv + f64::from(rep.escaped) * quantum).abs() <= in_flight + 1e-6,
        "field change {dv} m³, escaped {} m³, in flight ≤ {in_flight} m³",
        f64::from(rep.escaped) * quantum
    );
    // Against gravity-regime π-scaling (Holsapple 1993) and simple-crater
    // shape (Melosh 1989). The bands are the laboratory scatter, not a fit:
    // - the final apparent crater is the transient one widened by rim
    //   collapse, D ≈ 1.1–1.3 D_transient for simple craters, and dry-sand
    //   π-scaling fits scatter by about ±20% in radius;
    // - depth over apparent diameter between Holsapple's transient 0.125 and
    //   Melosh's transient 1/3 (final simple craters ≈ 0.2);
    // - a raised rim outside the apparent radius;
    // - cavity volume within a factor of 2 of the predicted transient volume.
    let deepest = -profile.iter().cloned().fold(f64::MAX, f64::min);
    let ratio = apparent / crater.radius;
    assert!(
        (0.9..=1.6).contains(&ratio),
        "apparent radius {apparent} m is {ratio}× the transient {} m",
        crater.radius
    );
    let d_over_d = deepest / (2.0 * apparent);
    assert!(
        (0.1..=1.0 / 3.0).contains(&d_over_d),
        "depth/diameter {d_over_d}"
    );
    assert!(
        rim > apparent && rim_h > 0.0,
        "no raised rim ({rim} m, {rim_h} m)"
    );
    let area = f64::from(spacing * spacing);
    let cavity: f64 = after
        .iter()
        .zip(&before)
        .map(|(a, b)| f64::from((b - a).max(0.0)))
        .sum::<f64>()
        * area;
    let v_ratio = cavity / crater.volume;
    assert!(
        (0.5..=2.0).contains(&v_ratio),
        "cavity {cavity} m³ is {v_ratio}× the transient {} m³",
        crater.volume
    );
    eprintln!(
        "apparent/transient radius {ratio:.2}, depth/diameter {d_over_d:.3}, \
         cavity/transient volume {v_ratio:.2}"
    );
}
