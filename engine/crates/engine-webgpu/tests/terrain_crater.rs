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
use glam::{DVec3, Quat, Vec3};

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
    s.terrain_set_material(Some(&sand)).unwrap();
    let src = impact(&sand);
    let crater = src.crater(G).unwrap();
    let z = point_source_z(src.target.ejecta.mu);
    let field_model = src.field(G, z).unwrap();
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
    let plan = crater.region(&cfg, Vec3::ZERO, margin, 0, 0).unwrap();
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
