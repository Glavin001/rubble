//! Deformable-terrain MPM transfers (T2, `slang/kernels/terrain_mpm.slang`),
//! checked against what MLS-MPM guarantees with no stress: a uniform motion is
//! carried exactly, free fall gains g·dt per substep, P2G/G2P conserve linear
//! momentum, results do not depend on the order threads run in, material falling
//! on the heightfield bed stops on it, and limits fail by name.

use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::WorldCapacity;
use engine_terrain::layout::MpmParticle;
use engine_terrain::{RegionPlan, TerrainConfig};
use engine_webgpu::WgpuSim;
use glam::{Quat, Vec3};

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

fn config() -> TerrainConfig {
    TerrainConfig {
        max_particles: 1 << 16,
        max_blocks: 1 << 12,
        ..TerrainConfig::default()
    }
}

/// A region of radius 1 m (dx = 1/16 m) around `center`, and a cube of
/// particles (2³ per cell) of side `side` centred there.
fn cube(center: Vec3, side: f32, v: impl Fn(Vec3) -> Vec3) -> (RegionPlan, Vec<MpmParticle>) {
    let cfg = config();
    let plan = RegionPlan::around(&cfg, center, 1.0, 0.5, 0, 0).unwrap();
    let h = plan.dx / 2.0;
    let n = (side / h).round() as i32;
    let mut out = Vec::new();
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let x = center - Vec3::splat(side / 2.0)
                    + (Vec3::new(i as f32, j as f32, k as f32) + 0.5) * h;
                let mut p = MpmParticle::at_rest(x.to_array(), h * h * h, 1600.0);
                p.vj[..3].copy_from_slice(&v(x).to_array());
                out.push(p);
            }
        }
    }
    (plan, out)
}

/// Total momentum, summed in f64 (an f32 sum over thousands of particles
/// rounds by more than the transfers do).
fn momentum(ps: &[MpmParticle]) -> glam::DVec3 {
    ps.iter()
        .map(|p| Vec3::from_slice(&p.vj[..3]).as_dvec3() * f64::from(p.xm[3]))
        .sum()
}

#[test]
fn a_uniform_motion_is_carried_exactly() {
    let mut s = sim();
    s.terrain_enable(&config()).unwrap();
    let v0 = Vec3::new(0.3, -0.2, 0.1);
    let (plan, ps) = cube(Vec3::new(0.0, 0.0, 0.0), 0.5, |_| v0);
    s.terrain_load(&plan, &ps, None, 0.0).unwrap();
    let dt = 1.0e-3;
    s.terrain_step(20, dt, 0.0).unwrap();
    for (a, b) in ps.iter().zip(s.terrain_particles()) {
        let v = Vec3::from_slice(&b.vj[..3]);
        assert!((v - v0).length() < 1e-5, "velocity {v} (was {v0})");
        let moved = Vec3::from_slice(&b.xm[..3]) - Vec3::from_slice(&a.xm[..3]);
        assert!((moved - v0 * 20.0 * dt).length() < 1e-5, "moved {moved}");
        for r in b.c {
            assert!(Vec3::from_slice(&r[..3]).length() < 1e-3, "C row {r:?}");
        }
    }
}

#[test]
fn free_fall_gains_g_dt_per_substep() {
    let mut s = sim();
    s.terrain_enable(&config()).unwrap();
    let (plan, ps) = cube(Vec3::ZERO, 0.25, |_| Vec3::ZERO);
    s.terrain_load(&plan, &ps, None, 0.0).unwrap();
    let (dt, g, n) = (1.0e-3f32, -9.81f32, 30u32);
    s.terrain_step(n, dt, g).unwrap();
    for (a, b) in ps.iter().zip(s.terrain_particles()) {
        let vy = b.vj[1];
        assert!((vy - g * dt * n as f32).abs() < 1e-5, "vy {vy}");
        // Symplectic Euler: Δy = g·dt²·n(n+1)/2.
        let dy = b.xm[1] - a.xm[1];
        let want = g * dt * dt * (n * (n + 1)) as f32 / 2.0;
        assert!((dy - want).abs() < 1e-5, "fell {dy}, want {want}");
    }
}

#[test]
fn transfers_conserve_linear_momentum() {
    let mut s = sim();
    s.terrain_enable(&config()).unwrap();
    let field = |x: Vec3| {
        Vec3::new(
            (7.0 * x.y).sin(),
            (5.0 * x.z).cos(),
            (3.0 * x.x).sin() + x.y,
        )
    };
    let (plan, ps) = cube(Vec3::new(0.1, -0.05, 0.2), 0.5, field);
    s.terrain_load(&plan, &ps, None, 0.0).unwrap();
    let before = momentum(&ps);
    s.terrain_step(1, 1.0e-4, 0.0).unwrap();
    let after = momentum(&s.terrain_particles());
    let scale: f64 = ps.iter().map(|p| f64::from(p.xm[3])).sum();
    assert!(
        (after - before).length() < 1e-6 * scale,
        "momentum {before} -> {after}"
    );
}

#[test]
fn results_do_not_depend_on_thread_order() {
    let field = |x: Vec3| Vec3::new((9.0 * x.z).sin(), -0.5, (4.0 * x.x).cos());
    let mut runs = Vec::new();
    for _ in 0..2 {
        let mut s = sim();
        s.terrain_enable(&config()).unwrap();
        let (plan, ps) = cube(Vec3::new(0.0, 0.1, 0.0), 0.5, field);
        s.terrain_load(&plan, &ps, None, 0.0).unwrap();
        s.terrain_step(10, 1.0e-3, -9.81).unwrap();
        runs.push(s.terrain_particles());
    }
    assert!(
        bytemuck::cast_slice::<MpmParticle, u32>(&runs[0])
            == bytemuck::cast_slice::<MpmParticle, u32>(&runs[1]),
        "two runs of the same input differ"
    );
}

#[test]
fn material_falling_on_the_bed_stops_on_it() {
    let mut s = sim();
    s.terrain_enable(&config()).unwrap();
    let field = HeightField::new(33, 33, 0.125, 0.125, vec![0.0; 33 * 33]).unwrap();
    let mut arena = HullArena::new();
    let (_, bed) = s
        .add_height_field(
            &mut arena,
            &field,
            Vec3::new(0.0, -0.5, 0.0),
            Quat::IDENTITY,
            0,
        )
        .unwrap();
    let (plan, ps) = cube(Vec3::new(0.0, -0.2, 0.0), 0.25, |_| Vec3::ZERO);
    s.terrain_load(&plan, &ps, Some(bed), 0.5).unwrap();
    // 0.6 s of 0.5 ms substeps: a 0.2 m drop lands in about 0.2 s.
    for _ in 0..12 {
        s.terrain_step(100, 5.0e-4, -9.81).unwrap();
    }
    for p in s.terrain_particles() {
        // Bed at y = -0.5; grid contact holds material within a cell of it.
        assert!(
            p.xm[1] > -0.5 - plan.dx,
            "particle fell through the bed to {}",
            p.xm[1]
        );
        assert!(p.vj[1].abs() < 1e-3, "still moving at {} m/s", p.vj[1]);
    }
}

#[test]
fn limits_fail_by_name() {
    let mut s = sim();
    let small = TerrainConfig {
        max_blocks: 8,
        ..config()
    };
    s.terrain_enable(&small).unwrap();
    // The plan's box is checked against max_blocks too; plan with the roomy config.
    let (plan, ps) = cube(Vec3::ZERO, 0.5, |_| Vec3::ZERO);
    s.terrain_load(&plan, &ps, None, 0.0).unwrap();
    let err = s.terrain_step(1, 1.0e-3, 0.0).unwrap_err();
    assert!(err.to_string().contains("max_blocks"), "{err}");

    // 5 m/s leaves the region (1.5 m from the centre) within 0.4 s; with no
    // parcels to carry it on, that is an error.
    let mut s = sim();
    s.terrain_enable(&TerrainConfig {
        max_parcels: 0,
        ..config()
    })
    .unwrap();
    let (plan, ps) = cube(Vec3::ZERO, 0.25, |_| Vec3::new(5.0, 0.0, 0.0));
    s.terrain_load(&plan, &ps, None, 0.0).unwrap();
    let err = s.terrain_step(400, 1.0e-3, 0.0).unwrap_err();
    assert!(err.to_string().contains("left the active region"), "{err}");

    // 200 m/s moves 3.2 cells per 1 ms substep: past the Courant bound (0.3).
    let mut s = sim();
    s.terrain_enable(&config()).unwrap();
    let (plan, ps) = cube(Vec3::ZERO, 0.25, |_| Vec3::new(200.0, 0.0, 0.0));
    s.terrain_load(&plan, &ps, None, 0.0).unwrap();
    let err = s.terrain_step(1, 1.0e-3, 0.0).unwrap_err();
    assert!(err.to_string().contains("CFL"), "{err}");
}

/// A heap of dry sand dropped on a flat bed spreads and comes to rest with its
/// flanks at or below the friction angle (a granular pile cannot stand steeper
/// than its angle of repose, which for cohesionless sand is about φ), and does
/// not spread flat (it holds a slope near φ, unlike a fluid).
#[test]
fn a_sand_column_settles_at_or_below_its_friction_angle() {
    use engine_terrain::{substeps_for, SandMaterial};
    let mut s = sim();
    let cfg = TerrainConfig {
        max_particles: 1 << 18,
        ..config()
    };
    s.terrain_enable(&cfg).unwrap();
    let sand = SandMaterial::default();
    s.terrain_set_material(Some(&sand)).unwrap();
    let field = HeightField::new(65, 65, 0.0625, 0.0625, vec![0.0; 65 * 65]).unwrap();
    let mut arena = HullArena::new();
    let (_, bed) = s
        .add_height_field(
            &mut arena,
            &field,
            Vec3::new(0.0, -0.6, 0.0),
            Quat::IDENTITY,
            0,
        )
        .unwrap();
    // A 0.4 m wide, 0.5 m tall column standing on the bed (y = -0.6).
    let plan = RegionPlan::around(&cfg, Vec3::new(0.0, -0.3, 0.0), 1.2, 0.3, 0, 0).unwrap();
    let h = plan.dx / 2.0;
    let mut ps = Vec::new();
    let (w, tall) = (0.4f32, 0.5f32);
    let (nw, nt) = ((w / h).round() as i32, (tall / h).round() as i32);
    for k in 0..nw {
        for j in 0..nt {
            for i in 0..nw {
                let x = Vec3::new(-w / 2.0, -0.6, -w / 2.0)
                    + (Vec3::new(i as f32, j as f32, k as f32) + 0.5) * h;
                ps.push(MpmParticle::at_rest(x.to_array(), h * h * h, sand.density));
            }
        }
    }
    s.terrain_load(&plan, &ps, Some(bed), sand.friction_angle.tan())
        .unwrap();
    // 3 s at 60 Hz, substeps from the P-wave speed plus a 5 m/s flow bound.
    let (n, dt) = substeps_for(&cfg, 1.0 / 60.0, plan.dx, sand.p_wave_speed(), 5.0).unwrap();
    let mut last = 0.0;
    let t0 = std::time::Instant::now();
    for _ in 0..180 {
        last = s.terrain_step(n, dt, -9.81).unwrap().max_speed;
    }
    let wall = t0.elapsed().as_secs_f32();
    let out = s.terrain_particles();
    // Height of the pile against distance from its axis: the top of the
    // particles in rings of 2 cells.
    let ring = 2.0 * plan.dx;
    let mut top = vec![-0.6f32; 64];
    for p in &out {
        let r = (p.xm[0] * p.xm[0] + p.xm[2] * p.xm[2]).sqrt();
        let k = ((r / ring) as usize).min(63);
        top[k] = top[k].max(p.xm[1]);
    }
    let phi = sand.friction_angle.to_degrees();
    let mut steepest = 0.0f32;
    for k in 0..63 {
        if top[k + 1] > -0.6 + 2.0 * plan.dx {
            steepest = steepest.max(((top[k] - top[k + 1]) / ring).atan().to_degrees());
        }
    }
    let peak = top.iter().cloned().fold(-1.0f32, f32::max) + 0.6;
    eprintln!(
        "pile: {} particles, peak {peak:.3} m, steepest flank {steepest:.1}° (φ = {phi:.0}°), \
         max speed {last:.4} m/s, {n} substeps of {dt:.2e} s per frame, {:.1} ms per frame",
        out.len(),
        wall * 1e3 / 180.0
    );
    assert!(last < 0.05, "the pile is still moving at {last} m/s");
    assert!(
        steepest <= phi + 5.0,
        "flank at {steepest}° stands steeper than φ = {phi}°"
    );
    assert!(
        peak > 0.1,
        "the pile spread flat (peak {peak} m): sand should hold a slope"
    );
}

/// Heights of heightfield `collider` as the GPU holds them (row-major).
fn gpu_heights(s: &WgpuSim, collider: u32, field: &HeightField) -> Vec<f32> {
    let base = engine_schema::convex::convex_params_base(s.collider(collider).params);
    let (nx, nz) = field.samples();
    let words = s.read_words(18, u64::from(base + 2) * 4, (nx * nz) as usize);
    words.into_iter().map(f32::from_bits).collect()
}

/// A mound of `peak` m on a flat field of 33×33 samples (12.5 cm apart) at y = -0.5.
fn mound(s: &mut WgpuSim, peak: f32, width: f32) -> (HeightField, u32, Vec3) {
    let field = HeightField::from_fn(33, 33, 0.125, 0.125, |x, z| {
        peak * (-(x * x + z * z) / (width * width)).exp()
    })
    .unwrap();
    let origin = Vec3::new(0.0, -0.5, 0.0);
    let mut arena = HullArena::new();
    let (_, collider) = s
        .add_height_field(&mut arena, &field, origin, Quat::IDENTITY, 0)
        .unwrap();
    (field, collider, origin)
}

#[test]
fn activating_then_baking_unmoved_material_restores_the_terrain() {
    let mut s = sim();
    let cfg = TerrainConfig {
        max_particles: 1 << 18,
        ..config()
    };
    s.terrain_enable(&cfg).unwrap();
    let (field, collider, origin) = mound(&mut s, 0.3, 0.5);
    let before = gpu_heights(&s, collider, &field);
    let plan = RegionPlan::spanning(
        &cfg,
        Vec3::new(-1.0, -0.8, -1.0),
        Vec3::new(1.0, 0.0, 1.0),
        0.05,
        0,
        0,
    )
    .unwrap();
    s.terrain_activate(&plan, collider, &field, origin, 1600.0, 0.5)
        .unwrap();
    let lowered = gpu_heights(&s, collider, &field);
    let ps = s.terrain_particles();
    assert!(!ps.is_empty(), "nothing was activated");
    // The particles hold exactly what the samples gave up: whole quanta.
    let quanta: u64 = ps.iter().map(|p| u64::from(p.f[1][3].to_bits())).sum();
    let (dx, dz) = field.cell_size();
    let removed: f64 = before
        .iter()
        .zip(&lowered)
        .map(|(a, b)| f64::from(a - b) * f64::from(dx * dz))
        .sum();
    let q = f64::from(ps[0].f[0][3]) / f64::from(ps[0].f[1][3].to_bits());
    assert!(
        (removed / q - quanta as f64).abs() < 1e-3 * quanta as f64,
        "heights lost {removed} m³, particles hold {} m³",
        quanta as f64 * q
    );
    let mass: f64 = ps.iter().map(|p| f64::from(p.xm[3])).sum();
    assert!((mass - 1600.0 * quanta as f64 * q).abs() < 1e-4 * mass);
    // Every particle sits between the bed and the surface it came from.
    for p in &ps {
        let (x, z) = (p.xm[0] - origin.x, p.xm[2] - origin.z);
        let (h, _) = field.surface(x, z).unwrap();
        assert!(
            p.xm[1] < origin.y + h + 0.13,
            "particle above its column: {:?}",
            p.xm
        );
    }
    s.terrain_bake().unwrap();
    let after = gpu_heights(&s, collider, &field);
    for (k, (a, b)) in before.iter().zip(&after).enumerate() {
        assert!(
            (a - b).abs() < 1e-5,
            "sample {k}: {a} before, {b} after bake"
        );
    }
    assert!(s.terrain_particles().is_empty());
}

/// The bake splats each activated particle bilinearly from its home sample
/// moved by its displacement. So material moved rigidly by a fraction of a
/// sample bakes as its columns moved: each column's volume is split by the
/// tent weights between the samples around its new centre (the heights become
/// the bilinear interpolation of the shifted columns, within one quantum per
/// contributing particle), the volume's centroid moves by the shift (to 1e-3
/// of a sample: the dithered rounding to whole quanta is unbiased, so it does
/// not add up over the column's particles, which all moved alike), and no
/// quantum is lost. Binning by the nearest sample instead moves whole lattice
/// rows at a time (here 5 per sample), so the centroid would move by a
/// multiple of 1/5 of a sample.
#[test]
fn material_moved_rigidly_bakes_as_its_columns_moved() {
    let mut s = sim();
    let cfg = TerrainConfig {
        max_particles: 1 << 18,
        ..config()
    };
    s.terrain_enable(&cfg).unwrap();
    let (field, collider, origin) = mound(&mut s, 0.3, 0.5);
    let before = gpu_heights(&s, collider, &field);
    let plan = RegionPlan::spanning(
        &cfg,
        Vec3::new(-1.0, -0.8, -1.0),
        Vec3::new(1.0, 0.0, 1.0),
        0.05,
        0,
        0,
    )
    .unwrap();
    s.terrain_activate(&plan, collider, &field, origin, 1600.0, 0.5)
        .unwrap();
    let lowered = gpu_heights(&s, collider, &field);
    let (nx, nz) = field.samples();
    let (dx, dz) = field.cell_size();
    let x0 = field.vertex(0, 0);
    let mut ps = s.terrain_particles();
    let q = f64::from(ps[0].f[0][3]) / f64::from(ps[0].f[1][3].to_bits());
    let qh = q / f64::from(dx * dz);
    // Particles per home sample (activation records where it emitted each).
    let mut count = vec![0u32; (nx * nz) as usize];
    for p in &ps {
        assert_ne!(
            p.f[2][3].to_bits() & engine_terrain::layout::MPM_PARTICLE_HOME,
            0,
            "an activated particle without its home"
        );
        let i = ((p.c[1][3] - x0.x) / dx).round() as u32;
        let j = ((p.c[2][3] - x0.z) / dz).round() as u32;
        count[(j * nx + i) as usize] += 1;
    }
    let shift = (0.3f32, 0.55f32);
    for p in &mut ps {
        p.xm[0] += shift.0 * dx;
        p.xm[2] += shift.1 * dz;
    }
    s.terrain_overwrite_particles(&ps).unwrap();
    s.terrain_bake().unwrap();
    let after = gpu_heights(&s, collider, &field);
    let column = |i: i64, j: i64| -> (f64, u32) {
        if i < 0 || j < 0 || i >= i64::from(nx) || j >= i64::from(nz) {
            return (0.0, 0);
        }
        let k = (j * i64::from(nx) + i) as usize;
        (f64::from(before[k] - lowered[k]), count[k])
    };
    let (wx, wz) = (
        [1.0 - f64::from(shift.0), f64::from(shift.0)],
        [1.0 - f64::from(shift.1), f64::from(shift.1)],
    );
    let mut worst = 0.0f64;
    let (mut m0, mut m1) = ([0.0f64; 2], [0.0f64; 2]);
    let (mut v0, mut v1) = (0.0f64, 0.0f64);
    for j in 0..nz {
        for i in 0..nx {
            let k = (j * nx + i) as usize;
            let (mut want, mut tol) = (f64::from(lowered[k]), 0.0f64);
            for (a, wa) in wx.iter().enumerate() {
                for (b, wb) in wz.iter().enumerate() {
                    let (c, n) = column(i64::from(i) - a as i64, i64::from(j) - b as i64);
                    want += wa * wb * c;
                    tol += f64::from(n) * qh;
                }
            }
            let err = (f64::from(after[k]) - want).abs();
            worst = worst.max(err);
            assert!(
                err <= tol + 1e-5,
                "sample ({i}, {j}): {} baked, {want} for the shifted columns (±{tol})",
                after[k]
            );
            let v = field.vertex(i, j);
            let (c, _) = column(i64::from(i), i64::from(j));
            let d = f64::from(after[k] - lowered[k]);
            v0 += c;
            v1 += d;
            m0[0] += c * f64::from(v.x);
            m0[1] += c * f64::from(v.z);
            m1[0] += d * f64::from(v.x);
            m1[1] += d * f64::from(v.z);
        }
    }
    assert!(
        ((v1 - v0) / qh).abs() < 1e-4 * v0 / qh,
        "volume {v0} before, {v1} after (heights × samples)"
    );
    let moved = [m1[0] / v1 - m0[0] / v0, m1[1] / v1 - m0[1] / v0];
    eprintln!(
        "rigid shift ({}, {}) samples: centroid moved ({:.5}, {:.5}) samples, worst height error {worst:.2e} m",
        shift.0,
        shift.1,
        moved[0] / f64::from(dx),
        moved[1] / f64::from(dz)
    );
    for (got, (want, d)) in moved.iter().zip([(shift.0, dx), (shift.1, dz)]) {
        assert!(
            (got / f64::from(d) - f64::from(want)).abs() < 1e-3,
            "centroid moved {} samples, the material {want}",
            got / f64::from(d)
        );
    }
    assert!(s.terrain_particles().is_empty());
}

/// A sand mound opened as an active region, flowing for 1.5 s and baked back:
/// no particle gains or loses quanta, and the heightfield ends with the volume it
/// started with. (How far a mound this steep slumps is the granular-validation
/// question; this test is about where the material goes, not how.)
#[test]
fn a_sand_mound_flows_and_bakes_back_with_its_volume() {
    use engine_terrain::{substeps_for, SandMaterial};
    let mut s = sim();
    let cfg = TerrainConfig {
        max_particles: 1 << 18,
        ..config()
    };
    s.terrain_enable(&cfg).unwrap();
    let sand = SandMaterial::default();
    s.terrain_set_material(Some(&sand)).unwrap();
    let (field, collider, origin) = mound(&mut s, 0.45, 0.3);
    let before = gpu_heights(&s, collider, &field);
    let plan = RegionPlan::spanning(
        &cfg,
        Vec3::new(-1.2, -0.75, -1.2),
        Vec3::new(1.2, 0.2, 1.2),
        0.05,
        0,
        0,
    )
    .unwrap();
    s.terrain_activate(
        &plan,
        collider,
        &field,
        origin,
        sand.density,
        sand.friction_angle.tan(),
    )
    .unwrap();
    let activated = s.terrain_particles();
    let quanta: u64 = activated
        .iter()
        .map(|p| u64::from(p.f[1][3].to_bits()))
        .sum();
    let home = |p: &MpmParticle| {
        (
            p.f[2][3].to_bits() & engine_terrain::layout::MPM_PARTICLE_HOME,
            p.c[1][3].to_bits(),
            p.c[2][3].to_bits(),
        )
    };
    let homes: Vec<_> = activated.iter().map(home).collect();
    let (n, dt) = substeps_for(&cfg, 1.0 / 60.0, plan.dx, sand.p_wave_speed(), 5.0).unwrap();
    let mut last = 0.0;
    for _ in 0..90 {
        last = s.terrain_step(n, dt, -9.81).unwrap().max_speed;
    }
    let moved: u64 = s
        .terrain_particles()
        .iter()
        .map(|p| u64::from(p.f[1][3].to_bits()))
        .sum();
    assert_eq!(
        moved, quanta,
        "particles lost or gained quanta while flowing"
    );
    // Flowing keeps each particle's record of where it was activated (the
    // bake's reference), bit for bit, its flag word included.
    assert!(homes.iter().all(|h| h.0 != 0));
    assert!(
        s.terrain_particles()
            .iter()
            .map(home)
            .eq(homes.iter().copied()),
        "a particle lost its home while flowing"
    );
    s.terrain_bake().unwrap();
    let after = gpu_heights(&s, collider, &field);
    let (dx, dz) = field.cell_size();
    let vol = |h: &[f32]| {
        h.iter()
            .map(|&v| f64::from(v) * f64::from(dx * dz))
            .sum::<f64>()
    };
    let (v0, v1) = (vol(&before), vol(&after));
    let peak = |h: &[f32]| h.iter().cloned().fold(f32::MIN, f32::max);
    eprintln!(
        "mound: peak {:.3} -> {:.3} m, volume {v0:.5} -> {v1:.5} m³, max speed {last:.4} m/s",
        peak(&before),
        peak(&after)
    );
    assert!(
        (v1 - v0).abs() < 1e-4 * v0.abs().max(1.0),
        "volume {v0} m³ before, {v1} m³ after"
    );
}

/// Sand thrown out of the region flies as parcels and lands on the terrain
/// around it: every quantum is accounted for (particles, parcels in flight,
/// deposited, escaped), and the heightfield outside the region rises by
/// exactly the deposited volume.
#[test]
fn ejecta_land_on_the_terrain_with_their_volume() {
    use engine_terrain::{substeps_for, SandMaterial};
    let mut s = sim();
    let cfg = TerrainConfig {
        max_particles: 1 << 18,
        ..config()
    };
    s.terrain_enable(&cfg).unwrap();
    let sand = SandMaterial::default();
    s.terrain_set_material(Some(&sand)).unwrap();
    let field = HeightField::new(65, 65, 0.125, 0.125, vec![0.0; 65 * 65]).unwrap();
    let mut arena = HullArena::new();
    let (_, bed) = s
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    let plan = RegionPlan::spanning(
        &cfg,
        Vec3::new(-0.6, -0.3, -0.6),
        Vec3::new(0.6, 0.9, 0.6),
        0.05,
        0,
        0,
    )
    .unwrap();
    s.terrain_activate(
        &plan,
        bed,
        &field,
        Vec3::ZERO,
        sand.density,
        sand.friction_angle.tan(),
    )
    .unwrap();
    let lowered = gpu_heights(&s, bed, &field);
    let quanta = |ps: &[MpmParticle]| -> u64 {
        ps.iter()
            .filter(|p| p.f[2][3].to_bits() & engine_terrain::layout::MPM_PARTICLE_DEAD == 0)
            .map(|p| u64::from(p.f[1][3].to_bits()))
            .sum()
    };
    // Throw the layer up and out, as a blast would.
    let mut ps = s.terrain_particles();
    let total = quanta(&ps);
    for p in &mut ps {
        let r = Vec3::new(p.xm[0], 0.0, p.xm[2]);
        let out = r.normalize_or_zero() * 3.0;
        p.vj[0] = out.x;
        p.vj[1] = 4.0;
        p.vj[2] = out.z;
    }
    s.terrain_load(&plan, &ps, Some(bed), sand.friction_angle.tan())
        .unwrap();
    let (n, dt) = substeps_for(&cfg, 1.0 / 60.0, plan.dx, sand.p_wave_speed(), 8.0).unwrap();
    let mut report = None;
    for _ in 0..60 {
        report = Some(s.terrain_step(n, dt, -9.81).unwrap());
    }
    let r = report.unwrap();
    let in_flight: u64 = s
        .terrain_parcels()
        .iter()
        .filter(|p| p.vm[3] > 0.0)
        .map(|p| u64::from(p.xq[3].to_bits()))
        .sum();
    let live = quanta(&s.terrain_particles());
    eprintln!(
        "ejecta: {} parcels, {} quanta deposited, {} escaped, {in_flight} in flight, {live} still particles (of {total})",
        r.parcels, r.deposited, r.escaped
    );
    assert!(r.deposited > 0, "nothing landed outside the region");
    assert_eq!(
        live + in_flight + u64::from(r.deposited) + u64::from(r.escaped),
        total,
        "quanta lost or made"
    );
    // The heights rose by the deposited quanta exactly (to f32).
    let after = gpu_heights(&s, bed, &field);
    let (dx, dz) = field.cell_size();
    let q = f64::from(ps[0].f[0][3]) / f64::from(ps[0].f[1][3].to_bits());
    // Parcels landing over the region come back as particles, so every height
    // change since activation is a deposit.
    let raised: f64 = after
        .iter()
        .zip(&lowered)
        .map(|(a, b)| f64::from(a - b) * f64::from(dx * dz))
        .sum();
    assert!(
        (raised / q - f64::from(r.deposited)).abs() < 1e-3 * f64::from(r.deposited),
        "heights rose by {raised} m³, parcels deposited {} m³",
        f64::from(r.deposited) * q
    );
}
