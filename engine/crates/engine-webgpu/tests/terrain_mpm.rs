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

/// Activate the mound, move every particle rigidly by `shift` samples, bake,
/// and check the bake against its definition (`terrain_mpm.slang`,
/// `mpmBakeReference`): each activated particle is splatted bilinearly at
/// u = x/spacing − w·o, o its rest offset in its home sample's cell and
/// w = max(0, 1 − |displacement|/fade), with fade = `bake_home_fade`.
/// - The heights equal that splat, within one quantum per particle reaching
///   the sample (each particle's share is rounded to whole quanta).
/// - The volume is exact, and its centroid is the splat's (the rounding is
///   unbiased) and moves with the shift: a column's rest offsets are
///   symmetric about its sample and all its particles moved alike, so their
///   w·o nearly cancel (only the extra quanta of `quanta mod n` break the
///   symmetry). Binning by the nearest sample instead moves whole lattice rows
///   at a time (here 5 per sample).
/// Returns the worst height error (m).
fn bake_after_rigid_shift(shift: (f32, f32)) -> f64 {
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
    let fade = f64::from(cfg.bake_home_fade);
    let mut ps = s.terrain_particles();
    let q = f64::from(ps[0].f[0][3]) / f64::from(ps[0].f[1][3].to_bits());
    let qh = q / f64::from(dx * dz);
    for p in &mut ps {
        assert_ne!(
            p.f[2][3].to_bits() & engine_terrain::layout::MPM_PARTICLE_HOME,
            0,
            "an activated particle without its home"
        );
        p.xm[0] += shift.0 * dx;
        p.xm[2] += shift.1 * dz;
    }
    // The splat's definition, in f64: volume per sample, and how many
    // particles reach each sample.
    let mut want = vec![0.0f64; (nx * nz) as usize];
    let mut reach = vec![0u32; (nx * nz) as usize];
    for p in &ps {
        let quanta = f64::from(p.f[1][3].to_bits());
        let axis = |x: f32, home: f32, x0: f32, d: f32| {
            let s0 = f64::from(home - x0) / f64::from(d);
            (f64::from(x - x0) / f64::from(d), s0 - (s0 + 0.5).floor())
        };
        let (ux, ox) = axis(p.xm[0] - origin.x, p.c[1][3], x0.x, dx);
        let (uz, oz) = axis(p.xm[2] - origin.z, p.c[2][3], x0.z, dz);
        let moved = f64::from(shift.0).hypot(f64::from(shift.1));
        let w = (1.0 - moved / fade).max(0.0);
        let (ux, uz) = (ux - w * ox, uz - w * oz);
        let (i0, j0) = (ux.floor(), uz.floor());
        let (tx, tz) = (ux - i0, uz - j0);
        for (a, wa) in [(0, 1.0 - tx), (1, tx)] {
            for (b, wb) in [(0, 1.0 - tz), (1, tz)] {
                let (i, j) = (i0 as i64 + a, j0 as i64 + b);
                if wa * wb > 0.0 && i >= 0 && j >= 0 && i < i64::from(nx) && j < i64::from(nz) {
                    let k = (j * i64::from(nx) + i) as usize;
                    want[k] += wa * wb * quanta;
                    reach[k] += 1;
                }
            }
        }
    }
    s.terrain_overwrite_particles(&ps).unwrap();
    s.terrain_bake().unwrap();
    let after = gpu_heights(&s, collider, &field);
    let mut worst = 0.0f64;
    let (mut m0, mut m1, mut mw) = ([0.0f64; 2], [0.0f64; 2], [0.0f64; 2]);
    let (mut v0, mut v1, mut vw) = (0.0f64, 0.0f64, 0.0f64);
    for j in 0..nz {
        for i in 0..nx {
            let k = (j * nx + i) as usize;
            let d = f64::from(after[k] - lowered[k]);
            let err = (d - want[k] * qh).abs();
            let tol = f64::from(reach[k]) * qh;
            worst = worst.max(err);
            assert!(
                err <= tol + 1e-5,
                "shift {shift:?}, sample ({i}, {j}): raised {d} m, the splat {} m (±{tol})",
                want[k] * qh
            );
            let v = field.vertex(i, j);
            let c = f64::from(before[k] - lowered[k]);
            v0 += c;
            v1 += d;
            m0[0] += c * f64::from(v.x);
            m0[1] += c * f64::from(v.z);
            m1[0] += d * f64::from(v.x);
            m1[1] += d * f64::from(v.z);
            vw += want[k];
            mw[0] += want[k] * f64::from(v.x);
            mw[1] += want[k] * f64::from(v.z);
        }
    }
    assert!(
        ((v1 - v0) / qh).abs() < 1e-4 * v0 / qh,
        "volume {v0} before, {v1} after (heights × samples)"
    );
    let moved = [m1[0] / v1 - m0[0] / v0, m1[1] / v1 - m0[1] / v0];
    let defined = [mw[0] / vw - m0[0] / v0, mw[1] / vw - m0[1] / v0];
    eprintln!(
        "rigid shift ({}, {}) samples: centroid moved ({:.5}, {:.5}) samples, worst height error {worst:.2e} m",
        shift.0,
        shift.1,
        moved[0] / f64::from(dx),
        moved[1] / f64::from(dz)
    );
    for ((got, def), (want, d)) in moved
        .iter()
        .zip(defined)
        .zip([(shift.0, dx), (shift.1, dz)])
    {
        let (got, def) = (got / f64::from(d), def / f64::from(d));
        // The whole-quanta rounding is unbiased: the baked centroid is the
        // splat's to well under a quantum's share.
        assert!(
            (got - def).abs() < 1e-4,
            "centroid moved {got} samples, the splat {def}"
        );
        // And the splat's moves with the material: to within the lattice's
        // own asymmetry (a column's `quanta mod n` extra quanta go to its
        // first particles, which keeps part of their offsets in the centroid).
        assert!(
            (got - f64::from(want)).abs() < 2e-3,
            "centroid moved {got} samples, the material {want}"
        );
    }
    assert!(s.terrain_particles().is_empty());
    worst
}

/// Material moved rigidly bakes as the splat defines, keeps its volume and
/// moves its centroid by the shift: within the home fade (here 0.63 of the
/// one-spacing fade, so the rest offsets are partly taken out) and beyond it
/// (each particle at its own position). Zero shift is the identity
/// (`activating_then_baking_unmoved_material_restores_the_terrain`).
#[test]
fn material_moved_rigidly_bakes_as_the_splat_defines() {
    bake_after_rigid_shift((0.3, 0.55));
    bake_after_rigid_shift((1.3, -0.55));
}

/// A flat layer opened by a crater-like flow and baked with home fade `fade`.
/// Every particle moves radially from r to r' = √(r² + r₀²) (r₀ = 0.5 m, four
/// samples): the map keeps area (r' dr' = r dr), so it opens a hole of radius
/// r₀ and leaves the layer around it exactly as thick as before, while the
/// material near the centre is stretched round the hole many times over, as a
/// crater's flow stretches it. Over the samples from r₀ + 2 samples to 1.4 m,
/// returns the rms and largest deviation of the raise from the layer's
/// thickness, and how far the two axis lines through the centre stand above
/// the lines beside them, all relative to the thickness.
fn opened_layer_error(fade: f32) -> (f64, f64, f64) {
    let mut s = sim();
    let cfg = TerrainConfig {
        max_particles: 1 << 18,
        bake_home_fade: fade,
        ..config()
    };
    s.terrain_enable(&cfg).unwrap();
    let field = HeightField::new(65, 65, 0.125, 0.125, vec![0.0; 65 * 65]).unwrap();
    let mut arena = HullArena::new();
    let (_, bed) = s
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    let plan = RegionPlan::spanning(
        &cfg,
        Vec3::new(-1.6, -0.3, -1.6),
        Vec3::new(1.6, 0.1, 1.6),
        0.05,
        0,
        0,
    )
    .unwrap();
    s.terrain_activate(&plan, bed, &field, Vec3::ZERO, 1600.0, 0.5)
        .unwrap();
    let lowered = gpu_heights(&s, bed, &field);
    let r0 = 0.5f32;
    let mut ps = s.terrain_particles();
    for p in &mut ps {
        let r = p.xm[0].hypot(p.xm[2]);
        let k = (r * r + r0 * r0).sqrt() / r.max(1e-6);
        p.xm[0] *= k;
        p.xm[2] *= k;
    }
    s.terrain_overwrite_particles(&ps).unwrap();
    s.terrain_bake().unwrap();
    let after = gpu_heights(&s, bed, &field);
    let (n, _) = field.samples();
    let (dx, _) = field.cell_size();
    let c = n / 2;
    let raise =
        |i: u32, j: u32| f64::from(after[(j * n + i) as usize] - lowered[(j * n + i) as usize]);
    // The layer's thickness: what activation took from a sample in the window.
    let thick = -f64::from(lowered[(c * n + c) as usize]);
    let within = |i: u32, j: u32| {
        let r = field.vertex(i, j).length();
        r >= r0 + 2.0 * dx && r <= 1.4
    };
    let (mut sum2, mut max, mut count) = (0.0f64, 0.0f64, 0u32);
    let (mut ridge, mut lines) = (0.0f64, 0u32);
    for j in 1..n - 1 {
        for i in 1..n - 1 {
            if !within(i, j) {
                continue;
            }
            let e = raise(i, j) - thick;
            sum2 += e * e;
            max = max.max(e.abs());
            count += 1;
            if j == c {
                ridge += raise(i, j) - 0.5 * (raise(i, j - 1) + raise(i, j + 1));
                lines += 1;
            }
            if i == c {
                ridge += raise(i, j) - 0.5 * (raise(i - 1, j) + raise(i + 1, j));
                lines += 1;
            }
        }
    }
    let rms = (sum2 / f64::from(count)).sqrt();
    (rms / thick, max / thick, ridge / f64::from(lines) / thick)
}

/// Material a crater's flow stretched bakes without ridges: a flat layer
/// opened by an area-keeping radial map must bake back as thick as it was
/// around the hole. With the home correction kept however far particles
/// moved, each column's particles go back to their rest offsets around where
/// the column's centre went, and stretched columns clump into ridges (the two
/// axis lines, whose centres do not move sideways, keep all of theirs). With
/// the fade (one sample spacing), particles that moved splat where they are.
/// The bounds leave room for the bilinear splat's own ripple (measured: rms
/// 0.7%, max 2.2%, axis ridge 1.6% of the thickness; with the home correction
/// kept: 4.7%, 18% and 13.5%).
#[test]
fn an_opened_layer_bakes_flat_without_axis_ridges() {
    let fade = TerrainConfig::default().bake_home_fade;
    let (rms, max, ridge) = opened_layer_error(fade);
    let (rms_home, max_home, ridge_home) = opened_layer_error(1e9);
    eprintln!(
        "opened layer, relative to its thickness: fade {fade}: error rms {rms:.4}, max {max:.4}, \
         axis ridge {ridge:.4}; home correction kept: rms {rms_home:.4}, max {max_home:.4}, \
         axis ridge {ridge_home:.4}"
    );
    assert!(rms < 0.02 && max < 0.06, "error rms {rms}, max {max}");
    assert!(ridge.abs() < 0.05, "axis ridge {ridge} of the thickness");
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
