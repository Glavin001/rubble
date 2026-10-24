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

    // 5 m/s leaves the region (1.5 m from the centre) within 0.4 s.
    let mut s = sim();
    s.terrain_enable(&config()).unwrap();
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
