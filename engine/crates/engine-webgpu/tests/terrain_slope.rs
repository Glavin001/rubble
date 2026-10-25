//! Slope failure (T5): a dry-sand slope steeper than its friction angle fails
//! and runs out until it stands near repose (at most φ plus a little for the
//! grid), while one below φ holds (infinite-slope stability: a cohesionless
//! slope is stable while tan β ≤ tan φ).

use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::WorldCapacity;
use engine_terrain::{substeps_for, RegionPlan, SandMaterial, TerrainConfig};
use engine_webgpu::WgpuSim;
use glam::{Quat, Vec3};

const N: u32 = 33;
const SPACING: f32 = 0.0625;

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

fn gpu_heights(s: &WgpuSim, collider: u32) -> Vec<f32> {
    let base = engine_schema::convex::convex_params_base(s.collider(collider).params);
    s.read_words(18, u64::from(base + 2) * 4, (N * N) as usize)
        .into_iter()
        .map(f32::from_bits)
        .collect()
}

/// A slope of `angle` across x ∈ [-0.5, 0.5] m between a plateau and a floor,
/// on 0.3 m of sand, run for 2 s and baked: the before and after profiles
/// along the middle row.
fn run(angle_deg: f32) -> (Vec<f32>, Vec<f32>) {
    let mut s = sim();
    let cfg = TerrainConfig {
        max_particles: 1 << 19,
        max_blocks: 1 << 13,
        ..TerrainConfig::default()
    };
    s.terrain_enable(&cfg).unwrap();
    let sand = SandMaterial::default();
    s.terrain_set_material(Some(&sand)).unwrap();
    let t = angle_deg.to_radians().tan();
    let field =
        HeightField::from_fn(N, N, SPACING, SPACING, |x, _| (t * (0.5 - x)).clamp(0.0, t)).unwrap();
    let mut arena = HullArena::new();
    let (_, collider) = s
        .add_height_field(&mut arena, &field, Vec3::ZERO, Quat::IDENTITY, 0)
        .unwrap();
    let before = gpu_heights(&s, collider);
    let plan = RegionPlan::spanning(
        &cfg,
        Vec3::new(-1.1, -0.3, -1.1),
        Vec3::new(1.1, t + 0.4, 1.1),
        0.05,
        0,
        0,
    )
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
    let mut speed = 2.0f32;
    for _ in 0..120 {
        let (n, dt) = substeps_for(
            &cfg,
            1.0 / 60.0,
            plan.dx,
            sand.p_wave_speed(),
            (1.5 * speed).max(2.0),
        )
        .unwrap();
        speed = s.terrain_step(n, dt, -9.81).unwrap().max_speed;
    }
    s.terrain_bake().unwrap();
    let after = gpu_heights(&s, collider);
    let row = |h: &[f32]| {
        (0..N)
            .map(|i| h[((N / 2) * N + i) as usize])
            .collect::<Vec<_>>()
    };
    (row(&before), row(&after))
}

/// The face's angle (degrees) between where the profile falls through 90% and
/// 10% of its height: the slope as a whole, as infinite-slope stability
/// treats it. (The thin runout front stops short and steep where it meets
/// the bed: the low-aspect gap documented for column collapse.)
fn face_angle(h: &[f32]) -> f32 {
    let top = h[1..].iter().cloned().fold(f32::MIN, f32::max);
    let cross = |level: f32| -> f32 {
        let i = (1..h.len() - 1)
            .find(|&i| h[i + 1] < level)
            .expect("the profile falls");
        let f = (h[i] - level) / (h[i] - h[i + 1]);
        (i as f32 + f) * SPACING
    };
    let (x0, x1) = (cross(0.9 * top), cross(0.1 * top));
    (0.8 * top / (x1 - x0)).atan().to_degrees()
}

#[test]
#[ignore = "validation run (minutes): cargo test --release ... -- --ignored"]
fn a_slope_steeper_than_repose_fails_and_one_below_it_holds() {
    let phi = SandMaterial::default().friction_angle;
    let (b, a) = run(40.0);
    let moved = b
        .iter()
        .zip(&a)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    let after = face_angle(&a);
    eprintln!(
        "40° slope: face {:.1}° after (φ = {:.0}°), largest change {moved:.3} m",
        after,
        phi.to_degrees()
    );
    assert!(moved > 0.03, "the over-steep slope did not fail");
    assert!(
        after <= phi.to_degrees() + 2.0,
        "its face still stands at {after}°"
    );

    let (b, a) = run(25.0);
    let moved = b
        .iter()
        .zip(&a)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    eprintln!("25° slope: largest change {moved:.4} m");
    assert!(moved < 0.02, "a sub-repose slope moved {moved} m");
}
