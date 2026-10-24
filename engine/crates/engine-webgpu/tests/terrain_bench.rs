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
