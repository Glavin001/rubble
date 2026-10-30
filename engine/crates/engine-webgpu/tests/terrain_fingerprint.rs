//! Fingerprints of terrain runs through every `WgpuSim` terrain entry point
//! (load, activate, source, uncoupled and coupled substeps, ejecta, bake).
//! Ignored: run by hand before and after a change to the terrain host and
//! compare the printed hashes, which must be equal (a refactor of the command
//! sequence is bit-identical).
//!
//! `cargo test --release -p engine-webgpu --test terrain_fingerprint -- --ignored --nocapture --test-threads=1`

use engine_schema::convex::HullArena;
use engine_schema::heightfield::HeightField;
use engine_schema::{AvbdParams, BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_terrain::layout::MpmParticle;
use engine_terrain::zmodel::point_source_z;
use engine_terrain::{
    substeps_for, CraterTarget, ImpactSource, RegionPlan, SandMaterial, TerrainConfig,
};
use engine_webgpu::WgpuSim;
use glam::{DVec3, Quat, Vec3};

const FRAME: f32 = 1.0 / 60.0;

fn sim() -> WgpuSim {
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
    .expect("wgpu adapter");
    let mut p = AvbdParams {
        dt: FRAME,
        iterations: 10,
        ..AvbdParams::default()
    };
    p.gravity_y = -9.81;
    p.sleep_time = 1.0e9;
    s.set_params(p);
    s.set_material(
        0,
        Material {
            friction: 0.5,
            restitution: 0.0,
            contact_offset: 0.01,
            _pad: 0.0,
        },
    );
    s
}

/// FNV-1a over 32-bit words.
fn hash(words: impl IntoIterator<Item = u32>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn hash_pod<T: bytemuck::Pod>(v: &[T]) -> u64 {
    hash(bytemuck::cast_slice::<T, u32>(v).iter().copied())
}

fn gpu_heights(s: &WgpuSim, collider: u32, field: &HeightField) -> Vec<u32> {
    let base = engine_schema::convex::convex_params_base(s.collider(collider).params);
    let (nx, nz) = field.samples();
    s.read_words(18, u64::from(base + 2) * 4, (nx * nz) as usize)
}

/// An impact into a flat sand field: activation, the Z-model source, coupled
/// frames with a falling box (ejecta thrown and landing), then a bake.
#[test]
#[ignore = "fingerprint: run by hand before and after a terrain host change"]
fn fingerprint_impact_coupled_and_bake() {
    let mut s = sim();
    let cfg = TerrainConfig {
        max_particles: 1 << 18,
        max_blocks: 1 << 13,
        max_parcels: 1 << 16,
        max_substeps: 1 << 12,
        cells_per_radius: 8.0,
        min_cell_size: 0.002,
        ..TerrainConfig::default()
    };
    s.terrain_enable(&cfg).unwrap();
    let sand = SandMaterial::default();
    s.terrain_set_material(Some(&sand)).unwrap();
    let src = ImpactSource {
        impactor_radius: 0.005,
        impactor_density: 2700.0,
        velocity: DVec3::new(0.0, -2000.0, 0.0),
        target: CraterTarget::sand(&sand).unwrap(),
    };
    let crater = src.crater(9.81).unwrap();
    let model = src
        .field(9.81, point_source_z(src.target.ejecta.mu))
        .unwrap();
    let spacing = crater.radius as f32 / 8.0;
    let n = 65u32;
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
    s.terrain_apply_source(&model, Vec3::ZERO).unwrap();
    // A box dropping into the region.
    let h = Vec3::splat(0.02);
    let inertia = Vec3::splat(2.0 * h.x * h.x) * (0.05 / 3.0);
    let mut state = BodyState::dynamic(Vec3::new(0.06, 0.08, 0.0), Quat::IDENTITY, 0.05, inertia);
    state.linear_vel = [0.0, -1.0, 0.0];
    let body = s.add_body(state);
    s.add_collider(Collider {
        body,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 1.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [h.x, h.y, h.z, h.length()],
    });
    let mut speed = 200.0f32;
    let mut reports = Vec::new();
    for _ in 0..12 {
        let (n, dt) = substeps_for(
            &cfg,
            FRAME,
            plan.dx,
            sand.p_wave_speed(),
            1.5 * speed + 20.0,
        )
        .unwrap();
        let (_, r) = s.step_with_terrain(n, dt).unwrap();
        speed = r.max_speed;
        reports.push(r);
    }
    let particles = s.terrain_particles();
    let parcels = s.terrain_parcels();
    let bodies = s.download_bodies();
    s.terrain_bake().unwrap();
    let heights = gpu_heights(&s, bed, &field);
    println!("impact reports {reports:?}");
    println!(
        "impact particles {:016x} parcels {:016x} bodies {:016x} heights {:016x} ({} particles, {} parcels)",
        hash_pod(&particles),
        hash_pod(&parcels),
        hash_pod(&bodies),
        hash(heights),
        particles.len(),
        parcels.len()
    );
}

/// Host-loaded material, uncoupled substeps with no bed.
#[test]
#[ignore = "fingerprint: run by hand before and after a terrain host change"]
fn fingerprint_loaded_cube_uncoupled() {
    let mut s = sim();
    let cfg = TerrainConfig {
        max_particles: 1 << 16,
        max_blocks: 1 << 12,
        ..TerrainConfig::default()
    };
    s.terrain_enable(&cfg).unwrap();
    s.terrain_set_material(Some(&SandMaterial::default()))
        .unwrap();
    let dx = 0.05;
    let plan = RegionPlan::spanning(&cfg, Vec3::splat(-0.6), Vec3::splat(0.6), dx, 0, 0).unwrap();
    let mut ps = Vec::new();
    let sp = dx / 2.0;
    for i in 0..12 {
        for j in 0..12 {
            for k in 0..12 {
                let x = Vec3::new(i as f32, j as f32, k as f32) * sp - Vec3::splat(0.15);
                let mut p = MpmParticle::at_rest(x.to_array(), sp * sp * sp, 1600.0);
                p.vj = [0.3 * x.y, -0.5, 0.2 * x.x, 0.0];
                ps.push(p);
            }
        }
    }
    s.terrain_load(&plan, &ps, None, 0.5).unwrap();
    let (n, dt) =
        substeps_for(&cfg, FRAME, dx, SandMaterial::default().p_wave_speed(), 5.0).unwrap();
    let mut reports = Vec::new();
    for _ in 0..6 {
        reports.push(s.terrain_step(n, dt, -9.81).unwrap());
    }
    let particles = s.terrain_particles();
    println!("cube reports {reports:?}");
    println!("cube particles {:016x}", hash_pod(&particles));
}
