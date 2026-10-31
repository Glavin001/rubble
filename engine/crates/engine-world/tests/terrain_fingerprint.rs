//! Fingerprint of a terrain run through the World API (impact, pipelined
//! coupled frames beside a resting box, the bake on settling, the
//! mirror copy). Ignored: run by hand before and after a change to the
//! terrain host and compare the printed hash, which must be equal.
//!
//! `cargo test --release -p engine-world --test terrain_fingerprint -- --ignored --nocapture --test-threads=1`

use rubble::*;

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

#[test]
#[ignore = "fingerprint: run by hand before and after a terrain host change"]
fn fingerprint_world_impact_to_bake() {
    let def = WorldDef {
        capacity: WorldCapacity {
            max_bodies: 64,
            max_colliders: 128,
            max_contacts: 4096,
            max_joints: 16,
            max_candidates: 8192,
            max_colors: 32,
            max_commands: 256,
            max_hull_vertices: 64,
        },
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    w.set_gravity([0.0, -9.81, 0.0]).unwrap();
    let tdef = TerrainDef {
        config: TerrainConfig {
            max_particles: 1 << 20,
            max_blocks: 1 << 14,
            max_parcels: 1 << 18,
            max_substeps: 1 << 13,
            cells_per_radius: 8.0,
            min_cell_size: 0.002,
            ..TerrainConfig::default()
        },
        region_margin: 0.6,
        settle_speed: 0.05,
        settle_frames: 3,
        ..TerrainDef::default()
    };
    w.enable_terrain(&tdef).unwrap();
    let impact = ImpactSource {
        impactor_radius: 0.005,
        impactor_density: 2700.0,
        velocity: glam::DVec3::new(0.0, -2000.0, 0.0),
        target: CraterTarget::sand(&tdef.material).unwrap(),
    };
    let spacing = 0.147 / 8.0;
    let n = 97u32;
    let g = w.create_body(&BodyDef::default()).unwrap();
    let field = HeightField::new(n, n, spacing, spacing, vec![0.0; (n * n) as usize]).unwrap();
    let ground = w
        .create_height_field_shape(g, &ShapeDef::default(), &field, Transform::IDENTITY)
        .unwrap();
    w.terrain_impact(ground, &impact, [0.0, 0.0, 0.0]).unwrap();
    // A box resting outside the region (a rigid body in the same world).
    let b = w.create_body(&BodyDef::dynamic([0.75, 0.05, 0.0])).unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [0.02; 3])
        .unwrap();
    let mut reports = Vec::new();
    let mut frames = 0;
    while w.terrain_region().is_some() && frames < 240 {
        w.step().unwrap();
        for f in w.poll().unwrap() {
            if let Some(r) = f.terrain {
                reports.push(r);
            }
        }
        frames += 1;
    }
    w.step().unwrap();
    w.poll_blocking().unwrap();
    let heights = w.read_height_field(ground).unwrap();
    let t = w.body_transform(b).unwrap().unwrap();
    println!(
        "world {frames} frames, {} reports, last {:?}",
        reports.len(),
        reports.last()
    );
    println!(
        "world reports {:016x} heights {:016x} box {:016x}",
        hash(reports.iter().flat_map(|r| {
            [
                r.blocks,
                r.max_speed.to_bits(),
                r.particles,
                r.parcels,
                r.deposited,
                r.escaped,
                r.airborne,
            ]
        })),
        hash(heights.iter().map(|h| h.to_bits())),
        hash(
            t.position
                .iter()
                .chain(t.rotation.iter())
                .map(|v| v.to_bits())
        )
    );
}
