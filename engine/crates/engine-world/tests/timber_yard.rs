//! The timber yard (`engine-webgpu/timber.html`): Shardwright's timber-frame house and two
//! plank walls (`fracture/assets/timber-yard.glb`, baked by `scripts/bake-fracture-assets.sh`)
//! as one threshold destructible, with support loss and crushing on, stand under their own
//! weight, at every level.
#![cfg(feature = "stress")]

use rubble::*;
use std::time::Instant;

const GLB: &[u8] = include_bytes!("../../engine-webgpu/fracture/assets/timber-yard.glb");

/// Crushing for the yard's wood (the stress import's `timber`: structural softwood, the
/// stress oracle's preset); the concrete sills are anchored.
fn crush() -> Crush {
    let p = stress_ref::crush::presets::softwood();
    Crush {
        materials: vec![(
            "timber".into(),
            CrushMaterial {
                cap_pressure: p.cap_pressure,
                cohesion: p.cohesion,
                friction_slope: p.friction_slope,
                crush_energy: p.crush_energy,
                crush_viscosity: p.crush_viscosity,
            },
        )],
        ..Crush::default()
    }
}

fn capacity(asset: &DestructibleAssetDesc) -> WorldCapacity {
    let stress = asset.stress.as_ref().unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let scene = stress.threshold_scene(1.0, solver.dt).unwrap();
    let chunks: usize = scene.bodies.iter().map(|b| b.chunks.len()).sum();
    let mut c = stress_rubble::capacity_for(&scene);
    c.max_bodies = c.max_bodies.max(((chunks + 64) as u32).next_power_of_two());
    c.max_contacts = stress_rubble::threshold::contact_capacity(chunks, c.max_contacts, 1 << 30);
    c.max_candidates = c.max_candidates.max(c.max_contacts);
    c
}

#[test]
fn yard_stands_under_its_own_weight() {
    for level in 0..=2 {
        stands(level);
    }
}

fn stands(level: u8) {
    let asset = DestructibleAssetDesc::from_glb(GLB, level).unwrap();
    let solver = asset.solver_for(StrengthModel::Threshold).unwrap();
    let def = WorldDef {
        capacity: capacity(&asset),
        gravity: [0.0, solver.gravity_y, 0.0],
        solver,
        enable_sleep: false,
        contacts: ContactStorage::Canonical,
        ..WorldDef::default()
    };
    let mut w = World::new(GpuContext::Default, &def).unwrap();
    let a = w.create_destructible_asset(&asset).unwrap();
    let id = w
        .create_destructible(&DestructibleDef {
            model: StrengthModel::Threshold,
            asleep: false,
            support_loss: true,
            crush: Some(crush()),
            ..DestructibleDef::new(a)
        })
        .unwrap();
    let frames = (5.0 / solver.dt).round() as usize;
    let (mut broken, mut crushed) = (0usize, 0usize);
    let t0 = Instant::now();
    for _ in 0..frames {
        let f = w.step_sync().unwrap();
        for e in &f.destruction_events {
            match e {
                DestructionEvent::BondsBroken { bonds, .. } => broken += bonds.len(),
                DestructionEvent::ChunkCrushed { .. } => crushed += 1,
                _ => {}
            }
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / frames as f64;
    let stats = w.destructible_stats(id).unwrap();
    eprintln!(
        "yard L{level}: {} chunks, {frames} frames at {:.1} ms/frame (step_sync); {broken} bonds broken, {crushed} crushed; {stats:?}",
        asset.chunks.len(),
        ms
    );
    assert_eq!(
        broken, 0,
        "L{level}: the yard breaks nothing under its own weight"
    );
    assert_eq!(
        crushed, 0,
        "L{level}: nothing is crushed under its own weight"
    );
}
