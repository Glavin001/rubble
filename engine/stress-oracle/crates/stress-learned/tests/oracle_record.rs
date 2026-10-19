//! Record two frames from a tiny builder scene and validate the event.

use std::path::PathBuf;

use stress_learned::{validate_event, OracleSession, RecordMeta};
use stress_ref::builders;
use stress_ref::world::World;

#[test]
fn record_bond_tension_two_frames() {
    let mut scene = builders::bond_tension();
    scene.sim.record_peak_utilization = true;
    scene.sim.duration = scene.sim.frame_dt * 2.0;
    let mut world = World::new(&scene);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("learned_impact_oracle_tension");
    let meta = RecordMeta {
        event_id: "oracle_tension".into(),
        structure_type: "bond_tension".into(),
        load_type: "axial_pull".into(),
        notes: "integration test".into(),
    };
    let mut session = OracleSession::begin(&dir, &world, meta, true).expect("begin");
    for _ in 0..2 {
        session.begin_frame(&mut world).expect("begin_frame");
        world.step_frame();
        session.end_frame(&mut world).expect("end_frame");
    }
    let path = session.finish().expect("finish");
    let man = validate_event(&path).expect("validate");
    assert_eq!(man.n_frames, 2);
    assert!(man.n_chunks >= 2);
    assert!(man.n_bonds >= 1);
}

#[test]
fn wall_impact_records_contact_load_bins() {
    use stress_learned::catalog::{self, Catalog};

    let mut scene = builders::wall_impact(20.0, "bin_wall");
    scene.sim.record_peak_utilization = true;
    scene.sim.duration = scene.sim.frame_dt; // one frame
    let mut world = World::new(&scene);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("learned_impact_contact_bins");
    let meta = RecordMeta {
        event_id: "bin_wall".into(),
        structure_type: "wall_b5".into(),
        load_type: "ram_20mps".into(),
        notes: "contact bin test".into(),
    };
    let mut session = OracleSession::begin(&dir, &world, meta, false).expect("begin");
    session.begin_frame(&mut world).expect("begin_frame");
    world.step_frame();
    session.end_frame(&mut world).expect("end_frame");
    let path = session.finish().expect("finish");
    let frame = path.join("frames/000000");
    let cat = Catalog::load(&frame.join("catalog.json")).expect("catalog");
    let loads = catalog::read_f32(&frame, cat.arrays.get("chunk_load_bins").unwrap()).expect("loads");
    let contact = catalog::read_i32(&frame, cat.arrays.get("chunk_contact").unwrap()).expect("contact");
    let load_norm: f32 = loads.iter().map(|x| x.abs()).sum();
    let n_contact: i32 = contact.iter().sum();
    assert!(n_contact > 0, "expected some chunks marked contacted, got {n_contact}");
    assert!(load_norm > 0.0, "expected non-zero contact/scripted load bins, got {load_norm}");
}
