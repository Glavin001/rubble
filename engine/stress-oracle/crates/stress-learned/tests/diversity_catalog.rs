//! Diversity v1/v2 catalog shape (no bulk oracle generate).

use std::collections::HashSet;
use std::path::PathBuf;

use stress_learned::generate::{diversity_jobs, diversity_v2_jobs, prove_jobs, Job};
use stress_learned::{record_event, RecordMeta};
use stress_ref::builders;

#[test]
fn diversity_v1_counts_and_tags() {
    let jobs = diversity_jobs();
    assert_eq!(jobs.len(), 35, "DIVERSITY.md ≈35 events");
    let frames: u32 = jobs.iter().map(|j| j.frames).sum();
    // Soft×4×4 + push/ram/punch + ring×16×2 + houses + cants + D ≈ 260–270.
    assert!(
        (250..=280).contains(&frames),
        "expected ~260 frames, got {frames}"
    );

    let ids: HashSet<_> = jobs.iter().map(|j| j.event_id.as_str()).collect();
    assert_eq!(ids.len(), jobs.len(), "duplicate event_id");

    assert!(jobs.iter().any(|j| j.event_id.starts_with("wall_soft_")));
    assert!(jobs.iter().any(|j| j.event_id.starts_with("wall_push_")));
    assert!(jobs.iter().any(|j| j.event_id.starts_with("wall_ram_")));
    assert!(jobs.iter().any(|j| j.event_id.starts_with("wall_punch_")));
    assert!(jobs.iter().any(|j| j.event_id.starts_with("wall_ring_")));
    assert!(jobs.iter().any(|j| j.frames == 16), "ring-out 16-frame clips");

    assert!(jobs.iter().filter(|j| j.structure_type == "masonry_house").count() >= 6);
    assert!(jobs.iter().any(|j| j.event_id == "arch_keystone"));
    assert!(jobs.iter().any(|j| j.event_id == "arch_stand"));
    assert!(jobs.iter().any(|j| j.event_id == "frame_sudden"));
    assert!(jobs.iter().any(|j| j.event_id == "frame_gradual"));

    let cants: Vec<_> = jobs.iter().filter(|j| j.structure_type == "cantilever").collect();
    assert_eq!(cants.len(), 9);
    assert!(cants.iter().all(|j| j.scene.sim.fracture), "cantilevers must fracture");
    assert!(cants.iter().any(|j| j.load_type == "tip_light"));
    assert!(cants.iter().any(|j| j.load_type == "tip_break_root"));
    assert!(cants.iter().any(|j| j.load_type == "tip_break_midspan"));
    assert!(cants.iter().any(|j| j.load_type == "tip_overload"));

    // Holdout keys: punch speeds and house structure are distinct.
    assert!(jobs.iter().any(|j| j.load_type.contains("45")));
    assert!(jobs.iter().any(|j| j.structure_type == "masonry_house"));
}

#[test]
fn prove_still_seventeen_with_fracture_cantilevers() {
    let jobs = prove_jobs();
    assert_eq!(jobs.len(), 17);
    let cants: Vec<_> = jobs.iter().filter(|j| j.structure_type == "cantilever").collect();
    assert_eq!(cants.len(), 3);
    assert!(cants.iter().all(|j| j.scene.sim.fracture));
}

#[test]
fn crackling_cantilever_can_break() {
    let scene = builders::cantilever_crackling(6, 2.0, false, "test_crack_cant");
    assert!(scene.sim.fracture);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("learned_impact_crack_cant");
    let meta = RecordMeta {
        event_id: "crack_cant".into(),
        structure_type: "cantilever".into(),
        load_type: "tip_break_root".into(),
        notes: "fracture smoke".into(),
    };
    let path = record_event(scene, &dir, meta, 4, true).expect("record");
    let man = stress_learned::validate_event(&path).expect("validate");
    assert_eq!(man.n_frames, 4);

    use stress_learned::catalog::{self, Catalog};
    let mut any_fail = false;
    let mut max_peak = 0.0f32;
    for fi in 0..man.n_frames {
        let frame = catalog::frame_dir(&path, fi);
        let cat = Catalog::load(&frame.join("catalog.json")).unwrap();
        let fail = catalog::read_i32(&frame, cat.arrays.get("bond_fail_class").unwrap()).unwrap();
        let peak = catalog::read_f32(&frame, cat.arrays.get("bond_peak_util").unwrap()).unwrap();
        any_fail |= fail.iter().any(|&c| c > 0);
        max_peak = max_peak.max(peak.iter().cloned().fold(0.0, f32::max));
    }
    assert!(
        any_fail || max_peak >= 0.95,
        "expected a break or near-fail util on crackling cantilever (max_peak={max_peak})"
    );
}

#[test]
fn diversity_job_scenes_build() {
    // Cheap: construct every scene once (World::new is heavier; just Scene build).
    let jobs: Vec<Job> = diversity_jobs();
    for j in &jobs {
        assert!(!j.scene.bodies.is_empty(), "{} empty bodies", j.event_id);
        assert!(j.frames >= 1);
    }
}

#[test]
fn diversity_v2_counts_and_new_ids() {
    let jobs = diversity_v2_jobs();
    assert!(jobs.len() >= 90, "v2 should be a large expand, got {}", jobs.len());
    let frames: u32 = jobs.iter().map(|j| j.frames).sum();
    assert!(frames >= 700, "v2 expects ~900+ frames, got {frames}");

    let ids: HashSet<_> = jobs.iter().map(|j| j.event_id.as_str()).collect();
    assert_eq!(ids.len(), jobs.len(), "duplicate event_id");
    assert!(ids.iter().all(|id| id.starts_with("v2_")), "v2 ids must be new");

    // No overlap with v1 event ids.
    let v1: HashSet<_> = diversity_jobs().iter().map(|j| j.event_id.clone()).collect();
    for j in &jobs {
        assert!(!v1.contains(&j.event_id), "v2 id {} collides with v1", j.event_id);
    }

    assert!(jobs.iter().any(|j| j.event_id.contains("wall_ang_")), "angled walls");
    assert!(jobs.iter().any(|j| j.frames >= 24), "24+ frame ring-outs");
    assert!(jobs.iter().any(|j| j.frames == 32), "32-frame ring-outs");
    assert!(jobs.iter().any(|j| j.event_id.starts_with("v2_wall_soft_")));
    assert!(jobs.iter().filter(|j| j.structure_type == "masonry_house").count() >= 15);
    assert!(jobs.iter().filter(|j| j.structure_type == "cantilever").count() >= 30);
    assert!(jobs.iter().any(|j| j.structure_type == "masonry_wall"));
    assert!(jobs.iter().any(|j| j.structure_type == "spall"));

    let cants: Vec<_> = jobs.iter().filter(|j| j.structure_type == "cantilever").collect();
    assert!(cants.iter().all(|j| j.scene.sim.fracture));
}

#[test]
fn diversity_v2_job_scenes_build() {
    let jobs: Vec<Job> = diversity_v2_jobs();
    for j in &jobs {
        assert!(!j.scene.bodies.is_empty(), "{} empty bodies", j.event_id);
        assert!(j.frames >= 1);
    }
}
