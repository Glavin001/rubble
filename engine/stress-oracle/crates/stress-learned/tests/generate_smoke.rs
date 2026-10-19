use std::path::PathBuf;

use stress_learned::generate::{run_jobs, smoke_jobs};

#[test]
fn smoke_jobs_write_manifest() {
    // Skip walls in unit test time budget: run only the two tiny jobs.
    let jobs: Vec<_> = smoke_jobs()
        .into_iter()
        .filter(|j| j.structure_type != "wall_b5")
        .collect();
    assert!(jobs.len() >= 2);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("learned_impact_gen_smoke");
    let man = run_jobs(&dir, "smoke_test", &jobs).expect("generate");
    assert_eq!(man.events.len(), jobs.len());
    assert!(dir.join("manifest.json").exists());
    for e in &man.events {
        assert!(dir.join(&e.path).join("manifest.json").exists());
    }
}
