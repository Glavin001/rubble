//! Reproducible parameter catalog with inspectable recipes before/after generation.
use std::{fs, path::PathBuf};
use stress_learned::{
    generate::{run_jobs_with, RunJobsOpts},
    permutations::{permutation_jobs, projectile_spin_jobs, subdivide_frames},
    layouts::{layout_jobs, scale_layout_job},
    forcing::{forcing_jobs, point_force_pilot_jobs, point_impulse_pilot_jobs, point_impulse_range_jobs},
};
fn main() {
    let a: Vec<_> = std::env::args().skip(1).collect();
    let val = |s: &str| a.windows(2).find(|v| v[0] == s).map(|v| v[1].clone());
    let out = PathBuf::from(val("--out").expect("--out DIR required"));
    let seed = val("--seed").map(|x| x.parse().unwrap()).unwrap_or(101);
    let catalog = val("--catalog").unwrap_or_else(||"permutations-v3".into());
    let mut jobs = match catalog.as_str() {
        "permutations-v3" => permutation_jobs(seed),
        "projectile-spin-v22" => projectile_spin_jobs(seed),
        "layouts-v4" => layout_jobs(seed),
        "scale-layout-v4" => vec![scale_layout_job(seed)],
        "forcing-v8" => forcing_jobs(seed),
        "point-force-pilot-v9" => point_force_pilot_jobs(seed),
        "point-impulse-pilot-v10" => point_impulse_pilot_jobs(seed),
        "point-impulse-range-v11" => point_impulse_range_jobs(seed),
        _ => panic!("unknown catalog: {catalog}"),
    };
    if let Some(f) = val("--family") {
        jobs.retain(|j| j.structure_type == f);
    }
    if let Some(prefix) = val("--event-prefix") {
        jobs.retain(|j| j.event_id.starts_with(&prefix));
    }
    if let Some(ids) = val("--event-ids") {
        let ids: std::collections::BTreeSet<_> = ids.split(',').map(str::trim).collect();
        assert!(!ids.contains(""), "--event-ids contains an empty ID");
        for id in &ids { assert!(jobs.iter().any(|j| j.event_id == *id), "unknown/filtered event ID: {id}"); }
        jobs.retain(|j| ids.contains(j.event_id.as_str()));
    }
    if let Some(n) = val("--limit") {
        jobs.truncate(n.parse().unwrap());
    }
    assert!(!jobs.is_empty(), "no selected jobs");
    if let Some(frames) = val("--frames") {
        let frames: u32 = frames.parse().expect("--frames must be a positive integer");
        assert!(frames > 0, "--frames must be positive");
        for job in &mut jobs {
            job.frames = frames;
            job.scene.sim.duration = frames as f64 * job.scene.sim.frame_dt;
            job.event_id = format!("{}_f{frames}", job.event_id);
            job.scene.name = job.event_id.clone();
            // Keep split_group as the original physical recipe, so extending a
            // clip cannot imply a new independent train/test scenario.
        }
    }
    if let Some(factor) = val("--subdivide-frames") {
        let factor: u32 = factor.parse().expect("--subdivide-frames must be a positive integer");
        for job in &mut jobs { subdivide_frames(job, factor).expect("subdivide recording frames"); }
    }
    fs::create_dir_all(&out).unwrap();
    let recipes:Vec<_>=jobs.iter().map(|j|serde_json::json!({"event_id":j.event_id,"frames":j.frames,"qs_probe":j.qs_probe,"parameters":j.scene.oracle["learned_impact"],"scene":j.scene,"solver_methods":j.scene.sim.methods,"rigid_preview":a.iter().any(|x| x == "--rigid-preview")})).collect();
    let path = out.join("recipes.json");
    let bytes = serde_json::to_vec_pretty(&recipes).unwrap();
    if path.exists() {
        assert_eq!(
            fs::read(&path).unwrap(),
            bytes,
            "recipe changed; choose a new output directory"
        );
    } else {
        fs::write(path, bytes).unwrap();
    }
    eprintln!(
        "{} recipes / {} frames",
        jobs.len(),
        jobs.iter().map(|j| j.frames).sum::<u32>()
    );
    if a.iter().any(|x| x == "--plan-only") {
        return;
    }
    let opts = RunJobsOpts {
        rigid_preview: a.iter().any(|x| x == "--rigid-preview"),
        always_qs: false,
        resume: true,
        threads: Some(val("--jobs").map(|x| x.parse().unwrap()).unwrap_or(8)),
    };
    let set = if catalog == "permutations-v3" { "permutations_v3" } else { &catalog };
    run_jobs_with(&out, set, &jobs, &opts).expect("record permutations");
}
