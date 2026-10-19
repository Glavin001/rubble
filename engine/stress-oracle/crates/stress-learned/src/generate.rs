//! Catalogued event generation for M1/M2 (smoke and mini sets).

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use stress_ref::builders;
use stress_ref::scene::Scene;

use crate::oracle::{record_event, RecordMeta};

#[derive(Clone, Debug)]
pub struct Job {
    pub event_id: String,
    pub structure_type: String,
    pub load_type: String,
    pub frames: u32,
    pub qs_probe: bool,
    pub scene: Scene,
}

#[derive(Clone, Debug, Serialize)]
pub struct ManifestEntry {
    pub event_id: String,
    pub structure_type: String,
    pub load_type: String,
    pub path: String,
    pub n_frames: u32,
    pub n_chunks: u32,
    pub n_bonds: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct DatasetManifest {
    pub schema_version: String,
    pub set: String,
    pub events: Vec<ManifestEntry>,
}

/// Tiny fast set for CI / smoke (~seconds).
pub fn smoke_jobs() -> Vec<Job> {
    vec![
        job("tension_00", "bond_tension", "axial_pull", 2, true, builders::bond_tension()),
        job("cantilever_00", "cantilever", "tip_static", 2, true, builders::cantilever(6, "gen_cantilever")),
        job("wall_10mps_00", "wall_b5", "ram_10mps", 1, false, builders::wall_impact(10.0, "gen_wall_10")),
        job("wall_20mps_00", "wall_b5", "ram_20mps", 1, false, builders::wall_impact(20.0, "gen_wall_20")),
    ]
}

/// Larger mini catalog (still opt-in). Walls dominate cost.
pub fn mini_jobs() -> Vec<Job> {
    let mut jobs = smoke_jobs();
    for (i, speed) in [5.0, 15.0, 30.0, 40.0].into_iter().enumerate() {
        jobs.push(job(
            &format!("wall_{speed}mps_{i:02}"),
            "wall_b5",
            &format!("ram_{speed}mps"),
            2,
            false,
            builders::wall_impact(speed, &format!("gen_wall_{speed}")),
        ));
    }
    jobs.push(job(
        "cantilever_01",
        "cantilever",
        "tip_static",
        4,
        true,
        builders::cantilever(10, "gen_cantilever_10"),
    ));
    jobs.push(job(
        "tension_01",
        "bond_tension",
        "axial_pull",
        4,
        true,
        builders::bond_tension(),
    ));
    jobs
}

fn job(id: &str, structure: &str, load: &str, frames: u32, qs: bool, scene: Scene) -> Job {
    Job {
        event_id: id.into(),
        structure_type: structure.into(),
        load_type: load.into(),
        frames,
        qs_probe: qs,
        scene,
    }
}

/// Run jobs into `out_dir/<event_id>/`, write `manifest.json` at the root.
pub fn run_jobs(out_dir: &Path, set_name: &str, jobs: &[Job]) -> Result<DatasetManifest, String> {
    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let mut entries = Vec::new();
    for (i, job) in jobs.iter().enumerate() {
        let dest = out_dir.join(&job.event_id);
        eprintln!(
            "[{}/{}] {} · {} / {} · {} frames · qs={}",
            i + 1,
            jobs.len(),
            job.event_id,
            job.structure_type,
            job.load_type,
            job.frames,
            job.qs_probe
        );
        let meta = RecordMeta {
            event_id: job.event_id.clone(),
            structure_type: job.structure_type.clone(),
            load_type: job.load_type.clone(),
            notes: format!("generated set={set_name}"),
        };
        let path = record_event(job.scene.clone(), &dest, meta, job.frames, job.qs_probe)?;
        let man = crate::validate::validate_event(&path).map_err(|e| e.to_string())?;
        entries.push(ManifestEntry {
            event_id: job.event_id.clone(),
            structure_type: job.structure_type.clone(),
            load_type: job.load_type.clone(),
            path: job.event_id.clone(),
            n_frames: man.n_frames,
            n_chunks: man.n_chunks,
            n_bonds: man.n_bonds,
        });
    }
    let manifest = DatasetManifest {
        schema_version: crate::SCHEMA_VERSION.into(),
        set: set_name.into(),
        events: entries,
    };
    let text = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
    fs::write(out_dir.join("manifest.json"), text).map_err(|e| e.to_string())?;
    Ok(manifest)
}

pub fn resolve_pack(p: &Path) -> PathBuf {
    if p.is_absolute() || p.exists() {
        return p.to_path_buf();
    }
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cand = here.join("../../packs").join(p);
    if cand.exists() {
        return cand;
    }
    let cand2 = here.join("../../").join(p);
    if cand2.exists() {
        return cand2;
    }
    p.to_path_buf()
}
