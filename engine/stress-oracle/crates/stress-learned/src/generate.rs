//! Catalogued event generation for M1/M2 (smoke and mini sets).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rayon::prelude::*;
use serde::Serialize;
use stress_ref::builders;
use stress_ref::scene::Scene;
use stress_ref::showcases;

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

/// Larger mini catalog (still opt-in). Walls dominate cost; other structures add diversity.
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
    // Diverse structure types (M2 held-out probes).
    jobs.push(job(
        "shear_00",
        "bond_shear",
        "mohr_coulomb_ramp",
        3,
        true,
        builders::bond_shear(),
    ));
    jobs.push(job(
        "support_loss_00",
        "support_loss",
        "sudden_remove",
        4,
        true,
        builders::support_loss(0.0, "gen_support_loss"),
    ));
    jobs.push(job(
        "frame_col_00",
        "frame_column",
        "sudden_remove",
        3,
        true,
        builders::frame_column_removal(0.0, "gen_frame_col"),
    ));
    jobs.push(job(
        "arch_00",
        "masonry_arch",
        "keystone_remove",
        4,
        true,
        showcases::arch(Some(0.05), "gen_arch"),
    ));
    jobs.push(job(
        "bar_wave_00",
        "bar_wave",
        "prestress_release",
        3,
        true,
        builders::bar_wave(1.0, "gen_bar_wave"),
    ));
    jobs.push(job(
        "panel_00",
        "pressure_panel",
        "blast_peak",
        2,
        true,
        builders::pressure_panel(5e4, "gen_panel"),
    ));
    jobs.push(job(
        "masonry_10mps_00",
        "masonry_wall",
        "ram_10mps",
        1,
        false,
        builders::masonry_wall(10.0, "gen_masonry_10"),
    ));
    jobs.push(job(
        "spall_20mps_00",
        "spall",
        "ram_20mps",
        1,
        false,
        builders::spall(20.0, "gen_spall_20"),
    ));
    jobs
}

/// Diversity prove set (small): push/ram/punch walls, crackling cantilevers,
/// loop (arch), building frame, masonry house. Hold out whole structure types.
///
/// ETA (Mac, measured wall ≈5–8 s/frame; house TBD, budget ~1–3 min/event @ 1–2 frames):
/// ~**8–15 min** for the full prove catalog.
pub fn prove_jobs() -> Vec<Job> {
    let mut jobs = Vec::new();
    // Walls: push / ram / punch regimes (same geometry, different dynamics).
    for (speed, tag) in [
        (5.0, "push"),
        (8.0, "push"),
        (12.0, "ram"),
        (15.0, "ram"),
        (25.0, "punch"),
        (40.0, "punch"),
    ] {
        jobs.push(job(
            &format!("wall_{tag}_{speed}mps"),
            "wall_b5",
            &format!("ram_{speed}mps"),
            2,
            false,
            builders::wall_impact(speed, &format!("prove_wall_{speed}")),
        ));
    }
    // Crackling cantilevers (fracture on; tip load held — see DIVERSITY.md).
    for (n, frames, scale) in [(6usize, 3u32, 1.8f64), (10, 4, 1.8), (14, 4, 1.8)] {
        jobs.push(job(
            &format!("cantilever_n{n}"),
            "cantilever",
            "tip_break_root",
            frames,
            true,
            builders::cantilever_crackling(n, scale, false, &format!("prove_cantilever_{n}")),
        ));
    }
    // Loop topology (compression ring → crack when keystone removed).
    jobs.push(job(
        "arch_keystone",
        "masonry_arch",
        "keystone_remove",
        4,
        true,
        showcases::arch(Some(0.05), "prove_arch"),
    ));
    // Building-like frame (progressive collapse path).
    jobs.push(job(
        "frame_sudden",
        "frame_column",
        "sudden_remove",
        3,
        true,
        builders::frame_column_removal(0.0, "prove_frame"),
    ));
    // Actual house structure (openings, lintels, footings).
    for (speed, frames) in [(10.0, 2u32), (20.0, 2u32)] {
        jobs.push(job(
            &format!("house_impact_{speed}mps"),
            "masonry_house",
            &format!("car_{speed}mps"),
            frames,
            false,
            showcases::masonry_house(
                showcases::HouseEvent::Impact { speed },
                &format!("prove_house_{speed}"),
            ),
        ));
    }
    jobs.push(job(
        "house_settle",
        "masonry_house",
        "settlement",
        2,
        true,
        showcases::masonry_house(showcases::HouseEvent::Settlement, "prove_house_settle"),
    ));
    // Cheap analytic anchors.
    jobs.push(job("shear_00", "bond_shear", "mohr_coulomb_ramp", 2, true, builders::bond_shear()));
    jobs.push(job("tension_00", "bond_tension", "axial_pull", 2, true, builders::bond_tension()));
    jobs.push(job(
        "support_loss_00",
        "support_loss",
        "sudden_remove",
        3,
        true,
        builders::support_loss(0.0, "prove_support_loss"),
    ));
    jobs
}

/// Diversity v1 catalog (~35 events / ~230 frames). See `learned-impact/DIVERSITY.md`.
///
/// Holdouts are train-time (`--holdout-structure-type` / load band); this set tags
/// `structure_type` / `load_type` so punch / house / long-cantilever splits are clean.
pub fn diversity_jobs() -> Vec<Job> {
    let mut jobs = Vec::new();

    // A. Walls — soft / push / ram / punch / ring-out (~14 events).
    for (speed, frames, band) in [
        (1.0, 4u32, "soft"),
        (2.0, 4, "soft"),
        (3.0, 4, "soft"),
        (4.0, 4, "soft"),
        (5.0, 8, "push"),
        (8.0, 8, "push"),
        (12.0, 8, "ram"),
        (15.0, 8, "ram"),
        (18.0, 8, "ram"),
        (25.0, 8, "punch"),
        (35.0, 8, "punch"),
        (45.0, 8, "punch"),
        (10.0, 16, "ring"),
        (20.0, 16, "ring"),
    ] {
        jobs.push(job(
            &format!("wall_{band}_{speed}mps"),
            "wall_b5",
            &format!("ram_{speed}mps"),
            frames,
            false,
            builders::wall_impact(speed, &format!("div_wall_{speed}")),
        ));
    }

    // B. Houses — openings + car (~6 events, ~56 frames).
    jobs.push(job(
        "house_settle",
        "masonry_house",
        "settlement",
        8,
        true,
        showcases::masonry_house(showcases::HouseEvent::Settlement, "div_house_settle"),
    ));
    for (speed, frames, tag) in [
        (8.0, 8u32, "car"),
        (12.0, 8, "car"),
        (20.0, 8, "car"),
        (30.0, 8, "punch"),
        (15.0, 16, "ring"),
    ] {
        jobs.push(job(
            &format!("house_{tag}_{speed}mps"),
            "masonry_house",
            &format!("car_{speed}mps"),
            frames,
            false,
            showcases::masonry_house(
                showcases::HouseEvent::Impact { speed },
                &format!("div_house_{speed}"),
            ),
        ));
    }

    // C. Crackling cantilevers (~9 events). short/mid/long = n 6/10/14.
    let arms = [("short", 6usize), ("mid", 10), ("long", 14)];
    for &(tag, n) in &arms {
        jobs.push(job(
            &format!("cantilever_{tag}_light"),
            "cantilever",
            "tip_light",
            4,
            true,
            builders::cantilever_crackling(n, 0.25, false, &format!("div_cant_{tag}_light")),
        ));
    }
    for &(tag, n) in &arms {
        jobs.push(job(
            &format!("cantilever_{tag}_root"),
            "cantilever",
            "tip_break_root",
            8,
            true,
            builders::cantilever_crackling(n, 1.8, false, &format!("div_cant_{tag}_root")),
        ));
    }
    for &(tag, n) in &[("mid", 10usize), ("long", 14)] {
        jobs.push(job(
            &format!("cantilever_{tag}_midspan"),
            "cantilever",
            "tip_break_midspan",
            8,
            true,
            builders::cantilever_crackling(n, 1.2, true, &format!("div_cant_{tag}_mid")),
        ));
    }
    jobs.push(job(
        "cantilever_mid_overload",
        "cantilever",
        "tip_overload",
        8,
        true,
        builders::cantilever_crackling(10, 6.0, false, "div_cant_mid_overload"),
    ));

    // D. Loop + frame + anchors (~6 events).
    jobs.push(job(
        "arch_keystone",
        "masonry_arch",
        "keystone_remove",
        8,
        true,
        showcases::arch(Some(0.05), "div_arch_keystone"),
    ));
    jobs.push(job(
        "arch_stand",
        "masonry_arch",
        "stand",
        4,
        true,
        showcases::arch(None, "div_arch_stand"),
    ));
    jobs.push(job(
        "frame_sudden",
        "frame_column",
        "sudden_remove",
        8,
        true,
        builders::frame_column_removal(0.0, "div_frame_sudden"),
    ));
    jobs.push(job(
        "frame_gradual",
        "frame_column",
        "gradual_remove",
        8,
        true,
        builders::frame_column_removal(0.2, "div_frame_gradual"),
    ));
    jobs.push(job("shear_fail", "bond_shear", "mohr_coulomb_ramp", 4, true, builders::bond_shear()));
    jobs.push(job("tension_fail", "bond_tension", "axial_pull", 4, true, builders::bond_tension()));

    jobs
}

/// Wall-heavy M2 set: denser speed sweep + multi-frame rollouts.
/// ETA on Mac (measured ≈8 s/frame @ 840 chunks): ~12 short + 2×8-frame ≈ **4–6 min**.
pub fn wall_jobs() -> Vec<Job> {
    let mut jobs = Vec::new();
    // Short clips across speeds (2 frames each).
    for (i, speed) in [5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0]
        .into_iter()
        .enumerate()
    {
        jobs.push(job(
            &format!("wall_{speed}mps_s{i:02}"),
            "wall_b5",
            &format!("ram_{speed}mps"),
            2,
            false,
            builders::wall_impact(speed, &format!("gen_wall_sweep_{speed}")),
        ));
    }
    // Longer rollouts at representative punch / push-over speeds.
    for (speed, frames) in [(10.0, 8u32), (20.0, 8u32)] {
        jobs.push(job(
            &format!("wall_{speed}mps_roll{frames}"),
            "wall_b5",
            &format!("ram_{speed}mps_rollout"),
            frames,
            false,
            builders::wall_impact(speed, &format!("gen_wall_roll_{speed}")),
        ));
    }
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

/// Options for [`run_jobs`].
#[derive(Clone, Debug)]
pub struct RunJobsOpts {
    /// Override every job's `qs_probe` to true (store a real QS solve on every frame).
    pub always_qs: bool,
    /// Skip event dirs that already validate (resume).
    pub resume: bool,
    /// Rayon thread count; `None` = all logical CPUs.
    pub threads: Option<usize>,
}

impl Default for RunJobsOpts {
    fn default() -> Self {
        Self { always_qs: false, resume: true, threads: None }
    }
}

/// Run jobs into `out_dir/<event_id>/` **in parallel**, write `manifest.json` at the root.
///
/// Events are independent; Rayon uses all cores by default (`opts.threads` or
/// `RAYON_NUM_THREADS`). Manifest event order matches the input `jobs` slice.
pub fn run_jobs(out_dir: &Path, set_name: &str, jobs: &[Job]) -> Result<DatasetManifest, String> {
    run_jobs_with(out_dir, set_name, jobs, &RunJobsOpts::default())
}

pub fn run_jobs_with(
    out_dir: &Path,
    set_name: &str,
    jobs: &[Job],
    opts: &RunJobsOpts,
) -> Result<DatasetManifest, String> {
    fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    if let Some(n) = opts.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .ok(); // already initialized is fine
    }
    let n_threads = rayon::current_num_threads();
    let total_frames: u32 = jobs.iter().map(|j| j.frames).sum();
    eprintln!(
        "parallel generate · {} events · {} frames · {} rayon threads · always_qs={} resume={}",
        jobs.len(),
        total_frames,
        n_threads,
        opts.always_qs,
        opts.resume
    );

    let done = AtomicUsize::new(0);
    let frames_done = AtomicUsize::new(0);
    let total = jobs.len();
    let t_all = Instant::now();
    // Shared timing for ETA: (completed_jobs, sum_job_secs) for recorded (non-skip) work.
    let timing = Mutex::new((0usize, 0.0f64));
    let set_name = set_name.to_string();
    let out_dir = out_dir.to_path_buf();
    let always_qs = opts.always_qs;
    let resume = opts.resume;

    // Preserve job order in the manifest: collect (index, entry).
    // Each job owns its Scene/World — no shared mutable oracle state across threads.
    let mut indexed: Vec<(usize, Result<ManifestEntry, String>)> = jobs
        .par_iter()
        .enumerate()
        .map(|(i, job)| {
            let dest = out_dir.join(&job.event_id);
            let qs = always_qs || job.qs_probe;
            let k = done.fetch_add(1, Ordering::Relaxed) + 1;
            if resume && dest.join("manifest.json").exists() {
                if let Ok(man) = crate::validate::validate_event(&dest) {
                    frames_done.fetch_add(man.n_frames as usize, Ordering::Relaxed);
                    eprintln!(
                        "[{k}/{total}] {} · skip (resume) · {} frames · {} chunks · {} bonds · {}",
                        job.event_id,
                        man.n_frames,
                        man.n_chunks,
                        man.n_bonds,
                        eta_str(&timing, &t_all, total, frames_done.load(Ordering::Relaxed), total_frames as usize)
                    );
                    return (
                        i,
                        Ok(ManifestEntry {
                            event_id: job.event_id.clone(),
                            structure_type: job.structure_type.clone(),
                            load_type: job.load_type.clone(),
                            path: job.event_id.clone(),
                            n_frames: man.n_frames,
                            n_chunks: man.n_chunks,
                            n_bonds: man.n_bonds,
                        }),
                    );
                }
            }
            eprintln!(
                "[{k}/{total}] {} · {} / {} · {} frames · qs={} · start",
                job.event_id, job.structure_type, job.load_type, job.frames, qs
            );
            let meta = RecordMeta {
                event_id: job.event_id.clone(),
                structure_type: job.structure_type.clone(),
                load_type: job.load_type.clone(),
                notes: format!("generated set={set_name}"),
            };
            let t_job = Instant::now();
            let result = (|| {
                let path = record_event(job.scene.clone(), &dest, meta, job.frames, qs)?;
                let man = crate::validate::validate_event(&path).map_err(|e| e.to_string())?;
                let secs = t_job.elapsed().as_secs_f64();
                {
                    let mut g = timing.lock().unwrap();
                    g.0 += 1;
                    g.1 += secs;
                }
                frames_done.fetch_add(man.n_frames as usize, Ordering::Relaxed);
                let per_frame = secs / man.n_frames.max(1) as f64;
                eprintln!(
                    "[{k}/{total}] {} · done in {:.1}s ({:.2}s/frame) · {} frames · {} chunks · {} bonds · {}",
                    job.event_id,
                    secs,
                    per_frame,
                    man.n_frames,
                    man.n_chunks,
                    man.n_bonds,
                    eta_str(&timing, &t_all, total, frames_done.load(Ordering::Relaxed), total_frames as usize)
                );
                Ok(ManifestEntry {
                    event_id: job.event_id.clone(),
                    structure_type: job.structure_type.clone(),
                    load_type: job.load_type.clone(),
                    path: job.event_id.clone(),
                    n_frames: man.n_frames,
                    n_chunks: man.n_chunks,
                    n_bonds: man.n_bonds,
                })
            })();
            (i, result)
        })
        .collect();

    indexed.sort_by_key(|(i, _)| *i);
    let mut entries = Vec::with_capacity(indexed.len());
    for (_, r) in indexed {
        entries.push(r?);
    }

    let manifest = DatasetManifest {
        schema_version: crate::SCHEMA_VERSION.into(),
        set: set_name,
        events: entries,
    };
    let text = serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?;
    fs::write(out_dir.join("manifest.json"), text).map_err(|e| e.to_string())?;
    eprintln!(
        "parallel generate finished · {} events · {:.1}s wall",
        manifest.events.len(),
        t_all.elapsed().as_secs_f64()
    );
    Ok(manifest)
}

fn eta_str(
    timing: &Mutex<(usize, f64)>,
    t_all: &Instant,
    total_jobs: usize,
    frames_done: usize,
    total_frames: usize,
) -> String {
    let (n_timed, sum_secs) = *timing.lock().unwrap();
    let wall = t_all.elapsed().as_secs_f64();
    if n_timed == 0 {
        return format!("wall={:.0}s", wall);
    }
    let remaining_jobs = total_jobs.saturating_sub(n_timed);
    let eta = if frames_done > 0 && total_frames > frames_done {
        let rate = frames_done as f64 / wall.max(1e-6);
        (total_frames - frames_done) as f64 / rate.max(1e-6)
    } else {
        let avg = sum_secs / n_timed as f64;
        let threads = rayon::current_num_threads().max(1) as f64;
        remaining_jobs as f64 * avg / threads
    };
    format!(
        "wall={:.0}s · ETA≈{:.0}s · frames {}/{}",
        wall, eta, frames_done, total_frames
    )
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
