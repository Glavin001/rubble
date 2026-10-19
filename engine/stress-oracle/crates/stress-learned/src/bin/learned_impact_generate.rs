//! Generate smoke / mini / walls / prove / diversity learned-impact event sets (parallel).
//!
//! ```text
//! cargo run --release -p stress-learned --bin learned_impact_generate -- \
//!   --prove --confirm-prove --out /tmp/li-prove-set
//!
//! cargo run --release -p stress-learned --bin learned_impact_generate -- \
//!   --diversity --confirm-diversity --out /tmp/li-diversity-v1
//!
//! cargo run --release -p stress-learned --bin learned_impact_generate -- \
//!   --diversity-v2 --confirm-diversity --out /tmp/li-diversity-v2 --jobs 16
//!
//! # All logical CPUs (default). Cap with --jobs N or RAYON_NUM_THREADS.
//! # Re-run skips valid event dirs. Force a real QS probe on every frame with --always-qs.
//! ```

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use stress_learned::generate::{
    diversity_jobs, diversity_v2_jobs, mini_jobs, prove_jobs, run_jobs_with, smoke_jobs, wall_jobs,
    RunJobsOpts,
};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let diversity_v2 = has(&args, "--diversity-v2");
    let diversity = has(&args, "--diversity") || has(&args, "--diversity-v1");
    let prove = has(&args, "--prove");
    let walls = has(&args, "--walls");
    let mini = has(&args, "--mini");
    let smoke =
        has(&args, "--smoke") || (!mini && !walls && !prove && !diversity && !diversity_v2);

    if mini && !has(&args, "--confirm-mini") {
        eprintln!("refusing --mini without --confirm-mini (~1 min serial; less in parallel)");
        std::process::exit(2);
    }
    if walls && !has(&args, "--confirm-walls") {
        eprintln!("refusing --walls without --confirm-walls");
        std::process::exit(2);
    }
    if prove && !has(&args, "--confirm-prove") {
        eprintln!("refusing --prove without --confirm-prove");
        std::process::exit(2);
    }
    if (diversity || diversity_v2) && !has(&args, "--confirm-diversity") {
        eprintln!(
            "refusing --diversity / --diversity-v2 without --confirm-diversity \
             (v1 ~35 events / ~230 frames; v2 ~100+ events / ~900+ frames; parallel with --jobs)"
        );
        std::process::exit(2);
    }
    let out = flag_val(&args, "--out").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(if diversity_v2 {
            "/tmp/li-diversity-v2"
        } else if diversity {
            "/tmp/li-diversity-v1"
        } else if prove {
            "/tmp/li-prove-set"
        } else if walls {
            "/tmp/li-wall-set"
        } else if mini {
            "/tmp/li-mini-set"
        } else {
            "/tmp/li-smoke-set"
        })
    });
    if looks_bulk(&out) && !has(&args, "--confirm-bulk") {
        eprintln!("refusing {}: pass a temp dir or --confirm-bulk", out.display());
        std::process::exit(2);
    }

    let (set_name, jobs) = if diversity_v2 {
        ("diversity_v2", diversity_v2_jobs())
    } else if diversity {
        ("diversity_v1", diversity_jobs())
    } else if prove {
        ("prove", prove_jobs())
    } else if walls {
        ("walls", wall_jobs())
    } else if mini {
        ("mini", mini_jobs())
    } else {
        let _ = smoke;
        ("smoke", smoke_jobs())
    };

    let threads = flag_val(&args, "--jobs").and_then(|s| s.parse().ok());
    let opts = RunJobsOpts {
        rigid_preview: has(&args, "--rigid-preview"),
        always_qs: has(&args, "--always-qs"),
        resume: !has(&args, "--no-resume"),
        threads,
    };

    let heavy_frames: u32 = jobs
        .iter()
        .filter(|j| {
            j.structure_type.contains("wall")
                || j.structure_type == "spall"
                || j.structure_type == "masonry_wall"
                || j.structure_type == "masonry_house"
                || j.structure_type == "frame_column"
        })
        .map(|j| j.frames)
        .sum();
    let house_frames: u32 = jobs
        .iter()
        .filter(|j| j.structure_type == "masonry_house")
        .map(|j| j.frames)
        .sum();
    let other_heavy = heavy_frames.saturating_sub(house_frames);
    let total_frames: u32 = jobs.iter().map(|j| j.frames).sum();
    // Serial ETA; parallel ≈ / min(events, cores) for wall-dominated sets.
    let eta_serial = other_heavy as f64 * 6.0 + house_frames as f64 * 8.0;
    let workers = threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8)
    });
    let eta_par = eta_serial / (workers.min(jobs.len()).max(1) as f64);
    eprintln!(
        "generating set={set_name} · {} events · {} frames · heavy_frames≈{} → {}  (ETA serial≈{:.0}s · parallel≈{:.0}s @ {} workers)",
        jobs.len(),
        total_frames,
        heavy_frames,
        out.display(),
        eta_serial,
        eta_par,
        workers
    );
    let t0 = Instant::now();
    let man = run_jobs_with(&out, set_name, &jobs, &opts).unwrap_or_else(|e| {
        eprintln!("generate failed: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "ok set={set_name} · {} events · {:.1}s wall ({:.1} min) · manifest {}",
        man.events.len(),
        t0.elapsed().as_secs_f64(),
        t0.elapsed().as_secs_f64() / 60.0,
        out.join("manifest.json").display()
    );
}

fn has(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn flag_val<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
}

fn looks_bulk(p: &std::path::Path) -> bool {
    let s = p.to_string_lossy().to_lowercase();
    s.contains("datasets") || s.contains("bulk") || s.ends_with("learned-impact/v1")
}
