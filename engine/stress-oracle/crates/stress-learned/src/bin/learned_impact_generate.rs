//! Generate a smoke or mini learned-impact event set.
//!
//! ```text
//! # ~4 events, ~10–20 s wall (default)
//! cargo run --release -p stress-learned --bin learned_impact_generate -- \
//!   --smoke --out /tmp/li-smoke-set
//!
//! # Larger mini catalog (needs --confirm-mini; minutes of wall time)
//! cargo run --release -p stress-learned --bin learned_impact_generate -- \
//!   --mini --confirm-mini --out /tmp/li-mini-set
//! ```

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use stress_learned::generate::{mini_jobs, run_jobs, smoke_jobs};

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let smoke = has(&args, "--smoke") || !has(&args, "--mini");
    let mini = has(&args, "--mini");
    if mini && !has(&args, "--confirm-mini") {
        eprintln!("refusing --mini without --confirm-mini (several wall impacts; minutes of CPU)");
        std::process::exit(2);
    }
    let out = flag_val(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(if mini { "/tmp/li-mini-set" } else { "/tmp/li-smoke-set" }));
    if looks_bulk(&out) && !has(&args, "--confirm-bulk") {
        eprintln!("refusing {}: pass a temp dir or --confirm-bulk", out.display());
        std::process::exit(2);
    }

    let (set_name, jobs) = if mini {
        ("mini", mini_jobs())
    } else {
        let _ = smoke;
        ("smoke", smoke_jobs())
    };
    eprintln!("generating set={set_name} · {} events → {}", jobs.len(), out.display());
    let t0 = Instant::now();
    let man = run_jobs(&out, set_name, &jobs).unwrap_or_else(|e| {
        eprintln!("generate failed: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "ok set={set_name} · {} events · {:.1}s wall · manifest {}",
        man.events.len(),
        t0.elapsed().as_secs_f64(),
        out.join("manifest.json").display()
    );
}

fn has(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn flag_val<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).map(|s| s.as_str()))
}

fn looks_bulk(p: &std::path::Path) -> bool {
    let s = p.to_string_lossy().to_lowercase();
    s.contains("datasets") || s.contains("bulk") || s.ends_with("learned-impact/v1")
}
