//! Write one synthetic tiny event and validate it.
//!
//!   cargo run --release -p stress-learned --bin learned_impact_smoke -- /tmp/learned-impact-smoke
//!
//! Refuses paths that look like bulk dataset roots unless `--confirm-bulk` is passed
//! (bulk generation is not implemented here).

use std::env;
use std::path::PathBuf;

fn main() {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let confirm_bulk = args.iter().any(|a| a == "--confirm-bulk");
    args.retain(|a| a != "--confirm-bulk");
    let out = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/learned-impact-smoke"));
    if !confirm_bulk && looks_bulk(&out) {
        eprintln!(
            "refusing {}: looks like a bulk dataset path; pass a temp dir, or --confirm-bulk (still writes one smoke event only)",
            out.display()
        );
        std::process::exit(2);
    }
    let path = stress_learned::recorder::write_synthetic_tiny(&out).unwrap_or_else(|e| {
        eprintln!("smoke write failed: {e}");
        std::process::exit(1);
    });
    let man = stress_learned::validate_event(&path).unwrap_or_else(|e| {
        eprintln!("validate failed: {e}");
        std::process::exit(1);
    });
    println!(
        "ok {} · {} frames · {} chunks · {} bonds · K={}",
        path.display(),
        man.n_frames,
        man.n_chunks,
        man.n_bonds,
        man.k
    );
}

fn looks_bulk(p: &std::path::Path) -> bool {
    let s = p.to_string_lossy().to_lowercase();
    s.contains("datasets") || s.ends_with("learned-impact/v1") || s.contains("bulk")
}
