//! Record one oracle event for learned-impact (smoke / M1).
//!
//! ```text
//! cargo run --release -p stress-learned --bin learned_impact_record -- \
//!   --scene wall --frames 2 --out /tmp/li-wall
//! ```

use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

use stress_learned::generate::resolve_pack;
use stress_learned::{record_event, validate_event, RecordMeta};
use stress_ref::builders;
use stress_ref::scene::Scene;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let confirm_bulk = has(&args, "--confirm-bulk");
    let qs = !has(&args, "--no-qs");
    let frames: u32 = flag_val(&args, "--frames").and_then(|s| s.parse().ok()).unwrap_or(2);
    let out = flag_val(&args, "--out").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/tmp/learned-impact-record"));
    if !confirm_bulk && looks_bulk(&out) {
        eprintln!("refusing {}: pass a temp dir or --confirm-bulk", out.display());
        std::process::exit(2);
    }

    let (scene, structure_type, load_type) = load_scene(&args);
    let t0 = Instant::now();
    let meta = RecordMeta {
        event_id: out.file_name().and_then(|s| s.to_str()).unwrap_or("event").into(),
        structure_type,
        load_type,
        notes: format!("oracle record · frames={frames} · qs_probe={qs}"),
    };
    let path = record_event(scene, &out, meta, frames, qs).unwrap_or_else(|e| {
        eprintln!("record failed: {e}");
        std::process::exit(1);
    });
    let man = validate_event(&path).unwrap_or_else(|e| {
        eprintln!("validate: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "ok {} · {} frames · {} chunks · {} bonds · {:.1}s wall",
        path.display(),
        man.n_frames,
        man.n_chunks,
        man.n_bonds,
        t0.elapsed().as_secs_f64()
    );
}

fn load_scene(args: &[String]) -> (Scene, String, String) {
    if let Some(pack) = flag_val(args, "--pack") {
        let path = resolve_pack(Path::new(pack));
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("pack");
        let scene = stress_ref::scene_pack::import(&path, stem, "structure").unwrap_or_else(|e| {
            eprintln!("import {}: {e}", path.display());
            std::process::exit(1);
        });
        return (scene, stem.into(), "pack_default".into());
    }
    let kind = flag_val(args, "--scene").unwrap_or("wall");
    match kind {
        "wall" => {
            let speed: f64 = flag_val(args, "--speed").and_then(|s| s.parse().ok()).unwrap_or(20.0);
            (builders::wall_impact(speed, "learned_wall"), "wall_b5".into(), format!("ram_{speed}mps"))
        }
        "cantilever" => (builders::cantilever(8, "learned_cantilever"), "cantilever".into(), "tip_static".into()),
        "tension" => (builders::bond_tension(), "bond_tension".into(), "axial_pull".into()),
        other => {
            eprintln!("unknown --scene {other} (wall|cantilever|tension) or use --pack");
            std::process::exit(2);
        }
    }
}

fn has(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn flag_val<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).map(|s| s.as_str()))
}

fn looks_bulk(p: &Path) -> bool {
    let s = p.to_string_lossy().to_lowercase();
    s.contains("datasets") || s.contains("bulk") || s.ends_with("learned-impact/v1")
}
