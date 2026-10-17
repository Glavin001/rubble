//! Full destruction of a Shardwright asset (a `shardwright-stress/1` export, see
//! `tools/shardwright-export`) in the standalone world: the intact asset at rest, and
//! three loads: the ram of `stress-gpu`'s `scenes::ram_at`, frac-sim's default
//! projectile, and the sudden removal of one authored component (frac-sim's `--remove`).
//! Writes the observation (`stress-observation/1`) of each case, for `stress-ref
//! compare` against another backend's run of the same scene, and the scene itself.
//!
//!   cargo run --release --example shardwright_destruction -- <export.json> <case> [opts]
//!
//! `<case>`: `idle`, `ram`, `projectile` or `remove`. Options: `--duration S`,
//! `--remove NAME` (default `f0_col_21`), `--out DIR` (writes `DIR/<case>.scene.json`
//! and `DIR/<case>.obs.json`), `--every N` (progress every N frames, default 6),
//! `--feature NAME=on|off`, `--set path=json`.
//!
//! Every frame is timed whole; the first frame (gravity prestress, first factorization)
//! and the world's construction are reported separately from the peak of the rest.

use stress_ref::shardwright::{case_from_args, run_timed};
use stress_ref::world::World;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (scene, case, out, every) = case_from_args(&args)?;
    let t = std::time::Instant::now();
    let world = World::new(&scene);
    let obs = run_timed(&scene, world, t.elapsed().as_secs_f64(), every);
    if let Some(dir) = &out {
        std::fs::write(dir.join(format!("{case}.obs.json")), obs.to_json_pretty() + "\n").map_err(|e| e.to_string())?;
    }
    Ok(())
}
