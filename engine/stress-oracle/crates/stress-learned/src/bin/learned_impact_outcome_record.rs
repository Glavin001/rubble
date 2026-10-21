//! Record one impact-outcome event (plan P1), or check a recorded one replays.
//!
//!   learned_impact_outcome_record --recipes RECIPES.json --event ID --out DIR [--oracle NAME] [--no-baselines]
//!   learned_impact_outcome_record --m5 m5_free_wall|m5_wall_row --out DIR
//!   learned_impact_outcome_record --replay EVENT_DIR
//!   learned_impact_outcome_record --regenerate EVENT_DIR --out DIR
//!
//! `--replay` re-runs the event's recipe under the oracle configuration its `context.json`
//! recorded and fails unless the end fingerprint matches `replay.json`.
use std::path::PathBuf;

use stress_learned::oracle_config::OracleConfig;
use stress_learned::outcome::{record_outcome_event, replay_fingerprint, OutcomeRecipe, RecordOptions};
use stress_learned::procedural::m5_recipes;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let val = |s: &str| a.windows(2).find(|v| v[0] == s).map(|v| v[1].clone());
    if let Some(ev) = val("--regenerate") {
        let out = PathBuf::from(val("--out").expect("--out DIR"));
        let r = stress_learned::outcome::regenerate(&PathBuf::from(ev), &out).unwrap_or_else(|e| panic!("regenerate: {e}"));
        println!("{}", serde_json::to_string_pretty(&r).unwrap());
        return;
    }
    if let Some(ev) = val("--replay") {
        let ev = PathBuf::from(ev);
        let replay: serde_json::Value = serde_json::from_slice(&std::fs::read(ev.join("replay.json")).expect("replay.json")).unwrap();
        let (fp, frames) = replay_fingerprint(&ev).expect("replay");
        let want = replay["end_fingerprint"].as_str().unwrap();
        println!("{}: {frames} frames, fingerprint {fp} (recorded {want})", ev.display());
        if fp != want {
            eprintln!("replay diverged");
            std::process::exit(1);
        }
        return;
    }
    let out = PathBuf::from(val("--out").expect("--out DIR"));
    let recipe: OutcomeRecipe = if let Some(m) = val("--m5") {
        m5_recipes().into_iter().find(|r| r.event_id == m).expect("unknown --m5")
    } else {
        let path = val("--recipes").expect("--recipes or --m5");
        let plan: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).expect("recipes")).unwrap();
        let id = val("--event").expect("--event ID");
        let r = plan["recipes"].as_array().expect("recipes array").iter().find(|r| r["event_id"] == id.as_str()).expect("unknown --event");
        serde_json::from_value(r.clone()).expect("recipe")
    };
    let mut opts = RecordOptions::new(OracleConfig::named(&val("--oracle").unwrap_or_else(|| "a12_realtime".into())).expect("--oracle"));
    opts.baselines = !a.iter().any(|x| x == "--no-baselines");
    if let Some(s) = val("--max-seconds") {
        opts.stop.max_seconds = s.parse().expect("--max-seconds");
    }
    std::fs::create_dir_all(&out).unwrap();
    let beat = |b: &stress_learned::outcome::Beat| {
        if b.frame % 30 == 0 {
            eprintln!("[beat] {} frame {} t={:.3} s phase={} breaks={}", b.event_id, b.frame, b.time, b.phase, b.breaks);
        }
    };
    let s = record_outcome_event(&recipe, &out, &opts, &beat).unwrap_or_else(|e| panic!("{}: {e}", recipe.event_id));
    println!("{}", serde_json::to_string_pretty(&s).unwrap());
}
