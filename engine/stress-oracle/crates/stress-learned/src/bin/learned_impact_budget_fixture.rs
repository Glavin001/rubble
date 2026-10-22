//! The event energy budget (`stress_learned::energy_budget`) before every call of the given
//! call-event directories, from the calls' own energy scalars, as a JSON fixture for parity
//! with P2's Python decode: per call `E_ref`, the ledger's gain so far and the remaining budget,
//! each also as its f64 bits (hex) for a bit-for-bit comparison, and the call's own change of
//! its island's mechanical energy.
//!
//!   learned_impact_budget_fixture OUT.json EVENT_DIR [EVENT_DIR ...]

use serde_json::json;

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().expect("OUT.json");
    let mut calls = Vec::new();
    let bits = |x: f64| format!("{:016x}", x.to_bits());
    for dir in args {
        let ev = stress_learned::outcome_model::read_event(std::path::Path::new(&dir)).unwrap_or_else(|e| panic!("{dir}: {e}"));
        let energy: Vec<_> = ev.calls.iter().map(|c| &c.energy).collect();
        let ledger = stress_learned::energy_budget::ledger_of_calls(&energy).unwrap_or_else(|e| panic!("{dir}: {e}"));
        for (i, ((l, r), c)) in ledger.iter().zip(&ev.calls).enumerate() {
            let d = stress_learned::energy_budget::call_delta(&c.energy).unwrap();
            calls.push(json!({
                "event": ev.manifest["event_id"], "dir": dir, "call_index": i, "frame": c.frame,
                "E_budget_ref": l.e_ref, "E_budget_gain": l.gain, "E_budget_remaining": r, "delta": d,
                "E_budget_ref_bits": bits(l.e_ref), "E_budget_gain_bits": bits(l.gain), "E_budget_remaining_bits": bits(*r), "delta_bits": bits(d),
            }));
        }
    }
    std::fs::write(&out, serde_json::to_vec(&json!({"tol": stress_learned::energy_budget::TOL, "calls": calls})).unwrap()).unwrap();
    eprintln!("{} calls -> {out}", calls.len());
}
