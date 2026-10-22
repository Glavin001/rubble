//! The energy envelope (`stress_learned::kin_envelope`) of every call of the given call-event
//! directories, as a JSON fixture for parity with P2's Python decode: per call its energy and
//! every node's radii, each also as its f64 bits (hex) for a bit-for-bit comparison.
//!
//!   learned_impact_envelope_fixture OUT.json EVENT_DIR [EVENT_DIR ...]

use serde_json::{json, Value};

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().expect("OUT.json");
    let mut calls = Vec::new();
    for dir in args {
        let ev = stress_learned::outcome_model::read_event(std::path::Path::new(&dir)).unwrap_or_else(|e| panic!("{dir}: {e}"));
        for (i, c) in ev.calls.iter().enumerate() {
            let e = stress_learned::kin_envelope::envelope(&ev.manifest, c).unwrap_or_else(|e| panic!("{dir} call {i}: {e}"));
            let bits = |x: f64| format!("{:016x}", x.to_bits());
            let pair = |r: (f64, f64)| json!({"dx": r.0, "drot": r.1, "dx_bits": bits(r.0), "drot_bits": bits(r.1)});
            calls.push(json!({
                "event": ev.manifest["event_id"], "dir": dir, "call_index": i,
                "energy": e.energy, "energy_bits": e.energy.map(bits),
                "chunk_radii": e.chunk_radii.iter().map(|r| r.map_or(Value::Null, pair)).collect::<Vec<_>>(),
                "impactor_radii": e.impactor_radii.iter().map(|r| r.map_or(Value::Null, pair)).collect::<Vec<_>>(),
            }));
        }
    }
    std::fs::write(&out, serde_json::to_vec(&json!({"tol": stress_learned::kin_envelope::TOL, "calls": calls})).unwrap()).unwrap();
    eprintln!("{} calls -> {out}", calls.len());
}
