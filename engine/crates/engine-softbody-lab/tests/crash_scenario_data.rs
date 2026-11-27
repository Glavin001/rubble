//! Scenarios as data (`crash_reference::desc`) are executed by the same code as the Rust
//! ones: S02 exported as data (`to_desc`), through JSON and back (`ScenarioDesc::scenario`),
//! builds the same fleet and steps to the same state, word for word, on the CPU reference.
//! Also: every committed data scenario of the Reference Studio's workspace builds and runs a
//! little (no NaN). `--release` (rigging the 4x4).

use engine_softbody::crash_reference::{self as cr, Contact, CpuRunner, Jitter, Runner, ScenarioDesc};

fn words(r: &mut CpuRunner) -> Vec<u32> {
    let x = r.sample().unwrap();
    x.pos.iter().chain(&x.vel).flatten().flat_map(|p| p.to_array()).map(f32::to_bits).chain(x.beams.iter().flatten().flat_map(|b| [b.rest.to_bits(), u32::from(b.broken)])).collect()
}

#[test]
fn s02_as_data_is_s02() {
    let code = cr::get("S02").expect("S02");
    let templates = engine_softbody_authoring::crash_templates::templates_for(&code).expect("templates");
    let desc = code.to_desc(&templates).expect("to_desc");
    assert_eq!(desc.bodies.len(), 1);
    assert_eq!(desc.bodies[0].speed_kmh, 50.0, "S02's 50 km/h is stated as it is in Rust");
    let text = serde_json::to_string_pretty(&desc).unwrap();
    let back: ScenarioDesc = serde_json::from_str(&text).unwrap();
    assert_eq!(back, desc, "the description survives JSON");
    back.check().expect("check");
    let data = back.scenario();
    assert!(data.recreated);
    let (a, b) = (code.build(&templates, Jitter::run(0), Contact::Edge).unwrap(), data.build(&templates, Jitter::run(0), Contact::Edge).unwrap());
    assert_eq!(a.fleet.len(), b.fleet.len());
    for (x, y) in a.layout.bodies.iter().zip(&b.layout.bodies) {
        assert_eq!((x.at, x.yaw.to_bits(), x.speed.to_bits(), x.vel), (y.at, y.yaw.to_bits(), y.speed.to_bits(), y.vel));
    }
    assert_eq!(a.layout.boxes, b.layout.boxes);
    // The data scenario's first contact is computed (the SUV's nose box to the wall's face at
    // its speed): S02 states 0.1 s.
    println!("first contact: code {} s, data {} s", a.layout.impact_s, b.layout.impact_s);
    assert!((b.layout.impact_s - 0.1).abs() < 1e-4, "{}", b.layout.impact_s);
    let (mut ra, mut rb) = (CpuRunner::new(&a), CpuRunner::new(&b));
    assert_eq!(words(&mut ra), words(&mut rb), "spawn state");
    // Through first contact (0.1 s) to 0.2 s: 400 steps.
    for k in 0..50 {
        ra.frame(8).unwrap();
        rb.frame(8).unwrap();
        assert_eq!(words(&mut ra), words(&mut rb), "state words after {} steps", (k + 1) * 8);
    }
}

/// The data scenarios the studio has committed: each builds and runs 0.1 s with a finite
/// state; their criteria are the law-based ones only.
#[test]
fn workspace_scenarios_build_and_run() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../apps/reference-studio/workspace/scenarios");
    let list = cr::desc::load_dir(&dir).expect("load");
    for d in list {
        let s = d.scenario();
        let templates = engine_softbody_authoring::crash_templates::templates_for(&s).expect("templates");
        let built = s.build(&templates, Jitter::run(0), d.contact).expect("build");
        let mut r = CpuRunner::new(&built);
        r.frame(200).unwrap();
        let x = r.sample().unwrap();
        assert!(x.pos.iter().flatten().all(|p| p.is_finite()), "{}", d.id);
        assert!((s.criteria)().iter().all(|c| c.basis != cr::Basis::Estimate), "{}: law-based criteria only", d.id);
    }
}
