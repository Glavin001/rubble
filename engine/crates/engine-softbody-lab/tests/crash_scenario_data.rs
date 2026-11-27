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

/// The studio's scene (`crash_reference::studio`, what the page's wasm binds): S02 as data
/// placed, run by the CPU path to its end, its recording through bytes and back, positions
/// at a time, Rubble's outline through a hand-set camera and the comparison table, all from
/// the recording.
#[test]
fn studio_records_and_outlines_from_the_recording() {
    use engine_softbody::crash_reference::{outline::FrameCamera, shape::Schema, studio::*};
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../apps/reference-studio/workspace");
    let desc: ScenarioDesc = serde_json::from_str(&std::fs::read_to_string(root.join("scenarios/s02-data.json")).unwrap()).unwrap();
    let schema: Schema = serde_json::from_str(&std::fs::read_to_string(root.join("structures/vehicle-side.json")).unwrap()).unwrap();
    let mut st = Studio::new();
    let src = SceneSource { desc: Some(desc.clone()), id: None, contact: None };
    for d in st.missing(&src).unwrap() {
        st.templates.insert(d.clone(), engine_softbody_authoring::crash_templates::template(&d).unwrap());
    }
    let info = st.place_now(&src).unwrap();
    println!("placed in {:.1} ms: {} bodies, {} samples, impact {} s", info.place_ms, info.bodies.len(), info.samples, info.impact_s);
    assert_eq!(info.bodies[0].triangles.len() % 3, 0);
    assert!(!info.bodies[0].triangles.is_empty());
    assert_eq!(st.spawn_positions().unwrap().len(), 3 * info.bodies[0].nodes as usize);
    // Live edit: the wall moved 1 m further; the first contact moves by 1 m at 13.9 m/s.
    let mut moved = desc.clone();
    moved.boxes[0].centre[2] += 1.0;
    let t = std::time::Instant::now();
    let info2 = st.place_now(&SceneSource { desc: Some(moved), id: None, contact: None }).unwrap();
    println!("re-placed in {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    assert!((info2.impact_s - info.impact_s - 1.0 / (50.0 / 3.6)).abs() < 1e-3);
    assert_ne!(info2.hash, info.hash);
    st.place_now(&src).unwrap();
    st.cpu_start().unwrap();
    let t = std::time::Instant::now();
    while !st.cpu_step(25).unwrap().done {}
    println!("CPU run in {:.1} s", t.elapsed().as_secs_f32());
    let status = st.recording_status().unwrap();
    assert!(status.present && status.fresh && status.has_report);
    assert_eq!(status.samples, info.samples);
    assert_eq!(status.executor, "cpu-fallback");
    let report = st.report().unwrap().unwrap();
    assert!(report.failures().is_empty(), "{}", report.table());
    let bytes = st.recording_bytes().unwrap();
    println!("recording: {} KB", bytes.len() / 1024);
    let at_70 = st.positions_at_ms(70.0).unwrap();
    st.load_recording(&bytes, None).unwrap();
    assert_eq!(st.positions_at_ms(70.0).unwrap(), at_70, "the recording survives its bytes");
    let cam = FrameCamera { eye: [4.6, 1.1, -0.3], target: [0.0, 0.8, -0.3], up: [0.0, 1.0, 0.0], fov_y_deg: 60.0, region: None };
    let spec = |ms: f32| RubbleSpec { at_ms: ms, camera: Some(cam.clone()), width: 1280.0, height: 720.0, body: 0 };
    let inst = st.outline(&schema, &spec(70.0)).unwrap();
    assert!(inst["elements"].as_object().is_some_and(|e| !e.is_empty()), "{inst}");
    assert!(inst["loops"].as_object().is_some_and(|e| !e.is_empty()));
    // Rubble against itself: before and peak, through the camera.
    let src = |ms: f32| ShapeSpec { tracing: None, rubble: Some(spec(ms)) };
    let table = st.compare(&schema, &[CompareSource { name: "rubble".into(), before: src(-10.0), now: src(70.0) }]).unwrap();
    assert!(table["rows"].as_array().is_some_and(|r| !r.is_empty()), "{table}");
    println!("{}", serde_json::to_string(&table["rows"]).unwrap());
}
