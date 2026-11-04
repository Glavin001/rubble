//! `WorldDef::gpu_timing`: each step's frame carries its GPU time from
//! timestamp queries, without waiting; an all-asleep world skips the step and
//! times nothing.

use rubble::*;

fn world(mode: GpuTiming) -> (World, BodyId) {
    // `gpu_timing` asks the default device for timestamp queries.
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            gpu_timing: mode,
            // A dispatch per colour and dual, so each iteration's primal and dual are
            // timed apart (the solve loop takes them in one pass while few are awake).
            features: rubble::EngineFeatures {
                solve_loop: false,
                ..rubble::EngineFeatures::default()
            },
            ..WorldDef::default()
        },
    )
    .unwrap();
    let g = w
        .create_body(&BodyDef {
            position: [0.0, -0.5, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(g, &ShapeDef::default(), [20.0, 0.5, 20.0])
        .unwrap();
    let b = w
        .create_body(&BodyDef {
            body_type: BodyType::Dynamic,
            position: [0.0, 2.0, 0.0],
            ..BodyDef::default()
        })
        .unwrap();
    w.create_box_shape(b, &ShapeDef::default(), [0.5, 0.5, 0.5])
        .unwrap();
    (w, b)
}

#[test]
fn frames_carry_the_step_and_phase_gpu_times() {
    let (mut w, _) = world(GpuTiming::Step);
    let mut timed = Vec::new();
    for _ in 0..6 {
        w.step().unwrap();
        for f in w.poll_blocking().unwrap() {
            // Native Metal leaves the device's first timestamp pair unwritten:
            // that frame has no time rather than a wrong one.
            if f.step > 0 {
                assert!(f.gpu_time.is_some(), "step {} timed", f.step);
            }
            timed.extend(f.gpu_time);
        }
    }
    assert!(timed.len() >= 5);
    for t in &timed {
        assert_eq!(t.segments.len(), 1);
        assert_eq!(t.segments[0].0, "step");
        assert!(t.total_ms > 0.0 && t.total_ms <= t.span_ms + 1e-9, "{t:?}");
    }

    w.set_gpu_timing(GpuTiming::Phases).unwrap();
    w.step().unwrap();
    let f = w.poll_blocking().unwrap().pop().unwrap();
    let t = f.gpu_time.expect("a timed step");
    // Native Metal leaves a per-phase step's last pass unwritten (reported).
    let names: Vec<&str> = t
        .segments
        .iter()
        .map(|s| s.0)
        .chain(t.unmeasured.iter().copied())
        .collect();
    for phase in [
        "broadphase",
        "narrowphase",
        "warm",
        "graph",
        "coloring",
        "topology",
        "primal",
        "dual",
        "sleep",
        "pose",
    ] {
        assert!(names.contains(&phase), "{phase} in {names:?}");
    }
    let iterations = w.solver().iterations as usize;
    assert_eq!(names.iter().filter(|n| **n == "primal").count(), iterations);
    assert!(t.total_ms <= t.span_ms + 1e-9);

    w.set_gpu_timing(GpuTiming::Off).unwrap();
    w.step().unwrap();
    assert!(w.poll_blocking().unwrap().pop().unwrap().gpu_time.is_none());
}

#[test]
fn an_asleep_world_times_nothing() {
    let (mut w, b) = world(GpuTiming::Step);
    let mut last = None;
    for _ in 0..400 {
        w.step().unwrap();
        for f in w.poll_blocking().unwrap() {
            last = Some((f.ran, f.gpu_time.is_some()));
        }
    }
    assert!(!w.body_is_awake(b).unwrap(), "the box fell asleep");
    assert_eq!(
        last,
        Some((false, false)),
        "skipped steps carry no GPU time"
    );
}
