//! `WgpuSim::kernel_replay`: per-kernel isolated GPU timing by replay. Measurement only: the
//! replay leaves every buffer of the step exactly as it found it.

use rubble::*;

fn pile() -> World {
    // Timestamp queries come with `gpu_timing` (the replay itself does not use the mode).
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef {
            gpu_timing: GpuTiming::Step,
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
    w.create_box_shape(g, &ShapeDef::default(), [20.0, 0.5, 20.0]).unwrap();
    for i in 0..60 {
        let b = w
            .create_body(&BodyDef {
                body_type: BodyType::Dynamic,
                position: [
                    (i % 5) as f32 * 1.05 - 2.0,
                    0.6 + (i / 25) as f32 * 1.05,
                    ((i / 5) % 5) as f32 * 1.05 - 2.0,
                ],
                ..BodyDef::default()
            })
            .unwrap();
        w.create_box_shape(b, &ShapeDef::default(), [0.5, 0.5, 0.5]).unwrap();
    }
    w
}

#[test]
fn replay_times_every_dispatch_and_restores_the_state() {
    let mut w = pile();
    for _ in 0..20 {
        w.step().unwrap();
    }
    w.poll_blocking().unwrap();
    let before = w.sim().buffer_digests();
    let report = w
        .sim_mut()
        .kernel_replay(engine_webgpu::KernelReplayOptions {
            reps: 3,
            whole_step_only: false,
        })
        .unwrap();
    let after = w.sim().buffer_digests();
    assert_eq!(before, after, "the replay restores every buffer");
    eprintln!("{}", report.table(12));
    assert!(!report.dispatches.is_empty());
    assert_eq!(report.whole_step_ms.len(), 3);
    for d in &report.dispatches {
        assert_eq!(d.isolated_ms.len(), 3, "{}", d.kernel);
        assert_eq!(d.in_sequence_ms.len(), 3, "{}", d.kernel);
    }
    assert!(report.dispatches.iter().any(|d| d.indirect), "the plan has indirect dispatches");
    let whole = engine_webgpu::replay::median(&report.whole_step_ms);
    assert!(whole > 0.0);
    // The world steps on after a replay.
    for _ in 0..5 {
        w.step().unwrap();
    }
    assert!(!w.poll_blocking().unwrap().is_empty());
}
