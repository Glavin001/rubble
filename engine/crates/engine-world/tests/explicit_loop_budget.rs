//! The runaway guards in the World (explicit_loop_cap.slang's pass budget): a World step
//! whose explicit frame or coupling passes stop at their budget fails, with the loop-cap
//! error, in the step that ran it, and fast. A tiny budget (`RUBBLE_LOOP_BUDGET`, read by
//! the planner and the coupling at each frame) stands in for a runaway: a sound frame
//! needs far more passes. Its own test binary: the variable is process-wide.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::Vec3;
use rubble::*;

fn sedan() -> ExplicitBeams {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan");
    ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, ExplicitBeamsSettings::default()).unwrap()
}

#[test]
fn a_world_step_past_the_pass_budget_fails_fast() {
    let mut town = rubble::explicit_scene::crash_town_at(
        GpuContext::Default,
        &sedan(),
        1,
        1,
        15.0,
        8.0,
        30,
        &rubble::explicit_scene::town_coupling(),
        Some(1.0 / 60.0),
        None,
    )
    .unwrap();
    let w = &mut town.world;
    // The host's budget: steps run.
    for _ in 0..3 {
        w.step().unwrap();
        w.poll_blocking().unwrap();
    }
    // SAFETY: this test binary's only test; the planner reads it at each frame.
    unsafe { std::env::set_var("RUBBLE_LOOP_BUDGET", "64") };
    let t = std::time::Instant::now();
    let mut failed = None;
    for k in 0..4 {
        let r = w.step().and_then(|_| w.poll_blocking().map(|_| ()));
        if let Err(e) = r {
            failed = Some((k, e.to_string()));
            break;
        }
    }
    unsafe { std::env::remove_var("RUBBLE_LOOP_BUDGET") };
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let (k, e) = failed.expect("a World step past its pass budget fails");
    assert_eq!(k, 0, "the step that ran past its budget fails: {e}");
    assert!(e.contains("stopped") || e.contains("pass budget"), "{e}");
    assert!(ms < 5000.0, "the failing step took {ms} ms");
}
