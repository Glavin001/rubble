//! Passes rewritten to do less work give the same bodies as the ones they replace, on
//! the same GPU state: `ccd_wake` and `ccd_unhold` a thread per collider
//! (`EngineFeatures::ccd_by_collider`), and prediction and velocity extraction over the
//! awake list (`EngineFeatures::awake_body_passes`).
//!
//! A brick wall settles and sleeps; compound bodies (two colliders, one off the body's
//! origin, so a body's extent is its larger collider's) sleep beside it; then fast CCD
//! balls (one a compound) are fired through. For every step of the strikes, the step
//! runs from the same saved state with each variant up to a phase marker, and the body
//! records must be equal (the reference run twice first, so a comparison only counts
//! where the reference itself repeats).

use engine_schema::{AvbdParams, BodyState, WorldCapacity, BODY_CCD};
use engine_webgpu::WgpuSim;
mod support;
use support::make_body_collider;

fn add(
    sim: &mut WgpuSim,
    pos: [f32; 3],
    half: [f32; 3],
    mass: f32,
    sphere: bool,
    v: [f32; 3],
) -> u32 {
    let id = sim.body_count();
    let (mut b, c) = make_body_collider(id, pos, half, mass, [0.0, 0.0, 0.0, 1.0], sphere);
    b.linear_vel = v;
    b.angular_vel = [0.0; 3];
    sim.add_body(b);
    sim.add_collider(c);
    id
}

/// A second box collider on `body`, `offset` from its origin.
fn add_part(sim: &mut WgpuSim, body: u32, offset: [f32; 3], half: [f32; 3]) {
    let (_, mut c) = make_body_collider(body, [0.0; 3], half, 1.0, [0.0, 0.0, 0.0, 1.0], false);
    c.local_pos = offset;
    sim.add_collider(c);
}

fn scene() -> WgpuSim {
    let mut sim = WgpuSim::new(WorldCapacity {
        max_bodies: 8192,
        max_colliders: 8192,
        max_contacts: 262_144,
        max_candidates: 262_144,
        max_colors: 32,
        ..WorldCapacity::small()
    })
    .unwrap();
    let mut p = AvbdParams::default();
    p.iterations = 4;
    p.dt = 1.0 / 60.0;
    sim.set_params(p);
    add(&mut sim, [0.0; 3], [60.0, 0.5, 60.0], 0.0, false, [0.0; 3]);
    for course in 0..8 {
        for row in 0..2 {
            for i in 0..30 {
                let x = -15.0 + i as f32 * 1.02 + (course % 2) as f32 * 0.5;
                add(
                    &mut sim,
                    [x, 0.75 + 0.5 * course as f32, row as f32 * 0.52],
                    [0.5, 0.25, 0.25],
                    2.0,
                    false,
                    [0.0; 3],
                );
            }
        }
    }
    // Compound sleepers behind the wall: a block with a part 0.9 m off its origin.
    for k in 0..6 {
        let b = add(
            &mut sim,
            [-6.0 + 2.5 * k as f32, 0.8, 3.0],
            [0.3, 0.3, 0.3],
            3.0,
            false,
            [0.0; 3],
        );
        add_part(&mut sim, b, [0.0, 0.0, 0.9], [0.25, 0.25, 0.25]);
    }
    sim
}

fn fire(sim: &mut WgpuSim, step: u32) {
    if step == 150 {
        // A fast CCD ball through the wall.
        let ball = add(
            sim,
            [-4.0, 1.4, -8.0],
            [0.3; 3],
            40.0,
            true,
            [0.0, 0.0, 70.0],
        );
        let mut b = sim.read_body(ball);
        b.flags |= BODY_CCD;
        sim.write_bodies(ball, &[b]);
    }
    if step == 190 {
        // A compound CCD body (a ball with a box part) the other way.
        let ball = add(
            sim,
            [5.0, 2.4, 9.0],
            [0.35; 3],
            60.0,
            true,
            [0.0, 0.0, -80.0],
        );
        add_part(sim, ball, [0.6, 0.0, 0.0], [0.2, 0.2, 0.2]);
        let mut b = sim.read_body(ball);
        b.flags |= BODY_CCD;
        sim.write_bodies(ball, &[b]);
    }
}

/// Steps 150..260 of the scene: each from the same state with `set(false)` (the
/// reference, twice) and `set(true)` up to every phase marker of `phases`, then the
/// real step with `set(true)`. `key` picks what is compared of each body.
fn compare<K: PartialEq + std::fmt::Debug>(
    set: &dyn Fn(&mut WgpuSim, bool),
    phases: &[&str],
    key: &dyn Fn(&BodyState) -> K,
) -> (usize, usize) {
    let mut sim = scene();
    let mut compared = 0;
    let mut changed = 0usize;
    for step in 0..260u32 {
        fire(&mut sim, step);
        if step >= 150 {
            let start = sim.save_state();
            let mut runs: Vec<Vec<Vec<K>>> = Vec::new();
            for on in [false, true, false] {
                set(&mut sim, on);
                let mut out = Vec::new();
                for phase in phases {
                    sim.restore_state(&start);
                    sim.debug_step_until(phase);
                    out.push(sim.download_bodies().iter().map(key).collect());
                }
                runs.push(out);
            }
            sim.restore_state(&start);
            set(&mut sim, true);
            let before: Vec<K> = sim.download_bodies().iter().map(key).collect();
            for (k, phase) in phases.iter().enumerate() {
                if runs[0][k] != runs[2][k] {
                    eprintln!("step {step} {phase}: the reference differs from itself; skipped");
                    continue;
                }
                let (a, b) = (&runs[0][k], &runs[1][k]);
                let diff: Vec<usize> = (0..a.len()).filter(|&i| a[i] != b[i]).collect();
                assert!(
                    diff.is_empty(),
                    "step {step} {phase}: bodies {diff:?} differ: {:?} vs {:?}",
                    diff.iter().map(|&i| &a[i]).collect::<Vec<_>>(),
                    diff.iter().map(|&i| &b[i]).collect::<Vec<_>>()
                );
                compared += 1;
                changed += (0..a.len()).filter(|&i| b[i] != before[i]).count();
            }
        }
        sim.step().unwrap();
    }
    (compared, changed)
}

#[test]
fn ccd_passes_by_collider_wake_and_release_the_same_bodies() {
    let flags = |b: &BodyState| (b.flags, b.sleep_timer.to_bits());
    let (compared, changed) = compare(
        &|s, on| s.set_ccd_by_collider(on),
        &["ccdToi", "topology"],
        &flags,
    );
    eprintln!("ccd: compared {compared} phases, {changed} body changes");
    assert!(compared >= 150, "compared {compared}");
}

#[test]
fn awake_list_body_passes_change_the_same_bodies() {
    let all = |b: &BodyState| bytemuck::bytes_of(b).to_vec();
    let (compared, changed) = compare(
        &|s, on| s.set_awake_body_passes(on),
        &["topology", "sleep"],
        &all,
    );
    eprintln!("body passes: compared {compared} phases, {changed} body changes");
    assert!(compared >= 150, "compared {compared}");
}
