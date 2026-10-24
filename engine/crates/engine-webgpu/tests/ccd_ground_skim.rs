//! A CCD ball skimming the ground (centre at or below its resting height, as
//! rubble presses it down) into the bottom course of a sleeping brick wall:
//! its centre must never cross a brick that did not move. The 512-ball field
//! (ccd_scale.rs) found balls doing that.

use engine_schema::{AvbdParams, BodyState, Collider, ShapeType, WorldCapacity, BODY_CCD, FLAG_SPEC_VELOCITY};
use engine_webgpu::WgpuSim;

#[path = "support/phase.rs"]
mod phase;

const DT: f32 = 1.0 / 60.0;

fn body(pos: [f32; 3], inv_mass: f32, inv_i: [f32; 3], flags: u32, v: [f32; 3]) -> BodyState {
    BodyState { position: pos, inv_mass, orientation: [0.0, 0.0, 0.0, 1.0], inv_inertia: inv_i, flags, linear_vel: v, sleep_timer: 0.0, angular_vel: [0.0; 3], generation: 0, initial_pos: pos, initial_pad: 0.0, initial_rot: [0.0, 0.0, 0.0, 1.0] }
}
fn collider(body: u32, shape: ShapeType, p: [f32; 4]) -> Collider {
    Collider { body, shape: shape as u32, material: 0, generation: 0, local_pos: [0.0; 3], density: 0.0, local_rot: [0.0, 0.0, 0.0, 1.0], params: p }
}

/// Phase-throughs of one skim (ball centre height `y`, speed `v`).
fn skim(browser: bool, hier: bool, y: f32, v: f32) -> Vec<String> {
    let cap = WorldCapacity { max_bodies: 2048, max_colliders: 2048, max_contacts: 1 << 18, max_candidates: 1 << 18, max_colors: 32, ..WorldCapacity::small() };
    let mut sim = if browser { WgpuSim::new_with_compact_contacts(cap, false).unwrap() } else { WgpuSim::new_with_persistent_contacts(cap).unwrap() };
    if browser {
        sim.use_control_modes();
    }
    sim.set_grid_levels(Some(hier));
    let mut p = AvbdParams::default();
    p.dt = DT;
    p.flags |= FLAG_SPEC_VELOCITY;
    sim.set_params(p);
    let g = sim.add_body(body([0.0, -1.0, 0.0], 0.0, [0.0; 3], 0, [0.0; 3]));
    sim.add_collider(collider(g, ShapeType::Box, [100.0, 1.0, 100.0, 141.0]));
    let (hx, hy, hz) = (0.2f32, 0.1f32, 0.1f32);
    let ii = [3.0 / (hy * hy + hz * hz), 3.0 / (hx * hx + hz * hz), 3.0 / (hx * hx + hy * hy)];
    // A wall across z = 0 (3 courses, 2 deep), asleep from the start.
    for course in 0..3 {
        for row in 0..2 {
            for i in 0..12 {
                let x = -2.4 + i as f32 * 0.4 + if course % 2 == 1 { 0.2 } else { 0.0 };
                let id = sim.add_body(body([x, hy + course as f32 * 2.0 * hy, row as f32 * 0.2], 1.0, ii, 1 | 4, [0.0; 3]));
                sim.add_collider(collider(id, ShapeType::Box, [hx, hy, hz, (hx * hx + hy * hy + hz * hz).sqrt()]));
            }
        }
    }
    let ib = 2.5 / (5.0 * 0.15 * 0.15);
    let ball = sim.add_body(body([0.07, y, -6.0], 0.2, [ib; 3], 9 | BODY_CCD, [0.0, 0.0, v]));
    sim.add_collider(collider(ball, ShapeType::Sphere, [0.15, 0.0, 0.0, 0.15]));
    let cols: Vec<Collider> = (0..sim.collider_count()).map(|i| sim.collider(i)).filter(|c| c.body != 0).collect();
    let mut prev = sim.download_bodies();
    let mut out = Vec::new();
    for step in 0..30 {
        sim.step().unwrap();
        let now = sim.download_bodies();
        if let Some(why) = phase::crossing(&cols, &prev, &now, ball as usize) {
            out.push(format!("browser {browser} hier {hier} y {y} v {v} step {step}: {why}"));
        }
        prev = now;
    }
    out
}

#[test]
fn a_ccd_ball_skimming_the_ground_never_passes_the_bottom_course() {
    let mut failures = Vec::new();
    for browser in [false, true] {
        for hier in [false, true] {
            for y in [0.03f32, 0.08, 0.15] {
                for v in [50.0f32, 120.0, 250.0] {
                    failures.extend(skim(browser, hier, y, v));
                }
            }
        }
    }
    println!("{} phase-throughs over 36 skims", failures.len());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
