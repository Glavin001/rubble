//! `FLAG_POST_STAB` in World's step plan: the velocity is recorded after the iterations and an
//! α = 0 pass follows (as the fracture lab's planner runs it). Without those passes the velocity
//! was never extracted at all: no gravity, no response to contacts.

use engine_schema::{FLAG_CONTACT_GAP, FLAG_POST_STAB};
use rubble::*;

fn world(flags: u32) -> World {
    let mut w = World::new(GpuContext::Default, &WorldDef::default()).unwrap();
    w.set_solver(AvbdParams {
        flags,
        ..w.solver()
    });
    let g = w.create_body(&BodyDef::default()).unwrap();
    w.create_shape(g, &ShapeDef::default(), Geometry::Plane { normal: [0.0, 1.0, 0.0], offset: 0.0 })
        .unwrap();
    w
}

fn height_after(w: &mut World, ball: BodyId, steps: u32) -> f32 {
    for _ in 0..steps {
        w.step().unwrap();
    }
    w.poll_blocking().unwrap();
    w.body_transform(ball).unwrap().unwrap().position[1]
}

#[test]
fn a_post_stabilized_ball_falls_and_lands() {
    // The lab's set, so contacts target zero gap (otherwise a resting ball sinks in).
    for flags in [FLAG_POST_STAB, FLAG_POST_STAB | FLAG_CONTACT_GAP] {
        let mut w = world(flags);
        let ball = w.create_body(&BodyDef::dynamic([0.0, 10.0, 0.0])).unwrap();
        w.create_sphere_shape(ball, &ShapeDef::default(), [0.0; 3], 0.5).unwrap();
        let g = -w.solver().gravity_y;
        let dt = w.solver().dt;
        // Free fall for 0.5 s: 10 - g t² / 2 (implicit Euler falls one step's g dt² further).
        let t = 30.0 * dt;
        let y = height_after(&mut w, ball, 30);
        let expect = 10.0 - 0.5 * g * t * t;
        assert!((y - expect).abs() < g * dt * t + 1e-3, "flags {flags}: {y} after {t} s, not {expect}");
        // Then the ground stops it.
        let y = height_after(&mut w, ball, 180);
        assert!(y > 0.25 && y < 0.6, "flags {flags}: at {y}, not resting on the ground");
    }
}
