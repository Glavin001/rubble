//! A heavy CCD ball through a dynamic brick wall (the ring benchmark's 8 m,
//! 21 t ball): it ploughs through as the plain ball does and does not ride up
//! over the bricks. Rigid time-of-impact rows against light free bricks made
//! it yield to them; the post-solve sweep check against bricks the solve had
//! not pushed yet slowed it to a crawl.
use engine_schema::{AvbdParams, WorldCapacity, FLAG_REUSE_CONTACTS, BODY_CCD};
use engine_webgpu::WgpuSim;
mod support;
use support::make_body_collider;

#[test]
fn a_heavy_ccd_ball_ploughs_through_a_brick_wall() {
    let mut ends = Vec::new();
    for ccd in [false, true] {
        let mut sim = WgpuSim::new(WorldCapacity { max_bodies: 8192, max_colliders: 8192, max_contacts: 262144, max_candidates: 262144, max_colors: 32, ..WorldCapacity::small() }).unwrap();
        let mut p = AvbdParams::default();
        p.iterations = 4;
        p.flags |= FLAG_REUSE_CONTACTS;
        p.sleep_time = 1.0e6;
        sim.set_params(p);
        let mut add = |sim: &mut WgpuSim, pos, half, mass, sphere, ccd: bool| {
            let id = sim.body_count();
            let (mut b, c) = make_body_collider(id, pos, half, mass, [0.0, 0.0, 0.0, 1.0], sphere);
            if ccd { b.flags |= BODY_CCD; }
            sim.add_body(b); sim.add_collider(c); id
        };
        add(&mut sim, [0.0; 3], [60.0, 0.5, 60.0], 0.0, false, false);
        // A wall of bricks across z = 0 (3 rows, 10 courses).
        for course in 0..10 { for row in 0..2 { for i in 0..40 {
            add(&mut sim, [-20.0 + i as f32 * 1.02 + (course % 2) as f32 * 0.5, 0.75 + 0.5 * course as f32, row as f32 * 0.52], [0.5, 0.25, 0.25], 0.25, false, false);
        }}}
        let ball = add(&mut sim, [0.0, 8.5, -15.2], [8.0; 3], 10.0 * 4.0 / 3.0 * std::f32::consts::PI * 512.0, true, ccd);
        let mut top = 0.0f32;
        for _ in 0..120 {
            sim.step().unwrap();
            top = top.max(sim.read_body(ball).position[1]);
        }
        let b = sim.read_body(ball);
        println!("ccd {ccd}: ball z {:.2} (wall at 0), highest centre {top:.2} (rolling: 8.5)", b.position[2]);
        ends.push((b.position[2], top));
    }
    let (plain, ccd) = (ends[0], ends[1]);
    assert!(plain.0 > 30.0, "control: the plain ball rolls through ({})", plain.0);
    assert!(ccd.0 > 0.8 * plain.0, "the CCD ball should plough through as the plain one does: z {} vs {}", ccd.0, plain.0);
    assert!(ccd.1 < 9.5, "the CCD ball rode up over the bricks (centre {})", ccd.1);
}
