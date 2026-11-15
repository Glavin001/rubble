//! Retired debris on the GPU (`ExplicitBeamsGpu::set_retired`, `explicit_beams.slang`
//! nodeRetired) against the CPU reference (`ExplicitBeams::retired`): the same host
//! decision at the same step gives the same state bit for bit. Tiny structures with
//! anti-clip, two-body contact and fixed-point sums (the CPU computes the GPU's values):
//! a contacter retired before it reaches a triangle, a triangle retired before a node
//! crosses it, a node retired while it rests on a static box with surface contact.

use engine_softbody::explicit_beams::*;
use engine_softbody::explicit_contact::ExplicitBeamsWorld;
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

const EXACT: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: false,
    drag: false,
    turbulence: false,
    triangle_throttle: false,
    centred_damping: false,
    centred_objective: false,
};

fn node(p: Vec3, v: Vec3, flags: u32) -> Node {
    Node { pos: p, vel: v, force: Vec3::ZERO, mass: 1.0, friction: 1.0, flags }
}

/// The scene (`centred`: with centred damping and the falling triangle's beams): an immovable triangle in y = 0 (nodes 0..2) with contacters 3 and 4 falling
/// onto it; a triangle (5..7) falling at 2 m/s with contacter 8 rising into it; node 9
/// falling onto a static box (surface contact on).
fn scene(centred: bool) -> ExplicitBeams {
    let mut s = ExplicitBeams::empty(ExplicitBeamsSettings { centred_damping: centred, centred_objective: centred, ..EXACT });
    s.ground = -1.0e3;
    let fixed = NODE_IMMOVABLE | NODE_NO_GROUND;
    let free = NODE_NO_GROUND | NODE_CONTACTER;
    for p in [Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 2.0)] {
        s.nodes.push(node(p, Vec3::ZERO, fixed));
    }
    s.nodes.push(node(Vec3::new(0.1, 0.15, 0.2), Vec3::new(0.0, -3.0, 0.0), free));
    s.nodes.push(node(Vec3::new(-0.2, 0.15, 0.1), Vec3::new(0.0, -3.0, 0.0), free));
    for p in [Vec3::new(3.0, 1.0, -1.0), Vec3::new(5.0, 1.0, -1.0), Vec3::new(4.0, 1.0, 2.0)] {
        s.nodes.push(node(p, Vec3::new(0.0, -2.0, 0.0), NODE_NO_GROUND));
    }
    s.nodes.push(node(Vec3::new(4.0, 0.85, 0.0), Vec3::new(0.0, 4.0, 0.0), free));
    s.nodes.push(node(Vec3::new(-4.0, 0.08, 0.0), Vec3::new(0.0, -0.5, 0.0), 0));
    s.triangles.push(Triangle { nodes: [0, 1, 2], wait: 0, idle: 0 });
    s.triangles.push(Triangle { nodes: [5, 6, 7], wait: 0, idle: 0 });
    if centred {
        // The falling triangle held by JBeam beams (centred damping's correction acts on
        // JBeam elements), stretched a little so they ring.
        for (a, b) in [(5u32, 6u32), (6, 7), (7, 5)] {
            let l = (s.nodes[a as usize].pos - s.nodes[b as usize].pos).length();
            s.beams.push(Beam::new(a, b, ExplicitKind::JbeamPlain, ExplicitType::Deformable, 2.0e4, 40.0, 0.97 * l, 1.0e9, 1.0e9, 0.0));
        }
    }
    s.boxes.push(StaticBox::centered(Vec3::new(-4.0, -0.5, 0.0), Vec3::new(1.0, 0.5, 1.0)));
    s.anti_clip = true;
    s.two_body_contact = true;
    s.fixed_contact = true;
    s.surface_contact = true;
    s.finish();
    s
}

fn bits(v: &[Vec3]) -> Vec<[u32; 3]> {
    v.iter().map(|p| p.to_array().map(f32::to_bits)).collect()
}

/// Steps of one-step frames, the retirement `retire` (nodes) applied before frame `at`
/// on both; every frame positions and velocities bit for bit. Returns the CPU's final
/// positions.
fn run(retire: &[usize], at: u32, frames: u32, centred: bool) -> Option<Vec<Vec3>> {
    let s = scene(centred);
    let (_, device, queue) = default_device().ok()?;
    let mut g = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&s));
    let mut w = ExplicitBeamsWorld::new(vec![s.clone()]);
    let mut flags = vec![false; s.nodes.len()];
    for &i in retire {
        flags[i] = true;
    }
    for f in 0..frames {
        if f == at && !retire.is_empty() {
            w.vehicles[0].retired = flags.clone();
            g.set_retired(0, &flags).expect("retire");
        }
        w.frame(1);
        g.frame(1);
        let (gp, gv) = (g.positions(), g.velocities());
        let (cp, cv) = (w.vehicles[0].positions(), w.vehicles[0].nodes.iter().map(|n| n.vel).collect::<Vec<_>>());
        assert_eq!(bits(&gp[0]), bits(&cp), "positions after frame {f} (retired {retire:?} at {at})");
        assert_eq!(bits(&gv[0]), bits(&cv), "velocities after frame {f} (retired {retire:?} at {at})");
    }
    g.wait();
    Some(w.vehicles[0].positions())
}

#[test]
fn retired_nodes_step_like_the_cpu_bit_for_bit() {
    // Nothing retired: the reference run (the contacts happen).
    let Some(free) = run(&[], 0, 320, false) else {
        eprintln!("no GPU: skipped");
        return;
    };
    // Contacter 4 retired at step 20 (before it reaches the triangle at y = 0), the
    // falling triangle 5..7 at step 20 (before contacter 8 meets it), node 9 at step 200
    // (resting on the box).
    let a = run(&[4, 5, 6, 7], 20, 320, false).unwrap();
    let b = run(&[9], 200, 320, false).unwrap();
    // With centred damping (its correction skips retired nodes): the beam-held triangle
    // retired mid-fall, and nothing retired.
    run(&[], 0, 320, true).unwrap();
    let c = run(&[5, 6, 7], 20, 320, true).unwrap();
    assert!(c[8].y > c[5].y + 0.05, "centred: node 8 passed the retired triangle: {} vs {}", c[8].y, c[5].y);
    eprintln!(
        "free: node 4 at y {:.4}, node 8 at y {:.4}, node 9 at y {:.4}; retired: node 4 at y {:.4}, node 8 at y {:.4}; node 9 at y {:.4}",
        free[4].y, free[8].y, free[9].y, a[4].y, a[8].y, b[9].y
    );
    // Without retirement the contacter is caught at the triangle; retired it stays where
    // it was at step 20 (0.15 - 3 · 20 h ≈ 0.12 m).
    assert!(free[4].y > -0.02 && free[4].y < 0.05, "free contacter caught: {}", free[4].y);
    assert!(a[4].y > 0.1, "retired contacter held: {}", a[4].y);
    // Node 8 passes the retired triangle (no contact): it ends above it.
    assert!(a[8].y > a[5].y + 0.05, "node 8 passed the retired triangle: {} vs {}", a[8].y, a[5].y);
    assert!(free[8].y < free[5].y + 0.05, "node 8 caught by the free triangle: {} vs {}", free[8].y, free[5].y);
}
