//! A digest of a small World with explicit vehicles after a fixed number of steps: two
//! procedural sedans dropped onto dynamic boxes over a static slab, the vehicles
//! bit-exact. Prints one hash of the vehicles' node state, the bodies and the coupling's
//! records and asserts only that the state is finite and that two Worlds built alike in
//! this process agree. A change to the World's plumbing that must not change results
//! (the coupling's shared plan, `engine_exec::explicit_plan::coupling::world`) is checked
//! by running this before and after it and comparing the printed line.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody::procedural::{car_truck_bare, CarParams};
use glam::Vec3;
use rubble::*;

fn sedan(at: Vec3) -> ExplicitBeams {
    let text = car_truck_bare(&CarParams { minimass: 10.0, ..CarParams::sedan() });
    let settings = ExplicitBeamsSettings { turbulence: false, drag: false, ..Default::default() };
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    let mut car = ExplicitBeams::from_truck(&text, at + Vec3::new(0.0, -low, 0.0), Vec3::ZERO, settings).unwrap();
    // No ground of the vehicles' own: they land on the World's bodies.
    car.ground = -1.0e4;
    car
}

fn fnv(h: &mut u64, words: impl IntoIterator<Item = u32>) {
    for w in words {
        for b in w.to_le_bytes() {
            *h ^= u64::from(b);
            *h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
}

fn run(steps: u32, sleep: bool) -> (u64, u64, u64) {
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef { gravity: [0.0, engine_softbody::explicit_beams::GRAVITY, 0.0], ..WorldDef::default() },
    )
    .unwrap();
    let slab = w.create_body(&BodyDef { body_type: BodyType::Static, position: [0.0, -0.5, 0.0], ..BodyDef::default() }).unwrap();
    w.create_box_shape(slab, &ShapeDef::default(), [30.0, 0.5, 30.0]).unwrap();
    let mut bodies = Vec::new();
    for k in 0..12 {
        let x = -6.0 + 1.1 * k as f32;
        let b = w.create_body(&BodyDef::dynamic([x, 0.25, 0.3 * (k % 3) as f32])).unwrap();
        w.create_box_shape(b, &ShapeDef { density: 400.0, ..ShapeDef::default() }, [0.5, 0.25, 0.9]).unwrap();
        bodies.push(b);
    }
    let cars = [sedan(Vec3::new(-3.0, 0.9, 0.0)), sedan(Vec3::new(3.0, 0.9, 0.2))];
    let def = ExplicitVehiclesDef {
        bit_exact: true,
        sleep: sleep.then(engine_softbody::explicit_sleep::VehicleSleep::default),
        ..ExplicitVehiclesDef::default()
    };
    w.add_explicit_vehicles(&cars, &def).unwrap();
    for _ in 0..steps {
        w.step().unwrap();
        w.poll_blocking().unwrap();
    }
    let x = w.explicit_vehicles().unwrap();
    let (mut hv, mut hb, mut hc) = (0xcbf2_9ce4_8422_2325u64, 0xcbf2_9ce4_8422_2325u64, 0xcbf2_9ce4_8422_2325u64);
    for (p, v) in x.positions().iter().zip(&x.velocities()) {
        for (p, v) in p.iter().zip(v) {
            assert!(p.is_finite() && v.is_finite(), "a node is not finite");
            fnv(&mut hv, p.to_array().into_iter().chain(v.to_array()).map(f32::to_bits));
        }
    }
    for &b in &bodies {
        let s = w.read_body_state(b).unwrap();
        let words = s.transform.position.into_iter().chain(s.linear_velocity).chain(s.angular_velocity);
        fnv(&mut hb, words.map(f32::to_bits));
    }
    for car in w.explicit_window_impulses() {
        for (collider, j, m, r) in car {
            fnv(&mut hc, std::iter::once(collider).chain(j.into_iter().chain(m).chain(r).map(f32::to_bits)));
        }
    }
    (hv, hb, hc)
}

#[test]
fn a_small_world_with_explicit_vehicles_has_one_digest() {
    for sleep in [false, true] {
        let a = run(90, sleep);
        let b = run(90, sleep);
        println!(
            "explicit world digest (sleep {sleep}, 90 steps): vehicles {:016x} bodies {:016x} window impulses {:016x}",
            a.0, a.1, a.2
        );
        assert_eq!(a, b, "two Worlds built alike differ (sleep {sleep})");
    }
}

/// engine-cuda's `tests/world.rs` pile on this host, for comparing the two (printed, not
/// asserted beyond finiteness): over a static slab, a dynamic deck (3000 kg) with a sedan
/// dropped 0.4 m onto it, and six dynamic boxes with a second sedan dropped onto and among
/// them. Per ten steps each vehicle's lowest node and the deck's height.
#[test]
fn the_cuda_world_tests_pile_on_this_host() {
    let mut w = World::new(
        GpuContext::Default,
        &WorldDef { gravity: [0.0, engine_softbody::explicit_beams::GRAVITY, 0.0], enable_sleep: false, ..WorldDef::default() },
    )
    .unwrap();
    let slab = w.create_body(&BodyDef { body_type: BodyType::Static, position: [0.0, -0.5, 0.0], ..BodyDef::default() }).unwrap();
    w.create_box_shape(slab, &ShapeDef::default(), [30.0, 0.5, 30.0]).unwrap();
    let dynamic = |w: &mut World, pos: [f32; 3], half: [f32; 3], mass: f32| {
        let b = w.create_body(&BodyDef::dynamic(pos)).unwrap();
        let density = mass / (8.0 * half[0] * half[1] * half[2]);
        w.create_box_shape(b, &ShapeDef { density, ..ShapeDef::default() }, half).unwrap();
        b
    };
    let deck = dynamic(&mut w, [-3.5, 0.25, 0.0], [1.6, 0.25, 3.2], 3000.0);
    for k in 0..6 {
        dynamic(&mut w, [1.5 + 1.1 * k as f32, 0.25, 0.3 * (k % 3) as f32], [0.5, 0.25, 0.9], 360.0);
    }
    // The sedan's box middle at x, its lowest node at 0.9, its front face at z 2.2.
    let place = |x: f32| {
        let mut car = sedan(Vec3::ZERO);
        let (lo, hi) = car.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| (a.min(n.pos), b.max(n.pos)));
        let shift = Vec3::new(x - 0.5 * (lo.x + hi.x), 0.9 - lo.y, 2.2 - hi.z);
        for n in &mut car.nodes {
            n.pos += shift;
        }
        car
    };
    let cars = [place(-3.5), place(4.0)];
    let weight: Vec<f32> = cars.iter().map(|c| c.nodes.iter().map(|n| n.mass).sum::<f32>() * -engine_softbody::explicit_beams::GRAVITY).collect();
    w.add_explicit_vehicles(&cars, &ExplicitVehiclesDef { bit_exact: true, sleep: None, ..ExplicitVehiclesDef::default() }).unwrap();
    let dt = w.solver().dt;
    let mut tail = [0.0f32; 2];
    for step in 0..120 {
        w.step().unwrap();
        w.poll_blocking().unwrap();
        if step >= 80 {
            for (v, car) in w.explicit_window_impulses().iter().enumerate() {
                tail[v] += car.iter().map(|s| s.1[1]).sum::<f32>() / 40.0;
            }
        }
        if step % 10 == 9 {
            let lowest: Vec<f32> = w
                .explicit_vehicles()
                .unwrap()
                .positions()
                .iter()
                .map(|car| car.iter().map(|p| p.y).fold(f32::MAX, f32::min))
                .collect();
            let d = w.read_body_state(deck).unwrap();
            assert!(lowest.iter().all(|y| y.is_finite()));
            println!("wgpu World pile step {step}: lowest nodes {lowest:?}, deck y {:.4} vy {:.4}", d.transform.position[1], d.linear_velocity[1]);
        }
    }
    println!(
        "wgpu World pile: mean window impulse on the bodies {tail:?} N s (the weights' {:?}), used the broadphase {:?}",
        weight.iter().map(|w| -w * dt).collect::<Vec<_>>(),
        w.explicit_plan_used_broadphase()
    );
}
