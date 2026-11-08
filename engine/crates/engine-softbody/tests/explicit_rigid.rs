//! The explicit vehicle ↔ rigid body coupling's CPU reference
//! (`engine/docs/explicit-vehicles-in-world.md`): a static proxy is the static shape bit
//! for bit; momentum closes across the coupling; energy does not rise; a vehicle into a
//! free box of known mass gives the box a speed inside the exact plastic and elastic
//! bounds of the momentum balance.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, StaticShape};
use engine_softbody::explicit_rigid::{ProxyGeom, ProxySet, ProxyShape, RigidProxy};
use engine_softbody::procedural::{car_truck_bare, CarParams};
use glam::{Quat, Vec3};

const QUIET: ExplicitBeamsSettings = ExplicitBeamsSettings {
    fast_inv_sqrt: true,
    drag: false,
    turbulence: false,
    triangle_throttle: true,
    centred_damping: false,
};

/// The procedural sedan's bare body, its long axis along +Z, its front face at z = 0,
/// moving at `speed` along +Z, `ground` the flat ground's height.
fn sedan(speed: f32, ground: f32) -> ExplicitBeams {
    let p = CarParams {
        minimass: 10.0,
        ..CarParams::sedan()
    };
    let mut sim = ExplicitBeams::from_truck(&car_truck_bare(&p), Vec3::ZERO, Vec3::ZERO, QUIET).unwrap();
    let (lo, hi) = bounds(&sim);
    // Long axis along Z.
    if hi.x - lo.x > hi.z - lo.z {
        for n in &mut sim.nodes {
            n.pos = Vec3::new(-n.pos.z, n.pos.y, n.pos.x);
        }
    }
    let (lo, hi) = bounds(&sim);
    let shift = Vec3::new(-0.5 * (lo.x + hi.x), ground + 0.02 - lo.y, -hi.z);
    for n in &mut sim.nodes {
        n.pos += shift;
        n.vel = Vec3::new(0.0, 0.0, speed);
    }
    sim.ground = ground;
    sim
}

fn bounds(sim: &ExplicitBeams) -> (Vec3, Vec3) {
    sim.nodes
        .iter()
        .fold((Vec3::MAX, Vec3::MIN), |(a, b), n| (a.min(n.pos), b.max(n.pos)))
}

fn box_proxy(centre: Vec3, half: Vec3, rot: Quat, mass: f32) -> ProxySet {
    let (inv_mass, inv_inertia) = if mass.is_finite() {
        let i = Vec3::new(
            mass / 3.0 * (half.y * half.y + half.z * half.z),
            mass / 3.0 * (half.x * half.x + half.z * half.z),
            mass / 3.0 * (half.x * half.x + half.y * half.y),
        );
        (1.0 / mass, Vec3::ONE / i)
    } else {
        (0.0, Vec3::ZERO)
    };
    ProxySet {
        bodies: vec![RigidProxy {
            body: 0,
            pos: centre,
            rot,
            vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            inv_mass,
            inv_inertia,
            share: 1.0,
            count: 0,
        }],
        shapes: vec![ProxyShape::new(0, Vec3::ZERO, Quat::IDENTITY, ProxyGeom::Box { half }, 0, None)],
        frozen: false,
    }
}

fn node_momentum(sim: &ExplicitBeams) -> Vec3 {
    sim.nodes.iter().map(|n| n.vel * n.mass).sum()
}

fn node_energy(sim: &ExplicitBeams) -> f64 {
    // Kinetic plus gravitational (the node's weight m g at height y), plus every intact
    // beam's elastic energy ½ k (L − L₀)² about its current rest length.
    let strain: f64 = sim
        .beams
        .iter()
        .filter(|b| !b.broken)
        .map(|b| {
            let l = (sim.nodes[b.a as usize].pos - sim.nodes[b.b as usize].pos).length();
            0.5 * f64::from(b.k) * f64::from(l - b.rest).powi(2)
        })
        .sum();
    strain
        + sim
            .nodes
            .iter()
        .map(|n| {
            let m = f64::from(n.mass);
            0.5 * m * f64::from(n.vel.length_squared()) + m * 9.807 * f64::from((sim.origin + n.pos).y)
        })
        .sum::<f64>()
}

fn run_coupled(sim: &mut ExplicitBeams, set: ProxySet, steps: u32, mut each: impl FnMut(&ExplicitBeams, &ProxySet)) -> ProxySet {
    let mut set = Box::new(set);
    set.begin_window();
    sim.rigid = Some(set);
    for _ in 0..steps {
        sim.step();
        each(sim, sim.rigid.as_ref().unwrap());
    }
    *sim.rigid.take().unwrap()
}

/// An infinite-mass proxy at rest is the static shape: every node state the same, bit for
/// bit, through a 15 m/s crash into a turned wall.
#[test]
fn static_proxy_is_the_static_shape_bit_for_bit() {
    let rot = Quat::from_rotation_y(0.3);
    let (c, half) = (Vec3::new(0.0, 1.0, 2.0), Vec3::new(2.0, 1.0, 0.4));
    let mut a = sedan(15.0, 0.0);
    let mut b = a.clone();
    a.shapes.push(StaticShape::oriented_box(c, half, rot));
    for _ in 0..600 {
        a.step();
    }
    let set = run_coupled(&mut b, box_proxy(c, half, rot, f32::INFINITY), 600, |_, _| {});
    assert!(set.shapes[0].impulse.length() > 1000.0, "the wall was hit: {}", set.shapes[0].impulse);
    for (na, nb) in a.nodes.iter().zip(&b.nodes) {
        assert_eq!(na.pos, nb.pos);
        assert_eq!(na.vel, nb.vel);
        assert_eq!(na.force, nb.force);
    }
}

struct Crash {
    v0: f32,
    vehicle_mass: f32,
    box_mass: f32,
    /// Vehicle and box velocity along Z at the end.
    vehicle_v: f32,
    box_v: f32,
    /// Largest |ΔP| (horizontal, N s) over the run, and P0.
    worst_dp: f32,
    p0: f32,
    /// Largest rise of the total energy over the start (J) and the start's kinetic energy.
    worst_rise: f64,
    e_kin0: f64,
    /// The same vehicle's largest rise in free flight (no box) over the same time: the
    /// integrator's own energy oscillation, the floor any rise is compared with.
    free_rise: f64,
}

/// A vehicle flying (no ground: no horizontal external force) at `v0` into a free box of
/// `mass` floating 1 m ahead of it, for `seconds`.
fn crash(v0: f32, mass: f32, seconds: f32) -> Crash {
    let mut sim = sedan(v0, 0.0);
    sim.ground = -1.0e4;
    let steps = (seconds / engine_softbody::explicit_beams::STEP).round() as u32;
    let free_rise = {
        let mut free = sim.clone();
        let e0 = node_energy(&free);
        let mut worst = 0.0f64;
        for _ in 0..steps {
            free.step();
            worst = worst.max(node_energy(&free) - e0);
        }
        worst
    };
    let half = Vec3::new(1.2, 0.8, 0.5);
    let (lo, hi) = bounds(&sim);
    let centre = Vec3::new(0.0, 0.5 * (lo.y + hi.y), 1.0 + half.z);
    let set = box_proxy(centre, half, Quat::IDENTITY, mass);
    let vehicle_mass: f32 = sim.nodes.iter().map(|n| n.mass).sum();
    let horizontal = |p: Vec3| Vec3::new(p.x, 0.0, p.z);
    let p0 = horizontal(node_momentum(&sim));
    let e0 = node_energy(&sim);
    let e_kin0 = 0.5 * f64::from(vehicle_mass) * f64::from(v0 * v0);
    let (mut worst_dp, mut worst_rise) = (0.0f32, f64::MIN);
    let set = run_coupled(&mut sim, set, steps, |sim, set| {
        let pb = set.bodies[0].momentum().0;
        let p = horizontal(node_momentum(sim) + pb);
        worst_dp = worst_dp.max((p - p0).length());
        // The box's gravitational energy does not change (no gravity on it in the frame).
        let e = node_energy(sim) + set.bodies[0].kinetic_energy();
        worst_rise = worst_rise.max(e - e0);
    });
    Crash {
        v0,
        vehicle_mass,
        box_mass: mass,
        vehicle_v: node_momentum(&sim).z / vehicle_mass,
        box_v: set.bodies[0].vel.z,
        worst_dp,
        p0: p0.length(),
        worst_rise,
        e_kin0,
        free_rise,
    }
}

/// Horizontal momentum (no external horizontal force) closes to the fixed-point rounding
/// of the per-step sums through the crash; the total energy (kinetic and gravitational of
/// the nodes, the beams' elastic ½ k (L − L₀)², kinetic of the box) never rises above the
/// start's by more than the vehicle's own free-flight oscillation of that measure (the
/// explicit integrator's, with no box): the coupling adds no energy.
#[test]
fn momentum_closes_and_energy_does_not_rise() {
    for mass in [200.0, 1500.0, 6000.0] {
        let c = crash(15.0, mass, 0.4);
        println!(
            "box {mass} kg: |dP| {:.4} of {:.0} N s, energy rise {:.3} J (free flight {:.3}) of {:.0} J, v {:.3} / {:.3}",
            c.worst_dp, c.p0, c.worst_rise, c.free_rise, c.e_kin0, c.vehicle_v, c.box_v
        );
        assert!(c.box_v > 0.5, "the box was hit");
        assert!(c.worst_dp < 1e-4 * c.p0, "momentum: {} of {}", c.worst_dp, c.p0);
        assert!(c.worst_rise <= c.free_rise + 1e-5 * c.e_kin0, "energy rose by {} J (free {})", c.worst_rise, c.free_rise);
    }
}

/// Into a free box of known mass M at rest, a vehicle of mass m at v gives the box, once
/// they part, a speed between the perfectly plastic m v / (m + M) and the elastic
/// 2 m v / (m + M) (exact consequences of momentum and energy), and the vehicle the
/// balance: m v = m v' + M V'.
#[test]
fn box_of_known_mass_takes_the_momentum_exchange_physics_predicts() {
    for mass in [300.0, 1200.0, 4000.0] {
        let c = crash(12.0, mass, 1.2);
        let (m, big) = (c.vehicle_mass, c.box_mass);
        let plastic = m * c.v0 / (m + big);
        let elastic = 2.0 * plastic;
        let restitution = (c.box_v - c.vehicle_v) / c.v0;
        println!(
            "m {m:.0} kg, M {big} kg: V' {:.3} in [{plastic:.3}, {elastic:.3}], v' {:.3}, e {restitution:.3}",
            c.box_v, c.vehicle_v
        );
        // Still pushing at the end (a heavy box): within 0.5 % of the common speed.
        assert!(c.box_v >= plastic * 0.995 && c.box_v <= elastic, "box speed {}", c.box_v);
        // Parted, or still pushing together: the box is not slower than the vehicle's
        // centre of mass (beyond its body's own ringing, 0.5 % of the speed).
        assert!(c.box_v >= c.vehicle_v - 5e-3 * c.v0, "box {} behind vehicle {}", c.box_v, c.vehicle_v);
        let balance = m * c.vehicle_v + big * c.box_v;
        assert!((balance - m * c.v0).abs() < 1e-4 * m * c.v0, "balance {balance} vs {}", m * c.v0);
    }
}

/// The substeps matter: with the box frozen at its window-start velocity (a wall for the
/// frame) a light box would take a speed no physics allows; integrated at 2 kHz it never
/// passes the elastic bound. This is a measurement of the design's choice.
#[test]
fn a_light_box_moves_away_inside_the_frame() {
    let c = crash(15.0, 100.0, 0.3);
    let elastic = 2.0 * c.vehicle_mass * c.v0 / (c.vehicle_mass + c.box_mass);
    assert!(c.box_v <= elastic, "{} > {elastic}", c.box_v);
}



