//! The fast setting (bit-exact results off, `ExplicitBeamsGpu::set_bit_exact(false)`):
//! the same equations, contact rules, iterations and tolerances as the bit-exact
//! kernels, with the compilers free to fuse and reassociate the arithmetic
//! (`engine_webgpu::fast_float_wgsl`; vehicle-perf-first-principles.md §8.38). Its
//! results differ from the CPU reference in the last bits, and a crash amplifies last
//! bits, so "the same within a tolerance" needs a yardstick that is not chosen by hand.
//!
//! The yardstick: the bit-exact kernel (the CPU reference bit for bit:
//! `explicit_beams_gpu.rs`, `explicit_beams_bitwise.rs`) run twice, once as spawned and
//! once with every node's velocity moved by one unit in the last place (what another
//! machine's rounding does to a state). How far those two runs end apart is the scene's
//! own sensitivity to last bits. The fast setting must end no farther from the
//! bit-exact run than SPREAD_FACTOR times that, with a small absolute floor (a scene
//! that amplifies nothing still rounds differently). Per scene and run, physical
//! invariants too:
//! - every position finite;
//! - the mechanical energy (kinetic plus gravity's potential) never above its start by
//!   more than ENERGY_SLACK of the start's kinetic energy (no run injects energy; the
//!   beams' elastic energy is not counted, so it only lowers the sum);
//! - the fast setting is deterministic run to run on one machine (twice, every state
//!   word the same): its spread is zero, so the tolerance is the scene's own.
//! Measures compared: the largest node distance, the centroid, the broken beams, the
//! crush (the front-most node along the travel: the wall), the ride height (the mean
//! height: the parked car).

use engine_softbody::explicit_beams::{
    yaw_offset, ExplicitBeams, ExplicitBeamsSettings, StaticBox,
};
use engine_webgpu::{default_device, ExplicitBeamsGpu};
use glam::Vec3;

/// How many times the scene's own last-bit spread the fast setting may end from the
/// bit-exact run.
const SPREAD_FACTOR: f32 = 4.0;
/// The absolute floor of a distance's tolerance (m) and of the broken beams'.
const FLOOR_M: f32 = 0.002;
const FLOOR_BROKEN: usize = 2;
/// Energy rise allowed over the start: a fraction of the start's kinetic energy, plus
/// the weight times SPAWN_SLACK_M (the spawn state's own elastic energy: tyres and
/// suspension take the car's weight from a state without their load, and the beams'
/// elastic energy is not in the sum).
const ENERGY_SLACK: f64 = 0.01;
const SPAWN_SLACK_M: f64 = 0.02;
/// The yardstick's last-bit changes of the spawn state ([`nudged`]).
const NUDGES: u32 = 3;
const GRAVITY: f64 = 9.807;

fn sedan() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan")
}

/// Spawned at `at` (turned by `yaw` degrees), lowest node 2 cm up, tyres rolling.
fn spawn(text: &str, at: Vec3, yaw: f32, vel: Vec3) -> ExplicitBeams {
    let settings = ExplicitBeamsSettings {
        drag: true,
        turbulence: false,
        ..Default::default()
    };
    let at = Vec3::new(at.x, 0.0, at.z);
    let mut sim = if yaw == 0.0 {
        ExplicitBeams::from_truck_config(text, None, at, vel, settings)
    } else {
        ExplicitBeams::from_truck_config_yaw(text, None, at, yaw, vel, settings)
    }
    .expect("spawn");
    let lowest = sim.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
    for n in &mut sim.nodes {
        n.pos.y += 0.02 - lowest;
    }
    sim.roll_tyres(vel);
    sim
}

/// The sedan's forward axis is −x (its cinecam), its extent along it.
fn front(sim: &ExplicitBeams) -> f32 {
    sim.nodes.iter().map(|n| -n.pos.x).fold(f32::MIN, f32::max)
}

/// Last-bit changes of the spawn state (the yardstick's runs): k 0 every node's velocity
/// one unit in the last place up, k 1 down, k 2 every position up.
fn nudged(fleet: &[ExplicitBeams], k: u32) -> Vec<ExplicitBeams> {
    let up = |x: f32| {
        if x == 0.0 {
            f32::from_bits(1)
        } else if x > 0.0 {
            f32::from_bits(x.to_bits() + 1)
        } else {
            f32::from_bits(x.to_bits() - 1)
        }
    };
    let down = |x: f32| {
        if x == 0.0 {
            -f32::from_bits(1)
        } else if x > 0.0 {
            f32::from_bits(x.to_bits() - 1)
        } else {
            f32::from_bits(x.to_bits() + 1)
        }
    };
    let f = |v: Vec3, g: &dyn Fn(f32) -> f32| Vec3::new(g(v.x), g(v.y), g(v.z));
    fleet
        .iter()
        .map(|s| {
            let mut s = s.clone();
            for n in &mut s.nodes {
                match k {
                    0 => n.vel = f(n.vel, &up),
                    1 => n.vel = f(n.vel, &down),
                    _ => n.pos = f(n.pos, &up),
                }
            }
            s
        })
        .collect()
}

struct Run {
    pos: Vec<Vec<Vec3>>,
    broken: usize,
    /// The largest mechanical energy over the run less the start's (J), and the start's
    /// kinetic energy (J).
    rise: f64,
    ke0: f64,
    finite: bool,
    words: Vec<(&'static str, Vec<u32>)>,
}

fn energy(gpu: &ExplicitBeamsGpu) -> (f64, f64) {
    let (p, v, m) = (gpu.positions(), gpu.velocities(), gpu.masses());
    let mut ke = 0.0;
    let mut pe = 0.0;
    for ((p, v), m) in p.iter().zip(&v).zip(&m) {
        for ((p, v), m) in p.iter().zip(v).zip(m) {
            ke += 0.5 * f64::from(*m) * f64::from(v.length_squared());
            pe += f64::from(*m) * GRAVITY * f64::from(p.y);
        }
    }
    (ke, pe)
}

fn run(
    fleet: &[ExplicitBeams],
    bit_exact: bool,
    contact: bool,
    frames: u32,
    steps: u32,
) -> Option<Run> {
    let (_, device, queue) = default_device().ok()?;
    let mut gpu = ExplicitBeamsGpu::with_settings(&device, &queue, fleet, None, None, bit_exact);
    assert_eq!(gpu.bit_exact(), bit_exact);
    gpu.set_vehicle_contact(contact);
    let (ke0, pe0) = energy(&gpu);
    let mut rise = f64::MIN;
    let mut finite = true;
    for _ in 0..frames {
        gpu.frame(steps);
        gpu.wait();
        if let Err(e) = gpu.kernel_status() {
            panic!("kernel status: {e}");
        }
        let (ke, pe) = energy(&gpu);
        finite &= ke.is_finite() && pe.is_finite();
        rise = rise.max(ke + pe - ke0 - pe0);
    }
    let pos = gpu.positions();
    finite &= pos.iter().flatten().all(|p| p.is_finite());
    Some(Run {
        broken: gpu.beams().iter().flatten().filter(|b| b.broken).count(),
        pos,
        rise,
        ke0,
        finite,
        words: gpu.state_words(),
    })
}

fn largest(a: &Run, b: &Run) -> f32 {
    a.pos
        .iter()
        .zip(&b.pos)
        .flat_map(|(p, q)| p.iter().zip(q).map(|(p, q)| (*p - *q).length()))
        .fold(0.0, f32::max)
}

fn centroid(r: &Run) -> Vec3 {
    let all: Vec<Vec3> = r.pos.iter().flatten().copied().collect();
    all.iter().copied().sum::<Vec3>() / all.len() as f32
}

fn mean_y(r: &Run) -> f32 {
    centroid(r).y
}

fn front_of(r: &Run) -> f32 {
    r.pos[0].iter().map(|p| -p.x).fold(f32::MIN, f32::max)
}

/// The scene with the bit-exact kernel (as spawned and under each of NUDGES last-bit
/// changes) and the fast setting (twice); every check above. A measure's yardstick is
/// the largest difference any nudged run ends at from the bit-exact one (one nudge alone
/// is a single sample of a chaotic spread).
fn check(label: &str, fleet: &[ExplicitBeams], contact: bool, frames: u32, steps: u32) {
    let Some(exact) = run(fleet, true, contact, frames, steps) else {
        eprintln!("{label}: no GPU, skipped");
        return;
    };
    let nudges: Vec<Run> = (0..NUDGES)
        .map(|k| run(&nudged(fleet, k), true, contact, frames, steps).unwrap())
        .collect();
    let fast = run(fleet, false, contact, frames, steps).unwrap();
    let fast2 = run(fleet, false, contact, frames, steps).unwrap();
    // (Weight: the energy's slack below.)
    let weight: f64 = fleet
        .iter()
        .flat_map(|s| s.nodes.iter())
        .map(|n| f64::from(n.mass) * GRAVITY)
        .sum();
    for (name, r) in std::iter::once(("bit-exact", &exact))
        .chain(nudges.iter().map(|n| ("nudged", n)))
        .chain([("fast", &fast)])
    {
        assert!(
            r.finite,
            "{label}: {name}: a position or the energy is not finite"
        );
        let slack = ENERGY_SLACK * r.ke0 + SPAWN_SLACK_M * weight;
        assert!(
            r.rise <= slack,
            "{label}: {name}: mechanical energy rose {:.1} J over the start (allowed {:.1} J)",
            r.rise,
            slack
        );
    }
    let same = fast.words == fast2.words;
    let tol = |spread: f32| (SPREAD_FACTOR * spread).max(FLOOR_M);
    let spread_of = |m: &dyn Fn(&Run) -> f32| {
        nudges
            .iter()
            .map(|n| (m(n) - m(&exact)).abs())
            .fold(0.0f32, f32::max)
    };
    let spread = nudges
        .iter()
        .map(|n| largest(&exact, n))
        .fold(0.0f32, f32::max);
    let d = largest(&exact, &fast);
    let c_spread = nudges
        .iter()
        .map(|n| (centroid(&exact) - centroid(n)).length())
        .fold(0.0f32, f32::max);
    let c = (centroid(&exact) - centroid(&fast)).length();
    let b_spread = nudges
        .iter()
        .map(|n| exact.broken.abs_diff(n.broken))
        .max()
        .unwrap_or(0);
    let b = exact.broken.abs_diff(fast.broken);
    let h_spread = spread_of(&mean_y);
    let h = (mean_y(&exact) - mean_y(&fast)).abs();
    let f_spread = spread_of(&front_of);
    let f = (front_of(&exact) - front_of(&fast)).abs();
    eprintln!(
        "{label}: fast run to run the same: {same}; largest node difference fast {:.3} mm (last-bit spread {:.3}); centroid {:.3} mm ({:.3}); \
         broken beams bit-exact {} fast {} nudged {:?}; ride height {:.3} mm ({:.3}); front {:.3} mm ({:.3}); energy rise bit-exact {:.1} J, fast {:.1} J (start KE {:.0} J)",
        d * 1e3,
        spread * 1e3,
        c * 1e3,
        c_spread * 1e3,
        exact.broken,
        fast.broken,
        nudges.iter().map(|n| n.broken).collect::<Vec<_>>(),
        h * 1e3,
        h_spread * 1e3,
        f * 1e3,
        f_spread * 1e3,
        exact.rise,
        fast.rise,
        exact.ke0
    );
    assert!(
        same,
        "{label}: the fast setting is not deterministic run to run on one machine"
    );
    assert!(
        d <= tol(spread),
        "{label}: largest node difference {:.3} mm past {:.3} mm",
        d * 1e3,
        tol(spread) * 1e3
    );
    assert!(
        c <= tol(c_spread),
        "{label}: centroid {:.3} mm past {:.3} mm",
        c * 1e3,
        tol(c_spread) * 1e3
    );
    assert!(
        h <= tol(h_spread),
        "{label}: ride height {:.3} mm past {:.3} mm",
        h * 1e3,
        tol(h_spread) * 1e3
    );
    assert!(
        f <= tol(f_spread),
        "{label}: front {:.3} mm past {:.3} mm",
        f * 1e3,
        tol(f_spread) * 1e3
    );
    assert!(
        b <= (SPREAD_FACTOR as usize * b_spread).max(FLOOR_BROKEN),
        "{label}: broken beams {} against {} (last-bit spread {b_spread})",
        fast.broken,
        exact.broken
    );
}

/// The tiny scene: the sedan at rest on its ground (the ride height), 1 s.
#[test]
fn fast_parked_sedan_within_its_last_bit_spread() {
    let s = sedan();
    check(
        "parked",
        &[spawn(&s, Vec3::new(100.0, 0.0, 100.0), 0.0, Vec3::ZERO)],
        false,
        60,
        34,
    );
}

/// The sedan at 100 km/h into its own wall 1 m ahead (the crush), 1 s.
#[test]
fn fast_wall_crash_within_its_last_bit_spread() {
    let s = sedan();
    let speed = 27.78;
    let mut car = spawn(
        &s,
        Vec3::new(100.0, 0.0, 100.0),
        0.0,
        Vec3::new(-speed, 0.0, 0.0),
    );
    let face = front(&car) + 1.0;
    let mid = car.nodes.iter().map(|n| n.pos.z).sum::<f32>() / car.nodes.len() as f32;
    car.boxes.push(StaticBox::centered(
        Vec3::new(-(face + 0.5), 3.0, mid),
        Vec3::new(0.5, 3.0, 2.0),
    ));
    check("wall", &[car], false, 60, 34);
}

/// Two sedans head on at 50 km/h each, vehicle contact on (two-car contact), 1 s.
#[test]
fn fast_head_on_within_its_last_bit_spread() {
    let s = sedan();
    let speed = 13.9;
    let a = spawn(
        &s,
        Vec3::new(100.0, 0.0, 100.0),
        0.0,
        Vec3::new(-speed, 0.0, 0.0),
    );
    let turned = spawn(&s, Vec3::ZERO, 0.0, Vec3::ZERO);
    let lo = turned
        .nodes
        .iter()
        .map(|n| -yaw_offset(n.pos, 180.0).x)
        .fold(f32::MAX, f32::min);
    let at = Vec3::new(-(front(&a) + 0.5 - lo), 0.0, 100.0);
    let b = spawn(&s, at, 180.0, Vec3::new(speed, 0.0, 0.0));
    check("head-on", &[a, b], true, 60, 34);
}
