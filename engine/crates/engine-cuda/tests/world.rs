//! The World step on CUDA (`CudaWorld`: the rigid step, the explicit vehicles' coupling
//! and the explicit frame in one context on one stream; engine/docs/world-on-cuda.md C4),
//! with no fixture: scenes built here from the procedural sedan and boxes.
//!
//! * the coupling's kernels are built for CUDA, an entry a module, in the shared plan's
//!   order;
//! * two sedans dropped over a static slab, one onto a dynamic deck, one onto and among
//!   dynamic boxes: the state stays finite, no node goes through the slab or the deck,
//!   the bodies carry each vehicle's weight (its windows' impulses on them sum to its
//!   weight, downward), the deck stays put, boxes are pushed, the status words stay
//!   zero, and two runs agree bit for bit on the explicit state and the bodies;
//! * the same pile with a dynamic deck, step for step the wgpu World's on Metal (lowest
//!   nodes within a millimetre, the windows' impulses within 0.1 %);
//! * a flying sedan into a free box (engine-world's
//!   `a_vehicle_shoves_a_free_box_and_momentum_closes`): the momentum the vehicle loses
//!   the box gains through the World's impulse path (plan, frame, emit, rigid step), and
//!   the box ends between the plastic and elastic bounds;
//! * the explicit side against the CPU reference coupling (`ProxySet`), proxies written
//!   by the host: engine-metal's `metal_coupled_sedan_into_a_rigid_box_matches_cpu`.
//!
//! GPU tests run when `ENGINE_CUDA_DEVICE=1` (an NVIDIA host with nvcc at build time).

use engine_cuda::{CouplingDef, CudaExplicitBeams, CudaSim, CudaWorld, CUDA_COUPLING_KERNELS};
use engine_exec::explicit_plan::coupling::world as cw;
use engine_exec::explicit_plan::CouplingCapacity;
use engine_schema::{AvbdParams, BodyState, Collider, ShapeType, WorldCapacity};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, GRAVITY};
use engine_softbody::procedural::{car_truck_bare, CarParams};
use glam::{Quat, Vec3};

fn gpu() -> bool {
    if std::env::var("ENGINE_CUDA_DEVICE").as_deref() != Ok("1") {
        eprintln!("skipped: set ENGINE_CUDA_DEVICE=1 on an NVIDIA host (built with nvcc)");
        return false;
    }
    assert!(CudaWorld::has_ptx(), "ENGINE_CUDA_DEVICE=1 but this build embedded no device code for the World (no nvcc)");
    assert!(CudaWorld::available(), "ENGINE_CUDA_DEVICE=1 but no NVIDIA driver/device loaded");
    true
}

#[test]
fn coupling_entries_are_the_plans() {
    let names: Vec<&str> = cw::Entry::ALL.iter().map(|e| e.name()).collect();
    let built: Vec<&str> = CUDA_COUPLING_KERNELS.iter().map(|k| k.entry).collect();
    assert_eq!(built, names, "engine-cuda build.rs COUPLING_ENTRIES and the shared plan's entries");
    for (k, e) in CUDA_COUPLING_KERNELS.iter().zip(cw::Entry::ALL) {
        assert!(k.cu_bytes > 100, "{}: CUDA source is {} bytes", k.entry, k.cu_bytes);
        assert!(k.threads.iter().product::<u32>() > 0, "{}: empty thread group", k.entry);
        // Every binding the entry's passes bind is in its bank, and the bank names no
        // binding the module does not have.
        for &(binding, _) in e.set() {
            assert!(k.binds.iter().any(|b| b.2 == binding), "{}: binding {binding} not in the bank", k.entry);
        }
        for &(_, size, binding) in k.binds {
            let _ = cw::source(binding);
            assert!(size == 8 || size == 16, "{}: binding {binding} is {size} bytes", k.entry);
        }
    }
    if std::env::var("ENGINE_CUDA_DEVICE").as_deref() == Ok("1") || std::env::var("ENGINE_CUDA_REQUIRE_PTX").as_deref() == Ok("1") {
        for k in CUDA_COUPLING_KERNELS {
            assert!(k.image.len() > 1024, "{}: fatbin is {} bytes", k.entry, k.image.len());
        }
    }
}

fn settings() -> ExplicitBeamsSettings {
    ExplicitBeamsSettings { turbulence: false, drag: false, ..Default::default() }
}

fn sedan_text() -> String {
    car_truck_bare(&CarParams { minimass: 10.0, ..CarParams::sedan() })
}

/// The procedural sedan (its long axis along Z), its lowest node at `at.y`, its box's
/// middle at `at.x`, its front face at `at.z`, with no ground of its own.
fn sedan(at: Vec3, velocity: Vec3) -> ExplicitBeams {
    let text = sedan_text();
    let probe = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings()).unwrap();
    let (lo, hi) = probe.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| (a.min(n.pos), b.max(n.pos)));
    let shift = Vec3::new(at.x - 0.5 * (lo.x + hi.x), at.y - lo.y, at.z - hi.z);
    let mut car = ExplicitBeams::from_truck(&text, shift, velocity, settings()).unwrap();
    car.ground = -1.0e4;
    car
}

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 4096,
        max_colliders: 4096,
        max_contacts: 1 << 16,
        max_joints: 64,
        max_candidates: 1 << 17,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    }
}

fn fixed_box(body: u32, pos: Vec3, half: Vec3) -> (BodyState, Collider) {
    let mut state = BodyState::dynamic(pos, Quat::IDENTITY, 0.0, Vec3::ZERO);
    state.inv_mass = 0.0;
    state.inv_inertia = [0.0; 3];
    state.flags = 0;
    (state, box_collider(body, half, 0.0))
}

fn box_collider(body: u32, half: Vec3, density: f32) -> Collider {
    Collider {
        body,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [half.x, half.y, half.z, half.length()],
    }
}

fn dynamic_box(body: u32, pos: Vec3, half: Vec3, mass: f32, inertia: Option<Vec3>) -> (BodyState, Collider) {
    let i = inertia.unwrap_or(Vec3::new(
        mass / 3.0 * (half.y * half.y + half.z * half.z),
        mass / 3.0 * (half.x * half.x + half.z * half.z),
        mass / 3.0 * (half.x * half.x + half.y * half.y),
    ));
    let volume = 8.0 * half.x * half.y * half.z;
    (BodyState::dynamic(pos, Quat::IDENTITY, mass, i), box_collider(body, half, mass / volume))
}

fn rigid(bodies: &[(BodyState, Collider)]) -> CudaSim {
    let mut sim = CudaSim::new(capacity()).expect("CUDA device");
    // The explicit solver's own gravity: bodies and vehicles fall alike. No rigid sleep.
    sim.set_params(AvbdParams { gravity_y: GRAVITY, sleep_time: 1.0e6, ..AvbdParams::default() });
    sim.add_bodies(&bodies.iter().map(|b| b.0).collect::<Vec<_>>());
    sim.add_colliders(&bodies.iter().map(|b| b.1).collect::<Vec<_>>());
    sim
}

const BOXES: u32 = 6;
const BOX_HALF: Vec3 = Vec3::new(0.5, 0.25, 0.9);
const BOX_MASS: f32 = 360.0;
const DECK_HALF: Vec3 = Vec3::new(1.6, 0.25, 3.2);
const DECK_MASS: f32 = 3000.0;
const DECK_X: f32 = -3.5;

fn box_rest(k: u32) -> Vec3 {
    Vec3::new(1.5 + 1.1 * k as f32, BOX_HALF.y, 0.3 * (k % 3) as f32)
}

/// Over a static slab (its top at y = 0): a deck larger than a sedan (a second static
/// body, or with `free_deck` a dynamic one of 3000 kg lying on the slab) with a sedan
/// 0.4 m above it, and a row of six dynamic boxes with a second sedan 0.4 m above them
/// (narrower than the sedan: it lands on some and among the others).
fn pile(sleep: bool, free_deck: bool) -> (CudaWorld, Vec<ExplicitBeams>) {
    let deck_at = Vec3::new(DECK_X, DECK_HALF.y, 0.0);
    let mut bodies = vec![
        fixed_box(0, Vec3::new(0.0, -0.5, 0.0), Vec3::new(30.0, 0.5, 30.0)),
        if free_deck { dynamic_box(1, deck_at, DECK_HALF, DECK_MASS, None) } else { fixed_box(1, deck_at, DECK_HALF) },
    ];
    for k in 0..BOXES {
        bodies.push(dynamic_box(2 + k, box_rest(k), BOX_HALF, BOX_MASS, None));
    }
    let top = 2.0 * BOX_HALF.y;
    let cars = vec![sedan(Vec3::new(DECK_X, top + 0.4, 2.2), Vec3::ZERO), sedan(Vec3::new(4.0, top + 0.4, 2.2), Vec3::ZERO)];
    let def = CouplingDef { sleep: sleep.then(engine_softbody::explicit_sleep::VehicleSleep::default), ..CouplingDef::default() };
    let world = CudaWorld::new(rigid(&bodies), &cars, &def).expect("CudaWorld");
    (world, cars)
}

/// What a run of the pile leaves: the explicit state's words, the bodies, per vehicle its
/// lowest node over the run and its centre of mass at the end, and per step and vehicle
/// the vertical impulse its window put on the bodies (N s).
struct PileRun {
    words: Vec<(&'static str, Vec<u32>)>,
    bodies: Vec<BodyState>,
    lowest: Vec<f32>,
    centre_y: Vec<f32>,
    impulse_y: Vec<Vec<f32>>,
    weight: Vec<f32>,
    ring_waits: u64,
    /// Every tenth step: (step, each vehicle's lowest node then, the deck's height).
    trace: Vec<(u32, Vec<f32>, f32)>,
}

fn run_pile(sleep: bool, free_deck: bool, steps: u32) -> PileRun {
    let (mut w, cars) = pile(sleep, free_deck);
    let dt = AvbdParams::default().dt;
    let weight: Vec<f32> = cars.iter().map(|c| c.nodes.iter().map(|n| n.mass).sum::<f32>() * -GRAVITY).collect();
    let mut impulse_y = Vec::new();
    let mut lowest = vec![f32::MAX; cars.len()];
    let mut trace = Vec::new();
    for step in 0..steps {
        w.step(dt).unwrap_or_else(|e| panic!("step {step}: {e}"));
        // Waiting each step: the next step's plan has this step's counters (one late).
        w.wait().unwrap_or_else(|e| panic!("step {step}: {e}"));
        impulse_y.push(w.window_impulses().unwrap().iter().map(|car| car.iter().map(|s| s.1[1]).sum::<f32>()).collect::<Vec<f32>>());
        if step % 5 == 4 {
            let mut now = vec![f32::MAX; cars.len()];
            for (v, car) in w.explicit().positions().unwrap().iter().enumerate() {
                for p in car {
                    assert!(p.is_finite(), "step {step}: a node of vehicle {v} is not finite");
                    now[v] = now[v].min(p.y);
                }
                lowest[v] = lowest[v].min(now[v]);
            }
            if step % 10 == 9 {
                let deck = w.sim().download_bodies()[1];
                if std::env::var("WORLD_TRACE").is_ok() {
                    eprintln!("step {step}: lowest nodes {now:?}, deck y {:.4} vy {:.4}", deck.position[1], deck.linear_vel[1]);
                }
                trace.push((step, now.clone(), deck.position[1]));
            }
        }
    }
    assert_eq!(w.coupling_status().unwrap(), 0, "the coupling's status word");
    let pos = w.explicit().positions().unwrap();
    let centre_y = pos
        .iter()
        .zip(&cars)
        .map(|(p, c)| {
            let m: f32 = c.nodes.iter().map(|n| n.mass).sum();
            p.iter().zip(&c.nodes).map(|(p, n)| p.y * n.mass).sum::<f32>() / m
        })
        .collect();
    for v in w.explicit().velocities().unwrap().iter().flatten() {
        assert!(v.is_finite(), "a node's velocity is not finite");
    }
    let clock = w.clock();
    assert_eq!(clock.windows, u64::from(steps));
    assert_eq!(clock.steps, (f64::from(steps) * f64::from(dt) / cw::STEP).round() as u64, "explicit steps owed to World time");
    let ring_waits = w.ring_waits;
    let words = w.explicit().state_words().unwrap();
    let bodies = w.sim().download_bodies();
    PileRun { words, bodies, lowest, centre_y, impulse_y, weight, ring_waits, trace }
}

#[test]
fn cuda_world_vehicles_rest_on_bodies() {
    if !gpu() {
        return;
    }
    let steps = 120;
    let a = run_pile(false, false, steps);
    let dt = AvbdParams::default().dt;
    for (k, b) in a.bodies.iter().enumerate() {
        let finite = b.position.iter().chain(&b.linear_vel).chain(&b.angular_vel).chain(&b.orientation).all(|x| x.is_finite());
        assert!(finite, "body {k} is not finite: {b:?}");
    }
    // The bodies carry the vehicles: over the last third of the run each vehicle's
    // windows put its weight on the bodies under it, downward (momentum goes to the
    // bodies with the right sign and size; the vehicles are still settling, hence the band).
    let tail = &a.impulse_y[(2 * steps / 3) as usize..];
    let mean: Vec<f32> = (0..2).map(|v| tail.iter().map(|j| j[v]).sum::<f32>() / tail.len() as f32).collect();
    let want: Vec<f32> = a.weight.iter().map(|w| -w * dt).collect();
    let deck = &a.bodies[1];
    eprintln!(
        "CUDA World pile: lowest nodes {:?} m, centres {:?} m, mean window impulse on the bodies {mean:?} N s (the weights' {want:?}), deck at {:?}, ring waits {}",
        a.lowest, a.centre_y, deck.position, a.ring_waits
    );
    // Nothing through the slab (its top at 0), and the first vehicle never through the
    // deck (its top at 0.5; a node's contact band is millimetres). (A static deck: on a
    // dynamic one lying on the slab a vehicle sinks, on the wgpu host alike; see
    // `cuda_world_pile_is_the_wgpu_worlds`.)
    assert!(a.lowest[1] > -0.05, "vehicle 1: a node at {} m: through the slab", a.lowest[1]);
    assert!(a.lowest[0] > 2.0 * DECK_HALF.y - 0.05, "vehicle 0: a node at {} m: through the deck", a.lowest[0]);
    // The first vehicle rests on the deck: its centre of mass above the deck's top.
    assert!(a.centre_y[0] > 2.0 * DECK_HALF.y + 0.2 && a.centre_y[0] < 1.5, "vehicle 0: centre of mass at {} m", a.centre_y[0]);
    for v in 0..2 {
        assert!(mean[v] < 0.0, "vehicle {v} pushes the bodies up: {} N s a window", mean[v]);
        assert!((mean[v] - want[v]).abs() < 0.25 * want[v].abs(), "vehicle {v}: mean window impulse {} N s, its weight a window is {}", mean[v], want[v]);
    }
    // The first windows are free fall: no impulse before the vehicles reach the bodies.
    assert_eq!(a.impulse_y[0], vec![0.0, 0.0], "an impulse in the first window, 0.4 m above the bodies");
    // Boxes under the second vehicle were pushed, and none left the slab (the reaction is
    // applied once, with the right sign: a doubled or inverted impulse throws them).
    let moved = (0..BOXES)
        .filter(|&k| {
            let b = &a.bodies[2 + k as usize];
            (Vec3::from(b.position) - box_rest(k)).length() > 1e-3 || (Quat::from_array(b.orientation).w.abs() - 1.0).abs() > 1e-6
        })
        .count();
    assert!(moved >= 1, "{moved} boxes moved under the second vehicle");
    for k in 0..BOXES as usize {
        let y = a.bodies[2 + k].position[1];
        assert!(y > 0.0 && y < 1.0, "box {k} at y {y}");
    }

    // Two runs: the explicit state bit for bit (every state word), and the bodies.
    let b = run_pile(false, false, steps);
    for ((name, x), (_, y)) in a.words.iter().zip(&b.words) {
        let diff = x.iter().zip(y).filter(|(p, q)| p != q).count();
        assert_eq!(diff, 0, "two runs: {name}: {diff} of {} words differ", x.len());
    }
    let body_diff = a.bodies.iter().zip(&b.bodies).filter(|(p, q)| bytemuck::bytes_of(*p) != bytemuck::bytes_of(*q)).count();
    eprintln!("CUDA World pile, two runs: explicit state bit for bit; {body_diff} of {} bodies differ", a.bodies.len());
    assert_eq!(body_diff, 0, "two runs: {body_diff} bodies differ");
}

/// The pile with a dynamic deck, step for step what the wgpu World does with the same
/// scene (engine-world `tests/explicit_world_digest.rs`,
/// `the_cuda_world_tests_pile_on_this_host`; the numbers below are that test's on Metal,
/// an M3 Max, 2026-11-22, bit-exact vehicles): each vehicle's lowest node within a
/// millimetre at steps 19, 59 and 119, the deck's height, and each vehicle's mean window
/// impulse on the bodies over steps 80 to 119. The rigid step, the coupling and the
/// explicit frame of two hosts and two GPU vendors, held together.
///
/// (Both hosts show the first sedan sinking through the dynamic deck at about 0.2 m/s
/// while its windows put its full weight on it: within a window the explicit solver
/// integrates a proxy as a free body, and the deck's support by the slab reaches the
/// vehicle only as the deck's reset at the next window. A model property of the
/// staggered coupling, reported in world-on-cuda.md, not judged here.)
#[test]
fn cuda_world_pile_is_the_wgpu_worlds() {
    if !gpu() {
        return;
    }
    let a = run_pile(false, true, 120);
    // (step, vehicle 0's lowest node, vehicle 1's, the deck's height) on the wgpu host.
    let wgpu = [(19, 0.44457087f32, 0.3475625f32, 0.2482f32), (59, 0.31409833, -0.00024173231, 0.2455), (119, 0.11115019, -0.0006055629, 0.2430)];
    let mut worst = 0.0f32;
    for (step, y0, y1, deck) in wgpu {
        let (_, lowest, deck_y) = a.trace.iter().find(|t| t.0 == step).expect("a traced step");
        worst = worst.max((lowest[0] - y0).abs()).max((lowest[1] - y1).abs());
        assert!((lowest[0] - y0).abs() < 1e-3 && (lowest[1] - y1).abs() < 1e-3, "step {step}: lowest nodes {lowest:?}, the wgpu World's ({y0}, {y1})");
        assert!((deck_y - deck).abs() < 1e-3, "step {step}: the deck at {deck_y}, the wgpu World's at {deck}");
    }
    let tail = &a.impulse_y[80..];
    let mean: Vec<f32> = (0..2).map(|v| tail.iter().map(|j| j[v]).sum::<f32>() / tail.len() as f32).collect();
    let wgpu_mean = [-614.57007f32, -615.3169];
    eprintln!("CUDA World pile against the wgpu World's: lowest nodes within {:.2e} m at steps 19, 59, 119; mean window impulses {mean:?} N s (wgpu {wgpu_mean:?})", worst);
    for v in 0..2 {
        assert!((mean[v] - wgpu_mean[v]).abs() < 1e-3 * wgpu_mean[v].abs(), "vehicle {v}: mean window impulse {} N s, the wgpu World's {}", mean[v], wgpu_mean[v]);
    }
}

#[test]
fn cuda_world_vehicle_sleep_runs_the_asleep_bits() {
    if !gpu() {
        return;
    }
    // Vehicle sleep on (the World's default): cpl_sleep_mask runs every window and the
    // explicit state is the run's without it while no vehicle sleeps (1 s: none does).
    let a = run_pile(true, false, 60);
    let b = run_pile(false, false, 60);
    let mut differing = Vec::new();
    for ((name, x), (_, y)) in a.words.iter().zip(&b.words) {
        if x.len() != y.len() || x.iter().zip(y).any(|(p, q)| p != q) {
            differing.push(*name);
        }
    }
    eprintln!("CUDA World pile, vehicle sleep on against off: state sections that differ: {differing:?}");
    for (p, q) in a.bodies.iter().zip(&b.bodies) {
        for k in 0..3 {
            assert!((p.position[k] - q.position[k]).abs() < 1e-3, "a body differs with vehicle sleep on: {:?} vs {:?}", p.position, q.position);
        }
    }
    assert!(a.lowest[1] > -0.05, "a node at {} m: through the slab", a.lowest[1]);
}

#[test]
fn cuda_world_vehicle_shoves_a_free_box_and_momentum_closes() {
    if !gpu() {
        return;
    }
    let speed = 15.0f32;
    let car = sedan(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, speed));
    let hi_y = car.nodes.iter().map(|n| n.pos.y).fold(f32::MIN, f32::max);
    let (half, mass) = (Vec3::new(1.2, 0.7, 0.5), 1200.0f32);
    // A box that does not turn (a huge rotational inertia): the one-dimensional collision
    // whose bounds are exact. Nothing else in the world: both fly and fall alike.
    let free = dynamic_box(0, Vec3::new(0.0, 0.5 * (1.0 + hi_y), 0.8 + half.z), half, mass, Some(Vec3::splat(1.0e9)));
    let masses: Vec<f32> = car.nodes.iter().map(|n| n.mass).collect();
    let m: f32 = masses.iter().sum();
    let mut w = CudaWorld::new(rigid(&[free]), std::slice::from_ref(&car), &CouplingDef { sleep: None, ..CouplingDef::default() }).expect("CudaWorld");
    let horizontal = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
    let momentum = |w: &mut CudaWorld| -> Vec3 { w.explicit().velocities().unwrap()[0].iter().zip(&masses).map(|(v, m)| *v * *m).sum() };
    let p0 = horizontal(momentum(&mut w));
    let dt = AvbdParams::default().dt;
    let frames = (1.2 / dt).round() as u32;
    let mut worst = 0.0f32;
    for k in 0..frames {
        w.step(dt).unwrap();
        w.wait().unwrap();
        if k % 4 == 3 {
            let vb = Vec3::from(w.sim().download_bodies()[0].linear_vel);
            // The last window's impulses, applied by the next step's rigid step.
            let pending: Vec3 = w.window_impulses().unwrap()[0].iter().map(|s| Vec3::from(s.1)).sum();
            let p = horizontal(momentum(&mut w) + vb * mass + pending);
            worst = worst.max((p - p0).length());
        }
    }
    assert_eq!(w.coupling_status().unwrap(), 0);
    let vb = Vec3::from(w.sim().download_bodies()[0].linear_vel);
    let vv = momentum(&mut w).z / m;
    let plastic = m * speed / (m + mass);
    eprintln!(
        "CUDA World: box {:.3} m/s in [{plastic:.3}, {:.3}], vehicle {vv:.3}; worst |dP| {worst:.3} of {:.0} N s",
        vb.z,
        2.0 * plastic,
        p0.length()
    );
    assert!(vb.z > 0.5 * plastic, "the box was shoved: {vb:?}");
    assert!(worst < 2e-4 * p0.length(), "momentum: {worst} of {}", p0.length());
    assert!(vb.z >= 0.99 * plastic && vb.z <= 2.0 * plastic, "box speed {}", vb.z);
    // Kinetic energy is lost, never gained (deformation, damping).
    let e0 = 0.5 * m * speed * speed;
    let e1 = 0.5 * m * vv * vv + 0.5 * mass * vb.z * vb.z;
    assert!(e1 < e0, "energy {e1} after {e0}");
}

/// The explicit side of the coupling against the CPU reference (`ProxySet`, the proxies
/// written by the host: no World side): the sedan at 15 m/s into a free 1200 kg box, the
/// first frames of contact within 1 mm, horizontal momentum closed to 2e-4, box and
/// vehicle speeds within 1 % (engine-metal's test of the same name, the procedural sedan).
#[test]
fn cuda_coupled_sedan_into_a_rigid_box_matches_cpu() {
    use engine_softbody::explicit_rigid::{ProxyGeom, ProxySet, ProxyShape, RigidProxy};
    if !gpu() {
        return;
    }
    const STEPS: u32 = 17;
    let sim = sedan(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 15.0));
    let hi_y = sim.nodes.iter().map(|n| n.pos.y).fold(f32::MIN, f32::max);
    let (centre, half, mass) = (Vec3::new(0.0, 0.5 * (1.0 + hi_y), 1.3), Vec3::new(1.2, 0.7, 0.5), 1200.0f32);
    let inertia = Vec3::new(
        mass / 3.0 * (half.y * half.y + half.z * half.z),
        mass / 3.0 * (half.x * half.x + half.z * half.z),
        mass / 3.0 * (half.x * half.x + half.y * half.y),
    );
    let mut set = ProxySet {
        bodies: vec![RigidProxy {
            body: 7,
            pos: centre,
            rot: Quat::from_rotation_y(0.2),
            vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            inv_mass: 1.0 / mass,
            inv_inertia: Vec3::ONE / inertia,
            share: 1.0,
            count: 0,
        }],
        shapes: vec![ProxyShape::new(0, Vec3::ZERO, Quat::IDENTITY, ProxyGeom::Box { half }, 3, None)],
        frozen: false,
    };
    set.begin_window();
    let m: f32 = sim.nodes.iter().map(|n| n.mass).sum();
    let cap = CouplingCapacity { bodies: 4, shapes: 8, planes: 64 };
    let mut gpu = CudaExplicitBeams::with_coupling(std::slice::from_ref(&sim), Some(cap)).unwrap();
    gpu.write_proxies(0, &set).unwrap();
    gpu.set_coupling_on(true).unwrap();
    let mut cpu = sim.clone();
    cpu.rigid = Some(Box::new(set.clone()));
    let horizontal = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
    let p0 = horizontal(sim.nodes.iter().map(|n| n.vel * n.mass).sum());
    let mut first_contact = None;
    let mut worst_early = 0.0f32;
    for frame in 0..48 {
        gpu.frame(STEPS).unwrap();
        for _ in 0..STEPS {
            cpu.step();
        }
        if first_contact.is_none() && cpu.rigid.as_ref().unwrap().shapes[0].impulse.length() > 0.0 {
            first_contact = Some(frame);
        }
        if first_contact.is_some_and(|f| frame <= f + 1) {
            let pos = gpu.positions().unwrap();
            for (a, n) in pos[0].iter().zip(&cpu.nodes) {
                worst_early = worst_early.max((*a - (cpu.origin + n.pos)).length());
            }
        }
    }
    assert!(first_contact.is_some(), "the box was hit");
    let mut gset = set.clone();
    gpu.read_proxies(0, &mut gset).unwrap();
    let cset = cpu.rigid.take().unwrap();
    let gvel = gpu.velocities().unwrap();
    let masses: Vec<f32> = sim.nodes.iter().map(|n| n.mass).collect();
    let pg = horizontal(gvel[0].iter().zip(&masses).map(|(v, m)| *v * *m).sum::<Vec3>()) + horizontal(gset.bodies[0].momentum().0);
    let vg = gvel[0].iter().zip(&masses).map(|(v, m)| v.z * m).sum::<f32>() / m;
    let vc = cpu.nodes.iter().map(|n| n.vel.z * n.mass).sum::<f32>() / m;
    eprintln!(
        "CUDA coupled: early worst {worst_early:.2e} m; box GPU {:.4} CPU {:.4} m/s; vehicle GPU {vg:.4} CPU {vc:.4}; |dP| {:.4} of {:.0}",
        gset.bodies[0].vel.z,
        cset.bodies[0].vel.z,
        (pg - p0).length(),
        p0.length()
    );
    assert!(worst_early < 1e-3, "early frames apart by {worst_early} m");
    assert!((pg - p0).length() < 2e-4 * p0.length(), "CUDA momentum {pg} vs {p0}");
    let rel = |a: f32, b: f32| (a - b).abs() / b.abs().max(1e-3);
    assert!(rel(gset.bodies[0].vel.z, cset.bodies[0].vel.z) < 0.01);
    assert!(rel(vg, vc) < 0.01);
}

/// The explicit modules this file's GPU tests load, as `ENGINE_CUDA_EXPLICIT_MODULES`
/// takes them (printed; CPU only):
/// `cargo test -p engine-cuda --release --test world world_test_modules -- --nocapture`.
#[test]
fn world_test_modules() {
    use engine_exec::explicit_plan::{ExplicitUpload, Kernel};
    let cap = Some(CouplingDef::default().capacity);
    let car = sedan(Vec3::ZERO, Vec3::ZERO);
    let mut need = std::collections::BTreeSet::new();
    for sleep in [None, Some(engine_softbody::explicit_sleep::VehicleSleep::default())] {
        for fleet in [vec![car.clone()], vec![car.clone(), car.clone()]] {
            let layout = ExplicitUpload::with_sleep(&fleet, cap, sleep).layout;
            need.extend(CudaExplicitBeams::modules_for_layout(&layout).into_iter().map(str::to_string));
            if layout.sleep.is_some() {
                need.insert(Kernel::Sleep.module_for(layout.aero, layout.lanes, layout.surface).to_string());
            }
        }
    }
    let small = Some(CouplingCapacity { bodies: 4, shapes: 8, planes: 64 });
    need.extend(CudaExplicitBeams::modules_for(std::slice::from_ref(&car), small).into_iter().map(str::to_string));
    println!("ENGINE_CUDA_EXPLICIT_MODULES={}", need.into_iter().collect::<Vec<_>>().join(","));
}
