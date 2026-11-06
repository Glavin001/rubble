//! The explicit vehicle ↔ rigid body coupling on the GPU (`explicit_beams_rigid.slang`)
//! against its CPU reference (`engine_softbody::explicit_rigid`): the repo sedan flying
//! (no ground: no horizontal external force) into a free box.
//! * Momentum closes on the GPU (nodes plus box, horizontal) to the fixed-point rounding.
//! * CPU↔GPU parity: the first frames of contact agree to float noise (the GPU differs in
//!   summation order and in its fixed-point self-collision), and the exchange's outcome
//!   (box and vehicle speeds) agrees to 1 %.
//! * Coupling built but off, the GPU frame is bit for bit the uncoupled one.

use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody::explicit_rigid::{ProxyGeom, ProxySet, ProxyShape, RigidProxy};
use engine_webgpu::explicit_coupling::{read_proxies, write_proxies};
use engine_webgpu::{default_device, CouplingCapacity, ExplicitBeamsGpu};
use glam::{Quat, Vec3};

const STEPS: u32 = 17;
const CAP: CouplingCapacity = CouplingCapacity {
    bodies: 4,
    shapes: 8,
    planes: 64,
};

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let (_, d, q) = default_device().ok()?;
    Some((d, q))
}

fn sedan(speed: f32) -> ExplicitBeams {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/explicit-beams-acceptance/vehicles/rubble_sedan_ror.truck"
    ))
    .expect("sedan");
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        drag: false,
        turbulence: false,
        triangle_throttle: true,
        centred_damping: false,
    };
    let mut sim = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let b = |s: &ExplicitBeams| s.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| (a.min(n.pos), b.max(n.pos)));
    let (lo, hi) = b(&sim);
    if hi.x - lo.x > hi.z - lo.z {
        for n in &mut sim.nodes {
            n.pos = Vec3::new(-n.pos.z, n.pos.y, n.pos.x);
        }
    }
    let (lo, hi) = b(&sim);
    let shift = Vec3::new(-0.5 * (lo.x + hi.x), 1.0 - lo.y, -hi.z);
    for n in &mut sim.nodes {
        n.pos += shift;
        n.vel = Vec3::new(0.0, 0.0, speed);
    }
    sim.ground = -1.0e4;
    sim
}

fn free_box(centre: Vec3, half: Vec3, mass: f32) -> ProxySet {
    let i = Vec3::new(
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
            inv_inertia: Vec3::ONE / i,
            share: 1.0,
            count: 0,
        }],
        shapes: vec![ProxyShape::new(0, Vec3::ZERO, Quat::IDENTITY, ProxyGeom::Box { half }, 3, None)],
        frozen: false,
    };
    set.begin_window();
    set
}

fn setup(mass: f32) -> (ExplicitBeams, ProxySet, f32) {
    let sim = sedan(15.0);
    let hi_y = sim.nodes.iter().map(|n| n.pos.y).fold(f32::MIN, f32::max);
    let set = free_box(Vec3::new(0.0, 0.5 * (1.0 + hi_y), 0.8 + 0.5), Vec3::new(1.2, 0.7, 0.5), mass);
    let m: f32 = sim.nodes.iter().map(|n| n.mass).sum();
    (sim, set, m)
}

fn horizontal(v: Vec3) -> Vec3 {
    Vec3::new(v.x, 0.0, v.z)
}

#[test]
fn coupled_crash_matches_the_cpu_and_closes_momentum() {
    let Some((device, queue)) = device() else {
        eprintln!("skip: no GPU");
        return;
    };
    let (sim, set, m) = setup(1200.0);
    let mut gpu = ExplicitBeamsGpu::with_coupling(&device, &queue, std::slice::from_ref(&sim), Some(CAP));
    write_proxies(&gpu, &queue, 0, &set).unwrap();
    gpu.set_coupling_on(true);
    let mut cpu = sim.clone();
    cpu.rigid = Some(Box::new(set.clone()));
    let p0 = horizontal(sim.nodes.iter().map(|n| n.vel * n.mass).sum());
    let mut first_contact = None;
    let mut worst_early = 0.0f32;
    for frame in 0..48 {
        gpu.frame(STEPS);
        for _ in 0..STEPS {
            cpu.step();
        }
        let cset = cpu.rigid.as_ref().unwrap();
        if first_contact.is_none() && cset.shapes[0].impulse.length() > 0.0 {
            first_contact = Some(frame);
        }
        // The first two frames of contact: node positions within float noise.
        if first_contact.is_some_and(|f| frame <= f + 1) {
            let pos = gpu.positions();
            for (a, n) in pos[0].iter().zip(&cpu.nodes) {
                worst_early = worst_early.max((*a - (cpu.origin + n.pos)).length());
            }
        }
    }
    assert!(first_contact.is_some(), "the box was hit");
    let mut gset = set.clone();
    read_proxies(&gpu, &device, &queue, 0, &mut gset);
    let cset = cpu.rigid.take().unwrap();
    let gvel = gpu.velocities();
    let masses: Vec<f32> = sim.nodes.iter().map(|n| n.mass).collect();
    let pg = horizontal(gvel[0].iter().zip(&masses).map(|(v, m)| *v * *m).sum::<Vec3>()) + horizontal(gset.bodies[0].momentum().0);
    let vg = gvel[0].iter().zip(&masses).map(|(v, m)| v.z * m).sum::<f32>() / m;
    let vc = cpu.nodes.iter().map(|n| n.vel.z * n.mass).sum::<f32>() / m;
    println!(
        "early worst {worst_early:.2e} m; box GPU {:.4} CPU {:.4} m/s; vehicle GPU {vg:.4} CPU {vc:.4}; |dP| GPU {:.4} of {:.0}",
        gset.bodies[0].vel.z,
        cset.bodies[0].vel.z,
        (pg - p0).length(),
        p0.length()
    );
    assert!(worst_early < 1e-3, "early frames apart by {worst_early} m");
    assert!((pg - p0).length() < 2e-4 * p0.length(), "GPU momentum {pg} vs {p0}");
    let rel = |a: f32, b: f32| (a - b).abs() / b.abs().max(1e-3);
    assert!(rel(gset.bodies[0].vel.z, cset.bodies[0].vel.z) < 0.01);
    assert!(rel(vg, vc) < 0.01);
    // The impulse record matches the box's momentum change.
    let j = gset.shapes[0].impulse;
    assert!((horizontal(j) - horizontal(gset.bodies[0].vel * 1200.0)).length() < 1e-3 * j.length());
}

/// More proxy bodies than a workgroup has lanes (a lone sedan runs on 512 lanes): 599
/// small boxes far off, then the free box the sedan hits as body 600 (shape 600). Every
/// body is integrated, the 600th as the CPU does: it takes the strike's impulse and
/// moves off like the first test's box. (Before the fix the kernel integrated bodies
/// j < LANES only: the struck box stood still.)
#[test]
fn more_proxy_bodies_than_lanes_are_all_integrated() {
    let Some((device, queue)) = device() else {
        eprintln!("skip: no GPU");
        return;
    };
    let (sim, one, _) = setup(1200.0);
    let mut set = ProxySet {
        bodies: Vec::new(),
        shapes: Vec::new(),
        frozen: false,
    };
    for k in 0..599u32 {
        let mut b = one.bodies[0].clone();
        b.body = 100 + k;
        b.pos = Vec3::new(50.0 + 2.0 * (k % 30) as f32, 0.5, 50.0 + 2.0 * (k / 30) as f32);
        set.bodies.push(b);
        set.shapes.push(ProxyShape::new(k, Vec3::ZERO, Quat::IDENTITY, ProxyGeom::Box { half: Vec3::splat(0.4) }, 1000 + k, None));
    }
    set.bodies.push(one.bodies[0].clone());
    let mut struck = one.shapes[0].clone();
    struck.body = 599;
    set.shapes.push(struck);
    set.begin_window();
    let cap = CouplingCapacity {
        bodies: 600,
        shapes: 600,
        planes: 64,
    };
    let mut gpu = ExplicitBeamsGpu::with_coupling(&device, &queue, std::slice::from_ref(&sim), Some(cap));
    write_proxies(&gpu, &queue, 0, &set).unwrap();
    gpu.set_coupling_on(true);
    let mut cpu = sim.clone();
    cpu.rigid = Some(Box::new(set.clone()));
    for _ in 0..48 {
        gpu.frame(STEPS);
        for _ in 0..STEPS {
            cpu.step();
        }
    }
    let mut gset = set.clone();
    read_proxies(&gpu, &device, &queue, 0, &mut gset);
    let cset = cpu.rigid.take().unwrap();
    let (g, c) = (gset.bodies[599].vel.z, cset.bodies[599].vel.z);
    println!("body 600 of 600: GPU {g:.4} m/s, CPU {c:.4} m/s; window impulse GPU {:?}", gset.shapes[599].impulse);
    assert!(c > 1.0, "the CPU's box was not struck ({c} m/s)");
    assert!((g - c).abs() < 0.01 * c, "the 600th proxy body: GPU {g} m/s, CPU {c} m/s");
    // The far boxes are untouched on both.
    assert!(gset.bodies[..599].iter().all(|b| b.vel == Vec3::ZERO));
}

/// Contact precision does not depend on where the vehicle is: the first test's crash
/// (the sedan into a free 1,200 kg box) at the origin and with the vehicle's origin at
/// (4096, 0, 2048) m, the box written in the vehicle's frame (proxy records are relative
/// to its origin) and its centre on a 2^-11 m grid (so the box sits at the same place at
/// 4 km too, a float spacing there being 2^-11 m). Every node word and the box's whole
/// state end bit for bit equal. (With contact tested in world coordinates, as before this
/// test, a node's world position at 4 km rounded to 0.49 mm.)
#[test]
fn far_from_the_origin_the_coupling_is_the_origins_bit_for_bit() {
    let Some((device, queue)) = device() else {
        eprintln!("skip: no GPU");
        return;
    };
    let (sim, mut set, _) = setup(1200.0);
    let grid = |v: Vec3| (v * 2048.0).round() / 2048.0;
    set.bodies[0].pos = grid(set.bodies[0].pos);
    set.begin_window();
    let run = |origin: Vec3| {
        let mut s = sim.clone();
        s.origin = origin;
        let mut gpu = ExplicitBeamsGpu::with_coupling(&device, &queue, std::slice::from_ref(&s), Some(CAP));
        write_proxies(&gpu, &queue, 0, &set).unwrap();
        gpu.set_coupling_on(true);
        let mut nodes = Vec::new();
        for _ in 0..48 {
            gpu.frame(STEPS);
            nodes.push(gpu.state_words().into_iter().find(|(n, _)| *n == "nodes").unwrap().1);
        }
        let mut gset = set.clone();
        read_proxies(&gpu, &device, &queue, 0, &mut gset);
        (nodes, gset)
    };
    let (a_nodes, a_set) = run(Vec3::ZERO);
    let (b_nodes, b_set) = run(Vec3::new(4096.0, 0.0, 2048.0));
    assert!(a_set.bodies[0].vel.z > 1.0, "the box was not struck");
    for (f, (a, b)) in a_nodes.iter().zip(&b_nodes).enumerate() {
        let differ = a.iter().zip(b).filter(|(x, y)| x != y).count();
        assert_eq!(differ, 0, "frame {f}: {differ} node words differ at 4 km");
    }
    let bits = |b: &RigidProxy| {
        [b.pos, b.vel, b.ang_vel].map(|v| v.to_array().map(f32::to_bits))
    };
    assert_eq!(bits(&a_set.bodies[0]), bits(&b_set.bodies[0]), "the box differs at 4 km");
    assert_eq!(
        a_set.bodies[0].rot.to_array().map(f32::to_bits),
        b_set.bodies[0].rot.to_array().map(f32::to_bits)
    );
    println!(
        "48 frames at the origin and at 4 km: every node word and the box (now {:.4} m/s) bit for bit",
        a_set.bodies[0].vel.z
    );
}

/// Built with coupling room but off: every frame bit for bit the plain build's.
#[test]
fn coupling_off_is_bit_for_bit() {
    let Some((device, queue)) = device() else {
        return;
    };
    let (sim, _, _) = setup(1200.0);
    let mut a = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&sim));
    let mut b = ExplicitBeamsGpu::with_coupling(&device, &queue, std::slice::from_ref(&sim), Some(CAP));
    for _ in 0..10 {
        a.frame(STEPS);
        b.frame(STEPS);
    }
    assert_eq!(a.positions(), b.positions());
    assert_eq!(a.velocities(), b.velocities());
}
