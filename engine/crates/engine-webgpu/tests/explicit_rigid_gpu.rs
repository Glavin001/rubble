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
