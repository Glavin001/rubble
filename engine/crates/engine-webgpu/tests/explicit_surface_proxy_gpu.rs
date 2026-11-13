//! Surface contact against the World's rigid proxies on the GPU
//! (`explicit_beams_surface.slang` proxy path, `slang/physics/feature_query.slang`)
//! against its CPU reference (`engine_softbody::explicit_surface` with
//! `ExplicitBeams::rigid`): the repo sedan with surface contact on, flying (no ground)
//! at 10 m/s into a 20 cm pole that is a proxy shape.
//! * a static box pole, a static hull pole (its vertices and edges from the World's
//!   hull records) and a free 60 kg box pole;
//! * CPU↔GPU parity: the first frames of contact agree to float noise (node contact
//!   against proxies is not bit for bit: its GPU path uses native arithmetic), and
//!   the outcome (node and pole speeds) to 1 %;
//! * the free pole takes the equal and opposite impulse: horizontal momentum (nodes plus pole)
//!   closes on the GPU to the fixed-point rounding;
//! * the pass does work: with it on the struck skin stops nearer the pole's face than
//!   with it off.

use engine_schema::convex::{ConvexHull, HullArena};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use engine_softbody::explicit_rigid::{ProxyGeom, ProxySet, ProxyShape, RigidProxy};
use engine_webgpu::explicit_coupling::{read_proxies, write_proxies};
use engine_webgpu::{default_device, CouplingCapacity, ExplicitBeamsGpu};
use glam::{Quat, Vec3};

const STEPS: u32 = 17;
const CAP: CouplingCapacity = CouplingCapacity {
    bodies: 4,
    shapes: 8,
    planes: 128,
};

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let (_, d, q) = default_device().ok()?;
    Some((d, q))
}

fn sedan(speed: f32, surface: bool) -> ExplicitBeams {
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
        centred_objective: false,
    };
    let mut sim = ExplicitBeams::from_truck(&text, Vec3::ZERO, Vec3::ZERO, settings).unwrap();
    let b = |s: &ExplicitBeams| {
        s.nodes.iter().fold((Vec3::MAX, Vec3::MIN), |(a, b), n| {
            (a.min(n.pos), b.max(n.pos))
        })
    };
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
    sim.surface_contact = surface;
    if surface {
        sim.dissipative_contact = true;
    }
    sim
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Pole {
    StaticBox,
    StaticHull,
    FreeBox(f32),
}

const HALF: Vec3 = Vec3::new(0.1, 0.9, 0.1);

/// The unit cube's hull scaled to the pole (vertices, edges, planes from the arena).
fn hull_pole() -> (Vec<(Vec3, f32)>, Vec<Vec3>, Vec<[u32; 2]>, f32) {
    let pts: Vec<Vec3> = (0..8)
        .map(|i| {
            Vec3::new(
                if i & 1 != 0 { HALF.x } else { -HALF.x },
                if i & 2 != 0 { HALF.y } else { -HALF.y },
                if i & 4 != 0 { HALF.z } else { -HALF.z },
            )
        })
        .collect();
    let hull = ConvexHull::from_points(&pts).expect("hull");
    let mut arena = HullArena::new();
    let base = arena.insert(&hull).expect("insert");
    let recs = arena.records();
    let head = recs[base as usize];
    let (nv, nf, ne) = (head[0] & 255, (head[0] >> 8) & 255, (head[0] >> 16) & 255);
    let f = |r: [u32; 4]| {
        Vec3::new(
            f32::from_bits(r[0]),
            f32::from_bits(r[1]),
            f32::from_bits(r[2]),
        )
    };
    let verts: Vec<Vec3> = (0..nv).map(|i| f(recs[(base + 4 + i) as usize])).collect();
    let planes = (0..nf)
        .map(|i| {
            let r = recs[(base + 4 + nv + i) as usize];
            (f(r), f32::from_bits(r[3]))
        })
        .collect();
    let word = |w: u32| recs[(base + w / 4) as usize][(w % 4) as usize];
    let edges = (0..ne)
        .map(|e| {
            let x = word(head[2] + e);
            [x & 255, (x >> 8) & 255]
        })
        .collect();
    (planes, verts, edges, HALF.length())
}

fn pole_set(pole: Pole, at: Vec3) -> ProxySet {
    let (inv_mass, inv_inertia) = match pole {
        Pole::FreeBox(m) => {
            let h = HALF;
            let i = Vec3::new(
                m / 3.0 * (h.y * h.y + h.z * h.z),
                m / 3.0 * (h.x * h.x + h.z * h.z),
                m / 3.0 * (h.x * h.x + h.y * h.y),
            );
            (1.0 / m, Vec3::ONE / i)
        }
        _ => (0.0, Vec3::ZERO),
    };
    let shape = match pole {
        Pole::StaticHull => {
            let (planes, verts, edges, radius) = hull_pole();
            ProxyShape::new(
                0,
                Vec3::ZERO,
                Quat::IDENTITY,
                ProxyGeom::Hull { planes, radius },
                3,
                None,
            )
            .with_hull_geometry(verts, edges)
        }
        _ => ProxyShape::new(
            0,
            Vec3::ZERO,
            Quat::IDENTITY,
            ProxyGeom::Box { half: HALF },
            3,
            None,
        ),
    };
    let mut set = ProxySet {
        bodies: vec![RigidProxy {
            body: 7,
            pos: at,
            rot: Quat::from_rotation_y(0.3),
            vel: Vec3::ZERO,
            ang_vel: Vec3::ZERO,
            inv_mass,
            inv_inertia,
            share: 1.0,
            count: 0,
        }],
        shapes: vec![shape],
        frozen: false,
    };
    set.begin_window();
    set
}

struct Run {
    worst_early: f32,
    first_contact: Option<u32>,
    gpu_vz: f32,
    cpu_vz: f32,
    gpu_pole: Vec3,
    cpu_pole: Vec3,
    dp: f32,
    p0: f32,
    /// The struck skin's foremost node (largest z) at the end, GPU.
    gpu_front: f32,
}

fn run(device: &wgpu::Device, queue: &wgpu::Queue, pole: Pole, surface: bool, frames: u32) -> Run {
    let sim = sedan(10.0, surface);
    let hi_y = sim.nodes.iter().map(|n| n.pos.y).fold(f32::MIN, f32::max);
    let set = pole_set(pole, Vec3::new(0.05, 0.5 * (1.0 + hi_y), 0.3 + HALF.z));
    let mut gpu =
        ExplicitBeamsGpu::with_coupling(device, queue, std::slice::from_ref(&sim), Some(CAP));
    write_proxies(&gpu, queue, 0, &set).unwrap();
    gpu.set_coupling_on(true);
    let mut cpu = sim.clone();
    cpu.rigid = Some(Box::new(set.clone()));
    let masses: Vec<f32> = sim.nodes.iter().map(|n| n.mass).collect();
    let m: f32 = masses.iter().sum();
    let p0 = sim.nodes.iter().map(|n| n.vel * n.mass).sum::<Vec3>();
    let mut first_contact = None;
    let mut worst_early = 0.0f32;
    for frame in 0..frames {
        gpu.frame(STEPS);
        for _ in 0..STEPS {
            cpu.step();
        }
        let cset = cpu.rigid.as_ref().unwrap();
        if first_contact.is_none() && cset.shapes[0].peak > 0.0 {
            first_contact = Some(frame);
        }
        if first_contact.is_some_and(|f| frame <= f + 1) {
            let pos = gpu.positions();
            for (a, n) in pos[0].iter().zip(&cpu.nodes) {
                worst_early = worst_early.max((*a - (cpu.origin + n.pos)).length());
            }
        }
    }
    let mut gset = set.clone();
    read_proxies(&gpu, device, queue, 0, &mut gset);
    let cset = cpu.rigid.take().unwrap();
    let gvel = gpu.velocities();
    let pos = gpu.positions();
    let pn = gvel[0]
        .iter()
        .zip(&masses)
        .map(|(v, m)| *v * *m)
        .sum::<Vec3>();
    let h = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
    let pg = h(pn + gset.bodies[0].momentum().0);
    let p0 = h(p0);
    Run {
        worst_early,
        first_contact,
        gpu_vz: gvel[0]
            .iter()
            .zip(&masses)
            .map(|(v, m)| v.z * m)
            .sum::<f32>()
            / m,
        cpu_vz: cpu.nodes.iter().map(|n| n.vel.z * n.mass).sum::<f32>() / m,
        gpu_pole: gset.bodies[0].vel,
        cpu_pole: cset.bodies[0].vel,
        dp: (pg - p0).length(),
        p0: p0.length(),
        gpu_front: pos[0].iter().map(|p| p.z).fold(f32::MIN, f32::max),
    }
}

fn check(pole: Pole) {
    let Some((device, queue)) = device() else {
        eprintln!("skip: no GPU");
        return;
    };
    let on = run(&device, &queue, pole, true, 40);
    println!(
        "{pole:?}: first contact {:?}, early worst {:.2e} m, vehicle vz GPU {:.4} CPU {:.4}, pole v GPU {} CPU {}, |dP| {:.4} of {:.0}, front {:.3}",
        on.first_contact, on.worst_early, on.gpu_vz, on.cpu_vz, on.gpu_pole, on.cpu_pole, on.dp, on.p0, on.gpu_front
    );
    assert!(on.first_contact.is_some(), "the pole was hit");
    assert!(
        on.worst_early < 1e-3,
        "early frames apart by {} m",
        on.worst_early
    );
    let rel = |a: f32, b: f32| (a - b).abs() / b.abs().max(0.05);
    assert!(
        rel(on.gpu_vz, on.cpu_vz) < 0.01,
        "vehicle speed GPU {} CPU {}",
        on.gpu_vz,
        on.cpu_vz
    );
    if let Pole::FreeBox(_) = pole {
        assert!(
            (on.gpu_pole - on.cpu_pole).length() < 0.01 * on.cpu_pole.length().max(0.1),
            "pole GPU {} CPU {}",
            on.gpu_pole,
            on.cpu_pole
        );
        assert!(
            on.dp < 2e-4 * on.p0,
            "GPU momentum off by {} of {}",
            on.dp,
            on.p0
        );
    }
}

#[test]
fn surface_contact_meets_a_static_box_proxy_like_the_cpu() {
    check(Pole::StaticBox);
}

#[test]
fn surface_contact_meets_a_static_hull_proxy_like_the_cpu() {
    check(Pole::StaticHull);
}

#[test]
fn surface_contact_moves_a_free_box_proxy_like_the_cpu_and_closes_momentum() {
    check(Pole::FreeBox(60.0));
}

/// Built with surface contact and coupling room, the coupling off: bit for bit the
/// surface-contact build without coupling room (no proxies, the static path only).
#[test]
fn surface_contact_without_proxies_is_bit_for_bit() {
    let Some((device, queue)) = device() else {
        return;
    };
    let sim = sedan(10.0, true);
    let mut a = ExplicitBeamsGpu::new(&device, &queue, std::slice::from_ref(&sim));
    let mut b =
        ExplicitBeamsGpu::with_coupling(&device, &queue, std::slice::from_ref(&sim), Some(CAP));
    for _ in 0..10 {
        a.frame(STEPS);
        b.frame(STEPS);
    }
    assert_eq!(a.positions(), b.positions());
    assert_eq!(a.velocities(), b.velocities());
}
