//! How the procedural car crumples in a full-width rigid-wall crash (the NCAP / IIHS
//! frontal test at 56 km/h): how far the front folds (dynamically and at rest), how much
//! the firewall moves into the cabin, whether the bonnet tents up, and the cabin's
//! deceleration pulse. `crumple_probe` prints them for the default car and the
//! crash-structure car (`CarParams::crash_structure`).

use engine_schema::{BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_softbody::procedural::{car_truck_layout, CarLayout, CarParams, CAR_RING};
use engine_softbody::truck::load_truck;
use engine_softbody::{SoftCapacity, SoftHandle, SpawnOptions};
use engine_vehicle::VehicleCapacity;
use engine_webgpu::WgpuSim;
use glam::Vec3;

fn capacity() -> WorldCapacity {
    WorldCapacity {
        max_bodies: 4096,
        max_colliders: 8192,
        max_contacts: 65_536,
        max_joints: 512,
        max_candidates: 131_072,
        max_colors: 32,
        max_commands: 64,
        max_hull_vertices: 64,
    }
}

fn static_box(sim: &mut WgpuSim, pos: [f32; 3], half: [f32; 3]) {
    let body = sim.add_body(BodyState {
        position: pos,
        inv_mass: 0.0,
        orientation: [0.0, 0.0, 0.0, 1.0],
        inv_inertia: [0.0; 3],
        flags: 0,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: pos,
        initial_pad: 0.0,
        initial_rot: [0.0, 0.0, 0.0, 1.0],
    });
    sim.add_collider(Collider {
        body,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [
            half[0],
            half[1],
            half[2],
            (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt(),
        ],
    });
}

fn world() -> WgpuSim {
    let mut sim = WgpuSim::new_with_soft(capacity(), VehicleCapacity::default(), SoftCapacity::default())
        .expect("wgpu");
    sim.set_params(engine_softbody::soft_scene_params(240.0, 10));
    for m in [0, 9] {
        sim.set_material(
            m,
            Material {
                friction: if m == 0 { 1.0 } else { 0.8 },
                restitution: 0.0,
                contact_offset: 0.01,
                _pad: 0.0,
            },
        );
    }
    static_box(&mut sim, [0.0, -5.0, 0.0], [400.0, 5.0, 400.0]);
    sim
}

fn positions(sim: &WgpuSim, h: SoftHandle) -> Vec<Vec3> {
    sim.read_soft_particles(h)
        .iter()
        .map(|b| Vec3::from_array(b.position))
        .collect()
}

/// The mean of ring `k`'s eight nodes.
fn ring_centre(p: &[Vec3], k: usize) -> Vec3 {
    (0..CAR_RING).map(|j| p[k * CAR_RING + j]).sum::<Vec3>() / CAR_RING as f32
}

#[derive(Debug, Default)]
pub struct Crumple {
    /// Front bay (firewall ring to the front ring) shortening, m: the most during the
    /// crash, and at the end.
    pub crush_peak: f32,
    pub crush_end: f32,
    /// Cabin (rear seat ring to firewall ring) shortening at the end, m.
    pub cabin: f32,
    /// The most any bonnet panel node rose above its rest height (less the floor's lift), m.
    pub bonnet_rise: f32,
    /// The cabin's peak deceleration (5 ms average), g, and when it stopped (s).
    pub peak_g: f32,
    pub stop_s: f32,
    pub broken: usize,
    pub yielded: usize,
    /// Parts torn off (`PieceDetached::parts`).
    pub detached: Vec<String>,
}

fn crumple(p: &CarParams, speed: f32) -> Crumple {
    let (text, layout): (String, CarLayout) = car_truck_layout(p);
    let desc = load_truck(&text);
    let mut sim = world();
    // A rigid wall 0.4 m ahead of the bumper.
    let front = desc
        .nodes
        .iter()
        .map(|n| n.position[2])
        .fold(f32::MIN, f32::max);
    static_box(&mut sim, [0.0, 1.5, front + 0.4 + 0.5], [4.0, 1.5, 0.5]);
    let opts = SpawnOptions {
        velocity: [0.0, 0.0, speed],
        ..Default::default()
    };
    let h = sim
        .add_soft_vehicle(&desc, [0.0, 0.05, 0.0], [0.0, 0.0, 0.0, 1.0], &opts)
        .expect("spawn");
    let n = layout.stations;
    let firewall = layout.cabin_last;
    let seat = layout.cabin_first;
    let rest = positions(&sim, h);
    let bay = |q: &[Vec3]| ring_centre(q, n - 1).z - ring_centre(q, firewall).z;
    let cell = |q: &[Vec3]| ring_centre(q, firewall).z - ring_centre(q, seat).z;
    let bonnet = &layout.panels[0];
    // The most any bonnet node rose above its own rest height.
    let bonnet_y = |q: &[Vec3], r: &[Vec3]| {
        bonnet
            .iter()
            .map(|&i| q[i as usize].y - r[i as usize].y)
            .fold(f32::MIN, f32::max)
    };
    let cabin_z = |q: &[Vec3]| (seat..=firewall).map(|k| ring_centre(q, k).z).sum::<f32>() / (firewall - seat + 1) as f32;
    let (bay0, cell0) = (bay(&rest), cell(&rest));
    let mut out = Crumple::default();
    let dt = 1.0 / 240.0;
    let mut zs: Vec<f32> = vec![cabin_z(&rest)];
    for step in 0..(240.0 * 1.2) as usize {
        sim.step().expect("step");
        let q = positions(&sim, h);
        out.crush_peak = out.crush_peak.max(bay0 - bay(&q));
        // Tenting: the bonnet's top above where it was, less how much the floor moved up.
        let lift = (ring_centre(&q, firewall).y - ring_centre(&rest, firewall).y).max(0.0);
        out.bonnet_rise = out.bonnet_rise.max(bonnet_y(&q, &rest) - lift);
        zs.push(cabin_z(&q));
        for piece in sim.poll_soft_events().pieces {
            out.detached.extend(piece.parts);
        }
        let v = (zs[zs.len() - 1] - zs[zs.len() - 2]) / dt;
        if out.stop_s == 0.0 && v <= 0.0 && step > 2 {
            out.stop_s = step as f32 * dt;
        }
    }
    // Velocity, then its 5 ms (≈ CFC 60) average slope.
    let v: Vec<f32> = zs.windows(2).map(|w| (w[1] - w[0]) / dt).collect();
    for w in v.windows(3) {
        out.peak_g = out.peak_g.max((w[0] - w[2]) / (2.0 * dt) / 9.81);
    }
    let end = positions(&sim, h);
    assert!(end.iter().all(|q| q.is_finite()), "finite");
    out.crush_end = bay0 - bay(&end);
    out.cabin = cell0 - cell(&end);
    let beams = sim.read_soft_beams(h);
    out.broken = beams.iter().filter(|b| b.broken()).count();
    out.yielded = beams
        .iter()
        .filter(|b| b.flags & engine_softbody::BEAM_YIELDED != 0)
        .count();
    out
}

#[test]
fn crumple_probe() {
    let speed = 56.0 / 3.6;
    let stations = |k: usize| CarParams {
        stations: k,
        ..CarParams::default()
    };
    for (name, p) in [
        ("default", CarParams::default()),
        ("15 rings", stations(15)),
        ("21 rings", stations(21)),
        ("sedan", CarParams::sedan()),
    ] {
        let c = crumple(&p, speed);
        eprintln!(
            "{name:>8}: crush peak {:.2} end {:.2} m, cabin {:+.3} m, bonnet rise {:+.3} m, peak {:.0} g, stop {:.3} s, yielded {} broken {}, off {:?}",
            c.crush_peak, c.crush_end, c.cabin, c.bonnet_rise, c.peak_g, c.stop_s, c.yielded, c.broken, c.detached
        );
    }
}

/// The crash-test car (`CarParams::sedan`) in the 56 km/h full-width rigid-wall test: its
/// front folds about as far as a real small car's (0.5–0.7 m), its cabin keeps its length,
/// its bonnet folds up into a tent and stays on, and it stops in about a tenth of a second.
#[test]
fn the_sedan_crumples_its_front_and_keeps_its_cabin() {
    let c = crumple(&CarParams::sedan(), 56.0 / 3.6);
    eprintln!("{c:?}");
    assert!((0.5..0.75).contains(&c.crush_peak), "front crush {}", c.crush_peak);
    assert!(c.cabin.abs() < 0.03, "cabin shortened {}", c.cabin);
    assert!((0.05..0.3).contains(&c.bonnet_rise), "bonnet tent {}", c.bonnet_rise);
    assert!(!c.detached.iter().any(|p| p == "bonnet"), "the bonnet stays on: {:?}", c.detached);
    assert!((25.0..80.0).contains(&c.peak_g), "peak deceleration {} g", c.peak_g);
    assert!((0.06..0.14).contains(&c.stop_s), "stopped after {} s", c.stop_s);
}
