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
    let mut sim = WgpuSim::new_with_soft(
        capacity(),
        VehicleCapacity::default(),
        SoftCapacity::default(),
    )
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

#[derive(Clone, Debug, Default)]
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
    /// The cabin's fastest speed back off the wall after it stopped, m/s.
    pub rebound: f32,
    /// The least the cowl (the bay's first ring, top centre) stayed ahead of the roof's front
    /// edge (the cell's last ring, top centre), m; 0.2 at rest, below 0 when the bay slid in
    /// under the roof.
    pub cowl_gap: f32,
    pub broken: usize,
    pub yielded: usize,
    /// Parts torn off (`PieceDetached::parts`).
    pub detached: Vec<String>,
    /// The pulse: (time s, front crush m, cabin deceleration g), every 2 ms.
    pub pulse: Vec<(f32, f32, f32)>,
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
    let cabin_z = |q: &[Vec3]| {
        (seat..=firewall).map(|k| ring_centre(q, k).z).sum::<f32>() / (firewall - seat + 1) as f32
    };
    let (bay0, cell0) = (bay(&rest), cell(&rest));
    let mut out = Crumple {
        cowl_gap: f32::MAX,
        ..Default::default()
    };
    let (roof_front, cowl) = (firewall * CAR_RING + 7, (firewall + 1) * CAR_RING + 7);
    let dt = 1.0 / 240.0;
    let mut zs: Vec<f32> = vec![cabin_z(&rest)];
    let mut crush: Vec<f32> = Vec::new();
    for step in 0..(240.0 * 1.2) as usize {
        sim.step().expect("step");
        let q = positions(&sim, h);
        out.crush_peak = out.crush_peak.max(bay0 - bay(&q));
        crush.push(bay0 - bay(&q));
        out.cowl_gap = out.cowl_gap.min(q[cowl].z - q[roof_front].z);
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
    // 10 ms averages, so a node's jitter isn't counted as the car moving.
    out.rebound = v
        .windows(3)
        .map(|w| -(w.iter().sum::<f32>()) / 3.0)
        .skip((out.stop_s / dt) as usize)
        .fold(0.0, f32::max);
    for (i, w) in v.windows(3).enumerate() {
        let g = (w[0] - w[2]) / (2.0 * dt) / 9.81;
        out.peak_g = out.peak_g.max(g);
        if i < crush.len() && (i as f32 * dt) < 0.16 {
            out.pulse.push((i as f32 * dt, crush[i], g));
        }
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

/// The mean deceleration while the front crushes from 0.1 m to 0.4 m (the rails' plateau).
fn plateau(c: &Crumple) -> f32 {
    let mid: Vec<f32> = c
        .pulse
        .iter()
        .filter(|(_, x, _)| (0.1..0.4).contains(x))
        .map(|p| p.2)
        .collect();
    mid.iter().sum::<f32>() / mid.len().max(1) as f32
}

#[test]
fn crumple_probe() {
    let speed = 56.0 / 3.6;
    // RAIL_DEFORMS=d,… and CELL=deform_scale:strength_scale (the passenger cell's).
    let cell: Option<(f32, f32)> = std::env::var("CELL").ok().and_then(|v| {
        let (a, b) = v.split_once(':')?;
        Some((a.parse().ok()?, b.parse().ok()?))
    });
    let rails = |d: f32| {
        let mut p = CarParams {
            rail_deform: d,
            ..CarParams::sedan()
        };
        if let Some((a, b)) = cell {
            p.cell_deform_scale = a;
            p.cell_strength_scale = b;
        }
        p
    };
    let deforms: Vec<f32> = std::env::var("RAIL_DEFORMS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_default();
    let mut cars = vec![
        ("default".to_string(), CarParams::default()),
        ("sedan".to_string(), CarParams::sedan()),
    ];
    for d in deforms {
        cars.push((format!("rails {d}"), rails(d)));
    }
    // PLASTIC=c,…: the sedan with every beam's plastic coefficient c.
    for c in std::env::var("PLASTIC")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|x| x.parse::<f32>().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
    {
        cars.push((
            format!("plastic {c}"),
            CarParams {
                plastic_coef: c,
                ..CarParams::sedan()
            },
        ));
    }
    // SPRINGS=k,…: the sedan with every beam's spring rate k (N/m).
    for k in std::env::var("SPRINGS")
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|x| x.parse::<f32>().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
    {
        cars.push((
            format!("spring {k:.0e}"),
            CarParams {
                spring: k,
                ..CarParams::sedan()
            },
        ));
    }
    for (name, p) in cars {
        let c = crumple(&p, speed);
        eprintln!(
            "{name:>12}: crush peak {:.2} end {:.2} m, cabin {:+.3} m, bonnet rise {:+.3} m, peak {:.0} g, stop {:.3} s, rebound {:.2} m/s, cowl gap {:.3} m, yielded {} broken {}, off {:?}, plateau {:.0} g",
            c.crush_peak, c.crush_end, c.cabin, c.bonnet_rise, c.peak_g, c.stop_s, c.rebound, c.cowl_gap, c.yielded, c.broken, c.detached,
            plateau(&c)
        );
    }
}

/// The crash-test car (`CarParams::sedan`) in the 56 km/h full-width rigid-wall test: its
/// front folds about as far as a real small car's (0.45–0.7 m) at a steady resistance (its
/// rails), with no hard stop when the crush reaches the cell; its cabin keeps its length,
/// its bonnet folds up into a tent and stays on, its windscreen pillars hold, and it stops in
/// about a tenth of a second.
#[test]
fn the_sedan_crumples_its_front_and_keeps_its_cabin() {
    let c = crumple(&CarParams::sedan(), 56.0 / 3.6);
    let level = plateau(&c);
    eprintln!(
        "{:?}, plateau {level:.0} g",
        Crumple {
            pulse: Vec::new(),
            ..c.clone()
        }
    );
    assert!(
        (0.45..0.75).contains(&c.crush_peak),
        "front crush {}",
        c.crush_peak
    );
    assert!(c.cabin.abs() < 0.03, "cabin shortened {}", c.cabin);
    assert!(
        (0.05..0.3).contains(&c.bonnet_rise),
        "bonnet tent {}",
        c.bonnet_rise
    );
    assert!(
        !c.detached.iter().any(|p| p == "bonnet"),
        "the bonnet stays on: {:?}",
        c.detached
    );
    // The windscreen pillars hold the cowl ahead of the roof: the bay doesn't slide in under it.
    assert!(c.cowl_gap > 0.05, "cowl {} m ahead of the roof", c.cowl_gap);
    // A steady pulse: the rails hold ~20–35 g through the crush, and the peak stays near it.
    assert!((20.0..35.0).contains(&level), "plateau {level} g");
    assert!(
        c.peak_g < 1.5 * level && c.peak_g < 45.0,
        "peak {} g over a {level} g plateau",
        c.peak_g
    );
    assert!(
        (0.06..0.14).contains(&c.stop_s),
        "stopped after {} s",
        c.stop_s
    );
}

/// The sedan's crash pulse, printed: cabin deceleration against front crush (a crash
/// structure's force–displacement curve, force = mass × deceleration).
#[test]
fn sedan_pulse() {
    let c = crumple(&CarParams::sedan(), 56.0 / 3.6);
    for (t, x, g) in &c.pulse {
        eprintln!("t {:6.1} ms  crush {:5.2} m  {:6.1} g", t * 1000.0, x, g);
    }
}

/// Diagnostic: how the sedan's rail beams fared (yielded, shortened, broken).
#[test]
fn sedan_rails_report() {
    let p = CarParams::sedan();
    let (text, layout) = car_truck_layout(&p);
    let desc = load_truck(&text);
    let mut sim = world();
    let front = desc
        .nodes
        .iter()
        .map(|n| n.position[2])
        .fold(f32::MIN, f32::max);
    static_box(&mut sim, [0.0, 1.5, front + 0.4 + 0.5], [4.0, 1.5, 0.5]);
    let opts = SpawnOptions {
        velocity: [0.0, 0.0, 56.0 / 3.6],
        ..Default::default()
    };
    let h = sim
        .add_soft_vehicle(&desc, [0.0, 0.05, 0.0], [0.0, 0.0, 0.0, 1.0], &opts)
        .expect("spawn");
    for _ in 0..240 {
        sim.step().expect("step");
    }
    let rails: std::collections::HashSet<u32> = layout.rails.iter().copied().collect();
    let beams = sim.read_soft_beams(h);
    // Beams name global bodies; the car's particles start at the lowest.
    let base = beams.iter().map(|b| b.a.min(b.b)).min().unwrap_or(0);
    let (mut n, mut yielded, mut broken, mut short) = (0, 0, 0, 0.0f32);
    for b in &beams {
        if rails.contains(&(b.a - base)) && rails.contains(&(b.b - base)) {
            n += 1;
            if b.flags & engine_softbody::BEAM_YIELDED != 0 {
                yielded += 1;
            }
            if b.broken() {
                broken += 1;
            }
            short += (b.rest0 - b.rest).max(0.0);
        }
    }
    let q = positions(&sim, h);
    let r: Vec<Vec3> = layout.rails.iter().map(|&i| q[i as usize]).collect();
    eprintln!("rail beams {n}: yielded {yielded} broken {broken}, total shortening {short:.2} m; {} rail nodes, first ring z {:.2} last ring z {:.2}",
        r.len(), r[0].z, r[r.len() - 1].z);
}

/// Diagnostic: where the sedan's front springs back after its deepest crush. Each ring's
/// length to the next (along the car) at rest, at the deepest crush and at the end, and the
/// engine's, rails' and wheels' z.
#[test]
fn sedan_springback_report() {
    let p = CarParams::sedan();
    let (text, layout) = car_truck_layout(&p);
    let desc = load_truck(&text);
    let mut sim = world();
    let front = desc
        .nodes
        .iter()
        .map(|n| n.position[2])
        .fold(f32::MIN, f32::max);
    static_box(&mut sim, [0.0, 1.5, front + 0.4 + 0.5], [4.0, 1.5, 0.5]);
    let opts = SpawnOptions {
        velocity: [0.0, 0.0, 56.0 / 3.6],
        ..Default::default()
    };
    let h = sim
        .add_soft_vehicle(&desc, [0.0, 0.05, 0.0], [0.0, 0.0, 0.0, 1.0], &opts)
        .expect("spawn");
    let n = layout.stations;
    let rest = positions(&sim, h);
    let bay = |q: &[Vec3]| ring_centre(q, n - 1).z - ring_centre(q, layout.cabin_last).z;
    let mean = |q: &[Vec3], ids: &[u32]| {
        ids.iter().map(|&i| q[i as usize]).sum::<Vec3>() / ids.len().max(1) as f32
    };
    let (mut deepest, mut at) = (f32::MAX, rest.clone());
    let mut tracks = Vec::new();
    for step in 0..288 {
        sim.step().expect("step");
        let q = positions(&sim, h);
        if bay(&q) < deepest {
            deepest = bay(&q);
            at = q.clone();
        }
        if step % 6 == 0 {
            tracks.push((
                step,
                bay(&q),
                mean(&q, &layout.engine).z - ring_centre(&q, layout.cabin_last).z,
                mean(&q, &layout.rails).z - ring_centre(&q, layout.cabin_last).z,
                ring_centre(&q, layout.cabin_last).z,
            ));
        }
    }
    let end = positions(&sim, h);
    // Every beam on the cowl's top node (the bay's first ring, top centre).
    let beams = sim.read_soft_beams(h);
    // Beams name global bodies; the car's particles start at the lowest.
    let base = beams.iter().map(|b| b.a.min(b.b)).min().unwrap_or(0);
    let cowl = ((layout.cabin_last + 1) * CAR_RING + 7) as u32;
    for b in beams
        .iter()
        .filter(|b| b.a - base == cowl || b.b - base == cowl)
    {
        let o = (if b.a - base == cowl { b.b } else { b.a } - base) as usize;
        let len = |q: &[Vec3]| (q[o] - q[cowl as usize]).length();
        eprintln!("cowl beam to {o:4} (ring {:2} node {}) at {:?}: rest0 {:.3} rest {:.3} deepest {:.3} end {:.3} broken {} yielded {} spring {:.0e} deformC {:.0} deformT {:.0} strength {:.0} force {:.0} penalty {:.0e} kind {} short {} long {}",
            o / CAR_RING, o % CAR_RING, rest[o].to_array().map(|x| (x * 100.0).round() / 100.0), b.rest0, b.rest, len(&at), len(&end),
            b.broken(), b.flags & engine_softbody::BEAM_YIELDED != 0, b.spring, b.deform_c, b.deform_t, b.strength, b.force, b.penalty, b.kind, b.short_bound, b.long_bound);
    }
    eprintln!(
        "cabin {}..{} front bay {} stations {n}",
        layout.cabin_first, layout.cabin_last, layout.front_bay
    );
    for k in layout.cabin_first..n - 1 {
        let gap = |q: &[Vec3]| ring_centre(q, k + 1).z - ring_centre(q, k).z;
        if k + 1 == layout.cabin_last + 1 || gap(&at) < 0.1 {
            for j in 0..CAR_RING {
                let (a, b) = (k * CAR_RING + j, (k + 1) * CAR_RING + j);
                eprintln!(
                    "   node {j}: rest {:?} deepest d {:+.3?} -> {:+.3?}",
                    rest[a].to_array().map(|x| (x * 100.0).round() / 100.0),
                    (at[b] - at[a]).to_array(),
                    (end[b] - end[a]).to_array()
                );
            }
        }
        eprintln!(
            "ring {k:2}->{:2}: rest {:.3} deepest {:.3} end {:.3}  springback {:+.3}",
            k + 1,
            gap(&rest),
            gap(&at),
            gap(&end),
            gap(&end) - gap(&at)
        );
    }
    for (s, b, e, r, c) in tracks {
        eprintln!(
            "t {:5.1} ms bay {b:.3} engine {e:+.3} rails {r:+.3} firewall z {c:+.3}",
            s as f32 / 0.24
        );
    }
}
