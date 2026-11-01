//! Soft-body calibration harness (design: "Calibration and validation") on the
//! native wgpu host. Runs the standard tests on a procedural car with the
//! explicit oracle (`engine_softbody::explicit_ref`: f64 symplectic Euler at
//! ≥ 2 kHz, the Rigs of Rods / BeamNG scheme) as the reference, and at candidate
//! GPU settings, through the same scenes and the same metrics
//! (`engine_softbody::metrics`). Every vehicle spawns without a rigid core (the
//! oracle has none), standing on its node spheres.
//!
//! ```bash
//! cargo run -p engine-webgpu --release --bin soft_calibrate
//! # a custom sweep: step rate × iterations, Chebyshev ρ, particle sweeps per iteration
//! SETTINGS=120x10,240x10,960x30 CHEB=0 SWEEPS=2 cargo run -p engine-webgpu --release --bin soft_calibrate
//! # candidates with BDF2 particles; the default hatchback instead of the sedan
//! SETTINGS=240x10 BDF2=1 CAR=default cargo run -p engine-webgpu --release --bin soft_calibrate
//! # fit the global threshold scale per step rate to the oracle's 56 km/h crush
//! FIT=1 cargo run -p engine-webgpu --release --bin soft_calibrate
//! # crash islands: 2 substeps per step for bodies in contact
//! SUBSTEPS=2 SETTINGS=240x10 cargo run -p engine-webgpu --release --bin soft_calibrate
//! # the GPU scenes 10 km from the origin (f32 precision)
//! OFFSET=10000 SETTINGS=240x10 cargo run -p engine-webgpu --release --bin soft_calibrate
//! ```
use engine_schema::{BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_softbody::calibrate::*;
use engine_softbody::explicit_ref::{ExplicitSettings, ExplicitSim, Shape};
use engine_softbody::hires::{hires_car_desc, named, ring_nodes, HiresParams};
use engine_softbody::metrics::*;
use engine_softbody::procedural::{car_desc, CarParams, CAR_RING};
use engine_softbody::{
    soft_scene_params, SoftCapacity, SoftHandle, SoftSettings, SoftVehicleDesc, SpawnOptions,
};
use engine_vehicle::VehicleCapacity;
use engine_webgpu::WgpuSim;
use glam::{DQuat, DVec3, Quat, Vec3};

/// Friction the GPU uses between nodes (0.8) and the ground and walls (1.0): the
/// geometric mean.
const MU: f64 = 0.894;
/// Metrics sample rate (Hz).
const SAMPLE_HZ: f32 = 240.0;

#[derive(Clone, Copy)]
struct Setting {
    hz: f32,
    iterations: u32,
    threshold_scale: f32,
}

/// A static box: centre and half extents.
#[derive(Clone, Copy)]
struct Wall {
    center: Vec3,
    half: Vec3,
}

/// One scene running on the GPU or on the oracle.
trait Runner {
    /// Advances one sample period.
    fn frame(&mut self);
    fn time(&self) -> f64;
    fn positions(&self) -> Vec<Vec3>;
    fn velocities(&self) -> Vec<Vec3>;
    fn masses(&self) -> Vec<f32>;
    /// Broken flag per beam (desc order).
    fn broken(&self) -> Vec<bool>;
    fn yielded(&self) -> usize;
    fn kick(&mut self, node: u32, velocity: Vec3);
}

struct Gpu {
    sim: WgpuSim,
    h: SoftHandle,
    /// The whole scene sits here (`OFFSET=d` puts it at (d, 0, d)); positions are
    /// reported relative to it.
    offset: Vec3,
    steps: u32,
    dt: f64,
    t: f64,
}

fn env_f32(name: &str) -> Option<f32> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

impl Gpu {
    fn new(s: Setting, desc: &SoftVehicleDesc, at: Vec3, vel: Vec3, walls: &[Wall]) -> Self {
        let d = env_f32("OFFSET").unwrap_or(0.0);
        let offset = Vec3::new(d, 0.0, d);
        let cap = WorldCapacity {
            max_bodies: 4096,
            max_colliders: 8192,
            max_contacts: 65_536,
            max_joints: 512,
            max_candidates: 131_072,
            max_colors: 32,
            max_commands: 64,
            max_hull_vertices: 64,
        };
        let mut sim =
            WgpuSim::new_with_soft(cap, VehicleCapacity::default(), SoftCapacity::default())
                .expect("wgpu");
        let mut p = soft_scene_params(s.hz, s.iterations);
        p.sleep_time = 1.0e6;
        sim.set_params(p);
        let d = SoftSettings::default();
        sim.set_soft_settings(SoftSettings {
            threshold_scale: s.threshold_scale,
            chebyshev_rho: env_f32("CHEB").unwrap_or(d.chebyshev_rho),
            sweeps: env_f32("SWEEPS").map_or(d.sweeps, |v| v as u32),
            plasticity: std::env::var("PLASTIC").as_deref() != Ok("0"),
            bdf2: std::env::var("BDF2").as_deref() == Ok("1"),
            crash_substeps: env_f32("SUBSTEPS").map_or(d.crash_substeps, |v| v as u32),
            ..d
        });
        for (i, mu) in [(0, 1.0), (9, 0.8)] {
            sim.set_material(
                i,
                Material {
                    friction: mu,
                    restitution: 0.0,
                    contact_offset: 0.01,
                    _pad: 0.0,
                },
            );
        }
        let ground = Wall {
            center: Vec3::new(0.0, -5.0, 0.0),
            half: Vec3::new(400.0, 5.0, 400.0),
        };
        for w in std::iter::once(&ground).chain(walls) {
            let b = sim.add_body(BodyState::static_plane_support());
            sim.add_collider(Collider {
                body: b,
                shape: ShapeType::Box as u32,
                material: 0,
                generation: 0,
                local_pos: (w.center + offset).to_array(),
                density: 0.0,
                local_rot: [0.0, 0.0, 0.0, 1.0],
                params: [w.half.x, w.half.y, w.half.z, w.half.length()],
            });
        }
        let h = sim
            .add_soft_vehicle(
                desc,
                (at + offset).to_array(),
                [0.0, 0.0, 0.0, 1.0],
                &SpawnOptions {
                    rigid_core: false,
                    velocity: vel.to_array(),
                    ..Default::default()
                },
            )
            .unwrap();
        let steps = ((s.hz / SAMPLE_HZ).round() as u32).max(1);
        Self {
            sim,
            h,
            offset,
            steps,
            dt: 1.0 / f64::from(s.hz),
            t: 0.0,
        }
    }
}

impl Runner for Gpu {
    fn frame(&mut self) {
        for _ in 0..self.steps {
            self.sim.step().expect("step");
            self.t += self.dt;
        }
    }
    fn time(&self) -> f64 {
        self.t
    }
    fn positions(&self) -> Vec<Vec3> {
        self.sim
            .read_soft_particles(self.h)
            .iter()
            .map(|b| Vec3::from_array(b.position) - self.offset)
            .collect()
    }
    fn velocities(&self) -> Vec<Vec3> {
        self.sim
            .read_soft_particles(self.h)
            .iter()
            .map(|b| Vec3::from_array(b.linear_vel))
            .collect()
    }
    fn masses(&self) -> Vec<f32> {
        self.sim
            .read_soft_particles(self.h)
            .iter()
            .map(|b| {
                if b.inv_mass > 0.0 {
                    1.0 / b.inv_mass
                } else {
                    0.0
                }
            })
            .collect()
    }
    fn broken(&self) -> Vec<bool> {
        self.sim
            .read_soft_beams(self.h)
            .iter()
            .map(|b| b.broken())
            .collect()
    }
    fn yielded(&self) -> usize {
        self.sim
            .read_soft_beams(self.h)
            .iter()
            .filter(|b| b.flags & engine_softbody::BEAM_YIELDED != 0)
            .count()
    }
    fn kick(&mut self, node: u32, velocity: Vec3) {
        let i = self.sim.soft_host().entry(self.h).first_body + node;
        let mut b = self.sim.read_body(i);
        b.linear_vel = velocity.to_array();
        engine_vehicle::VehicleBackend::write_body(&mut self.sim, i, &b);
    }
}

struct Oracle {
    sim: ExplicitSim,
    per: u32,
}

impl Oracle {
    fn new(desc: &SoftVehicleDesc, at: Vec3, vel: Vec3, walls: &[Wall]) -> Self {
        let mut sim = ExplicitSim::new(ExplicitSettings::default());
        sim.add_vehicle(desc, at, Quat::IDENTITY, vel);
        sim.add_obstacle(
            Shape::Plane {
                normal: DVec3::Y,
                offset: 0.0,
            },
            MU,
        );
        for w in walls {
            sim.add_obstacle(
                Shape::Box {
                    center: w.center.as_dvec3(),
                    half: w.half.as_dvec3(),
                    rotation: DQuat::IDENTITY,
                },
                MU,
            );
        }
        let per = (sim.hz() / f64::from(SAMPLE_HZ)).round() as u32;
        Self { sim, per }
    }
}

impl Runner for Oracle {
    fn frame(&mut self) {
        for _ in 0..self.per {
            self.sim.substep();
        }
    }
    fn time(&self) -> f64 {
        self.sim.time
    }
    fn positions(&self) -> Vec<Vec3> {
        self.sim.positions()
    }
    fn velocities(&self) -> Vec<Vec3> {
        self.sim.body_velocities(0)
    }
    fn masses(&self) -> Vec<f32> {
        self.sim.mass.iter().map(|&m| m as f32).collect()
    }
    fn broken(&self) -> Vec<bool> {
        self.sim.beams.iter().map(|b| b.broken()).collect()
    }
    fn yielded(&self) -> usize {
        self.sim.yielded_beams(0)
    }
    fn kick(&mut self, node: u32, velocity: Vec3) {
        self.sim.v[node as usize] = velocity.as_dvec3();
    }
}

/// The car under test and the node sets the metrics read.
struct Car {
    name: String,
    desc: SoftVehicleDesc,
    /// Tap test: the kicked roof node and the floor node below it.
    tap: (u32, u32),
    /// Standing height (lowest node sphere on the ground).
    lift: f32,
    body: Vec<u32>,
    cabin: Vec<u32>,
    rear: Vec<u32>,
    /// The firewall ring's nodes (intrusion probes).
    firewall: Vec<u32>,
    engine: Vec<u32>,
}

impl Car {
    /// `CAR=default` (the hatchback), `CAR=hires` (the ~1,000-node JBeam car), or
    /// the sedan.
    fn from_env() -> Self {
        match std::env::var("CAR").as_deref() {
            Ok("hires") => Self::hires(&HiresParams::default()),
            Ok("default") => Self::procedural(&CarParams::default()),
            _ => Self::procedural(&CarParams::sedan()),
        }
    }

    fn procedural(p: &CarParams) -> Self {
        let (desc, layout) = car_desc(p);
        let shell = (layout.stations * CAR_RING) as u32;
        let body: Vec<u32> = desc.part("body").unwrap().nodes.clone();
        let ring_of = |i: u32| i as usize / CAR_RING;
        let cabin = (0..shell).filter(|&i| layout.cabin[ring_of(i)]).collect();
        let rear = (0..shell)
            .filter(|&i| ring_of(i) <= layout.cabin_first)
            .collect();
        let firewall = (0..shell)
            .filter(|&i| ring_of(i) == layout.cabin_last)
            .collect();
        let engine = desc
            .part("engine")
            .map(|p| p.nodes.clone())
            .unwrap_or_default();
        let front = ((layout.stations - 1) * CAR_RING) as u32;
        Self {
            name: p.name.clone(),
            tap: (front + 4, front),
            lift: 0.06 - desc.bounds().0[1],
            desc,
            body,
            cabin,
            rear,
            firewall,
            engine,
        }
    }

    fn hires(p: &HiresParams) -> Self {
        let (desc, layout) = hires_car_desc(p);
        let rings = |r: std::ops::RangeInclusive<usize>| -> Vec<u32> {
            r.flat_map(|k| ring_nodes(&desc, k)).collect()
        };
        let front = ring_nodes(&desc, layout.stations - 1);
        Self {
            name: p.name.clone(),
            tap: (front[6], front[1]),
            lift: 0.06 - desc.bounds().0[1],
            body: named(&desc, "s"),
            cabin: rings(layout.cabin_first..=layout.cabin_last),
            rear: rings(0..=layout.cabin_first),
            firewall: ring_nodes(&desc, layout.cabin_last),
            engine: named(&desc, "eng"),
            desc,
        }
    }

    fn rest(&self, at: Vec3) -> Vec<Vec3> {
        self.desc
            .nodes
            .iter()
            .map(|n| Vec3::from_array(n.position) + at)
            .collect()
    }
}

#[derive(Default, Debug, Clone)]
struct Crash {
    crush: (f32, f32),
    profile: Vec<f32>,
    broken: Vec<u32>,
    absorbed: f32,
    penetrations: PenetrationCount,
    intrusion: f32,
    engine_back: f32,
    peak_g: f64,
    stop: Option<f64>,
    rebound: f32,
}

#[derive(Default, Debug, Clone)]
struct Metrics {
    sag_rms: f32,
    tap_hz: f32,
    drop_yield: usize,
    crush30: (f32, f32),
    crash56: Crash,
    crush100: (f32, f32),
}

/// Rest 3 s, then a 1 m/s downward kick on the front-left roof node and the
/// dominant frequency of the roof-to-floor length (a structural mode).
fn rest_and_tap(car: &Car, r: &mut dyn Runner, at: Vec3) -> (f32, f32) {
    let rest = car.rest(at);
    for _ in 0..(3.0 * SAMPLE_HZ) as u32 {
        r.frame();
    }
    let sag = rms_max(&residual_displacement(&rest, &r.positions())).0;
    let (node, floor) = car.tap;
    r.kick(node, Vec3::new(0.0, -1.0, 0.0));
    let mut sig = Vec::new();
    for _ in 0..SAMPLE_HZ as u32 {
        r.frame();
        let x = r.positions();
        sig.push((x[node as usize] - x[floor as usize]).length());
    }
    (
        sag,
        dominant_frequency(&sig, 1.0 / SAMPLE_HZ, 0.5 * SAMPLE_HZ),
    )
}

fn drop_test(r: &mut dyn Runner) -> usize {
    for _ in 0..(2.0 * SAMPLE_HZ) as u32 {
        r.frame();
    }
    r.yielded()
}

const WALL: Wall = Wall {
    center: Vec3::new(0.0, 1.5, 3.0),
    half: Vec3::new(3.0, 1.5, 0.2),
};

fn wall_crash(car: &Car, r: &mut dyn Runner, at: Vec3, v: f32) -> Crash {
    let rest = car.rest(at);
    let mass = r.masses();
    let m: f32 = mass.iter().sum();
    let ke0 = 0.5 * m * v * v;
    let mut tr = PenetrationTracker::new(vec![TrackedBody::from_desc(&car.desc, 0)]);
    let mut decel = DecelTracker::default();
    let mut rebound: f32 = 0.0;
    let mut impact = None;
    for _ in 0..(1.5 * SAMPLE_HZ) as u32 {
        r.frame();
        let x = r.positions();
        tr.update(&x, Some(&[r.broken()]));
        let vc = centroid_velocity(&r.velocities(), &mass, &car.cabin);
        if impact.is_none() && vc.z < 0.97 * v {
            impact = Some(r.time());
        }
        rebound = rebound.min(vc.z);
        decel.push(r.time(), vc);
    }
    let now = r.positions();
    let ke1: f32 = r
        .velocities()
        .iter()
        .zip(&mass)
        .map(|(v, m)| 0.5 * m * v.length_squared())
        .sum();
    let body_rest: Vec<Vec3> = car.body.iter().map(|&i| rest[i as usize]).collect();
    let body_now: Vec<Vec3> = car.body.iter().map(|&i| now[i as usize]).collect();
    Crash {
        crush: rms_max(&residual_displacement(&body_rest, &body_now)),
        profile: crush_profile(&rest, &now, &car.body, &car.rear, Vec3::Z, Vec3::X, 5)
            .into_iter()
            .filter(|c| !c.is_nan())
            .collect(),
        broken: r
            .broken()
            .iter()
            .enumerate()
            .filter(|(_, b)| **b)
            .map(|(i, _)| i as u32)
            .collect(),
        absorbed: (ke0 - ke1) / ke0,
        penetrations: tr.max,
        intrusion: intrusion(&rest, &now, &car.cabin, &car.firewall),
        engine_back: if car.engine.is_empty() {
            0.0
        } else {
            block_travel(&rest, &now, &car.cabin, &car.engine, Vec3::Z).1
        },
        peak_g: decel.peak_decel_g(0.01),
        stop: impact.and_then(|t0| decel.time_to_stop(t0, 0.5)),
        rebound: -rebound,
    }
}

/// Builds a runner: the oracle (`None`) or the GPU at a setting.
fn runner(s: Option<Setting>, car: &Car, at: Vec3, vel: Vec3, walls: &[Wall]) -> Box<dyn Runner> {
    match s {
        Some(s) => Box::new(Gpu::new(s, &car.desc, at, vel, walls)),
        None => Box::new(Oracle::new(&car.desc, at, vel, walls)),
    }
}

fn metrics(car: &Car, s: Option<Setting>) -> Metrics {
    let still = Vec3::new(0.0, car.lift, 0.0);
    let (sag_rms, tap_hz) = rest_and_tap(car, &mut *runner(s, car, still, Vec3::ZERO, &[]), still);
    let high = Vec3::new(0.0, car.lift + 1.0, 0.0);
    let drop_yield = drop_test(&mut *runner(s, car, high, Vec3::ZERO, &[]));
    let start = Vec3::new(0.0, car.lift, -1.0);
    let crash = |kmh: f32| {
        let v = kmh / 3.6;
        wall_crash(
            car,
            &mut *runner(s, car, start, Vec3::new(0.0, 0.0, v), &[WALL]),
            start,
            v,
        )
    };
    Metrics {
        sag_rms,
        tap_hz,
        drop_yield,
        crush30: crash(30.0).crush,
        crash56: crash(56.0),
        crush100: crash(100.0).crush,
    }
}

fn cm(v: &[f32]) -> String {
    v.iter()
        .map(|c| format!("{:.0}", c * 100.0))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let car = Car::from_env();
    eprintln!(
        "{}: {} nodes, {} beams, {} triangles",
        car.name,
        car.desc.nodes.len(),
        car.desc.beams.len(),
        car.desc.triangles.len()
    );
    let t = std::time::Instant::now();
    let r = metrics(&car, None);
    eprintln!("oracle done in {:.1} s", t.elapsed().as_secs_f32());
    println!("| setting | sag RMS (mm) | tap (Hz) | 1 m drop: bent | crush 30 / 56 / 100 km/h max (cm) | 56 km/h: profile (cm) | Jaccard | KE absorbed | rebound (m/s) | peak (g) | stop (s) | intrusion (cm) | engine back (cm) | penetrations self/ext |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    let row = |name: &str, m: &Metrics| {
        let c = &m.crash56;
        println!(
            "| {name} | {:.2} | {:.1} | {} | {:.1} / {:.1} / {:.1} | {} | {:.2} | {:.0}% | {:.1} | {:.0} | {} | {:.1} | {:.1} | {}/{} |",
            m.sag_rms * 1e3,
            m.tap_hz,
            m.drop_yield,
            m.crush30.1 * 100.0,
            c.crush.1 * 100.0,
            m.crush100.1 * 100.0,
            cm(&c.profile),
            jaccard(&c.broken, &r.crash56.broken),
            c.absorbed * 100.0,
            c.rebound,
            c.peak_g,
            c.stop.map_or("—".into(), |s| format!("{s:.2}")),
            c.intrusion * 100.0,
            c.engine_back * 100.0,
            c.penetrations.self_,
            c.penetrations.external,
        );
    };
    row("oracle (explicit)", &r);
    let sweep: Vec<(f32, u32)> = match std::env::var("SETTINGS") {
        Ok(v) => v
            .split(',')
            .map(|p| {
                let (hz, it) = p.split_once('x').expect("SETTINGS=HZxITERS,...");
                (hz.parse().unwrap(), it.parse().unwrap())
            })
            .collect(),
        Err(_) => vec![
            (120.0, 10),
            (240.0, 10),
            (240.0, 20),
            (480.0, 10),
            (960.0, 30),
        ],
    };
    for (hz, it) in sweep {
        let s = Setting {
            hz,
            iterations: it,
            threshold_scale: 1.0,
        };
        let t = std::time::Instant::now();
        let m = metrics(&car, Some(s));
        eprintln!("{hz} Hz × {it} done in {:.1} s", t.elapsed().as_secs_f32());
        row(&format!("{hz} Hz × {it}"), &m);
    }
    if std::env::var("FIT").as_deref() != Ok("1") {
        return;
    }
    // Fit: one global threshold scale per step rate so the 56 km/h crush depth
    // (max residual) matches the oracle.
    println!();
    println!(
        "| step rate | fitted threshold scale | crush 56 km/h max (cm), oracle {:.1} |",
        r.crash56.crush.1 * 100.0
    );
    println!("|---|---|---|");
    let start = Vec3::new(0.0, car.lift, -1.0);
    let v = 56.0 / 3.6;
    let crush = |hz: f32, k: f32| {
        let s = Setting {
            hz,
            iterations: 10,
            threshold_scale: k,
        };
        wall_crash(
            &car,
            &mut *runner(Some(s), &car, start, Vec3::new(0.0, 0.0, v), &[WALL]),
            start,
            v,
        )
        .crush
        .1
    };
    for hz in [120.0, 240.0] {
        let target = r.crash56.crush.1;
        let scale = minimize_1d(|k| rel_error(crush(hz, k), target), 0.2, 3.0, 8);
        println!(
            "| {hz} Hz × 10 | {scale:.2} | {:.1} |",
            crush(hz, scale) * 100.0
        );
    }
}
