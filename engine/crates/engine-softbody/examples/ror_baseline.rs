//! A Rigs of Rods-style CPU baseline: what Rubble's node–beam vehicles cost and how
//! they crash when solved the way Rigs of Rods and BeamNG solve them.
//!
//! ```bash
//! cargo run -p engine-softbody --release --example ror_baseline -- car=sedan hz=2000
//! cargo run -p engine-softbody --release --example ror_baseline -- car=hires cars=1,10,40
//! ```
//! Explicit symplectic Euler in f32 at a fixed rate (Rigs of Rods: 2 kHz), every beam
//! through the same rules as the oracle and the GPU ([`model`]'s force, yield and
//! break, checked every substep), nodes against the ground and a wall by projection
//! (pushed out, normal velocity removed, Coulomb friction), as RoR's terrain contact.
//! No node–triangle contacts: this is a lower bound on RoR's per-substep cost.
//!
//! Reports the beam-only stable rate, the CPU cost per 60 fps frame for N cars (one
//! thread, and one car per thread as RoR runs actors), and the 56 km/h wall crush
//! profile (compare `oracle_crash`, same metric and scene).
use engine_softbody::hires::{hires_car_desc, HiresParams};
use engine_softbody::metrics::crush_profile;
use engine_softbody::model::{plastic_update, readout_force_rate, BeamT};
use engine_softbody::pack::{pack_soft, Placement, SpawnOptions};
use engine_softbody::procedural::{car_desc, CarParams};
use engine_softbody::SoftVehicleDesc;
use glam::{Quat, Vec3};
use std::time::Instant;

fn arg(name: &str) -> Option<String> {
    std::env::args()
        .skip(1)
        .find_map(|a| a.strip_prefix(&format!("{name}=")).map(str::to_string))
}

const NODE_RADIUS: f32 = 0.06;
const WALL_Z: f32 = 2.8;

/// One vehicle, structure of arrays.
#[derive(Clone)]
struct Car {
    x: Vec<Vec3>,
    v: Vec<Vec3>,
    f: Vec<Vec3>,
    inv_m: Vec<f32>,
    collide: Vec<bool>,
    beams: Vec<BeamT<f32>>,
    wall: bool,
}

impl Car {
    fn new(desc: &SoftVehicleDesc, pos: Vec3, vel: Vec3, wall: bool) -> Self {
        let opts = SpawnOptions {
            rigid_core: false,
            velocity: vel.to_array(),
            ..Default::default()
        };
        let p = pack_soft(desc, Placement { position: pos, rotation: Quat::IDENTITY }, &opts, 0, -1);
        let beams = p
            .beams
            .iter()
            .map(|b| {
                let mut e = b.clone();
                e.penalty = e.spring;
                e
            })
            .collect();
        Self {
            x: p.bodies.iter().take(desc.nodes.len()).map(|b| Vec3::from_array(b.position)).collect(),
            v: desc
                .nodes
                .iter()
                .map(|n| if n.fixed { Vec3::ZERO } else { vel })
                .collect(),
            f: vec![Vec3::ZERO; desc.nodes.len()],
            inv_m: desc
                .nodes
                .iter()
                .zip(&p.masses)
                .map(|(n, &m)| if n.fixed { 0.0 } else { 1.0 / m })
                .collect(),
            collide: desc.nodes.iter().map(|n| n.collision).collect(),
            beams,
            wall,
        }
    }

    /// Beam-only stable rate: per node ω = √(Σk/m), damped bound as the oracle's.
    fn stable_hz(&self) -> f32 {
        let n = self.x.len();
        let (mut ks, mut cs) = (vec![0.0f32; n], vec![0.0f32; n]);
        for b in &self.beams {
            for k in [b.a, b.b] {
                ks[k as usize] += b.spring;
                cs[k as usize] += b.damp;
            }
        }
        (0..n)
            .filter(|&i| self.inv_m[i] > 0.0)
            .map(|i| {
                let w = (ks[i] * self.inv_m[i]).sqrt();
                let zeta = cs[i] * self.inv_m[i] / (2.0 * w);
                w * (zeta + (zeta * zeta + 1.0).sqrt()) / 1.6
            })
            .fold(0.0, f32::max)
    }

    fn substep(&mut self, h: f32) {
        let g = Vec3::new(0.0, -9.81, 0.0);
        for (f, &w) in self.f.iter_mut().zip(&self.inv_m) {
            *f = if w > 0.0 { g / w } else { Vec3::ZERO };
        }
        for b in &mut self.beams {
            if b.broken() {
                continue;
            }
            let (ia, ib) = (b.a as usize, b.b as usize);
            let d = self.x[ia] - self.x[ib];
            let len = d.length();
            if len < 1e-6 {
                continue;
            }
            let dir = d / len;
            let rate = (self.v[ia] - self.v[ib]).dot(dir);
            let force = readout_force_rate(b, len, rate, 1.0);
            if plastic_update(b, len, force, 0.1, true, true).broke {
                continue;
            }
            let fa = dir * b.force;
            self.f[ia] += fa;
            self.f[ib] -= fa;
        }
        for i in 0..self.x.len() {
            let w = self.inv_m[i];
            if w == 0.0 {
                continue;
            }
            self.v[i] += self.f[i] * (h * w);
            self.x[i] += self.v[i] * h;
            if !self.collide[i] {
                continue;
            }
            // Ground (y = 0) and wall (z = WALL_Z, facing −z): project out, remove
            // the approach speed, Coulomb friction on the tangential velocity.
            let contact = |n: Vec3, pen: f32, mu: f32, x: &mut Vec3, v: &mut Vec3| {
                if pen <= 0.0 {
                    return;
                }
                *x += n * pen;
                let vn = v.dot(n);
                if vn < 0.0 {
                    let vt = *v - n * vn;
                    let s = vt.length();
                    *v = if s > 0.0 { vt * (1.0 - (mu * -vn / s).min(1.0)) } else { vt };
                }
            };
            let (x, v) = (&mut self.x[i], &mut self.v[i]);
            contact(Vec3::Y, NODE_RADIUS - x.y, 0.8, x, v);
            if self.wall && x.y < 3.0 {
                contact(-Vec3::Z, x.z - (WALL_Z - NODE_RADIUS), 1.0, x, v);
            }
        }
    }

    fn finite(&self) -> bool {
        self.x.iter().all(|p| p.is_finite() && p.length() < 1e4)
    }
}

fn main() {
    let car = arg("car").unwrap_or_else(|| "sedan".into());
    let hz: f32 = arg("hz").and_then(|v| v.parse().ok()).unwrap_or(2000.0);
    let counts: Vec<usize> = arg("cars")
        .unwrap_or_else(|| "1,10,40".into())
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();
    let (desc, body): (SoftVehicleDesc, Vec<u32>) = match car.as_str() {
        "hires" => {
            let (d, _) = hires_car_desc(&HiresParams::default());
            let b = (0..d.nodes.len() as u32).collect();
            (d, b)
        }
        _ => {
            let p = if car == "default" { CarParams::default() } else { CarParams::sedan() };
            let (d, _) = car_desc(&p);
            let b = d.part("body").unwrap().nodes.clone();
            (d, b)
        }
    };
    let lift = 0.06 - desc.bounds().0[1];
    let probe = Car::new(&desc, Vec3::new(0.0, lift, 0.0), Vec3::ZERO, false);
    let stable = probe.stable_hz();
    println!(
        "{car}: {} nodes, {} beams; beam-only stable rate {:.0} Hz; running {hz:.0} Hz",
        desc.nodes.len(),
        probe.beams.len(),
        stable
    );
    let h = 1.0 / hz;
    let per_frame = (hz / 60.0).round() as usize;

    // Cost: N cars resting and settling side by side, 0.25 s of simulated time.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    for &n in &counts {
        let make = || -> Vec<Car> {
            (0..n)
                .map(|k| Car::new(&desc, Vec3::new(k as f32 * 4.0, lift, 0.0), Vec3::ZERO, false))
                .collect()
        };
        let frames = 15;
        let mut cars = make();
        let t = Instant::now();
        for _ in 0..frames * per_frame {
            for c in &mut cars {
                c.substep(h);
            }
        }
        let single = t.elapsed().as_secs_f64() * 1e3 / frames as f64;
        let mut cars = make();
        let t = Instant::now();
        for _ in 0..frames {
            let chunk = n.div_ceil(threads);
            std::thread::scope(|s| {
                for part in cars.chunks_mut(chunk) {
                    s.spawn(move || {
                        for _ in 0..per_frame {
                            for c in part.iter_mut() {
                                c.substep(h);
                            }
                        }
                    });
                }
            });
        }
        let multi = t.elapsed().as_secs_f64() * 1e3 / frames as f64;
        println!(
            "  {n:>3} cars: {single:7.2} ms per 60 fps frame on 1 thread, {multi:7.2} ms on {} threads ({} substeps per frame){}",
            threads.min(n),
            per_frame,
            if cars.iter().all(Car::finite) { "" } else { "  UNSTABLE" }
        );
    }

    // Accuracy: 56 km/h into the wall, 1 s.
    let speed: f32 = arg("speed").and_then(|v| v.parse().ok()).unwrap_or(56.0);
    let spawn = Vec3::new(0.0, lift, -1.0);
    let mut c = Car::new(&desc, spawn, Vec3::new(0.0, 0.0, speed / 3.6), true);
    let rest: Vec<Vec3> = desc.nodes.iter().map(|n| Vec3::from_array(n.position) + spawn).collect();
    let t = Instant::now();
    for _ in 0..hz as usize {
        c.substep(h);
    }
    let wall_ms = t.elapsed().as_secs_f64() * 1e3;
    if !c.finite() {
        println!("  {speed} km/h wall: UNSTABLE at {hz:.0} Hz");
        return;
    }
    let broken = c.beams.iter().filter(|b| b.broken()).count();
    let prof = crush_profile(&rest, &c.x, &body, &body, Vec3::Z, Vec3::X, 5);
    println!(
        "  {speed} km/h wall: crush profile {:?} cm, {broken} beams broken, {wall_ms:.1} ms CPU for 1 s",
        prof.iter().map(|v| (v * 100.0).round()).collect::<Vec<_>>()
    );
}
