//! Drive parity: the hybrid dune buggy (`engine_softbody::buggy`) against the
//! rigid `dune_buggy` preset it is built from, on the same scripted course
//! (240 Hz × 10 for every build). The hybrid in rigid mode must handle like the
//! rigid buggy: same launch, top speed, step-steer yaw rate, slalom path and
//! jump, no judder, and no mode flips or bending while it only drives.

use engine_schema::{BodyState, Collider, Material, ShapeType, WorldCapacity};
use engine_softbody::buggy::dune_buggy_soft;
use engine_softbody::{SoftCapacity, SoftHandle, SoftSettings, SpawnOptions};
use engine_vehicle::{presets, VehicleCapacity, VehicleHandle, VehicleInput};
use engine_webgpu::WgpuSim;
use glam::{Quat, Vec3};

const HZ: f32 = 240.0;

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

fn static_box(sim: &mut WgpuSim, pos: [f32; 3], half: [f32; 3], rot: Quat) {
    let rot = rot.to_array();
    let body = sim.add_body(BodyState {
        position: pos,
        inv_mass: 0.0,
        orientation: rot,
        inv_inertia: [0.0; 3],
        flags: 0,
        linear_vel: [0.0; 3],
        sleep_timer: 0.0,
        angular_vel: [0.0; 3],
        generation: 0,
        initial_pos: pos,
        initial_pad: 0.0,
        initial_rot: rot,
    });
    sim.add_collider(Collider {
        body,
        shape: ShapeType::Box as u32,
        material: 0,
        generation: 0,
        local_pos: [0.0; 3],
        density: 0.0,
        local_rot: [0.0, 0.0, 0.0, 1.0],
        params: [half[0], half[1], half[2], Vec3::from_array(half).length()],
    });
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Build {
    Rigid,
    HybridSoft,
    HybridRigidMode,
}

impl Build {
    const ALL: [Build; 3] = [Build::Rigid, Build::HybridSoft, Build::HybridRigidMode];
    fn name(self) -> &'static str {
        match self {
            Build::Rigid => "rigid",
            Build::HybridSoft => "hybrid soft",
            Build::HybridRigidMode => "hybrid rigid-mode",
        }
    }
}

/// A buggy of one build in a fresh world (flat ground, plus `extra` statics).
struct Car {
    sim: WgpuSim,
    core: VehicleHandle,
    chassis: u32,
    soft: Option<SoftHandle>,
}

impl Car {
    fn new(build: Build, extra: impl FnOnce(&mut WgpuSim)) -> Self {
        let mut sim = WgpuSim::new_with_soft(
            capacity(),
            VehicleCapacity::default(),
            SoftCapacity::default(),
        )
        .expect("wgpu");
        sim.set_params(engine_softbody::soft_scene_params(HZ, 10));
        sim.set_soft_settings(SoftSettings {
            rigid_mode: build == Build::HybridRigidMode,
            ..SoftSettings::calibrated_for(HZ)
        });
        sim.set_material(
            0,
            Material {
                friction: 1.0,
                restitution: 0.0,
                contact_offset: 0.01,
                _pad: 0.0,
            },
        );
        sim.set_material(
            9,
            Material {
                friction: 0.8,
                restitution: 0.0,
                contact_offset: 0.01,
                _pad: 0.0,
            },
        );
        static_box(
            &mut sim,
            [0.0, -5.0, 0.0],
            [400.0, 5.0, 400.0],
            Quat::IDENTITY,
        );
        extra(&mut sim);
        let (core, soft) = match build {
            Build::Rigid => (
                sim.add_vehicle(
                    &presets::dune_buggy(),
                    [0.0, 0.6, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                )
                .expect("spawn"),
                None,
            ),
            _ => {
                let h = sim
                    .add_soft_vehicle(
                        &dune_buggy_soft(),
                        [0.0, 0.05, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        &SpawnOptions::default(),
                    )
                    .expect("spawn");
                (sim.soft_host().entry(h).core.expect("core").0, Some(h))
            }
        };
        let chassis = sim.vehicle_host().chassis_body(core);
        Car {
            sim,
            core,
            chassis,
            soft,
        }
    }

    fn step(&mut self, input: VehicleInput) -> Sample {
        self.sim.set_vehicle_input(self.core, input);
        self.sim.step().expect("step");
        let b = self.sim.read_body(self.chassis);
        let st = self.sim.vehicle_state(self.core);
        let q = Quat::from_array(b.orientation);
        Sample {
            pos: Vec3::from_array(b.position),
            vel: Vec3::from_array(b.linear_vel),
            pitch: (q * Vec3::Z).y.clamp(-1.0, 1.0).asin(),
            speed: st.forward_speed,
            yaw_rate: st.yaw_rate,
            contacts: st.contacts,
            rigid: self.soft.is_some_and(|h| self.sim.read_soft_rigid(h)),
            susp: {
                let w: Vec<f32> = st
                    .wheels
                    .iter()
                    .filter(|w| w.in_contact)
                    .map(|w| w.suspension_length)
                    .collect();
                w.iter().sum::<f32>() / w.len().max(1) as f32
            },
        }
    }

    fn settle(&mut self, steps: u32) -> Sample {
        let mut s = self.step(VehicleInput::default());
        for _ in 1..steps {
            s = self.step(VehicleInput::default());
        }
        s
    }

    /// Beams yielded and broken; bolted parts and core parts lost.
    fn damage(&self) -> (usize, usize, u32) {
        let st = self.sim.vehicle_state(self.core);
        let lost = st.parts_broken.count_ones() + st.wheels_broken.count_ones();
        match self.soft {
            None => (0, 0, lost),
            Some(h) => {
                let beams = self.sim.read_soft_beams(h);
                let y = beams
                    .iter()
                    .filter(|b| b.flags & engine_softbody::BEAM_YIELDED != 0)
                    .count();
                let br = beams.iter().filter(|b| b.broken()).count();
                let loose = self
                    .sim
                    .soft_host()
                    .entry(h)
                    .loose_parts(&self.sim.download_joints())
                    .len();
                (y, br, lost + loose as u32)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Sample {
    pos: Vec3,
    vel: Vec3,
    pitch: f32,
    speed: f32,
    yaw_rate: f32,
    contacts: u32,
    rigid: bool,
    /// Mean suspension length of the wheels in contact (m).
    susp: f32,
}

/// Throttle that holds `target` m/s.
fn cruise(speed: f32, target: f32, steer: f32) -> VehicleInput {
    let e = target - speed;
    VehicleInput {
        forward: (e * 0.5).clamp(0.0, 1.0),
        brake: (-e * 0.5).clamp(0.0, 1.0),
        steer,
        ..Default::default()
    }
}

/// High-frequency RMS of chassis vertical acceleration (m/s²): the per-step
/// vertical acceleration minus its 0.1 s moving average.
fn judder(vy: &[f32]) -> f32 {
    let a: Vec<f32> = vy.windows(2).map(|w| (w[1] - w[0]) * HZ).collect();
    let k = (HZ * 0.1) as usize;
    if a.len() <= 2 * k {
        return 0.0;
    }
    let hf: Vec<f32> = (k..a.len() - k)
        .map(|i| a[i] - a[i - k..=i + k].iter().sum::<f32>() / (2 * k + 1) as f32)
        .collect();
    (hf.iter().map(|x| x * x).sum::<f32>() / hf.len() as f32).sqrt()
}

#[derive(Clone, Debug, Default)]
struct Metrics {
    /// Suspension length at rest (soft settle, then after the switch to rigid).
    susp_rest: f32,
    /// Largest one-step chassis height change in the 0.25 s around a mode switch.
    switch_dy: f32,
    t15: f32,
    v6: f32,
    judder: f32,
    yaw_peak: f32,
    yaw_steady: f32,
    slalom: Vec<f32>,
    airtime: f32,
    land_pitch: f32,
    settle: f32,
    flips: u32,
    rigid_frac: f32,
    damage: (usize, usize, u32),
}

/// Straight-line launch, step steer, slalom and a ramp jump, each in a fresh
/// world. Counts mode flips over all driving after the settle.
fn course(build: Build) -> Metrics {
    let mut m = Metrics::default();
    let mut flips = 0u32;
    let (mut rigid_steps, mut steps) = (0u32, 0u32);
    let mut track = |s: &Sample, prev: &mut Option<bool>| {
        if let Some(p) = *prev {
            flips += u32::from(p != s.rigid);
        }
        *prev = Some(s.rigid);
        rigid_steps += u32::from(s.rigid);
        steps += 1;
    };

    // Launch: full throttle for 6 s after a 2 s settle.
    {
        let mut car = Car::new(build, |_| {});
        let mut ys = Vec::new();
        let mut switch_at = None;
        let mut rest = car.step(VehicleInput::default());
        for i in 1..480usize {
            let was = rest.rigid;
            rest = car.step(VehicleInput::default());
            ys.push(rest.pos.y);
            if was != rest.rigid && switch_at.is_none() {
                switch_at = Some(i);
            }
        }
        m.susp_rest = rest.susp;
        if let Some(k) = switch_at {
            let w = (HZ * 0.125) as usize;
            let (a, b) = (k.saturating_sub(w).max(1), (k + w).min(ys.len() - 1));
            m.switch_dy = (a..b)
                .map(|i| (ys[i] - ys[i - 1]).abs())
                .fold(0.0, f32::max);
        }
        let mut prev = None;
        let mut vy = Vec::new();
        m.t15 = f32::NAN;
        for i in 0..(HZ * 6.0) as u32 {
            let s = car.step(VehicleInput {
                forward: 1.0,
                ..Default::default()
            });
            track(&s, &mut prev);
            vy.push(s.vel.y);
            if m.t15.is_nan() && s.speed >= 15.0 {
                m.t15 = (i + 1) as f32 / HZ;
            }
            m.v6 = s.speed;
        }
        m.judder = judder(&vy);
        let d = car.damage();
        m.damage = (m.damage.0 + d.0, m.damage.1 + d.1, m.damage.2 + d.2);
    }

    // Step steer: hold 10 m/s, then full right lock for 2 s.
    {
        let mut car = Car::new(build, |_| {});
        let mut s = car.settle(480);
        let mut prev = None;
        for _ in 0..(HZ * 4.0) as u32 {
            s = car.step(cruise(s.speed, 10.0, 0.0));
            track(&s, &mut prev);
        }
        let mut yaw = Vec::new();
        for _ in 0..(HZ * 2.0) as u32 {
            s = car.step(cruise(s.speed, 10.0, 1.0));
            track(&s, &mut prev);
            yaw.push(s.yaw_rate.abs());
        }
        m.yaw_peak = yaw.iter().copied().fold(0.0, f32::max);
        let tail = &yaw[yaw.len() / 2..];
        m.yaw_steady = tail.iter().sum::<f32>() / tail.len() as f32;
        let d = car.damage();
        m.damage = (m.damage.0 + d.0, m.damage.1 + d.1, m.damage.2 + d.2);
    }

    // Slalom: 12 m/s, steer ±0.5 at 0.5 Hz for 6 s; lateral offset every 0.25 s.
    {
        let mut car = Car::new(build, |_| {});
        let mut s = car.settle(480);
        let mut prev = None;
        for _ in 0..(HZ * 4.0) as u32 {
            s = car.step(cruise(s.speed, 12.0, 0.0));
            track(&s, &mut prev);
        }
        let x0 = s.pos.x;
        let n = (HZ * 6.0) as u32;
        for i in 0..n {
            let t = i as f32 / HZ;
            s = car.step(cruise(
                s.speed,
                12.0,
                0.5 * (std::f32::consts::TAU * 0.5 * t).sin(),
            ));
            track(&s, &mut prev);
            if (i + 1) % (HZ as u32 / 4) == 0 {
                m.slalom.push(s.pos.x - x0);
            }
        }
        let d = car.damage();
        m.damage = (m.damage.0 + d.0, m.damage.1 + d.1, m.damage.2 + d.2);
    }

    // Jump: the mount test's kicker ramp 25 m ahead, full throttle.
    {
        let mut car = Car::new(build, |sim| {
            static_box(
                sim,
                [0.0, 0.5, 25.0],
                [3.0, 0.3, 3.0],
                Quat::from_axis_angle(Vec3::X, -0.25),
            );
        });
        car.settle(480);
        let mut prev = None;
        let (mut air, mut landed_at, mut airborne) = (0u32, None, false);
        let mut pitch_after = Vec::new();
        let mut vy_after = Vec::new();
        for i in 0..(HZ * 7.0) as u32 {
            let s = car.step(VehicleInput {
                forward: 1.0,
                ..Default::default()
            });
            track(&s, &mut prev);
            if s.contacts == 0 && s.pos.z > 20.0 {
                air += 1;
                airborne = true;
            } else if airborne && landed_at.is_none() && s.contacts >= 2 {
                landed_at = Some(i);
            }
            if landed_at.is_some() {
                pitch_after.push(s.pitch);
                vy_after.push(s.vel.y);
            }
        }
        m.airtime = air as f32 / HZ;
        m.land_pitch = pitch_after
            .iter()
            .take(HZ as usize)
            .map(|p| p.abs())
            .fold(0.0, f32::max)
            .to_degrees();
        // Settle: time after landing until vertical speed stays under 0.3 m/s.
        let last_big = vy_after
            .iter()
            .rposition(|v| v.abs() > 0.3)
            .map_or(0, |i| i + 1);
        m.settle = last_big as f32 / HZ;
        let d = car.damage();
        m.damage = (m.damage.0 + d.0, m.damage.1 + d.1, m.damage.2 + d.2);
    }

    m.flips = flips;
    m.rigid_frac = rigid_steps as f32 / steps.max(1) as f32;
    m
}

fn rel(a: f32, b: f32) -> f32 {
    (a - b).abs() / b.abs().max(1e-6)
}

/// Run by: cargo test -p engine-webgpu --release --test buggy_parity -- --nocapture
#[test]
fn hybrid_buggy_drives_like_the_rigid_buggy() {
    let runs: Vec<(Build, Metrics)> = Build::ALL.iter().map(|&b| (b, course(b))).collect();
    let rigid = runs[0].1.clone();
    eprintln!(
        "{:<18} {:>7} {:>7} {:>6} {:>6} {:>7} {:>7} {:>7} {:>8} {:>6} {:>6} {:>6} {:>5} {:>6} damage(y,b,lost)",
        "build", "susp m", "swΔy mm", "t15 s", "v6", "judder", "yawPk", "yawSt", "slalomΔ", "air s", "pitch°", "settle", "flips", "rigid%"
    );
    for (b, m) in &runs {
        let slalom_err = (m
            .slalom
            .iter()
            .zip(&rigid.slalom)
            .map(|(a, r)| (a - r) * (a - r))
            .sum::<f32>()
            / m.slalom.len().max(1) as f32)
            .sqrt();
        eprintln!(
            "{:<18} {:>7.3} {:>7.2} {:>6.2} {:>6.2} {:>7.2} {:>7.3} {:>7.3} {:>8.3} {:>6.2} {:>6.1} {:>6.2} {:>5} {:>5.0}% {:?}",
            b.name(),
            m.susp_rest,
            m.switch_dy * 1000.0,
            m.t15,
            m.v6,
            m.judder,
            m.yaw_peak,
            m.yaw_steady,
            slalom_err,
            m.airtime,
            m.land_pitch,
            m.settle,
            m.flips,
            m.rigid_frac * 100.0,
            m.damage
        );
    }
    let h = &runs[2].1;
    let slalom_err = (h
        .slalom
        .iter()
        .zip(&rigid.slalom)
        .map(|(a, r)| (a - r) * (a - r))
        .sum::<f32>()
        / h.slalom.len().max(1) as f32)
        .sqrt();
    assert_eq!(h.damage, (0, 0, 0), "driving and jumping does no damage");
    assert!(
        rel(h.v6, rigid.v6) < 0.05,
        "speed after 6 s: {} vs {}",
        h.v6,
        rigid.v6
    );
    assert!(
        rel(h.t15, rigid.t15) < 0.05,
        "0–15 m/s: {} vs {}",
        h.t15,
        rigid.t15
    );
    assert!(
        rel(h.yaw_steady, rigid.yaw_steady) < 0.05,
        "step-steer yaw rate: {} vs {}",
        h.yaw_steady,
        rigid.yaw_steady
    );
    assert!(
        rel(h.yaw_peak, rigid.yaw_peak) < 0.10,
        "step-steer peak yaw rate: {} vs {}",
        h.yaw_peak,
        rigid.yaw_peak
    );
    assert!(
        slalom_err < 0.25,
        "slalom path within 25 cm RMS ({slalom_err} m)"
    );
    assert!(
        h.judder <= rigid.judder * 1.5 + 0.5,
        "no extra judder: {} vs {}",
        h.judder,
        rigid.judder
    );
    assert!(
        h.rigid_frac > 0.9,
        "rigid while it only drives ({:.0}%)",
        h.rigid_frac * 100.0
    );
}
