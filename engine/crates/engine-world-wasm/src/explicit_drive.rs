// Driving explicit vehicles in the browser (`ExplicitSim`): raycast wheels
// (`engine_softbody::explicit_drive`) take the page's input per vehicle and report
// their wheels' poses for drawing and the camera. The page forwards keys and
// gamepad state only; every rule is in Rust and Slang.

use super::{err, ExplicitSim};
use engine_softbody::explicit_drive::{Drive, DriveInput};
use glam::Vec3;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

/// A vehicle's pose from its wheels (the last `readDrive`): centre, forward, up and
/// forward speed.
#[derive(Clone, Copy, Debug)]
pub(super) struct Pose {
    pub p: Vec3,
    pub fwd: Vec3,
    pub up: Vec3,
    pub speed: f32,
}

/// Per vehicle its last read pose (`None`: no raycast wheels, or not read yet).
pub(super) type Poses = Rc<RefCell<Vec<Option<Pose>>>>;

/// A self-driving car (the soft solver's traffic AI, softbody.html, in Rust): it
/// steers toward a target, a random point in the arena (wander) or the nearest
/// other car (chase); a stuck car reverses out; cruising cars brake for cars ahead.
#[derive(Clone, Debug)]
pub(super) struct AiCar {
    chase: bool,
    cruise: bool,
    target: Option<[f32; 2]>,
    t: f32,
    slow: f32,
    reverse: f32,
    kmh: f32,
    seed: u32,
}

/// `n` cars' AI: `mode` 'derby' (two in three wander, one chases), 'cruise' or 'off'.
pub(super) fn ai_cars(n: usize, mode: &str) -> Vec<AiCar> {
    if mode == "off" || mode == "0" {
        return Vec::new();
    }
    let cruise = mode == "cruise";
    (0..n)
        .map(|i| AiCar {
            chase: !cruise && i % 3 == 2,
            cruise,
            target: None,
            t: 0.0,
            slow: 0.0,
            reverse: 0.0,
            kmh: 35.0 + ((i * 17) % 45) as f32,
            seed: (i as u32).wrapping_mul(7919).wrapping_add(13),
        })
        .collect()
}

fn rand(c: &mut AiCar) -> f32 {
    c.seed = c.seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
    (c.seed >> 8) as f32 / 16_777_216.0
}

/// One AI car's input from the poses (`k` itself).
fn ai_input(c: &mut AiCar, k: usize, poses: &[Option<Pose>], dt: f32) -> DriveInput {
    let none = DriveInput::default();
    let Some(me) = poses[k] else { return none };
    let right = me.fwd.cross(me.up);
    let speed = me.speed.abs();
    c.t += dt;
    if me.up.y < 0.3 {
        return none; // on its side or roof
    }
    if c.chase {
        let mut best: Option<(f32, Pose)> = None;
        for (j, o) in poses.iter().enumerate() {
            if let (true, Some(o)) = (j != k, o) {
                let d = Vec3::new(o.p.x - me.p.x, 0.0, o.p.z - me.p.z).length();
                if best.is_none_or(|b| d < b.0) {
                    best = Some((d, *o));
                }
            }
        }
        if let Some((_, o)) = best {
            let v = o.fwd * o.speed;
            c.target = Some([o.p.x + v.x * 0.4, o.p.z + v.z * 0.4]);
        }
    } else if c
        .target
        .is_none_or(|t| c.t > 9.0 || Vec3::new(t[0] - me.p.x, 0.0, t[1] - me.p.z).length() < 8.0)
    {
        let x = (rand(c) - 0.5) * 120.0;
        let z = (rand(c) - 0.5) * 120.0;
        c.target = Some([x, z]);
        c.t = 0.0;
    }
    let t = c.target.unwrap_or([me.p.x, me.p.z]);
    let d = Vec3::new(t[0] - me.p.x, 0.0, t[1] - me.p.z);
    let ang = d.dot(right).atan2(d.dot(me.fwd));
    let mut steer = (ang * 1.6).clamp(-1.0, 1.0);
    let mut want = c.kmh / 3.6 * if ang.abs() > 1.2 { 0.45 } else { 1.0 };
    if c.cruise {
        for (j, o) in poses.iter().enumerate() {
            let (true, Some(o)) = (j != k, o) else {
                continue;
            };
            let dd = o.p - me.p;
            let ahead = dd.x * me.fwd.x + dd.z * me.fwd.z;
            let side = dd.x * right.x + dd.z * right.z;
            if ahead > 0.0 && ahead < 14.0 && side.abs() < ahead * 0.58 {
                want = want.min(((ahead - 6.0) * 0.6).max(0.0));
                let sg = if side > 0.0 {
                    1.0
                } else if side < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                steer = (steer - sg * 0.6).clamp(-1.0, 1.0);
            }
        }
    }
    let mut input = DriveInput {
        forward: if speed < want { 0.85 } else { 0.0 },
        steer,
        brake: if speed > want * 1.3 { 0.4 } else { 0.0 },
        handbrake: 0.0,
    };
    // Stuck against something: reverse out with the wheel turned the other way.
    c.slow = if speed < 0.8 { c.slow + dt } else { 0.0 };
    if c.slow > 1.6 && c.reverse <= 0.0 {
        c.reverse = 1.4;
        c.slow = 0.0;
    }
    if c.reverse > 0.0 {
        c.reverse -= dt;
        input = DriveInput {
            forward: -0.8,
            steer: -steer,
            brake: 0.0,
            handbrake: 0.0,
        };
    }
    input
}

/// A vehicle's pose from its wheels: centre, forward (the fixed wheels' mean
/// heading, else every wheel's), up, and the wheels as drawn.
pub(super) fn pose_json(d: &Drive) -> serde_json::Value {
    let n = d.wheels.len().max(1) as f32;
    let centre = d.wheels.iter().map(|w| w.centre).sum::<Vec3>() / n;
    let up = d
        .wheels
        .iter()
        .map(|w| w.up)
        .sum::<Vec3>()
        .normalize_or_zero();
    let heading = |w: &engine_softbody::explicit_drive::WheelState| w.axle.cross(w.up);
    let fixed: Vec3 = d
        .wheels
        .iter()
        .zip(&d.config.wheels)
        .filter(|(_, c)| c.max_steer == 0.0)
        .map(|(w, _)| heading(w))
        .sum();
    let fwd = if fixed.length_squared() > 1e-8 {
        fixed.normalize()
    } else {
        d.wheels
            .iter()
            .map(heading)
            .sum::<Vec3>()
            .normalize_or_zero()
    };
    let wheels: Vec<serde_json::Value> = d
        .wheels
        .iter()
        .zip(&d.config.wheels)
        .map(|(w, c)| {
            serde_json::json!({
                "centre": w.centre.to_array(), "axle": w.axle.to_array(), "up": w.up.to_array(),
                "spin": w.spin, "steer": w.steer, "omega": w.omega, "radius": c.radius,
                "width": c.width, "contact": w.contact, "load": w.load, "kappa": w.kappa,
                "alpha": w.alpha,
            })
        })
        .collect();
    serde_json::json!({
        "centre": centre.to_array(), "forward": fwd.to_array(), "up": up.to_array(),
        "speed": d.speed, "input": d.smoothed, "tractionCut": d.tc_cut, "wheels": wheels,
    })
}

#[wasm_bindgen]
impl ExplicitSim {
    /// Whether vehicle `vehicle` drives on raycast wheels.
    #[wasm_bindgen(js_name = drivable)]
    pub fn drivable(&self, vehicle: u32) -> bool {
        self.vehicles
            .get(vehicle as usize)
            .is_some_and(|v| v.sim.drive.is_some())
    }

    /// The driver's input for vehicle `vehicle` from the next frame on: forward
    /// −1..1 (negative reverses), steer −1..1 (positive right), brake and handbrake
    /// 0..1. A vehicle without raycast wheels ignores it.
    #[wasm_bindgen(js_name = setDriveInput)]
    pub fn set_drive_input(
        &mut self,
        vehicle: u32,
        forward: f32,
        steer: f32,
        brake: f32,
        handbrake: f32,
    ) -> Result<(), JsValue> {
        let input = DriveInput {
            forward,
            steer,
            brake,
            handbrake,
        };
        if ![forward, steer, brake, handbrake]
            .iter()
            .all(|x| x.is_finite())
        {
            return Err(err("setDriveInput: input not finite"));
        }
        let v = vehicle as usize;
        if v >= self.vehicles.len() {
            return Err(err(format!("setDriveInput: no vehicle {vehicle}")));
        }
        self.gpu.set_drive_input(v, input);
        if let Some(d) = self.vehicles[v].sim.drive.as_mut() {
            d.input = input;
        }
        Ok(())
    }

    /// The self-driving cars' input for the next frame (`traffic`), from the poses of
    /// the last `readDrive`; vehicle `user` (−1 none) is the page's and skipped.
    /// `dt`: seconds since the last call.
    pub fn autopilot(&mut self, dt: f32, user: i32) -> Result<(), JsValue> {
        let poses = self.drive_poses.borrow().clone();
        if poses.len() != self.vehicles.len() {
            return Ok(());
        }
        for k in 0..self.ai.len() {
            if k as i32 == user {
                continue;
            }
            let input = ai_input(&mut self.ai[k], k, &poses, dt);
            self.gpu.set_drive_input(k, input);
        }
        Ok(())
    }

    /// Vehicle `vehicle`'s raycast wheels' records in the misc buffer
    /// (`miscBuffer`, vec4 records), 4 per wheel in the drive state's order:
    /// its state (omega, spin angle, steer, length), centre (world), axle
    /// (unit, steered, left) and up, written every step. The showcase
    /// renderer places the wheels' pieces from them (`skin::wheel_vertex`).
    /// Empty without raycast wheels.
    #[wasm_bindgen(js_name = wheelRecords)]
    pub fn wheel_records(&self, vehicle: u32) -> Vec<u32> {
        use engine_exec::explicit_plan::drive::{VEHICLE_RECS, WHEEL_RECS};
        let (Some(l), Some(d)) = (
            self.gpu.layout().drive,
            self.vehicles
                .get(vehicle as usize)
                .and_then(|v| v.sim.drive.as_ref()),
        ) else {
            return Vec::new();
        };
        let base = l.vehicle(vehicle as usize) + VEHICLE_RECS;
        (0..d.wheels.len() as u32)
            .flat_map(|k| {
                let w = base + WHEEL_RECS * k;
                [w + 15, w + 17, w + 18, w + 19]
            })
            .collect()
    }

    /// Whether the scene's cars drive themselves.
    #[wasm_bindgen(js_name = selfDriving)]
    pub fn self_driving(&self) -> bool {
        !self.ai.is_empty()
    }

    /// A promise of every vehicle's drive state after the frames submitted so far:
    /// `{ step, vehicles: [null | { centre, forward, up, speed, input, tractionCut,
    /// wheels: [{ centre, axle, up, spin, steer, omega, radius, width, contact, load,
    /// kappa, alpha }] }] }` (world metres; `axle` the steered wheel's axis, pointing
    /// left).
    #[wasm_bindgen(js_name = readDrive)]
    pub fn read_drive(&self) -> Result<js_sys::Promise, JsValue> {
        let Some(l) = self.gpu.layout().drive else {
            return Err(err("readDrive: no vehicle has raycast wheels"));
        };
        let misc = self.gpu.misc_buffer().clone();
        let at = u64::from(l.vehicle(0)) * 16;
        let len = u64::from(l.stride) * 16 * self.vehicles.len() as u64;
        let drives: Vec<Option<Drive>> = self
            .vehicles
            .iter()
            .map(|v| v.sim.drive.as_deref().cloned())
            .collect();
        let stride = l.stride as usize * 4;
        let step = self.steps as f64;
        let poses = self.drive_poses.clone();
        Ok(crate::read::read_ranges_on(
            self.device.clone(),
            self.queue.clone(),
            vec![(misc, at, len)],
            move |w| {
                let mut read = Vec::with_capacity(drives.len());
                let list: Vec<serde_json::Value> = drives
                    .into_iter()
                    .enumerate()
                    .map(|(v, d)| match d {
                        Some(mut d) => {
                            engine_exec::explicit_plan::drive::decode(
                                &w[v * stride..(v + 1) * stride],
                                &mut d,
                            );
                            let j = pose_json(&d);
                            let f = |k: &str| {
                                let a = &j[k];
                                Vec3::new(
                                    a[0].as_f64().unwrap_or(0.0) as f32,
                                    a[1].as_f64().unwrap_or(0.0) as f32,
                                    a[2].as_f64().unwrap_or(0.0) as f32,
                                )
                            };
                            read.push(Some(Pose {
                                p: f("centre"),
                                fwd: f("forward"),
                                up: f("up"),
                                speed: d.speed,
                            }));
                            j
                        }
                        None => {
                            read.push(None);
                            serde_json::Value::Null
                        }
                    })
                    .collect();
                *poses.borrow_mut() = read;
                Ok(super::js(
                    serde_json::json!({ "step": step, "vehicles": list }),
                ))
            },
        ))
    }
}
