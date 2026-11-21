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
pub(super) use engine_softbody::drive_ai::Pose;

/// Per vehicle its last read pose (`None`: no raycast wheels, or not read yet).
pub(super) type Poses = Rc<RefCell<Vec<Option<Pose>>>>;

/// The traffic AI lives in `engine_softbody::drive_ai` (shared with the vehicle lab's
/// derby), unchanged.
pub(super) use engine_softbody::drive_ai::{ai_cars, ai_input, AiCar};

/// A vehicle's pose from its wheels: centre, forward (the fixed wheels' mean
/// heading, else every wheel's), up, and the wheels as drawn.
pub(super) fn pose_json(d: &Drive) -> serde_json::Value {
    let (centre, fwd, up) = engine_softbody::drive_ai::wheel_frame(d, None);
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
