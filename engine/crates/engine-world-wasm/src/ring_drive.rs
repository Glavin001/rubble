//! The brick ring drive (`brick-ring-drive.html`): `rubble::brick_ring_drive`'s ring of
//! bricks with drivable explicit cars on raycast wheels, on the page's device. The page
//! forwards the keys (`setDriveInput`) and draws from the World's buffers.

use crate::{err, res, WebWorld};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels};
use engine_softbody::explicit_drive::DriveInput;
use rubble::brick_ring_drive::{brick_ring_drive, car_starts, ring_coupling, RingSpec};
use rubble::showcase_scene::jbeam_vehicle_on;
use rubble::GpuContext;
use wasm_bindgen::prelude::*;

fn num(v: &serde_json::Value, k: &str, d: f64) -> f64 {
    v.get(k).and_then(|x| x.as_f64()).unwrap_or(d)
}

fn js(v: serde_json::Value) -> JsValue {
    use serde::Serialize;
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .unwrap_or(JsValue::NULL)
}

/// The drive's ring from `spec.ring` (`{ scale, radius, courses, tier, rows, density }`,
/// each defaulting to [`drive_ring`]'s).
fn ring_of(spec: &serde_json::Value) -> RingSpec {
    let d = rubble::brick_ring_drive::drive_ring();
    let r = spec.get("ring").cloned().unwrap_or(serde_json::Value::Null);
    RingSpec {
        scale: num(&r, "scale", d.scale as f64) as f32,
        radius: num(&r, "radius", d.radius as f64) as f32,
        courses: num(&r, "courses", d.courses as f64) as u32,
        tier: num(&r, "tier", d.tier as f64) as u32,
        rows: num(&r, "rows", d.rows as f64) as u32,
        density: num(&r, "density", d.density as f64) as f32,
        ball: r.get("ball").and_then(|b| b.as_bool()).unwrap_or(false),
    }
}

/// `{ bricks, inner (m, the wall's inner face from the centre), height (m), half: [x, y,
/// z], ground, cars: [{ front, speed, turns }] }` of the drive for `spec` (as
/// `createBrickRingDrive`'s).
#[wasm_bindgen(js_name = brickRingLayout)]
pub fn brick_ring_layout(spec: JsValue) -> Result<JsValue, JsValue> {
    let spec: serde_json::Value = serde_wasm_bindgen::from_value(spec).map_err(err)?;
    let ring = ring_of(&spec);
    let cars = car_starts(
        &ring,
        num(&spec, "count", 1.0) as usize,
        num(&spec, "runUp", 30.0) as f32,
        num(&spec, "speed", 0.0) as f32,
    );
    Ok(js(serde_json::json!({
        "bricks": ring.bricks().len(),
        "inner": ring.inner(),
        "height": ring.height(),
        "half": ring.half(),
        "ground": rubble::brick_ring_drive::GROUND,
        "rows": ring.rows,
        "scale": ring.scale,
        "density": ring.density,
        "cars": cars.iter().map(|c| serde_json::json!({ "front": c.front, "speed": c.speed, "turns": c.turns })).collect::<Vec<_>>(),
    })))
}

#[wasm_bindgen]
impl WebWorld {
    /// The brick ring drive on a new device. `spec`: `{ vehicles: [{ jbeam: [[name, text]],
    /// pc } | { truck }] (car k is `vehicles[k % len]`, on raycast wheels), count (1 or 2),
    /// speed (m/s at spawn), runUp (m from the wall), ring: { scale, radius, courses, tier,
    /// rows, density }, sleep (default true), shapes (coupling room per car), gpuTiming ('off' | 'step' |
    /// 'phases'), drag (default true) }`. No vehicles: the ring alone.
    #[wasm_bindgen(js_name = createBrickRingDrive)]
    pub async fn create_brick_ring_drive(spec: JsValue) -> Result<WebWorld, JsValue> {
        let spec: serde_json::Value = serde_wasm_bindgen::from_value(spec).map_err(err)?;
        let drag = spec.get("drag").and_then(|x| x.as_bool()).unwrap_or(true);
        let settings = ExplicitBeamsSettings {
            fast_inv_sqrt: true,
            drag,
            turbulence: drag,
            ..Default::default()
        };
        let list = spec
            .get("vehicles")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut templates = Vec::with_capacity(list.len());
        for v in &list {
            templates.push(
                if let Some(truck) = v.get("truck").and_then(|t| t.as_str()) {
                    ExplicitBeams::from_truck_raycast(
                        truck,
                        None,
                        glam::Vec3::ZERO,
                        0.0,
                        glam::Vec3::ZERO,
                        settings,
                    )
                    .map_err(err)?
                    .0
                } else if let Some(files) = v.get("jbeam").and_then(|f| f.as_array()) {
                    let files: Vec<(&str, &str)> = files
                        .iter()
                        .map(|f| {
                            (
                                f.get(0).and_then(|x| x.as_str()).unwrap_or(""),
                                f.get(1).and_then(|x| x.as_str()).unwrap_or(""),
                            )
                        })
                        .collect();
                    let pc = v.get("pc").and_then(|t| t.as_str());
                    res(jbeam_vehicle_on(&files, pc, settings, JbeamWheels::Raycast))?.0
                } else {
                    return Err(err(
                        "createBrickRingDrive: a vehicle needs `truck` or `jbeam`",
                    ));
                },
            );
        }
        let ring = ring_of(&spec);
        let count = if templates.is_empty() {
            0
        } else {
            num(&spec, "count", 1.0) as usize
        };
        let cars = car_starts(
            &ring,
            count,
            num(&spec, "runUp", 30.0) as f32,
            num(&spec, "speed", 0.0) as f32,
        );
        let coupling = ring_coupling(num(
            &spec,
            "shapes",
            rubble::brick_ring_drive::DRIVE_SHAPES as f64,
        ) as u32);
        let timing = match spec.get("gpuTiming").and_then(|t| t.as_str()) {
            Some(t) => Some(crate::defs::gpu_timing(t)?),
            None => None,
        };
        let (info, device, queue) = crate::web_device().await?;
        if count > 0 {
            // The vehicles' kernels compiled without blocking first (as `createVehicleTown`).
            let fleet: Vec<ExplicitBeams> = (0..count)
                .map(|k| templates[k % templates.len()].clone())
                .collect();
            let specs =
                engine_webgpu::ExplicitBeamsGpu::compute_specs(&fleet, Some(coupling.capacity));
            crate::warm_pipelines(&device, &specs, std::rc::Rc::new(|_: &str, _: f64| {})).await;
        }
        let s = res(brick_ring_drive(
            GpuContext::Shared {
                info,
                device,
                queue,
            },
            ring,
            |d| {
                if let Some(t) = timing {
                    d.gpu_timing = t;
                }
                // `sleep` (default true): bodies sleep after 0.5 s still (demo.html's
                // ?sleep=on); false: the demo's no-sleep ring.
                if spec.get("sleep").and_then(|x| x.as_bool()).unwrap_or(true) {
                    d.solver.sleep_time = 0.5;
                }
            },
            &templates,
            &cars,
            &coupling,
        ))?;
        crate::creation_check(&s.world).await?;
        Ok(WebWorld::of(s.world))
    }

    /// Whether explicit vehicle `vehicle` drives on raycast wheels.
    #[wasm_bindgen(js_name = explicitDrivable)]
    pub fn explicit_drivable(&self, vehicle: u32) -> bool {
        self.world
            .explicit_vehicles_as_added()
            .get(vehicle as usize)
            .is_some_and(|v| v.drive.is_some())
    }

    /// Explicit vehicle `vehicle`'s driver input from the next World step on: forward
    /// −1..1 (negative reverses), steer −1..1 (positive right), brake and handbrake 0..1.
    #[wasm_bindgen(js_name = setExplicitDriveInput)]
    pub fn set_explicit_drive_input(
        &mut self,
        vehicle: u32,
        forward: f32,
        steer: f32,
        brake: f32,
        handbrake: f32,
    ) -> Result<(), JsValue> {
        if ![forward, steer, brake, handbrake]
            .iter()
            .all(|x| x.is_finite())
        {
            return Err(err("setExplicitDriveInput: input not finite"));
        }
        let x = self
            .world
            .explicit_vehicles_mut()
            .ok_or_else(|| err("setExplicitDriveInput: no explicit vehicles"))?;
        if vehicle as usize >= x.vehicle_count() {
            return Err(err(format!("setExplicitDriveInput: no vehicle {vehicle}")));
        }
        x.set_drive_input(
            vehicle as usize,
            DriveInput {
                forward,
                steer,
                brake,
                handbrake,
            },
        );
        Ok(())
    }

    /// Explicit vehicle `vehicle`'s raycast wheels' records in the misc buffer
    /// (`explicitMiscBuffer`, vec4 records), 4 per wheel: state (omega, spin angle, steer,
    /// length), centre (world), axle (unit, steered, left) and up. The showcase renderer
    /// places the wheels' pieces from them. Empty without raycast wheels.
    #[wasm_bindgen(js_name = explicitWheelRecords)]
    pub fn explicit_wheel_records(&self, vehicle: u32) -> Vec<u32> {
        use engine_exec::explicit_plan::drive::{VEHICLE_RECS, WHEEL_RECS};
        let (Some(l), Some(d)) = (
            self.world
                .explicit_vehicles()
                .and_then(|g| g.layout().drive),
            self.world
                .explicit_vehicles_as_added()
                .get(vehicle as usize)
                .and_then(|v| v.drive.as_ref()),
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
}
