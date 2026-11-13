//! The vehicle test lab (`vehicle-lab.html`): `rubble::vehicle_lab` on the page's device.
//! The page forwards its controls as JSON commands, copies the car's buffers back each
//! step and hands the bytes to [`WebVehicleLab::ingest`]; every number it shows comes
//! from [`WebVehicleLab::stats`].

use crate::{err, res, WebWorld};
use engine_softbody::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings, JbeamWheels};
use rubble::showcase_scene::jbeam_vehicle_on;
use rubble::vehicle_lab::{lab_car, Lab, LabOptions, MATERIALS};
use rubble::GpuContext;
use wasm_bindgen::prelude::*;

fn js(v: &serde_json::Value) -> JsValue {
    use serde::Serialize;
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .unwrap_or(JsValue::NULL)
}

/// The lab's state beside its World.
#[wasm_bindgen]
pub struct WebVehicleLab {
    lab: Lab,
}

/// The vehicle lab on a new device: `[WebWorld, WebVehicleLab]`. `spec`: `{ vehicle:
/// { truck } | { jbeam: [[name, text]], pc } (on raycast wheels), drag (default true),
/// centredDamping (the solver's centred damping correction; default its own),
/// gpuTiming ('off' | 'step' | 'phases'), scene ('flat' | 'mountain'), surface (the car's
/// opt-in surface contact), parked (a second car broadside ahead), shapes and planes (the
/// coupling's room per car), sleep, mountain: { seed, steepness, roughness, … }, parts:
/// { names, start, beams } }` (`rubble::vehicle_lab::LabOptions::from_json`).
#[wasm_bindgen(js_name = createVehicleLab)]
pub async fn create_vehicle_lab(spec: JsValue) -> Result<js_sys::Array, JsValue> {
    let spec: serde_json::Value = serde_wasm_bindgen::from_value(spec).map_err(err)?;
    let drag = spec.get("drag").and_then(|x| x.as_bool()).unwrap_or(true);
    let d = ExplicitBeamsSettings::default();
    let settings = ExplicitBeamsSettings {
        fast_inv_sqrt: true,
        drag,
        turbulence: drag,
        centred_damping: spec
            .get("centredDamping")
            .and_then(|x| x.as_bool())
            .unwrap_or(d.centred_damping),
        ..d
    };
    let v = spec
        .get("vehicle")
        .ok_or_else(|| err("createVehicleLab: no vehicle"))?;
    let template = if let Some(truck) = v.get("truck").and_then(|t| t.as_str()) {
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
            "createVehicleLab: the vehicle needs `truck` or `jbeam`",
        ));
    };
    let opts = LabOptions::from_json(&spec);
    let timing = match spec.get("gpuTiming").and_then(|t| t.as_str()) {
        Some(t) => Some(crate::defs::gpu_timing(t)?),
        None => None,
    };
    let (info, device, queue) = crate::web_device().await?;
    {
        // The vehicles' kernels compiled without blocking first (as `createVehicleTown`).
        let car = lab_car(&template, &opts, None);
        let fleet = vec![car; if opts.parked { 2 } else { 1 }];
        let specs = engine_webgpu::ExplicitBeamsGpu::compute_specs(&fleet, Some(opts.capacity));
        crate::warm_pipelines(&device, &specs, std::rc::Rc::new(|_: &str, _: f64| {})).await;
    }
    let (world, lab) = res(Lab::build(
        GpuContext::Shared {
            info,
            device,
            queue,
        },
        &template,
        opts,
        |d| {
            if let Some(t) = timing {
                d.gpu_timing = t;
            }
        },
    ))?;
    crate::creation_check(&world).await?;
    Ok(js_sys::Array::of2(
        &JsValue::from(WebWorld { world }),
        &JsValue::from(WebVehicleLab { lab }),
    ))
}

/// The projectile material presets: `[[name, kg/m³]]`.
#[wasm_bindgen(js_name = vehicleLabMaterials)]
pub fn vehicle_lab_materials() -> JsValue {
    js(&serde_json::json!(MATERIALS
        .iter()
        .map(|m| serde_json::json!([m.0, m.1]))
        .collect::<Vec<_>>()))
}

#[wasm_bindgen]
impl WebVehicleLab {
    /// Where to read the car each step, as byte (offset, size) pairs: its nodes
    /// (`explicitNodesBuffer`), its origin record and its coupling records
    /// (`explicitMiscBuffer`), its beams (`explicitBeamsBuffer`).
    #[wasm_bindgen(js_name = readPlan)]
    pub fn read_plan(&self) -> Vec<f64> {
        let p = self.lab.read_plan();
        [p.nodes, p.origin, p.coupling, p.beams]
            .iter()
            .flat_map(|r| [r.0 as f64, r.1 as f64])
            .collect()
    }

    /// Before each `world.step()`: due script events and burst shots, the plates' stops,
    /// the mountain's chunks.
    #[wasm_bindgen(js_name = beforeStep)]
    pub fn before_step(&mut self, world: &mut WebWorld) -> Result<(), JsValue> {
        res(self.lab.before_step(&mut world.world))
    }

    /// One step's readback ([`Self::read_plan`]'s ranges as words; `beams` when read).
    pub fn ingest(
        &mut self,
        nodes: &[u32],
        origin: &[u32],
        coupling: &[u32],
        beams: Option<Vec<u32>>,
    ) -> Result<(), JsValue> {
        res(self.lab.ingest(nodes, origin, coupling, beams.as_deref()))
    }

    /// One command (`rubble::vehicle_lab::Bench::command`): fire, plates, plateControl,
    /// obstacle, clear, drive.
    pub fn command(&mut self, world: &mut WebWorld, cmd: JsValue) -> Result<JsValue, JsValue> {
        let cmd: serde_json::Value = serde_wasm_bindgen::from_value(cmd).map_err(err)?;
        Ok(js(&res(self.lab.command(&mut world.world, &cmd))?))
    }

    /// Start over in place (`rubble::vehicle_lab::Lab::restart`): the car back to its rest
    /// shape at a new start, every lab body gone, the scene laid out again; no device,
    /// buffer or pipeline is made. `spec`: `{ scene, mountain, start, crash }`.
    pub fn restart(&mut self, world: &mut WebWorld, spec: JsValue) -> Result<(), JsValue> {
        let spec: serde_json::Value = serde_wasm_bindgen::from_value(spec).map_err(err)?;
        res(self.lab.restart(&mut world.world, &spec))
    }

    /// A scripted run (`{ events: [{ t, …command }], duration }`).
    #[wasm_bindgen(js_name = setScript)]
    pub fn set_script(&mut self, script: JsValue) -> Result<(), JsValue> {
        let script: serde_json::Value = serde_wasm_bindgen::from_value(script).map_err(err)?;
        res(self.lab.bench.set_script(&script))
    }

    /// The lab's numbers (`rubble::vehicle_lab::Bench::stats`).
    pub fn stats(&self) -> JsValue {
        js(&self.lab.bench.stats())
    }

    /// The stats as the offline example's text table.
    pub fn table(&self) -> String {
        self.lab.bench.table()
    }

    /// Plate `i`'s (travel m, force N) samples, flat.
    #[wasm_bindgen(js_name = plateSeries)]
    pub fn plate_series(&self, i: u32) -> Vec<f32> {
        self.lab.bench.plate_series(i as usize)
    }

    /// The car now (`Bench::pose`): centre, velocity, heading, speed, node depth, readings.
    pub fn pose(&self) -> Vec<f64> {
        self.lab.bench.pose().to_vec()
    }

    /// Whether per-step readings matter now (`Bench::busy`): a projectile in its first
    /// seconds, a plate, a crash before its first contact, a script.
    pub fn busy(&self) -> bool {
        self.lab.bench.busy()
    }

    /// The lab's World time (s).
    pub fn time(&self) -> f64 {
        self.lab.bench.t
    }
}
