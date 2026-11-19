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
/// opt-in surface contact), parked (a second car broadside ahead), cars (1 to 8 copies of the car: the multi-car
/// benchmark), shapes and planes (the
/// coupling's room per car), sleep, mountain: { seed, steepness, roughness, … }, parts:
/// { names, start, beams } }` (`rubble::vehicle_lab::LabOptions::from_json`).
#[wasm_bindgen(js_name = createVehicleLab)]
pub async fn create_vehicle_lab(spec: JsValue, progress: Option<js_sys::Function>) -> Result<js_sys::Array, JsValue> {
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
        let fleet = vec![car; opts.cars as usize + usize::from(opts.parked)];
        let mut specs = engine_webgpu::ExplicitBeamsGpu::compute_specs(&fleet, Some(opts.capacity));
        if opts.self_split {
            engine_webgpu::ExplicitBeamsGpu::prefer_self_split(&mut specs);
        }
        crate::warm_pipelines(&device, &specs, crate::hear_fn(progress)).await.map_err(err)?;
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
    /// (`explicitMiscBuffer`), its beams (`explicitBeamsBuffer`), then the other cars'
    /// nodes (`explicitNodesBuffer`, read with the beams).
    #[wasm_bindgen(js_name = readPlan)]
    pub fn read_plan(&self) -> Vec<f64> {
        let p = self.lab.read_plan();
        [p.nodes, p.origin, p.coupling, p.beams, p.fleet_nodes]
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

    /// One step's readback ([`Self::read_plan`]'s ranges as words; `beams` and
    /// `fleet_nodes` when read), copied at the lab's time `taken` ([`Self::time`] when the
    /// copy was encoded; none: now). A readback arrives frames later: the camera needs the
    /// state's own time.
    pub fn ingest(
        &mut self,
        nodes: &[u32],
        origin: &[u32],
        coupling: &[u32],
        beams: Option<Vec<u32>>,
        taken: Option<f64>,
        fleet_nodes: Option<Vec<u32>>,
    ) -> Result<(), JsValue> {
        res(self.lab.ingest_at(nodes, origin, coupling, beams.as_deref(), taken, fleet_nodes.as_deref()))
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

    /// Vehicles in the World (the lab's cars and the parked car).
    #[wasm_bindgen(js_name = vehicleCount)]
    pub fn vehicle_count(&self) -> u32 {
        self.lab.vehicle_count() as u32
    }

    /// The camera's readback of car `k` (`Lab::follow_plan`): node buffer byte offset and
    /// bytes, misc buffer byte offset of its 16-byte origin record; empty: no such car.
    #[wasm_bindgen(js_name = followPlan)]
    pub fn follow_plan(&self, k: u32) -> Vec<f64> {
        self.lab.follow_plan(k as usize).map_or(Vec::new(), |p| p.iter().map(|&x| x as f64).collect())
    }

    /// Car `k`'s nodes and origin record copied at the lab's time `taken`
    /// (`Lab::ingest_follow`): what the camera follows.
    #[wasm_bindgen(js_name = ingestFollow)]
    pub fn ingest_follow(&mut self, k: u32, nodes: &[u32], origin: &[u32], taken: f64) -> Result<(), JsValue> {
        res(self.lab.ingest_follow(k as usize, nodes, origin, taken))
    }

    /// The followed car (`Lab::follow_pose`): centre, velocity, heading, speed, the
    /// state's time, readings, the car.
    #[wasm_bindgen(js_name = followPose)]
    pub fn follow_pose(&self) -> Vec<f64> {
        self.lab.follow_pose().to_vec()
    }

    /// How far the camera may stand from the target toward the eye with no rock, wall or
    /// ground between (`Bench::clear_view`).
    #[wasm_bindgen(js_name = clearView)]
    pub fn clear_view(&self, tx: f64, ty: f64, tz: f64, ex: f64, ey: f64, ez: f64) -> f64 {
        use glam::DVec3;
        self.lab.bench.clear_view(DVec3::new(tx, ty, tz), DVec3::new(ex, ey, ez))
    }

    /// Whether per-step readings matter now (`Bench::busy`): a projectile in its first
    /// seconds, a plate, a crash before its first contact, a script.
    pub fn busy(&self) -> bool {
        self.lab.bench.busy()
    }

    /// The state now as a fixture file (`Lab::fixture_bytes`: the format `vehicle_lab
    /// --save-fixtures` writes and `--from-fixture` / `fast-check.sh FILE` replay
    /// natively), resolved once the snapshot's copy is mapped. The snapshot is taken
    /// now (after the work submitted so far).
    pub fn fixture(&self, world: &WebWorld) -> Result<js_sys::Promise, JsValue> {
        let g = world.world.explicit_vehicles().ok_or_else(|| err("fixture: no explicit vehicles"))?;
        let staged = g.stage_snapshot(&g.snapshot());
        // (The host state now; only the snapshot's bytes wait for the map.)
        let head = self.lab.fixture_bytes(&[]);
        Ok(wasm_bindgen_futures::future_to_promise(async move {
            crate::read::map_read(staged.buffer()).await?;
            let mut out = head;
            out.extend(staged.bytes());
            Ok(js_sys::Uint8Array::from(out.as_slice()).into())
        }))
    }

    /// The lab's World time (s).
    pub fn time(&self) -> f64 {
        self.lab.bench.t
    }

    /// The derby's readback each frame (`Lab::derby_plan`): every car's origin record and
    /// drive records in `explicitMiscBuffer`, as byte (offset, size) pairs; empty outside
    /// the Derby scene.
    #[wasm_bindgen(js_name = derbyPlan)]
    pub fn derby_plan(&self, world: &WebWorld) -> Vec<f64> {
        self.lab.derby_plan(&world.world).iter().map(|&x| x as f64).collect()
    }

    /// One [`Self::derby_plan`] reading: the cars' poses for the derby's AI and round.
    #[wasm_bindgen(js_name = ingestDerby)]
    pub fn ingest_derby(&mut self, world: &WebWorld, origins: &[u32], drive: &[u32]) {
        self.lab.ingest_derby(&world.world, origins, drive);
    }

    /// The derby's round (`rubble::vehicle_lab::derby::DerbyGame::json`): time, cars left,
    /// each car's state, the winner; null outside the Derby scene.
    pub fn derby(&self) -> JsValue {
        js(&self.lab.derby_json())
    }
}
