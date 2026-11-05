//! Vehicles and soft-body vehicles (`World::add_vehicle`, `add_soft_vehicle`,
//! `vehicle_damage`, ...), and the read-only records renderers and HUDs need.

use crate::defs::{self, Fields};
use crate::{err, res, WebWorld};
use rubble::*;
use wasm_bindgen::prelude::*;

fn from_js<T: serde::de::DeserializeOwned>(what: &str, v: JsValue) -> Result<T, JsValue> {
    let v: serde_json::Value = serde_wasm_bindgen::from_value(v).map_err(err)?;
    serde_json::from_value(v).map_err(|e| err(format!("{what}: {e}")))
}

fn js(v: serde_json::Value) -> JsValue {
    use serde::Serialize;
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .unwrap_or(JsValue::NULL)
}

#[wasm_bindgen]
impl WebWorld {
    /// The world a vehicle scene needs (`rubble::play::vehicles::world_def`) on its own
    /// device: `def` is the scene and the page's switches (`VehicleSceneDef`, camelCase).
    #[wasm_bindgen(js_name = createVehicleScene)]
    pub async fn create_vehicle_scene(def: JsValue) -> Result<WebWorld, JsValue> {
        let def: rubble::play::vehicles::VehicleSceneDef = from_js("VehicleSceneDef", def)?;
        let wd = res(rubble::play::vehicles::world_def(&def))?;
        let (info, device, queue) = crate::web_device().await?;
        let world = res(World::new(
            GpuContext::Shared {
                info,
                device,
                queue,
            },
            &wd,
        ))?;
        crate::creation_check(&world).await?;
        Ok(WebWorld { world })
    }

    /// Place a vehicle scene in this (empty) world (`rubble::play::vehicles::place`).
    /// Returns `Placed` (camelCase): each collider's page kind, the vehicles, the
    /// cannonballs, the street's blocks and parked cars, the range's pads, meteors and
    /// markers, the run's meteors, the bridge's bolts.
    #[wasm_bindgen(js_name = placeVehicleScene)]
    pub fn place_vehicle_scene(&mut self, def: JsValue) -> Result<JsValue, JsValue> {
        let def: rubble::play::vehicles::VehicleSceneDef = from_js("VehicleSceneDef", def)?;
        let placed = res(rubble::play::vehicles::place(&mut self.world, &def))?;
        Ok(js(serde_json::to_value(placed).map_err(err)?))
    }

    /// A vehicle (`VehicleDesc`, as engine-vehicle serialises it: the presets
    /// JSON) with its chassis origin (its centre of mass) at `position`.
    #[wasm_bindgen(js_name = addVehicle)]
    pub fn add_vehicle(
        &mut self,
        desc: JsValue,
        position: &[f32],
        rotation: &[f32],
    ) -> Result<u32, JsValue> {
        let desc: VehicleDesc = from_js("VehicleDesc", desc)?;
        Ok(res(self
            .world
            .add_vehicle(&desc, crate::vec3(position)?, crate::vec4(rotation)?))?
        .0)
    }

    /// Driver input `{ forward, steer, brake, handbrake, gear }` (missing:
    /// 0, automatic gearbox).
    #[wasm_bindgen(js_name = setVehicleInput)]
    pub fn set_vehicle_input(&mut self, vehicle: u32, input: JsValue) -> Result<(), JsValue> {
        let mut f = Fields::of("VehicleInput", input)?;
        let mut i = VehicleInput::default();
        f.f32("forward", &mut i.forward)?;
        f.f32("steer", &mut i.steer)?;
        f.f32("brake", &mut i.brake)?;
        f.f32("handbrake", &mut i.handbrake)?;
        if let Some(g) = f.take("gear") {
            let g = g
                .as_i64()
                .ok_or_else(|| err("VehicleInput.gear: expected an integer or null"))?;
            i.gear = GearRequest::Gear(g as i32);
        }
        f.finish()?;
        self.world.set_vehicle_input(VehicleHandle(vehicle), i);
        Ok(())
    }

    /// Sever or switch off a wheel or part: `kind` is `'detachWheel'`,
    /// `'disableWheel'`, `'detachPart'` or `'disablePart'`.
    #[wasm_bindgen(js_name = vehicleDamage)]
    pub fn vehicle_damage(&mut self, vehicle: u32, kind: &str, index: u32) -> Result<(), JsValue> {
        let d = match kind {
            "detachWheel" => VehicleDamage::DetachWheel(index),
            "disableWheel" => VehicleDamage::DisableWheel(index),
            "detachPart" => VehicleDamage::DetachPart(index),
            "disablePart" => VehicleDamage::DisablePart(index),
            _ => return Err(err(format!("vehicleDamage: unknown kind {kind}"))),
        };
        res(self.world.vehicle_damage(VehicleHandle(vehicle), d))
    }

    /// Functional overrides `{ wheelMask, driveMask, steerMask, brakeMask,
    /// engineConnected }` (missing: all on).
    #[wasm_bindgen(js_name = vehicleSetFunctionalState)]
    pub fn vehicle_set_functional_state(
        &mut self,
        vehicle: u32,
        state: JsValue,
    ) -> Result<(), JsValue> {
        let mut f = Fields::of("FunctionalState", state)?;
        let mut s = FunctionalState::default();
        f.u32("wheelMask", &mut s.wheel_mask)?;
        f.u32("driveMask", &mut s.drive_mask)?;
        f.u32("steerMask", &mut s.steer_mask)?;
        f.u32("brakeMask", &mut s.brake_mask)?;
        f.bool("engineConnected", &mut s.engine_connected)?;
        f.finish()?;
        res(self
            .world
            .vehicle_set_functional_state(VehicleHandle(vehicle), s))
    }

    #[wasm_bindgen(js_name = vehicleCount)]
    pub fn vehicle_count(&self) -> u32 {
        self.world.sim().vehicle_host().vehicle_count()
    }

    /// Wheel records in use (the instance count for drawing every wheel).
    #[wasm_bindgen(js_name = wheelCount)]
    pub fn wheel_count(&self) -> u32 {
        self.world.sim().vehicle_host().wheel_count()
    }

    /// `{ desc, slots: { vehicle, wheelBase, partBase, ... }, wheelSlots,
    /// chassis, firstPartBody, origin }`: where a vehicle's records and bodies
    /// live (body slots), and its design-space centre of mass at spawn.
    #[wasm_bindgen(js_name = vehicleEntry)]
    pub fn vehicle_entry(&self, vehicle: u32) -> Result<JsValue, JsValue> {
        let host = self.world.sim().vehicle_host();
        if vehicle >= host.vehicle_count() {
            return Err(err(format!("no vehicle {vehicle}")));
        }
        let v = VehicleHandle(vehicle);
        let s = host.slots(v);
        Ok(js(serde_json::json!({
            "desc": host.desc(v),
            "slots": {"vehicle": s.vehicle, "wheelBase": s.wheel_base, "curvePointBase": s.curve_point_base,
                "diffBase": s.diff_base, "antiRollBase": s.anti_roll_base, "partBase": s.part_base,
                "bodyBase": s.body_base, "colliderBase": s.collider_base, "aabbBlock": s.aabb_block, "jointBase": s.joint_base},
            "wheelSlots": host.wheel_slots(v),
            "chassis": host.chassis_body(v),
            "firstPartBody": host.part_body(v, 0),
            "origin": host.origin(v),
        })))
    }

    /// Region bases (16-byte slots) of the vehicle pool: `{ vehicleBase,
    /// wheelBase, curveBase, diffBase, antiRollBase, partBase, rayBase,
    /// totalSlots }`.
    #[wasm_bindgen(js_name = vehicleLayout)]
    pub fn vehicle_layout(&self) -> JsValue {
        let l = self.world.sim().vehicle_host().capacity.layout();
        js(serde_json::json!({
            "vehicleBase": l.vehicle_base, "wheelBase": l.wheel_base, "curveBase": l.curve_base,
            "diffBase": l.diff_base, "antiRollBase": l.anti_roll_base, "partBase": l.part_base,
            "rayBase": l.ray_base, "totalSlots": l.total_slots,
        }))
    }

    /// The vehicle pool `GPUBuffer` (vehicle, wheel and part records), for
    /// renderers.
    #[wasm_bindgen(js_name = vehiclePoolBuffer)]
    pub fn vehicle_pool_buffer(&self) -> JsValue {
        crate::gpu_buffer(self.world.sim().vehicle_pool_buffer())
    }

    /// A copy of the slots `vehicleMountLoads` decodes, resolved once mapped
    /// (never waits in place).
    #[wasm_bindgen(js_name = readMountLoads)]
    pub fn read_mount_loads(&self, vehicle: u32) -> Result<js_sys::Promise, JsValue> {
        let host = self.world.sim().vehicle_host();
        if vehicle >= host.vehicle_count() {
            return Err(err(format!("no vehicle {vehicle}")));
        }
        let pool = self.world.sim().vehicle_pool_buffer().clone();
        let ranges = host
            .mount_load_slots(VehicleHandle(vehicle))
            .iter()
            .map(|&(first, n)| (pool.clone(), u64::from(first) * 16, u64::from(n) * 16))
            .collect();
        Ok(crate::read::read_ranges(&self.world, ranges))
    }

    /// Mount loads from a `readMountLoads` copy: parts, then wheels, `{ index,
    /// wheel, force, torque, breakForce, breakTorque, peakRatio }`.
    #[wasm_bindgen(js_name = vehicleMountLoads)]
    pub fn vehicle_mount_loads(&self, vehicle: u32, words: &[u32]) -> Result<JsValue, JsValue> {
        let host = self.world.sim().vehicle_host();
        let v = VehicleHandle(vehicle);
        let [(_, parts), (_, wheels)] = host.mount_load_slots(v);
        let slots: Vec<[u32; 4]> = words
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        if slots.len() != (parts + wheels) as usize {
            return Err(err("vehicleMountLoads: not this vehicle's copy"));
        }
        let (p, w) = slots.split_at(parts as usize);
        let loads = host.decode_mount_loads(v, p, w);
        Ok(js(serde_json::to_value(loads).map_err(err)?))
    }

    // ---- soft bodies ----

    /// A soft vehicle (`SoftVehicleDesc`, serde JSON) at `position`; `options`:
    /// `SpawnOptions` (serde JSON, missing fields default). With wheels its
    /// rigid core is a vehicle (`softVehicle`).
    #[wasm_bindgen(js_name = addSoftVehicle)]
    pub fn add_soft_vehicle(
        &mut self,
        desc: JsValue,
        position: &[f32],
        rotation: &[f32],
        options: JsValue,
    ) -> Result<u32, JsValue> {
        let desc: SoftVehicleDesc = from_js("SoftVehicleDesc", desc)?;
        let options: SpawnOptions = if options.is_undefined() || options.is_null() {
            SpawnOptions::default()
        } else {
            from_js("SpawnOptions", options)?
        };
        Ok(res(self.world.add_soft_vehicle(
            &desc,
            crate::vec3(position)?,
            crate::vec4(rotation)?,
            &options,
        ))?
        .0)
    }

    /// The vehicle driving a soft body's rigid core (`undefined`: none).
    #[wasm_bindgen(js_name = softVehicle)]
    pub fn soft_vehicle(&self, soft: u32) -> Option<u32> {
        self.world.soft_vehicle(SoftHandle(soft)).map(|v| v.0)
    }

    /// `{ name, firstBody, particles, nodes, firstBeam, beams, firstTri, tris,
    /// mass, warnings, core: { vehicle, chassis } | null, rigidParts: [{ name,
    /// body, joints: [start, end) }] }`.
    #[wasm_bindgen(js_name = softEntry)]
    pub fn soft_entry(&self, soft: u32) -> Result<JsValue, JsValue> {
        let host = self.world.sim().soft_host();
        let e = host
            .entries
            .get(soft as usize)
            .ok_or_else(|| err(format!("no soft body {soft}")))?;
        let d = &e.desc;
        Ok(js(serde_json::json!({
            "name": d.name,
            "firstBody": e.first_body, "particles": e.particles, "nodes": d.nodes.len(),
            "firstBeam": e.first_beam, "beams": e.beams,
            "firstTri": e.first_tri, "tris": e.tris,
            "mass": d.total_mass(), "warnings": d.warnings,
            "core": e.core.map(|(v, chassis)| serde_json::json!({"vehicle": v.0, "chassis": chassis})),
            "rigidParts": e.rigid_parts.iter().zip(&d.rigid_parts)
                .map(|((body, j), p)| serde_json::json!({"name": p.name, "body": body, "joints": [j.start, j.end]}))
                .collect::<Vec<_>>(),
        })))
    }

    /// The soft solver's settings (`SoftSettings`, serde JSON).
    #[wasm_bindgen(js_name = softSettings)]
    pub fn soft_settings(&self) -> JsValue {
        js(serde_json::to_value(self.world.sim().soft_host().settings).unwrap_or_default())
    }

    /// Replace the soft solver's settings: the fields given (snake_case, as
    /// `SoftSettings` serialises) over the current ones.
    #[wasm_bindgen(js_name = setSoftSettings)]
    pub fn set_soft_settings(&mut self, settings: JsValue) -> Result<(), JsValue> {
        let mut cur = serde_json::to_value(self.world.sim().soft_host().settings).map_err(err)?;
        let given = defs::object(settings)?;
        let obj = cur.as_object_mut().expect("SoftSettings is an object");
        for (k, v) in given {
            if !obj.contains_key(&k) {
                return Err(err(format!("SoftSettings: unknown field {k}")));
            }
            obj.insert(k, v);
        }
        let s: SoftSettings = serde_json::from_value(cur).map_err(err)?;
        res(self.world.set_soft_settings(s))
    }
}

/// `SoftSettings::calibrated_for(hz)`: the demos' soft tuning for a step rate.
#[wasm_bindgen(js_name = softCalibratedSettings)]
pub fn soft_calibrated_settings(hz: f32) -> JsValue {
    js(serde_json::to_value(SoftSettings::calibrated_for(hz)).unwrap_or_default())
}
