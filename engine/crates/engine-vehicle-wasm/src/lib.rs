//! wasm-bindgen face of `engine-vehicle` for the WebGPU demo (`engine-webgpu/vehicle.mjs`).
//!
//! The browser runs the same [`VehicleHost`] and packer as the native hosts. Every
//! call that touches GPU records returns an op stream (see [`Recorder`]) that the
//! JS backend replays into its buffers; reads come back through a JS callback.
//! Build: `wasm-pack build crates/engine-vehicle-wasm --target web
//! --out-dir ../engine-webgpu/generated/vehicle_wasm` (run by `wgsl_export`).

use engine_schema::vehicle::{BODY_VEHICLE, LAYOUT};
use engine_schema::{BodyState, Collider, Joint, JOINT_TYRE};
use engine_vehicle::pack::{self, Slot, WHEEL_RIM_MATERIAL};
use engine_vehicle::{
    FunctionalState, GearRequest, VehicleBackend, VehicleCapacity, VehicleDesc, VehicleHandle, VehicleHost,
    VehicleInput, WheelDesc,
};
use serde_json::json;
use wasm_bindgen::prelude::*;

/// Op codes of the stream: `[op, arg, len, words × len]`.
pub const OP_APPEND_BODY: u32 = 1;
pub const OP_APPEND_COLLIDER: u32 = 2;
pub const OP_APPEND_JOINT: u32 = 3;
/// `arg` = body index.
pub const OP_WRITE_BODY: u32 = 4;
/// `arg` = collider index.
pub const OP_WRITE_COLLIDER: u32 = 5;
/// `arg` = first float4 slot of gVehicleData.
pub const OP_WRITE_SLOTS: u32 = 6;
/// `arg` = byte offset into gVehicleData.
pub const OP_WRITE_BYTES: u32 = 7;

/// [`VehicleBackend`] that records writes for a JS host to replay.
pub struct Recorder<'a> {
    pub bodies: u32,
    pub colliders: u32,
    pub joints: u32,
    pub ops: Vec<u32>,
    read: Option<&'a dyn Fn(u32, u32) -> Vec<u32>>,
}

impl<'a> Recorder<'a> {
    pub fn new(counts: [u32; 3], read: Option<&'a dyn Fn(u32, u32) -> Vec<u32>>) -> Self {
        Self { bodies: counts[0], colliders: counts[1], joints: counts[2], ops: Vec::new(), read }
    }

    fn op(&mut self, op: u32, arg: u32, words: &[u32]) {
        self.ops.extend_from_slice(&[op, arg, words.len() as u32]);
        self.ops.extend_from_slice(words);
    }
}

impl VehicleBackend for Recorder<'_> {
    fn body_count(&self) -> u32 {
        self.bodies
    }
    fn collider_count(&self) -> u32 {
        self.colliders
    }
    fn joint_count(&self) -> u32 {
        self.joints
    }
    fn append_bodies(&mut self, bodies: &[BodyState]) {
        for b in bodies {
            self.op(OP_APPEND_BODY, self.bodies, bytemuck::cast_slice(std::slice::from_ref(b)));
            self.bodies += 1;
        }
    }
    fn append_colliders(&mut self, colliders: &[Collider]) {
        for c in colliders {
            self.op(OP_APPEND_COLLIDER, self.colliders, bytemuck::cast_slice(std::slice::from_ref(c)));
            self.colliders += 1;
        }
    }
    fn append_joints(&mut self, joints: &[Joint]) {
        for j in joints {
            self.op(OP_APPEND_JOINT, self.joints, bytemuck::cast_slice(std::slice::from_ref(j)));
            self.joints += 1;
        }
    }
    fn write_body(&mut self, index: u32, body: &BodyState) {
        self.op(OP_WRITE_BODY, index, bytemuck::cast_slice(std::slice::from_ref(body)));
    }
    fn write_collider(&mut self, index: u32, collider: &Collider) {
        self.op(OP_WRITE_COLLIDER, index, bytemuck::cast_slice(std::slice::from_ref(collider)));
    }
    fn write_slots(&mut self, slot: u32, data: &[Slot]) {
        self.op(OP_WRITE_SLOTS, slot, data.as_flattened());
    }
    fn write_curve_points(&mut self, byte_offset: u64, points: &[[f32; 2]]) {
        let words: Vec<u32> = points.as_flattened().iter().map(|x| x.to_bits()).collect();
        self.op(OP_WRITE_BYTES, byte_offset as u32, &words);
    }
    fn read_slots(&self, slot: u32, count: u32) -> Vec<Slot> {
        let read = self.read.expect("this call needs a gVehicleData reader");
        let words = read(slot, count);
        assert_eq!(words.len(), count as usize * 4, "reader returned the wrong length");
        words.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect()
    }
}

fn err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

fn parse<T: serde::de::DeserializeOwned>(json: &str) -> Result<T, JsError> {
    serde_json::from_str(json).map_err(err)
}

/// `read(slot, count)` → Uint32Array of `count` float4 slots of gVehicleData.
fn js_reader(read: &js_sys::Function) -> impl Fn(u32, u32) -> Vec<u32> + '_ {
    move |slot, count| {
        let v = read.call2(&JsValue::NULL, &slot.into(), &count.into()).expect("vehicle data reader threw");
        js_sys::Uint32Array::from(v).to_vec()
    }
}

/// Layout constants (`slang/physics/vehicle_layout.slang`) plus the record ids the
/// demo needs, as a JSON object.
#[wasm_bindgen]
pub fn layout_constants() -> String {
    let mut m: serde_json::Map<String, serde_json::Value> = LAYOUT.iter().map(|&(n, v)| (n.into(), v.into())).collect();
    m.insert("BODY_VEHICLE".into(), BODY_VEHICLE.into());
    m.insert("JOINT_TYRE".into(), JOINT_TYRE.into());
    m.insert("WHEEL_RIM_MATERIAL".into(), WHEEL_RIM_MATERIAL.into());
    serde_json::Value::Object(m).to_string()
}

/// Op codes of the stream (JSON object, `OP_*` names).
#[wasm_bindgen]
pub fn op_codes() -> String {
    json!({"OP_APPEND_BODY": OP_APPEND_BODY, "OP_APPEND_COLLIDER": OP_APPEND_COLLIDER, "OP_APPEND_JOINT": OP_APPEND_JOINT,
        "OP_WRITE_BODY": OP_WRITE_BODY, "OP_WRITE_COLLIDER": OP_WRITE_COLLIDER, "OP_WRITE_SLOTS": OP_WRITE_SLOTS,
        "OP_WRITE_BYTES": OP_WRITE_BYTES})
    .to_string()
}

/// Pool sizes a descriptor needs (JSON, camelCase).
#[wasm_bindgen]
pub fn requirements(desc_json: &str) -> Result<String, JsError> {
    let r = pack::requirements(&parse::<VehicleDesc>(desc_json)?);
    Ok(json!({"wheelSlots": r.wheel_slots, "curvePoints": r.curve_points, "diffs": r.diffs,
        "antiRolls": r.anti_rolls, "parts": r.parts, "bodies": r.bodies, "colliders": r.colliders})
    .to_string())
}

#[wasm_bindgen]
pub struct WasmVehicleHost {
    host: VehicleHost,
}

impl WasmVehicleHost {
    fn record(&mut self, f: impl FnOnce(&mut VehicleHost, &mut Recorder)) -> Vec<u32> {
        let mut r = Recorder::new([0; 3], None);
        f(&mut self.host, &mut r);
        r.ops
    }
}

#[wasm_bindgen]
impl WasmVehicleHost {
    /// `cap` = [maxVehicles, maxWheels, maxCurvePoints, maxDiffs, maxAntiRolls, maxParts, maxRays].
    #[wasm_bindgen(constructor)]
    pub fn new(cap: &[u32], aabb_tail: u32) -> WasmVehicleHost {
        let capacity = VehicleCapacity {
            max_vehicles: cap[0],
            max_wheels: cap[1],
            max_curve_points: cap[2],
            max_diffs: cap[3],
            max_anti_rolls: cap[4],
            max_parts: cap[5],
            max_rays: cap[6],
        };
        WasmVehicleHost { host: VehicleHost::new(capacity, aabb_tail) }
    }

    /// Region bases (float4 slots) inside gVehicleData (JSON, camelCase).
    pub fn layout(&self) -> String {
        let l = self.host.capacity.layout();
        json!({"vehicleBase": l.vehicle_base, "wheelBase": l.wheel_base, "curveBase": l.curve_base,
            "diffBase": l.diff_base, "antiRollBase": l.anti_roll_base, "partBase": l.part_base,
            "rayBase": l.ray_base, "totalSlots": l.total_slots})
        .to_string()
    }

    #[wasm_bindgen(getter)]
    pub fn vehicle_count(&self) -> u32 {
        self.host.vehicle_count()
    }

    #[wasm_bindgen(getter)]
    pub fn wheel_count(&self) -> u32 {
        self.host.wheel_count()
    }

    #[wasm_bindgen(getter)]
    pub fn dirty(&self) -> bool {
        self.host.dirty
    }

    #[wasm_bindgen(setter)]
    pub fn set_dirty(&mut self, v: bool) {
        self.host.dirty = v;
    }

    #[wasm_bindgen(getter)]
    pub fn full_scan_queries(&self) -> bool {
        self.host.full_scan_queries
    }

    #[wasm_bindgen(setter)]
    pub fn set_full_scan_queries(&mut self, v: bool) {
        self.host.full_scan_queries = v;
    }

    /// Pack and allocate a vehicle; `counts` = [bodies, colliders, joints] already in the world.
    pub fn add_vehicle(&mut self, counts: &[u32], desc_json: &str, position: &[f32], rotation: &[f32]) -> Result<Vec<u32>, JsError> {
        let desc: VehicleDesc = parse(desc_json)?;
        let mut r = Recorder::new([counts[0], counts[1], counts[2]], None);
        let pos = [position[0], position[1], position[2]];
        let rot = [rotation[0], rotation[1], rotation[2], rotation[3]];
        self.host.add_vehicle(&mut r, &desc, pos, rot).map_err(err)?;
        Ok(r.ops)
    }

    /// Attach a wheel (design-space position) into the next spare slot.
    pub fn add_wheel(&mut self, h: u32, wheel_json: &str, read: &js_sys::Function) -> Result<Vec<u32>, JsError> {
        let wheel: WheelDesc = parse(wheel_json)?;
        let reader = js_reader(read);
        let mut r = Recorder::new([0; 3], Some(&reader));
        self.host.add_wheel(&mut r, VehicleHandle(h), &wheel).map_err(err)?;
        Ok(r.ops)
    }

    /// Replace a wheel's tuning in place.
    pub fn set_wheel_params(&mut self, h: u32, wheel: u32, wheel_json: &str, read: &js_sys::Function) -> Result<Vec<u32>, JsError> {
        let desc: WheelDesc = parse(wheel_json)?;
        let reader = js_reader(read);
        let mut r = Recorder::new([0; 3], Some(&reader));
        self.host.set_wheel_params(&mut r, VehicleHandle(h), wheel, &desc);
        Ok(r.ops)
    }

    /// Driver input; `gear` undefined = automatic.
    pub fn set_input(&mut self, h: u32, forward: f32, steer: f32, brake: f32, handbrake: f32, gear: Option<i32>) -> Vec<u32> {
        let gear = gear.map_or(GearRequest::Auto, GearRequest::Gear);
        let input = VehicleInput { forward, steer, brake, handbrake, gear };
        self.record(|host, r| host.set_input(r, VehicleHandle(h), input))
    }

    pub fn set_functional_state(&mut self, h: u32, wheel_mask: u32, drive_mask: u32, steer_mask: u32, brake_mask: u32, engine_connected: bool) -> Vec<u32> {
        let s = FunctionalState { wheel_mask, drive_mask, steer_mask, brake_mask, engine_connected };
        self.record(|host, r| host.set_functional_state(r, VehicleHandle(h), s))
    }

    pub fn detach_wheel(&mut self, h: u32, wheel: u32) -> Vec<u32> {
        self.record(|host, r| host.detach_wheel(r, VehicleHandle(h), wheel))
    }

    pub fn disable_wheel(&mut self, h: u32, wheel: u32) -> Vec<u32> {
        self.record(|host, r| host.disable_wheel(r, VehicleHandle(h), wheel))
    }

    pub fn detach_part(&mut self, h: u32, part: u32) -> Vec<u32> {
        self.record(|host, r| host.detach_part(r, VehicleHandle(h), part))
    }

    pub fn disable_part(&mut self, h: u32, part: u32) -> Vec<u32> {
        self.record(|host, r| host.disable_part(r, VehicleHandle(h), part))
    }

    pub fn wake(&mut self, h: u32) -> Vec<u32> {
        self.record(|host, r| host.wake(r, VehicleHandle(h)))
    }

    /// Decoded outputs (JSON, camelCase) read through `read(slot, count)`.
    pub fn read_state(&self, h: u32, read: &js_sys::Function) -> String {
        let reader = js_reader(read);
        let r = Recorder::new([0; 3], Some(&reader));
        // Through `Value` so f32s print as their exact f64 value (JS sees the same
        // numbers a Float32Array view would give).
        serde_json::to_value(self.host.read_state(&r, VehicleHandle(h))).expect("state serialises").to_string()
    }

    /// Bookkeeping the demo reads (JSON): descriptor, slots, wheel slots, bodies.
    pub fn entry(&self, h: u32) -> String {
        let v = VehicleHandle(h);
        let s = self.host.slots(v);
        json!({
            "desc": self.host.desc(v),
            "slots": {"vehicle": s.vehicle, "wheelBase": s.wheel_base, "curvePointBase": s.curve_point_base,
                "diffBase": s.diff_base, "antiRollBase": s.anti_roll_base, "partBase": s.part_base,
                "bodyBase": s.body_base, "colliderBase": s.collider_base, "aabbBlock": s.aabb_block, "jointBase": s.joint_base},
            "wheelSlots": self.host.wheel_slots(v),
            "chassis": self.host.chassis_body(v),
            "firstPartBody": self.host.part_body(v, 0),
        })
        .to_string()
    }

    pub fn chassis_body(&self, h: u32) -> u32 {
        self.host.chassis_body(VehicleHandle(h))
    }

    pub fn wheel_body(&self, h: u32, wheel: u32) -> u32 {
        self.host.wheel_body(VehicleHandle(h), wheel)
    }

    pub fn part_body(&self, h: u32, part: u32) -> u32 {
        self.host.part_body(VehicleHandle(h), part)
    }

    /// Copy of the bookkeeping (pair with the GPU records for a scene reset).
    pub fn save(&self) -> WasmVehicleHost {
        WasmVehicleHost { host: self.host.clone() }
    }

    pub fn restore(&mut self, snap: &WasmVehicleHost) {
        self.host = snap.host.clone();
        self.host.dirty = true;
    }
}

#[cfg(test)]
mod tests;
