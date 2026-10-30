//! wasm-bindgen face of `engine-vehicle` for the WebGPU demo (`engine-webgpu/vehicle.mjs`).
//!
//! The browser runs the same [`VehicleHost`] and packer as the native hosts. Every
//! call that touches GPU records returns an op stream (see [`Recorder`]) that the
//! JS backend replays into its buffers; reads come back through a JS callback.
//! Build: `wasm-pack build crates/engine-vehicle-wasm --target web
//! --out-dir ../engine-webgpu/generated/vehicle_wasm` (run by `wgsl_export`).

use engine_schema::vehicle::{BODY_VEHICLE, LAYOUT};
use engine_schema::{BodyState, Collider, Joint, JOINT_FLAG_TYRE};
use engine_vehicle::pack::{self, Slot, WHEEL_RIM_MATERIAL};
use engine_vehicle::{
    FunctionalState, GearRequest, VehicleBackend, VehicleCapacity, VehicleDesc, VehicleHandle,
    VehicleHost, VehicleInput, WheelDesc,
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
/// `arg` = first float4 slot of gSoftData (binding 32).
pub const OP_WRITE_SOFT_SLOTS: u32 = 8;

/// [`VehicleBackend`] that records writes for a JS host to replay.
pub struct Recorder<'a> {
    pub bodies: u32,
    pub colliders: u32,
    pub joints: u32,
    pub ops: Vec<u32>,
    read: Option<&'a dyn Fn(u32, u32) -> Vec<u32>>,
    /// Records appended or written through this recorder (host mirrors for
    /// `SoftBackend::host_body` / `host_collider`).
    body_mirror: std::collections::HashMap<u32, BodyState>,
    collider_mirror: std::collections::HashMap<u32, Collider>,
}

impl<'a> Recorder<'a> {
    pub fn new(counts: [u32; 3], read: Option<&'a dyn Fn(u32, u32) -> Vec<u32>>) -> Self {
        Self {
            bodies: counts[0],
            colliders: counts[1],
            joints: counts[2],
            ops: Vec::new(),
            read,
            body_mirror: Default::default(),
            collider_mirror: Default::default(),
        }
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
            self.op(
                OP_APPEND_BODY,
                self.bodies,
                bytemuck::cast_slice(std::slice::from_ref(b)),
            );
            self.body_mirror.insert(self.bodies, *b);
            self.bodies += 1;
        }
    }
    fn append_colliders(&mut self, colliders: &[Collider]) {
        for c in colliders {
            self.op(
                OP_APPEND_COLLIDER,
                self.colliders,
                bytemuck::cast_slice(std::slice::from_ref(c)),
            );
            self.collider_mirror.insert(self.colliders, *c);
            self.colliders += 1;
        }
    }
    fn append_joints(&mut self, joints: &[Joint]) {
        for j in joints {
            self.op(
                OP_APPEND_JOINT,
                self.joints,
                bytemuck::cast_slice(std::slice::from_ref(j)),
            );
            self.joints += 1;
        }
    }
    fn write_body(&mut self, index: u32, body: &BodyState) {
        self.op(
            OP_WRITE_BODY,
            index,
            bytemuck::cast_slice(std::slice::from_ref(body)),
        );
        self.body_mirror.insert(index, *body);
    }
    fn write_collider(&mut self, index: u32, collider: &Collider) {
        self.op(
            OP_WRITE_COLLIDER,
            index,
            bytemuck::cast_slice(std::slice::from_ref(collider)),
        );
        self.collider_mirror.insert(index, *collider);
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
        assert_eq!(
            words.len(),
            count as usize * 4,
            "reader returned the wrong length"
        );
        words
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect()
    }
}

impl engine_softbody::SoftBackend for Recorder<'_> {
    fn write_soft_slots(&mut self, slot: u32, data: &[Slot]) {
        if !data.is_empty() {
            self.op(OP_WRITE_SOFT_SLOTS, slot, data.as_flattened());
        }
    }
    fn read_soft_slots(&self, slot: u32, count: u32) -> Vec<Slot> {
        self.read_slots(slot, count)
    }
    fn host_body(&self, index: u32) -> BodyState {
        *self
            .body_mirror
            .get(&index)
            .expect("body not written through this recorder")
    }
    fn host_collider(&self, index: u32) -> Collider {
        *self
            .collider_mirror
            .get(&index)
            .expect("collider not written through this recorder")
    }
}

mod soft;
pub use soft::*;
mod terrain;
pub use terrain::*;

fn err(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

fn parse<T: serde::de::DeserializeOwned>(json: &str) -> Result<T, JsError> {
    serde_json::from_str(json).map_err(err)
}

/// `read(slot, count)` → Uint32Array of `count` float4 slots of gVehicleData.
fn js_reader(read: &js_sys::Function) -> impl Fn(u32, u32) -> Vec<u32> + '_ {
    move |slot, count| {
        let v = read
            .call2(&JsValue::NULL, &slot.into(), &count.into())
            .expect("vehicle data reader threw");
        js_sys::Uint32Array::from(v).to_vec()
    }
}

/// Layout constants (`slang/physics/vehicle_layout.slang`) plus the record ids the
/// demo needs, as a JSON object.
#[wasm_bindgen]
pub fn layout_constants() -> String {
    let mut m: serde_json::Map<String, serde_json::Value> =
        LAYOUT.iter().map(|&(n, v)| (n.into(), v.into())).collect();
    m.insert("BODY_VEHICLE".into(), BODY_VEHICLE.into());
    m.insert("JOINT_FLAG_TYRE".into(), JOINT_FLAG_TYRE.into());
    m.insert(
        "JOINT_BYTES".into(),
        (std::mem::size_of::<Joint>() as u32).into(),
    );
    m.insert("WHEEL_RIM_MATERIAL".into(), WHEEL_RIM_MATERIAL.into());
    serde_json::Value::Object(m).to_string()
}

/// Character / query / projectile pool layout (`character_layout.slang`), the
/// continuous-collision words and flags a browser host writes (impulse records,
/// CCD and kinematic body flags, the event ring), as a JSON object.
#[wasm_bindgen]
pub fn motion_constants() -> String {
    use engine_schema as s;
    let mut m: serde_json::Map<String, serde_json::Value> =
        s::character::LAYOUT.iter().map(|&(n, v)| (n.into(), v.into())).collect();
    for (n, v) in [
        ("BODY_CCD", s::BODY_CCD),
        ("BODY_KINEMATIC", s::BodyFlags::Kinematic as u32),
        ("FLAG_SPEC_VELOCITY", s::FLAG_SPEC_VELOCITY),
        ("IMPULSE_BASE", s::IMPULSE_BASE),
        ("IMPULSE_COUNT", s::IMPULSE_COUNT),
        ("IMPULSE_FIRST", s::IMPULSE_FIRST),
        ("IMPULSE_WORDS", s::IMPULSE_WORDS),
        ("IMPULSE_CAP", s::IMPULSE_CAP),
        ("IMPULSE_AT_POINT", s::IMPULSE_AT_POINT),
        ("IMPULSE_SET_VELOCITY", s::IMPULSE_SET_VELOCITY),
        ("CCD_FALLBACKS", s::CCD_FALLBACKS),
        ("CCD_BODY_CAP", s::CCD_BODY_CAP),
        ("CCD_PAIR_CAP", s::CCD_PAIR_CAP),
        ("FAST_CAP", s::FAST_CAP),
        ("SCRATCH_WORDS", s::SCRATCH_WORDS),
        ("EVENT_RING_BASE", s::EVENT_RING_BASE),
        ("EVENT_RING_CAPACITY", s::EVENT_RING_CAPACITY),
        ("NO_INDEX", s::character::NO_INDEX),
    ] {
        m.insert(n.into(), v.into());
    }
    serde_json::Value::Object(m).to_string()
}

/// A character's record (16 float4 slots as words) from a `CharacterDesc` JSON
/// (snake_case fields; missing ones take the defaults).
#[wasm_bindgen]
pub fn character_record(desc_json: &str, inner_body: u32, x: f32, y: f32, z: f32) -> Result<Vec<u32>, JsError> {
    let mut v: serde_json::Value = parse(desc_json)?;
    let mut base = serde_json::to_value(engine_schema::character::CharacterDesc::default()).map_err(err)?;
    if let (Some(b), Some(o)) = (base.as_object_mut(), v.as_object_mut()) {
        for (k, val) in o.iter() {
            b.insert(k.clone(), val.clone());
        }
    }
    let d: engine_schema::character::CharacterDesc = serde_json::from_value(base).map_err(err)?;
    Ok(d.record(inner_body, [x, y, z]).iter().flatten().copied().collect())
}

/// A hull arena holding the unit cube (box pairs read it) and one heightfield,
/// as `[base, params × 4 (f32 bits), records…]`: the host uploads the records to
/// binding 18 and gives the heightfield collider those params
/// (engine_schema::heightfield; samples row-major, `nx × nz`).
#[wasm_bindgen]
pub fn height_field_arena(
    nx: u32,
    nz: u32,
    dx: f32,
    dz: f32,
    heights: &[f32],
) -> Result<Vec<u32>, JsError> {
    use engine_schema::heightfield::{height_field_collider_params, HeightField};
    let field = HeightField::new(nx, nz, dx, dz, heights.to_vec())
        .map_err(|e| JsError::new(&e.to_string()))?;
    let mut arena = engine_schema::convex::HullArena::new();
    let base = arena
        .insert_height_field(&field)
        .map_err(|e| JsError::new(&e.to_string()))?;
    let mut out = vec![base];
    out.extend(height_field_collider_params(base, &field).map(f32::to_bits));
    out.extend(arena.records().iter().flatten());
    Ok(out)
}

/// Default character descriptor (JSON).
#[wasm_bindgen]
pub fn character_default() -> String {
    serde_json::to_string(&engine_schema::character::CharacterDesc::default()).unwrap()
}

/// Op codes of the stream (JSON object, `OP_*` names).
#[wasm_bindgen]
pub fn op_codes() -> String {
    json!({"OP_APPEND_BODY": OP_APPEND_BODY, "OP_APPEND_COLLIDER": OP_APPEND_COLLIDER, "OP_APPEND_JOINT": OP_APPEND_JOINT,
        "OP_WRITE_BODY": OP_WRITE_BODY, "OP_WRITE_COLLIDER": OP_WRITE_COLLIDER, "OP_WRITE_SLOTS": OP_WRITE_SLOTS,
        "OP_WRITE_BYTES": OP_WRITE_BYTES, "OP_WRITE_SOFT_SLOTS": OP_WRITE_SOFT_SLOTS})
    .to_string()
}

/// Design-space centre of mass of a descriptor: where `add_vehicle` puts the
/// chassis body's origin (place a vehicle by a design point with it).
#[wasm_bindgen]
pub fn center_of_mass(desc_json: &str) -> Result<Vec<f32>, JsError> {
    Ok(parse::<VehicleDesc>(desc_json)?.center_of_mass().to_vec())
}

/// Pool sizes a descriptor needs (JSON, camelCase).
#[wasm_bindgen]
pub fn requirements(desc_json: &str) -> Result<String, JsError> {
    let r = pack::requirements(&parse::<VehicleDesc>(desc_json)?);
    Ok(
        json!({"wheelSlots": r.wheel_slots, "curvePoints": r.curve_points, "diffs": r.diffs,
        "antiRolls": r.anti_rolls, "parts": r.parts, "bodies": r.bodies, "colliders": r.colliders})
        .to_string(),
    )
}

#[wasm_bindgen]
pub struct WasmVehicleHost {
    pub(crate) host: VehicleHost,
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
        WasmVehicleHost {
            host: VehicleHost::new(capacity, aabb_tail),
        }
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
    pub fn add_vehicle(
        &mut self,
        counts: &[u32],
        desc_json: &str,
        position: &[f32],
        rotation: &[f32],
    ) -> Result<Vec<u32>, JsError> {
        let desc: VehicleDesc = parse(desc_json)?;
        let mut r = Recorder::new([counts[0], counts[1], counts[2]], None);
        let pos = [position[0], position[1], position[2]];
        let rot = [rotation[0], rotation[1], rotation[2], rotation[3]];
        self.host
            .add_vehicle(&mut r, &desc, pos, rot)
            .map_err(err)?;
        Ok(r.ops)
    }

    /// Attach a wheel (design-space position) into the next spare slot.
    pub fn add_wheel(
        &mut self,
        h: u32,
        wheel_json: &str,
        read: &js_sys::Function,
    ) -> Result<Vec<u32>, JsError> {
        let wheel: WheelDesc = parse(wheel_json)?;
        let reader = js_reader(read);
        let mut r = Recorder::new([0; 3], Some(&reader));
        self.host
            .add_wheel(&mut r, VehicleHandle(h), &wheel)
            .map_err(err)?;
        Ok(r.ops)
    }

    /// Replace a wheel's tuning in place.
    pub fn set_wheel_params(
        &mut self,
        h: u32,
        wheel: u32,
        wheel_json: &str,
        read: &js_sys::Function,
    ) -> Result<Vec<u32>, JsError> {
        let desc: WheelDesc = parse(wheel_json)?;
        let reader = js_reader(read);
        let mut r = Recorder::new([0; 3], Some(&reader));
        self.host
            .set_wheel_params(&mut r, VehicleHandle(h), wheel, &desc);
        Ok(r.ops)
    }

    /// Driver input; `gear` undefined = automatic.
    pub fn set_input(
        &mut self,
        h: u32,
        forward: f32,
        steer: f32,
        brake: f32,
        handbrake: f32,
        gear: Option<i32>,
    ) -> Vec<u32> {
        let gear = gear.map_or(GearRequest::Auto, GearRequest::Gear);
        let input = VehicleInput {
            forward,
            steer,
            brake,
            handbrake,
            gear,
        };
        self.record(|host, r| host.set_input(r, VehicleHandle(h), input))
    }

    pub fn set_functional_state(
        &mut self,
        h: u32,
        wheel_mask: u32,
        drive_mask: u32,
        steer_mask: u32,
        brake_mask: u32,
        engine_connected: bool,
    ) -> Vec<u32> {
        let s = FunctionalState {
            wheel_mask,
            drive_mask,
            steer_mask,
            brake_mask,
            engine_connected,
        };
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
        serde_json::to_value(self.host.read_state(&r, VehicleHandle(h)))
            .expect("state serialises")
            .to_string()
    }

    /// Mount loads (JSON array of `MountLoad`, parts then wheels) from the
    /// vehicle records `read(slot, count)` returns.
    pub fn read_mount_loads(&self, h: u32, read: &js_sys::Function) -> String {
        let reader = js_reader(read);
        let r = Recorder::new([0; 3], Some(&reader));
        serde_json::to_value(self.host.read_mount_loads(&r, VehicleHandle(h)))
            .expect("loads serialise")
            .to_string()
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
            // Design-space centre of mass at spawn: design geometry sits at p − origin in body space.
            "origin": self.host.origin(v),
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
        WasmVehicleHost {
            host: self.host.clone(),
        }
    }

    pub fn restore(&mut self, snap: &WasmVehicleHost) {
        self.host = snap.host.clone();
        self.host.dirty = true;
    }
}

/// D6 ball joint record (`engine_schema::Joint`, 288 B) as words. `break_force` /
/// `break_torque` ≤ 0 mean unbreakable; `penalty` > 0 seeds the linear rows'
/// starting stiffness (heavy bolted bodies sag under the default kStart).
#[wasm_bindgen]
pub fn ball_joint_words(
    a: u32,
    b: u32,
    anchor_a: &[f32],
    anchor_b: &[f32],
    break_force: f32,
    break_torque: f32,
    penalty: f32,
) -> Vec<u32> {
    let big = |v: f32| if v > 0.0 { v } else { engine_schema::JOINT_BIG };
    let mut j = engine_schema::JointDesc::ball(
        a,
        b,
        glam::Vec3::from_slice(anchor_a),
        glam::Vec3::from_slice(anchor_b),
    )
    .with_break(big(break_force), big(break_torque))
    .to_gpu();
    if penalty > 0.0 {
        j.penalty[..3].fill(penalty);
    }
    bytemuck::cast_slice(&[j]).to_vec()
}

/// The shared rigid step graph (`engine_exec::rigid_plan`) for a browser host:
/// buffer sizes, per-step host uploads and the command list, on the canonical
/// contact path (the `ENGINE_CONTACTS=canonical` controls of `WgpuSim`).
#[wasm_bindgen]
pub struct RigidStep {
    cap: engine_schema::WorldCapacity,
    vehicles: VehicleCapacity,
    soft: engine_softbody::SoftCapacity,
    params: engine_schema::AvbdParams,
    last: engine_schema::StepCounters,
    steps: u32,
    parity: u32,
    sleep_rounds: u32,
    kernels: Vec<&'static str>,
    /// Directed debris (bindings 37-41), allocated at construction.
    guide: Option<GuideState>,
}

/// Host side of directed debris: the capacity, the settings' params and how
/// many zones the last upload held.
struct GuideState {
    cap: engine_schema::guidance::GuidanceCapacity,
    params: engine_schema::guidance::GuideParamsGpu,
    enabled: bool,
    zones: u32,
}

/// Command words: `[op, a, b, c, d, e]` per command. Copy (0): src, src offset,
/// dst, dst offset, size (src `u32::MAX` = the host counter upload). Dispatch (1):
/// kernel index, threads, pass. Indirect (2): kernel index, work slot, pass.
pub const CMD_WORDS: usize = 6;

impl RigidStep {
    fn modes(&self) -> engine_exec::rigid_plan::StepModes {
        // Colours and colouring rounds: last step's counts plus headroom, the
        // rest in the single-workgroup tails (same result; native WebGPU, Metal
        // and CUDA defaults). Empty dispatches cost a barrier each: 32 rounds and
        // 32 colours × iterations per step otherwise.
        engine_exec::rigid_plan::StepModes {
            sparse_start: 32,
            color_headroom: Some(2),
            color_round_headroom: Some(2),
            tail_ok: self.cap.max_bodies <= (1 << 16) * 8,
            ..Default::default()
        }
    }

    fn inputs(&self, counts: &[u32]) -> engine_exec::rigid_plan::StepInputs {
        use engine_exec::rigid_plan as rp;
        let (bodies, colliders, joints, vehicles, wheels) =
            (counts[0], counts[1], counts[2], counts[3], counts[4]);
        let c = |i: usize| counts.get(i).copied().unwrap_or(0);
        // counts[5..15]: soft counts (WasmSoftHost::counts), counts[15]: collision filters
        // in use, counts[16]: the soft dual pass has work, counts[17]: particle
        // sweeps per iteration, counts[18..22]: motion (below).
        let soft = rp::SoftCounts {
            bodies: c(5),
            beams: c(6),
            colors: c(7),
            max_color: c(8),
            groups: c(9),
            particles: c(10),
            chebyshev: c(11) != 0,
            particles_total: c(12),
            triangles: c(13),
            contacts: c(14) != 0,
            dual: c(16) != 0,
            sweeps: c(17),
        };
        let prev_n = self.last.contact_count.min(self.cap.max_contacts).max(256);
        rp::StepInputs {
            body_count: bodies,
            collider_count: colliders,
            joint_count: joints,
            joint_hash_slots: engine_schema::joint_hash_size(joints),
            has_convex: false,
            has_height_field: c(18) & 8 != 0,
            // The browser hosts no event shapes yet (World runs natively).
            soil: false,
            reuse_contacts: self.params.flags & engine_schema::FLAG_REUSE_CONTACTS != 0,
            max_colors: self.cap.max_colors,
            iterations: self.params.iterations,
            sleep_time: self.params.sleep_time,
            sleep_island_rounds: self.sleep_rounds,
            grid_hash_words: rp::grid_hash_words(&self.cap, colliders, false, false),
            warm_hash_words: u64::from(rp::warm_hash_slots(&self.cap, bodies, prev_n, false)),
            steps_done: self.steps,
            last: self.last,
            vehicle_count: vehicles,
            wheel_count: wheels,
            soft,
            command_count: 0,
            guidance: self.guide_active(),
            guide_slots: self.guide.as_ref().map_or(0, |g| g.cap.plan_budget),
            explosion_count: 0,
            // counts[18]: bit 0 CCD bodies, bit 1 kinematic bodies, bit 2 impulse
            // records this step, bit 3 heightfield colliders (has_height_field,
            // below); counts[19] projectiles, counts[20] characters, counts[21]
            // queries.
            motion: {
                let bits = c(18);
                rp::MotionInputs {
                    ccd: bits & 1 != 0,
                    kinematic: bits & 2 != 0,
                    impulses: bits & 4 != 0,
                    look_ahead: self.params.flags & engine_schema::FLAG_SPEC_VELOCITY != 0,
                    projectiles: c(19),
                    characters: c(20),
                    queries: c(21),
                    fast_list_off: false,
                }
            },
        }
    }

    fn guide_active(&self) -> bool {
        self.guide
            .as_ref()
            .is_some_and(|g| g.enabled && g.zones > 0)
    }

    fn kernel_index(&mut self, name: &'static str) -> u32 {
        match self.kernels.iter().position(|k| *k == name) {
            Some(i) => i as u32,
            None => {
                self.kernels.push(name);
                (self.kernels.len() - 1) as u32
            }
        }
    }
}

#[wasm_bindgen]
impl RigidStep {
    /// `world` = [maxBodies, maxColliders, maxContacts, maxJoints, maxCandidates,
    /// maxColors, maxCommands, maxHullVertices] (user joints: one tyre joint per
    /// wheel slot is added); `vehicles` as for [`WasmVehicleHost::new`].
    /// `soft` (optional) = [maxBodies, maxBeams, maxParticles, maxColors, maxGroups,
    /// maxTriangles, maxObstacles] of the soft-body arena (binding 32).
    /// `guidance` (optional) = [maxZones, planBudget]: directed debris
    /// (bindings 55-59, `set_guidance`).
    #[wasm_bindgen(constructor)]
    pub fn new(
        world: &[u32],
        vehicles: &[u32],
        soft: Option<Vec<u32>>,
        guidance: Option<Vec<u32>>,
    ) -> Result<RigidStep, JsError> {
        let mut cap = engine_schema::WorldCapacity {
            max_bodies: world[0],
            max_colliders: world[1],
            max_contacts: world[2],
            max_joints: world[3],
            max_candidates: world[4],
            max_colors: world[5],
            max_commands: world[6],
            max_hull_vertices: world[7],
        };
        let vehicles = VehicleCapacity {
            max_vehicles: vehicles[0],
            max_wheels: vehicles[1],
            max_curve_points: vehicles[2],
            max_diffs: vehicles[3],
            max_anti_rolls: vehicles[4],
            max_parts: vehicles[5],
            max_rays: vehicles[6],
        };
        cap.max_joints += vehicles.max_wheels;
        let soft = soft.map_or(
            engine_softbody::SoftCapacity {
                max_bodies: 0,
                ..Default::default()
            },
            |v| soft_capacity(&v),
        );
        let mut s = RigidStep {
            cap,
            vehicles,
            soft,
            params: engine_schema::AvbdParams::default(),
            last: engine_schema::StepCounters::default(),
            steps: 0,
            parity: 0,
            sleep_rounds: 8,
            kernels: Vec::new(),
            guide: None,
        };
        if let Some(g) = guidance {
            let cap = engine_schema::guidance::GuidanceCapacity {
                max_zones: g.first().copied().unwrap_or(256),
                plan_budget: g.get(1).copied().unwrap_or(256),
            };
            if cap.max_zones == 0
                || cap.plan_budget == 0
                || cap.plan_budget > engine_schema::guidance::GUIDE_BLOCK
            {
                return Err(JsError::new(&format!(
                    "guidance capacity needs zones > 0 and a plan budget in 1..={}",
                    engine_schema::guidance::GUIDE_BLOCK
                )));
            }
            s.guide = Some(GuideState {
                cap,
                params: Default::default(),
                enabled: false,
                zones: 0,
            });
            for k in engine_schema::guidance::GUIDE_ENTRIES {
                s.kernel_index(k);
            }
        }
        // Every kernel a step can emit (small and large scenes, joints, vehicles).
        // Small and large scenes; soft passes with every option on, both with the
        // fused particle sweep (small colours) and one dispatch per colour.
        // After the first step, colours and colouring rounds past last step's
        // counts run in the tail kernels.
        // Several soft bodies: one workgroup per body. The motion passes (CCD,
        // kinematics, impulses, projectiles, characters, queries) off and on.
        for (n, prefix, max_color, done, bodies, motion) in [
            (1u32, true, 1u32, 0u32, 1u32, 0u32),
            (4096, false, 1, 0, 1, 0),
            (4096, false, 4096, 0, 1, 0),
            (4096, false, 1, 1, 1, 0),
            (4096, false, 1, 0, 2, 0),
            // No soft bodies: vehicle_iterate as its own pass (rigid vehicle worlds).
            (4096, false, 1, 0, 0, 0),
            (4096, false, 1, 1, 0, 0),
            (4096, false, 1, 0, 1, 7),
            (4096, false, 1, 0, 0, 7),
            // Heightfield terrain: the convex stages take its pairs.
            (4096, false, 1, 0, 0, 15),
        ] {
            let mut i = s.inputs(&[
                n, n, 1, 1, 1, bodies, 1, 1, max_color, 1, 1, 1, 1, 1, 1, 1, 1, 1, motion, motion,
                motion, motion,
            ]);
            i.sleep_time = 0.5;
            i.steps_done = done;
            i.motion.look_ahead = motion != 0;
            for c in engine_exec::rigid_plan::encode_step(&s.modes(), &i, false, prefix) {
                match c {
                    engine_exec::rigid_plan::Cmd::Dispatch { kernel, .. }
                    | engine_exec::rigid_plan::Cmd::Indirect { kernel, .. } => {
                        s.kernel_index(kernel);
                    }
                    _ => {}
                }
            }
        }
        Ok(s)
    }

    /// Kernel entry names (JSON array); command words index into it.
    pub fn kernels(&self) -> String {
        serde_json::to_string(&self.kernels).unwrap()
    }

    /// Byte size of every binding (vehicle ground blocks and pool included).
    pub fn buffer_sizes(&self) -> Vec<f64> {
        let mut s = engine_exec::rigid_plan::buffer_sizes(&self.cap, false, false);
        // Collider AABBs, then a 4-entry block per vehicle and per wheel slot.
        s[9] += u64::from(4 * (self.vehicles.max_vehicles + self.vehicles.max_wheels)) * 32;
        s[31] = self.vehicles.bytes().max(16);
        if self.soft.max_bodies > 0 {
            s[32] = self.soft.bytes();
        }
        let mut out: Vec<f64> = s.iter().map(|&b| b as f64).collect();
        if let Some(g) = &self.guide {
            use engine_schema::guidance as gd;
            let n = u64::from(self.cap.max_bodies.max(1));
            out.resize(gd::GUIDE_SLOTS_BINDING + 1, 256.0);
            let size = |b: u64| b.max(256) as f64;
            out[gd::GUIDE_PARAMS_BINDING] = size(std::mem::size_of::<gd::GuideParamsGpu>() as u64);
            out[gd::GUIDE_ZONES_BINDING] =
                size(u64::from(g.cap.max_zones) * std::mem::size_of::<engine_schema::guidance::GuideZoneGpu>() as u64);
            out[gd::GUIDE_BODIES_BINDING] =
                size(n * std::mem::size_of::<gd::GuideBodyGpu>() as u64);
            out[gd::GUIDE_SCRATCH_BINDING] = size(g.cap.scratch_words(self.cap.max_bodies) * 4);
            out[gd::GUIDE_SLOTS_BINDING] =
                size(u64::from(g.cap.plan_budget) * std::mem::size_of::<gd::GuideSlotGpu>() as u64);
        }
        out
    }

    /// Enable, disable or retune directed debris (`GuidanceSettings` JSON,
    /// camelCase; missing fields take the defaults). Needs the `guidance`
    /// capacity of the constructor.
    pub fn set_guidance(&mut self, settings_json: &str) -> Result<(), JsError> {
        let s: engine_schema::guidance::GuidanceSettings =
            serde_json::from_str(settings_json).map_err(|e| JsError::new(&e.to_string()))?;
        let (gravity, dt) = (self.params.gravity_y, self.params.dt);
        let g = self
            .guide
            .as_mut()
            .ok_or_else(|| JsError::new("RigidStep was built without a guidance capacity"))?;
        g.params = s.to_params(gravity, dt).map_err(|e| JsError::new(&e))?;
        g.enabled = s.enabled;
        Ok(())
    }

    /// Zone records (`ZoneDesc` JSON array) for binding 56, written at offset 0.
    /// `body_count` bounds the anchors.
    pub fn set_guidance_zones(
        &mut self,
        zones_json: &str,
        body_count: u32,
    ) -> Result<Vec<u8>, JsError> {
        let zones: Vec<engine_schema::guidance::ZoneDesc> =
            serde_json::from_str(zones_json).map_err(|e| JsError::new(&e.to_string()))?;
        let g = self
            .guide
            .as_mut()
            .ok_or_else(|| JsError::new("RigidStep was built without a guidance capacity"))?;
        if zones.len() > g.cap.max_zones as usize {
            return Err(JsError::new(&format!(
                "{} guidance zones exceed the capacity of {}",
                zones.len(),
                g.cap.max_zones
            )));
        }
        let mut out = Vec::with_capacity(zones.len() * std::mem::size_of::<engine_schema::guidance::GuideZoneGpu>());
        for (k, z) in zones.iter().enumerate() {
            if let Some(b) = z.anchor_body {
                if b >= body_count {
                    return Err(JsError::new(&format!(
                        "zone {k} anchors to body {b}, past the scene"
                    )));
                }
            }
            let gpu = z
                .to_gpu()
                .map_err(|e| JsError::new(&format!("zone {k}: {e}")))?;
            out.extend_from_slice(bytemuck::bytes_of(&gpu));
        }
        g.zones = zones.len() as u32;
        Ok(out)
    }

    /// This step's guidance params (binding 55), empty while guidance is off
    /// or has no zones. `counts` as for `encode`.
    pub fn guide_params(&self, counts: &[u32]) -> Vec<u8> {
        if !self.guide_active() {
            return Vec::new();
        }
        let g = self.guide.as_ref().unwrap();
        let p = g.params.for_step(
            engine_schema::guidance::GuideStep {
                time: self.steps as f32 * self.params.dt,
                dt: self.params.dt,
                gravity_y: self.params.gravity_y,
                zones: g.zones,
                bodies: counts[0],
                joints: counts[2],
            },
            g.cap,
        );
        bytemuck::bytes_of(&p).to_vec()
    }

    /// `count` guidance records (binding 57) arming bodies for zones whose tags
    /// meet `tag_mask` (0: off); write them at the first body's record.
    pub fn guide_records(count: u32, tag_mask: u32) -> Vec<u8> {
        use engine_schema::guidance::GuideBodyGpu;
        let r = if tag_mask == 0 {
            GuideBodyGpu::default()
        } else {
            GuideBodyGpu::armed(1, tag_mask)
        };
        bytemuck::bytes_of(&r).repeat(count as usize)
    }

    /// Bodies per state then per end reason (`[u32; 12]`, as the lab's
    /// `guideStateCounts`) of a binding-39 readback.
    pub fn guide_state_counts(records: &[u8]) -> Vec<u32> {
        use engine_schema::guidance::{GuideBodyGpu, GuideEnd};
        let mut out = vec![0u32; 12];
        for c in records.chunks_exact(std::mem::size_of::<GuideBodyGpu>()) {
            let r: GuideBodyGpu = bytemuck::pod_read_unaligned(c);
            out[(r.state as usize).min(3)] += 1;
            out[4 + GuideEnd::of(&r) as usize] += 1;
        }
        out
    }

    /// Guidance bindings: params, zones, bodies, scratch, slots.
    pub fn guide_bindings() -> Vec<u32> {
        use engine_schema::guidance as gd;
        [
            gd::GUIDE_PARAMS_BINDING,
            gd::GUIDE_ZONES_BINDING,
            gd::GUIDE_BODIES_BINDING,
            gd::GUIDE_SCRATCH_BINDING,
            gd::GUIDE_SLOTS_BINDING,
        ]
        .map(|b| b as u32)
        .to_vec()
    }

    /// Bytes of one body's guidance record (the bodies binding).
    pub fn guide_record_bytes() -> u32 {
        std::mem::size_of::<engine_schema::guidance::GuideBodyGpu>() as u32
    }

    /// Byte offset of the zone claim counters in the scratch binding.
    pub fn guide_claims_offset() -> u32 {
        engine_schema::guidance::GUIDE_SCRATCH_CLAIMS * 4
    }

    /// First AABB entry past the colliders (the vehicle host's ground blocks).
    pub fn aabb_tail(&self) -> u32 {
        self.cap.max_bodies + self.cap.max_colliders
    }

    /// Pass uniform table (binding 22).
    pub fn pass_table(&self) -> Vec<u8> {
        engine_exec::rigid_plan::pass_table()
    }

    /// `AvbdParams` as its 80 raw bytes (`engine_schema::AvbdParams` field order).
    pub fn set_params(&mut self, bytes: &[u8]) {
        self.params = bytemuck::pod_read_unaligned(
            &bytes[..std::mem::size_of::<engine_schema::AvbdParams>()],
        );
    }

    /// Island gate rounds per step (0 disables the gate).
    pub fn set_sleep_rounds(&mut self, rounds: u32) {
        self.sleep_rounds = rounds;
    }

    /// Last read counters (binding 7 words); sizes the warm hash and dispatches.
    pub fn set_last(&mut self, words: &[u32]) {
        let n = std::mem::size_of::<engine_schema::StepCounters>() / 4;
        if words.len() >= n {
            self.last = bytemuck::pod_read_unaligned(bytemuck::cast_slice(&words[..n]));
        }
    }

    pub fn steps(&self) -> u32 {
        self.steps
    }

    /// Start over (scene reset): step 0, contact parity 0, no counters.
    pub fn reset(&mut self) {
        self.steps = 0;
        self.parity = 0;
        self.last = engine_schema::StepCounters::default();
    }

    /// Host uploads for this step: 256 B of host words (binding 21 at 0), the
    /// host uniform (binding 6 at 0), then the counter block (the copy source
    /// `u32::MAX` in the commands).
    pub fn host_upload(&self, counts: &[u32]) -> Vec<u8> {
        use engine_exec::rigid_plan as rp;
        let i = self.inputs(counts);
        let s = rp::host_words(
            &self.cap,
            &self.modes(),
            &i,
            &rp::HostState {
                contacts_parity: self.parity,
                grid_levels: 0,
                has_filters: counts.get(15).is_some_and(|&f| f != 0),
                record_hash: false,
                np_defer_limit: 0,
                persist_list_limit: 0,
            },
        );
        let mut out = bytemuck::cast_slice::<u32, u8>(&s).to_vec();
        out.extend_from_slice(bytemuck::bytes_of(&rp::host_uniforms(&self.params, &s)));
        out.extend_from_slice(bytemuck::bytes_of(&rp::step_counters(&i)));
        out
    }

    /// Commands of one pipelined step for `counts` = [bodies, colliders, joints,
    /// vehicles, wheels, soft counts × 10, filters]; advances the step and contact parity.
    pub fn encode(&mut self, counts: &[u32]) -> Vec<u32> {
        use engine_exec::rigid_plan::Cmd;
        let i = self.inputs(counts);
        let cmds = engine_exec::rigid_plan::encode_step(&self.modes(), &i, false, true);
        let mut out = Vec::with_capacity(cmds.len() * CMD_WORDS);
        for c in cmds {
            let w = match c {
                Cmd::Copy {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => [0, src, src_offset, dst, dst_offset, size],
                Cmd::Dispatch {
                    kernel,
                    threads,
                    pass,
                } => [
                    1,
                    self.kernel_index(kernel),
                    threads.min(u64::from(u32::MAX)) as u32,
                    pass,
                    0,
                    0,
                ],
                Cmd::Indirect { kernel, slot, pass } => {
                    [2, self.kernel_index(kernel), slot, pass, 0, 0]
                }
                Cmd::Phase { .. } => continue,
            };
            out.extend_from_slice(&w);
        }
        self.parity ^= 1;
        self.steps += 1;
        out
    }
}

pub mod gauntlet;
pub mod meteors;
#[cfg(test)]
mod tests;
