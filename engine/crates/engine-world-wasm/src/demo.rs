//! What the benchmark page (`demo.html`) needs beyond the shared bindings:
//! scenes stored as RUBSNAP1 records (`World::import_snapshot`), its built
//! scenes in one batch of world calls, GPU step timing, joints it edits
//! (the drag spring, motors and servos), and raw record reads for its checks.

use crate::defs::{self, Fields};
use crate::{bits, err, handle, id, res, WebWorld};
use rubble::*;
use wasm_bindgen::prelude::*;

/// Floats per body in [`WebWorld::create_bodies`].
pub const BODY_RECORD: usize = 24;

#[wasm_bindgen]
impl WebWorld {
    /// Build a RUBSNAP1 snapshot's scene (`engine_schema::WorldSnapshot`, as
    /// `scene_export` writes it) through `World::import_snapshot`. Returns `{
    /// bodies, shapes, joints }`: the handles by record index (`null` for a
    /// broken joint record).
    #[wasm_bindgen(js_name = importSnapshot)]
    pub fn import_snapshot(&mut self, bytes: &[u8]) -> Result<JsValue, JsValue> {
        let snap = engine_schema::WorldSnapshot::from_bytes(bytes)
            .ok_or_else(|| err("not a RUBSNAP1 WorldSnapshot, or truncated"))?;
        let import = res(self.world.import_snapshot(&snap))?;
        let o = js_sys::Object::new();
        let handles = |ids: Vec<f64>| js_sys::Float64Array::from(ids.as_slice());
        let bodies = handles(import.bodies.iter().map(|b| handle(b.to_bits())).collect());
        let shapes = handles(import.shapes.iter().map(|s| handle(s.to_bits())).collect());
        let joints = js_sys::Array::new();
        for j in &import.joints {
            joints.push(&j.map_or(JsValue::NULL, |j| JsValue::from(handle(j.to_bits()))));
        }
        let _ = js_sys::Reflect::set(&o, &"bodies".into(), &bodies);
        let _ = js_sys::Reflect::set(&o, &"shapes".into(), &shapes);
        let _ = js_sys::Reflect::set(&o, &"joints".into(), &joints);
        Ok(o.into())
    }

    /// Many single-shape bodies in one call (`World::create_bodies`, then
    /// one shape and the mass data each), in record order. Per body,
    /// [`BODY_RECORD`] floats: type (0 static, 1 kinematic, 2 dynamic),
    /// bullet (0 / 1), position xyz, rotation xyzw, linear and angular
    /// velocity, mass, principal inertia xyz (centre at the origin, axes
    /// aligned; ignored unless dynamic), shape (0 sphere: radius; 2 box:
    /// half extents), its three numbers, then the shape's density. `shape`
    /// is the `ShapeDef` (material, filter, ...) every shape takes. Returns
    /// the body handles.
    #[wasm_bindgen(js_name = createBodies)]
    pub fn create_bodies(&mut self, records: &[f32], shape: JsValue) -> Result<Vec<f64>, JsValue> {
        if records.len() % BODY_RECORD != 0 {
            return Err(err(format!(
                "createBodies: {} floats is not whole records of {BODY_RECORD}",
                records.len()
            )));
        }
        let shape = defs::shape_def(shape)?;
        let rows: Vec<&[f32]> = records.chunks(BODY_RECORD).collect();
        let mut defs = Vec::with_capacity(rows.len());
        for r in &rows {
            let body_type = match r[0] as u32 {
                0 => BodyType::Static,
                1 => BodyType::Kinematic,
                2 => BodyType::Dynamic,
                t => return Err(err(format!("createBodies: body type {t}"))),
            };
            defs.push(BodyDef {
                body_type,
                is_bullet: r[1] != 0.0,
                position: [r[2], r[3], r[4]],
                rotation: [r[5], r[6], r[7], r[8]],
                linear_velocity: [r[9], r[10], r[11]],
                angular_velocity: [r[12], r[13], r[14]],
                ..BodyDef::default()
            });
        }
        let ids = res(self.world.create_bodies(&defs))?;
        // The records carry the mass: shapes leave it alone.
        let shape = ShapeDef {
            update_body_mass: false,
            ..shape
        };
        for (b, r) in ids.iter().zip(&rows) {
            let geometry = match r[19] as u32 {
                0 => Geometry::Sphere {
                    center: [0.0; 3],
                    radius: r[20],
                },
                2 => Geometry::Box {
                    half: [r[20], r[21], r[22]],
                    transform: Transform::IDENTITY,
                },
                s => return Err(err(format!("createBodies: shape {s}"))),
            };
            let def = ShapeDef {
                density: r[23],
                ..shape.clone()
            };
            res(self.world.create_shape(*b, &def, geometry))?;
            if r[0] as u32 == 2 {
                let mass = MassData::principal(r[15], [0.0; 3], [r[16], r[17], r[18]]);
                res(self.world.body_set_mass_data(*b, mass))?;
            }
        }
        Ok(ids.iter().map(|b| handle(b.to_bits())).collect())
    }

    /// Continuous collision for a static or kinematic body when it is made
    /// dynamic (`World::body_set_bullet`).
    #[wasm_bindgen(js_name = bodySetBullet)]
    pub fn body_set_bullet(&mut self, body: f64, bullet: bool) -> Result<(), JsValue> {
        res(self.world.body_set_bullet(id(body), bullet))
    }

    /// Time the rigid step on the GPU: `'off' | 'step' | 'phases'`
    /// (`World::set_gpu_timing`); frames then carry `gpuTimes()`.
    #[wasm_bindgen(js_name = setGpuTiming)]
    pub fn set_gpu_timing(&mut self, mode: &str) -> Result<(), JsValue> {
        res(self.world.set_gpu_timing(defs::gpu_timing(mode)?))
    }

    /// Keep sleeping bodies' resting contacts through sleep (`World::enable_sleep_archive`,
    /// default room). Canonical contacts only; turns the resting-pair carry off.
    #[wasm_bindgen(js_name = enableSleepArchive)]
    pub fn enable_sleep_archive(&mut self) -> Result<(), JsValue> {
        res(self.world.enable_sleep_archive(None))
    }

    /// The resting-pair carry from the next step on (`World::set_carry_rest`).
    #[wasm_bindgen(js_name = setCarryRest)]
    pub fn set_carry_rest(&mut self, on: bool) {
        self.world.set_carry_rest(on);
    }

    #[wasm_bindgen(js_name = carryRest)]
    pub fn carry_rest(&self) -> bool {
        self.world.carry_rest()
    }

    #[wasm_bindgen(js_name = destroyJoint)]
    pub fn destroy_joint(&mut self, joint: f64) -> Result<(), JsValue> {
        res(self.world.destroy_joint(JointId::from_bits(bits(joint))))
    }

    /// A joint's drive: motor speed and servo target (`World::joint_set_drive`).
    #[wasm_bindgen(js_name = jointSetDrive)]
    pub fn joint_set_drive(
        &mut self,
        joint: f64,
        velocity: f32,
        target: f32,
    ) -> Result<(), JsValue> {
        res(self
            .world
            .joint_set_drive(JointId::from_bits(bits(joint)), velocity, target))
    }

    /// Raw `BodyState` records (112 B each) of slots `[first, first + count)`
    /// as the queue leaves them, resolved once mapped (a `Uint32Array`): for
    /// checks that read whole records.
    #[wasm_bindgen(js_name = readBodyRecords)]
    pub fn read_body_records(&self, first: u32, count: u32) -> js_sys::Promise {
        let record = std::mem::size_of::<engine_schema::BodyState>() as u64;
        let b = self.world.sim().body_buffer().clone();
        crate::read::read_ranges(
            &self.world,
            vec![(b, u64::from(first) * record, u64::from(count) * record)],
        )
    }

    /// The last step's bodies per primal colour (`max_colors` words), resolved
    /// once mapped: a check that the colouring covered every awake body.
    #[wasm_bindgen(js_name = readColorCounts)]
    pub fn read_color_counts(&self) -> js_sys::Promise {
        let n = u64::from(self.world.sim().capacity().max_colors.min(64)) * 4;
        let b = self.world.sim().color_count_buffer().clone();
        crate::read::read_ranges(&self.world, vec![(b, 0, n)])
    }
}

/// A `JointDesc::ball` with optional stiffness and torque arm, for
/// `createBallJoint` (`{ linearStiffness, angularStiffness, torqueArm }`).
pub(crate) fn ball_extras(f: &mut Fields, mut desc: JointDesc) -> Result<JointDesc, JsValue> {
    let (mut lin, mut ang, mut arm) = (desc.lin_stiffness, desc.ang_stiffness, desc.torque_arm);
    f.f32("linearStiffness", &mut lin)?;
    f.f32("angularStiffness", &mut ang)?;
    f.f32("torqueArm", &mut arm)?;
    desc = desc.with_stiffness(lin, ang).with_torque_arm(arm);
    Ok(desc)
}
