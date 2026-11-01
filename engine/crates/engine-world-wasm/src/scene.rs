//! Scene controls: save and restore (a reset key), body type changes, fixed
//! material slots, ball joints and the counters pages time things by.

use crate::defs::{self, Fields};
use crate::{bits, err, handle, id, res, WebWorld};
use rubble::*;
use wasm_bindgen::prelude::*;

/// A saved world (`WebWorld.saveState`).
#[wasm_bindgen]
pub struct WebSnapshot {
    snapshot: WorldSnapshot,
}

fn body_type(s: &str) -> Result<BodyType, JsValue> {
    Ok(match s {
        "static" => BodyType::Static,
        "kinematic" => BodyType::Kinematic,
        "dynamic" => BodyType::Dynamic,
        _ => return Err(err(format!("unknown body type {s}"))),
    })
}

#[wasm_bindgen]
impl WebWorld {
    /// Copy the whole world (on the GPU) for `restoreState`.
    #[wasm_bindgen(js_name = saveState)]
    pub fn save_state(&self) -> Result<WebSnapshot, JsValue> {
        Ok(WebSnapshot { snapshot: res(self.world.save_state())? })
    }

    /// Back to a `saveState` snapshot (`World::restore_state`).
    #[wasm_bindgen(js_name = restoreState)]
    pub fn restore_state(&mut self, snapshot: &WebSnapshot) -> Result<(), JsValue> {
        res(self.world.restore_state(&snapshot.snapshot))
    }

    /// Steps the simulation has taken since it was created or last restored:
    /// its clock (guidance zones' time windows run on it).
    #[wasm_bindgen(js_name = simSteps)]
    pub fn sim_steps(&self) -> u32 {
        self.world.sim().steps_done()
    }

    /// Some body carries continuous collision: the step runs the CCD passes.
    #[wasm_bindgen(js_name = hasCcd)]
    pub fn has_ccd(&self) -> bool {
        self.world.sim().has_ccd()
    }

    /// The live body in slot `slot` (`undefined`: none).
    #[wasm_bindgen(js_name = bodyAt)]
    pub fn body_at(&self, slot: u32) -> Option<f64> {
        self.world.body_at(slot).map(|b| handle(b.to_bits()))
    }

    /// Change a body's type (`'static' | 'kinematic' | 'dynamic'`), rewriting
    /// it at a pose with velocities (`World::body_set_type`).
    #[wasm_bindgen(js_name = bodySetType)]
    pub fn body_set_type(&mut self, body: f64, kind: &str, position: &[f32], rotation: &[f32], linear: &[f32], angular: &[f32]) -> Result<(), JsValue> {
        let t = Transform::new(crate::vec3(position)?, crate::vec4(rotation)?);
        res(self.world.body_set_type(id(body), body_type(kind)?, t, crate::vec3(linear)?, crate::vec3(angular)?))
    }

    /// Pin GPU material slot `slot` to a `SurfaceMaterial` (`{ friction,
    /// restitution, contactOffset, ... }`), for vehicle part colliders, wheel
    /// rims and soft nodes, which name a slot.
    #[wasm_bindgen(js_name = setMaterialSlot)]
    pub fn set_material_slot(&mut self, slot: u32, material: JsValue) -> Result<(), JsValue> {
        let mut f = Fields::of("SurfaceMaterial", material)?;
        let mut m = SurfaceMaterial::default();
        defs::material(&mut f, &mut m)?;
        f.finish()?;
        res(self.world.set_material_slot(slot, m))
    }

    /// The GPU material slot a shape's collider uses.
    #[wasm_bindgen(js_name = shapeMaterialSlot)]
    pub fn shape_material_slot(&self, shape: f64) -> Result<u32, JsValue> {
        let c = res(self.world.shape_collider(ShapeId::from_bits(bits(shape))))?;
        Ok(self.world.sim().collider_record(c).material & 0xFFFF)
    }

    /// Size the broadphase grid for colliders of this bounding radius that
    /// will move (bodies parked static, launched later).
    #[wasm_bindgen(js_name = reserveMovingRadius)]
    pub fn reserve_moving_radius(&mut self, radius: f32) {
        self.world.reserve_moving_radius(radius);
    }

    /// A ball joint (`JointDesc::ball`, Rubble's D6 preset) at `anchorA` on A
    /// and `anchorB` on B (each from its body's origin), breaking past
    /// `breakForce` / `breakTorque` (≤ 0: never). Returns its handle.
    #[wasm_bindgen(js_name = createBallJoint)]
    pub fn create_ball_joint(&mut self, def: JsValue) -> Result<f64, JsValue> {
        let mut f = Fields::of("BallJoint", def)?;
        let (mut a, mut b) = (0u64, 0u64);
        let (mut anchor_a, mut anchor_b) = ([0.0f32; 3], [0.0f32; 3]);
        let (mut break_force, mut break_torque, mut user_data) = (0.0f32, 0.0f32, 0u64);
        f.u64("bodyA", &mut a)?;
        f.u64("bodyB", &mut b)?;
        f.floats("anchorA", &mut anchor_a)?;
        f.floats("anchorB", &mut anchor_b)?;
        f.f32("breakForce", &mut break_force)?;
        f.f32("breakTorque", &mut break_torque)?;
        f.u64("userData", &mut user_data)?;
        f.finish()?;
        let big = |v: f32| if v > 0.0 { v } else { engine_schema::JOINT_BIG };
        let desc = engine_schema::JointDesc::ball(0, 0, glam::Vec3::from(anchor_a), glam::Vec3::from(anchor_b))
            .with_break(big(break_force), big(break_torque));
        let def = JointDef::Custom { body_a: BodyId::from_bits(a), body_b: BodyId::from_bits(b), desc, user_data };
        Ok(handle(res(self.world.create_joint(&def))?.to_bits()))
    }

    /// Start a joint's linear rows at stiffness `k` (N/m) instead of `kStart`.
    #[wasm_bindgen(js_name = jointSetLinearPenalty)]
    pub fn joint_set_linear_penalty(&mut self, joint: f64, k: f32) -> Result<(), JsValue> {
        res(self.world.joint_set_linear_penalty(JointId::from_bits(bits(joint)), k))
    }
}
