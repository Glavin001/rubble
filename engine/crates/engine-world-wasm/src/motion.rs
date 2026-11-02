//! Characters and projectiles (`World::create_character`, `spawn_projectile`).

use crate::defs::Fields;
use crate::{bits, err, handle, res, WebWorld};
use rubble::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl WebWorld {
    /// A character with its capsule centre at `position`. `desc`: `CharacterDesc`
    /// fields (snake_case, as engine-schema serialises it); missing ones keep the
    /// defaults.
    #[wasm_bindgen(js_name = createCharacter)]
    pub fn create_character(&mut self, desc: JsValue, position: &[f32], user_data: f64) -> Result<f64, JsValue> {
        let mut d = serde_json::to_value(CharacterDesc::default()).map_err(err)?;
        let given = crate::defs::object(desc)?;
        let obj = d.as_object_mut().expect("CharacterDesc is an object");
        for (k, v) in given {
            if !obj.contains_key(&k) {
                return Err(err(format!("CharacterDesc: unknown field {k}")));
            }
            obj.insert(k, v);
        }
        let desc: CharacterDesc = serde_json::from_value(d).map_err(err)?;
        let def = CharacterDef { position: crate::vec3(position)?, desc, user_data: user_data as u64 };
        Ok(handle(res(self.world.create_character(&def))?.to_bits()))
    }

    #[wasm_bindgen(js_name = destroyCharacter)]
    pub fn destroy_character(&mut self, id: f64) -> Result<(), JsValue> {
        res(self.world.destroy_character(CharacterId::from_bits(bits(id))))
    }

    /// Desired velocity; `jump`: start a jump this step.
    #[wasm_bindgen(js_name = characterSetInput)]
    pub fn character_set_input(&mut self, id: f64, velocity: &[f32], jump: bool) -> Result<(), JsValue> {
        res(self.world.character_set_input(CharacterId::from_bits(bits(id)), crate::vec3(velocity)?, jump))
    }

    #[wasm_bindgen(js_name = characterTeleport)]
    pub fn character_teleport(&mut self, id: f64, position: &[f32]) -> Result<(), JsValue> {
        res(self.world.character_teleport(CharacterId::from_bits(bits(id)), crate::vec3(position)?))
    }

    /// The character's kinematic body.
    #[wasm_bindgen(js_name = characterBody)]
    pub fn character_body(&self, id: f64) -> Result<f64, JsValue> {
        Ok(handle(res(self.world.character_body(CharacterId::from_bits(bits(id))))?.to_bits()))
    }

    /// A projectile (`ProjectileDef`, camelCase: position, velocity, radius, mass,
    /// penetrationCost, drag, maxHitsPerStep, filter, ignoreBody, pushBodies,
    /// userData).
    #[wasm_bindgen(js_name = spawnProjectile)]
    pub fn spawn_projectile(&mut self, def: JsValue) -> Result<f64, JsValue> {
        let mut f = Fields::of("ProjectileDef", def)?;
        let mut d = ProjectileDef::default();
        f.floats("position", &mut d.position)?;
        f.floats("velocity", &mut d.velocity)?;
        f.f32("radius", &mut d.radius)?;
        f.f32("mass", &mut d.mass)?;
        f.f32("penetrationCost", &mut d.penetration_cost)?;
        f.f32("drag", &mut d.drag)?;
        f.u32("maxHitsPerStep", &mut d.max_hits_per_step)?;
        if let Some(mut g) = f.object("filter")? {
            d.filter = query_filter(&mut g)?;
            g.finish()?;
        }
        let mut ignore = 0u64;
        f.u64("ignoreBody", &mut ignore)?;
        d.ignore_body = BodyId::from_bits(ignore);
        f.bool("pushBodies", &mut d.push_bodies)?;
        f.u64("userData", &mut d.user_data)?;
        f.finish()?;
        Ok(handle(res(self.world.spawn_projectile(&d))?.to_bits()))
    }

    /// The character, query and projectile pool (`vec4` slots), for renderers.
    #[wasm_bindgen(js_name = motionBuffer)]
    pub fn motion_buffer(&self) -> JsValue {
        let (buffer, _, _) = self.world.sim().projectile_record_range();
        crate::gpu_buffer(buffer)
    }

    /// `{ projectileBase, projectileStride, projectileSlots }`: projectile `k`'s
    /// record starts at slot `projectileBase + k * projectileStride` of
    /// [`Self::motion_buffer`]; slots below `projectileSlots` may be in use
    /// (word 0 of slot 2 has bit 0 set while it flies).
    #[wasm_bindgen(js_name = projectileLayout)]
    pub fn projectile_layout(&self) -> JsValue {
        let cap = self.world.sim().character_capacity();
        let o = js_sys::Object::new();
        for (k, v) in [
            ("projectileBase", cap.projectile_base()),
            ("projectileStride", engine_schema::character::PROJ_STRIDE),
            ("projectileSlots", self.world.sim().projectile_count()),
        ] {
            let _ = js_sys::Reflect::set(&o, &k.into(), &v.into());
        }
        o.into()
    }
}

pub fn query_filter(f: &mut Fields) -> Result<QueryFilter, JsValue> {
    let d = QueryFilter::default();
    let (mut categories, mut mask) = (u32::from(d.categories), u32::from(d.mask));
    f.u32("categories", &mut categories)?;
    f.u32("mask", &mut mask)?;
    Ok(QueryFilter { categories: categories as u16, mask: mask as u16 })
}
