//! Definitions from plain JavaScript objects. Every field is optional and keeps
//! the Rust default (`BodyDef::default()`, ...) when absent, so defaults have one
//! source. Field names are the Rust ones in camelCase; an unknown field is an
//! error, never ignored.

use glam::Vec3;
use rubble::*;
use serde_json::{Map, Value};
use wasm_bindgen::JsValue;

pub type Obj = Map<String, Value>;

pub fn object(v: JsValue) -> Result<Obj, JsValue> {
    if v.is_undefined() || v.is_null() {
        return Ok(Obj::new());
    }
    match serde_wasm_bindgen::from_value::<Value>(v).map_err(crate::err)? {
        Value::Object(m) => Ok(m),
        other => Err(crate::err(format!("expected an object, got {other}"))),
    }
}

/// Reads the fields of one object; `finish` refuses any it did not read.
pub struct Fields {
    what: &'static str,
    map: Obj,
}

impl Fields {
    pub fn new(what: &'static str, map: Obj) -> Self {
        Self { what, map }
    }

    pub fn of(what: &'static str, v: JsValue) -> Result<Self, JsValue> {
        Ok(Self::new(what, object(v)?))
    }

    pub fn take(&mut self, key: &str) -> Option<Value> {
        self.map.remove(key).filter(|v| !v.is_null())
    }

    fn bad(&self, key: &str, want: &str) -> JsValue {
        crate::err(format!("{}.{key}: expected {want}", self.what))
    }

    pub fn f32(&mut self, key: &str, out: &mut f32) -> Result<(), JsValue> {
        if let Some(v) = self.take(key) {
            *out = v.as_f64().ok_or_else(|| self.bad(key, "a number"))? as f32;
        }
        Ok(())
    }

    pub fn u32(&mut self, key: &str, out: &mut u32) -> Result<(), JsValue> {
        if let Some(v) = self.take(key) {
            *out = v.as_u64().ok_or_else(|| self.bad(key, "an unsigned integer"))? as u32;
        }
        Ok(())
    }

    pub fn u64(&mut self, key: &str, out: &mut u64) -> Result<(), JsValue> {
        if let Some(v) = self.take(key) {
            *out = v.as_u64().ok_or_else(|| self.bad(key, "an unsigned integer"))?;
        }
        Ok(())
    }

    pub fn bool(&mut self, key: &str, out: &mut bool) -> Result<(), JsValue> {
        if let Some(v) = self.take(key) {
            *out = v.as_bool().ok_or_else(|| self.bad(key, "a boolean"))?;
        }
        Ok(())
    }

    pub fn string(&mut self, key: &str) -> Result<Option<String>, JsValue> {
        match self.take(key) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(Some(s)),
            Some(_) => Err(self.bad(key, "a string")),
        }
    }

    pub fn floats<const N: usize>(&mut self, key: &str, out: &mut [f32; N]) -> Result<(), JsValue> {
        if let Some(v) = self.take(key) {
            let a = v.as_array().filter(|a| a.len() == N).ok_or_else(|| self.bad(key, &format!("{N} numbers")))?;
            for (o, x) in out.iter_mut().zip(a) {
                *o = x.as_f64().ok_or_else(|| self.bad(key, "numbers"))? as f32;
            }
        }
        Ok(())
    }

    pub fn object(&mut self, key: &'static str) -> Result<Option<Fields>, JsValue> {
        match self.take(key) {
            None => Ok(None),
            Some(Value::Object(m)) => Ok(Some(Fields::new(key, m))),
            Some(_) => Err(self.bad(key, "an object")),
        }
    }

    pub fn finish(self) -> Result<(), JsValue> {
        match self.map.keys().next() {
            None => Ok(()),
            Some(k) => Err(crate::err(format!("{}: unknown field {k}", self.what))),
        }
    }
}

pub fn transform(f: &mut Fields, key: &str) -> Result<Transform, JsValue> {
    let mut t = Transform::IDENTITY;
    if let Some(v) = f.take(key) {
        let Value::Object(m) = v else {
            return Err(crate::err(format!("{key}: expected {{ position, rotation }}")));
        };
        let mut g = Fields::new("transform", m);
        g.floats("position", &mut t.position)?;
        g.floats("rotation", &mut t.rotation)?;
        g.finish()?;
    }
    Ok(t)
}

pub fn capacity(f: &mut Fields, c: &mut WorldCapacity) -> Result<(), JsValue> {
    f.u32("maxBodies", &mut c.max_bodies)?;
    f.u32("maxColliders", &mut c.max_colliders)?;
    f.u32("maxContacts", &mut c.max_contacts)?;
    f.u32("maxJoints", &mut c.max_joints)?;
    f.u32("maxCandidates", &mut c.max_candidates)?;
    f.u32("maxColors", &mut c.max_colors)?;
    f.u32("maxCommands", &mut c.max_commands)?;
    f.u32("maxHullVertices", &mut c.max_hull_vertices)
}

pub fn world_def(v: JsValue) -> Result<WorldDef, JsValue> {
    let mut f = Fields::of("WorldDef", v)?;
    let mut d = WorldDef::default();
    f.floats("gravity", &mut d.gravity)?;
    f.bool("enableSleep", &mut d.enable_sleep)?;
    f.u64("userData", &mut d.user_data)?;
    if let Some(mut c) = f.object("capacity")? {
        capacity(&mut c, &mut d.capacity)?;
        c.finish()?;
    }
    if let Some(s) = f.string("contacts")? {
        d.contacts = match s.as_str() {
            "persistent" => ContactStorage::Persistent,
            "canonical" => ContactStorage::Canonical,
            "compact" => ContactStorage::Compact,
            _ => return Err(crate::err(format!("WorldDef.contacts: unknown {s}"))),
        };
    }
    f.u32("readbackFrames", &mut d.readback_frames)?;
    f.f32("hitEventThreshold", &mut d.hit_event_threshold)?;
    if let Some(s) = f.string("backpressure")? {
        d.backpressure = match s.as_str() {
            "block" => Backpressure::Block,
            "drop" => Backpressure::Drop,
            _ => return Err(crate::err(format!("WorldDef.backpressure: unknown {s}"))),
        };
    }
    f.finish()?;
    Ok(d)
}

pub fn body_def(v: JsValue) -> Result<BodyDef, JsValue> {
    let mut f = Fields::of("BodyDef", v)?;
    let mut d = BodyDef::default();
    if let Some(s) = f.string("type")? {
        d.body_type = match s.as_str() {
            "static" => BodyType::Static,
            "kinematic" => BodyType::Kinematic,
            "dynamic" => BodyType::Dynamic,
            _ => return Err(crate::err(format!("BodyDef.type: unknown {s}"))),
        };
    }
    f.floats("position", &mut d.position)?;
    f.floats("rotation", &mut d.rotation)?;
    f.floats("linearVelocity", &mut d.linear_velocity)?;
    f.floats("angularVelocity", &mut d.angular_velocity)?;
    f.f32("linearDamping", &mut d.linear_damping)?;
    f.f32("angularDamping", &mut d.angular_damping)?;
    f.f32("gravityScale", &mut d.gravity_scale)?;
    f.bool("isBullet", &mut d.is_bullet)?;
    f.bool("isAwake", &mut d.is_awake)?;
    d.name = f.string("name")?;
    f.u64("userData", &mut d.user_data)?;
    f.finish()?;
    Ok(d)
}

pub fn material(f: &mut Fields, m: &mut SurfaceMaterial) -> Result<(), JsValue> {
    f.f32("friction", &mut m.friction)?;
    f.f32("restitution", &mut m.restitution)?;
    f.u64("userMaterialId", &mut m.user_material_id)?;
    f.u32("customColor", &mut m.custom_color)?;
    f.f32("contactOffset", &mut m.contact_offset)
}

pub fn shape_def(v: JsValue) -> Result<ShapeDef, JsValue> {
    let mut f = Fields::of("ShapeDef", v)?;
    let mut d = ShapeDef::default();
    if let Some(mut m) = f.object("material")? {
        material(&mut m, &mut d.material)?;
        m.finish()?;
    }
    f.f32("density", &mut d.density)?;
    if let Some(mut g) = f.object("filter")? {
        let (mut categories, mut mask, mut group) = (
            u32::from(d.filter.categories),
            u32::from(d.filter.mask),
            d.filter.group as f32,
        );
        g.u32("categories", &mut categories)?;
        g.u32("mask", &mut mask)?;
        g.f32("group", &mut group)?;
        g.finish()?;
        d.filter = Filter { categories: categories as u16, mask: mask as u16, group: group as i16 };
    }
    f.bool("isSensor", &mut d.is_sensor)?;
    f.bool("enableSensorEvents", &mut d.enable_sensor_events)?;
    f.bool("enableContactEvents", &mut d.enable_contact_events)?;
    f.bool("enableHitEvents", &mut d.enable_hit_events)?;
    f.bool("updateBodyMass", &mut d.update_body_mass)?;
    d.name = f.string("name")?;
    f.u64("userData", &mut d.user_data)?;
    f.finish()?;
    Ok(d)
}

fn points(f: &mut Fields, key: &str) -> Result<Vec<Vec3>, JsValue> {
    let v = f.take(key).ok_or_else(|| crate::err(format!("hull.{key}: missing")))?;
    let flat: Vec<f32> = serde_json::from_value(v).map_err(crate::err)?;
    if flat.len() % 3 != 0 {
        return Err(crate::err(format!("hull.{key}: not xyz triples")));
    }
    Ok(flat.chunks(3).map(|p| Vec3::new(p[0], p[1], p[2])).collect())
}

/// A tagged geometry: `{ sphere: {...} }`, `{ capsule }`, `{ box }`, `{ hull: {
/// points } }` (cooked by `ConvexHull::from_points`) or `{ plane }`.
pub fn geometry(v: JsValue) -> Result<Geometry, JsValue> {
    let map = object(v)?;
    let mut it = map.into_iter();
    let (Some((tag, Value::Object(body))), None) = (it.next(), it.next()) else {
        return Err(crate::err("Geometry: expected one tagged object, e.g. { box: { half } }"));
    };
    let mut f = Fields::new("Geometry", body);
    let g = match tag.as_str() {
        "sphere" => {
            let (mut center, mut radius) = ([0.0; 3], 0.5);
            f.floats("center", &mut center)?;
            f.f32("radius", &mut radius)?;
            Geometry::Sphere { center, radius }
        }
        "capsule" => {
            let (mut center, mut radius, mut half_len) = ([0.0; 3], 0.5, 0.5);
            f.floats("center", &mut center)?;
            f.f32("radius", &mut radius)?;
            f.f32("halfLen", &mut half_len)?;
            Geometry::Capsule { center, radius, half_len }
        }
        "box" => {
            let mut half = [0.5; 3];
            f.floats("half", &mut half)?;
            Geometry::Box { half, transform: transform(&mut f, "transform")? }
        }
        "hull" => {
            let pts = points(&mut f, "points")?;
            let hull = ConvexHull::from_points(&pts).map_err(|e| crate::err(format!("hull: {e:?}")))?;
            Geometry::Hull { hull, transform: transform(&mut f, "transform")? }
        }
        "plane" => {
            let (mut normal, mut offset) = ([0.0, 1.0, 0.0], 0.0);
            f.floats("normal", &mut normal)?;
            f.f32("offset", &mut offset)?;
            Geometry::Plane { normal, offset }
        }
        _ => return Err(crate::err(format!("Geometry: unknown kind {tag}"))),
    };
    f.finish()?;
    Ok(g)
}
