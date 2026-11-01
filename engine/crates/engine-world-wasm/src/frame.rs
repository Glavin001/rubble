//! A polled frame for JavaScript: bulk data as flat typed arrays, the rest as
//! plain objects.

use rubble::*;
use wasm_bindgen::prelude::*;

/// Floats per body move in [`WebFrame::moves`].
pub const MOVE_STRIDE: usize = 16;
const MOVE_ASLEEP: u32 = 1;
const MOVE_WOKE: u32 = 2;
const MOVE_FIRST: u32 = 4;

/// Every ready frame folded into one (`World::poll_coalesced`).
#[wasm_bindgen]
pub struct WebFrame {
    frame: CoalescedFrame,
}

impl WebFrame {
    pub(crate) fn new(_world: &World, frame: CoalescedFrame) -> Self {
        Self { frame }
    }

}

#[wasm_bindgen]
impl WebFrame {
    /// Whether any step's results arrived.
    #[wasm_bindgen(getter)]
    pub fn empty(&self) -> bool {
        self.frame.steps.is_none()
    }

    /// The last step folded in (-1: none).
    #[wasm_bindgen(getter, js_name = lastStep)]
    pub fn last_step(&self) -> f64 {
        self.frame.steps.map_or(-1.0, |(_, b)| b as f64)
    }

    /// Steps folded in.
    #[wasm_bindgen(getter, js_name = stepCount)]
    pub fn step_count(&self) -> u32 {
        self.frame.steps.map_or(0, |(a, b)| (b - a + 1) as u32)
    }

    /// Per moved body, [`MOVE_STRIDE`] floats: slot, flags (1 fell asleep, 2
    /// woke, 4 first report), position xyz, rotation xyzw, previous position
    /// xyz, previous rotation xyzw (the current pose on a first report).
    pub fn moves(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.frame.moves.len() * MOVE_STRIDE);
        for m in &self.frame.moves {
            let prev = m.previous.unwrap_or(m.transform);
            let flags = u32::from(m.fell_asleep) * MOVE_ASLEEP
                | u32::from(m.woke) * MOVE_WOKE
                | u32::from(m.previous.is_none()) * MOVE_FIRST;
            out.push(m.body.index() as f32);
            out.push(flags as f32);
            out.extend_from_slice(&m.transform.position);
            out.extend_from_slice(&m.transform.rotation);
            out.extend_from_slice(&prev.position);
            out.extend_from_slice(&prev.rotation);
        }
        out
    }

    /// The last step's counters (`StepCounters`) as an object, or `null`.
    pub fn counters(&self) -> JsValue {
        let Some(c) = self.frame.counters else {
            return JsValue::NULL;
        };
        let o = js_sys::Object::new();
        let set = |k: &str, v: f64| {
            let _ = js_sys::Reflect::set(&o, &k.into(), &v.into());
        };
        set("awake", f64::from(c.awake_count));
        set("contacts", f64::from(c.contact_count));
        set("colors", f64::from(c.color_count));
        set("bodies", f64::from(c.body_count));
        set("candidates", f64::from(c.candidate_count));
        set("manifolds", f64::from(c.manifold_count));
        set("overflow", f64::from(c.overflow_flags));
        o.into()
    }
}

fn js(v: serde_json::Value) -> JsValue {
    use serde::Serialize;
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible()).unwrap_or(JsValue::NULL)
}

fn id(bits: u64) -> f64 {
    bits as f64
}

fn ground(g: GroundState) -> &'static str {
    match g {
        GroundState::OnGround => "ground",
        GroundState::OnSteepGround => "steep",
        GroundState::InAir => "air",
    }
}

#[wasm_bindgen]
impl WebFrame {
    /// Every character as the last step left it: `{ id, position, velocity,
    /// ground ('ground' | 'steep' | 'air'), groundNormal, groundBody (handle or
    /// null), groundVelocity, push, userData }`.
    pub fn characters(&self) -> JsValue {
        js(self
            .frame
            .characters
            .iter()
            .map(|c| {
                serde_json::json!({
                    "id": id(c.character.to_bits()), "position": c.position, "velocity": c.velocity,
                    "ground": ground(c.ground), "groundNormal": c.ground_normal,
                    "groundBody": (!c.ground_body.is_null()).then(|| id(c.ground_body.to_bits())),
                    "groundVelocity": c.ground_velocity, "push": c.push, "userData": c.user_data,
                })
            })
            .collect())
    }

    /// `{ id, type ('landed' | 'supportLost' | 'crushed'), groundBody, userData }`.
    #[wasm_bindgen(js_name = characterEvents)]
    pub fn character_events(&self) -> JsValue {
        js(self
            .frame
            .character_events
            .iter()
            .map(|e| {
                let kind = match e.kind {
                    CharacterEventKind::Landed => "landed",
                    CharacterEventKind::SupportLost => "supportLost",
                    CharacterEventKind::Crushed => "crushed",
                };
                serde_json::json!({
                    "id": id(e.character.to_bits()), "type": kind,
                    "groundBody": (!e.ground_body.is_null()).then(|| id(e.ground_body.to_bits())),
                    "userData": e.user_data,
                })
            })
            .collect())
    }

    /// Every projectile in flight, and each one that stopped: `{ id, position,
    /// velocity, alive, energy, hits, userData }`.
    pub fn projectiles(&self) -> JsValue {
        js(self
            .frame
            .projectiles
            .iter()
            .map(|p| {
                serde_json::json!({
                    "id": id(p.projectile.to_bits()), "position": p.position, "velocity": p.velocity,
                    "alive": p.alive, "energy": p.energy, "hits": p.hits, "userData": p.user_data,
                })
            })
            .collect())
    }

    /// `{ id, shape, body, type ('passed' | 'stopped'), thickness, userData }`.
    #[wasm_bindgen(js_name = projectileEvents)]
    pub fn projectile_events(&self) -> JsValue {
        js(self
            .frame
            .projectile_events
            .iter()
            .map(|e| {
                let (kind, thickness) = match e.kind {
                    ProjectileEventKind::Passed { thickness } => ("passed", thickness),
                    ProjectileEventKind::Stopped => ("stopped", 0.0),
                };
                serde_json::json!({
                    "id": id(e.projectile.to_bits()), "shape": id(e.shape.to_bits()),
                    "body": id(e.body.to_bits()), "type": kind, "thickness": thickness,
                    "userData": e.user_data,
                })
            })
            .collect())
    }
}
