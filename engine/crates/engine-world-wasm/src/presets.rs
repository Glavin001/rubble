//! The demo scenes placed in Rust (`rubble::play::SceneDef`: rigid stacks, vehicles, soft
//! bodies, motion, terrain): a scene from a page's query or a preset's name, its world on
//! its own device (`createScene`, with the page's settings over the scene's own world
//! definition), and its placement (`placeScene`).

use crate::vehicle::{from_js, js};
use crate::{err, handle, res, WebWorld};
use rubble::play::{Family, Placed, SceneDef};
use rubble::*;
use wasm_bindgen::prelude::*;

fn family(name: &str) -> Result<Family, JsValue> {
    Ok(match name {
        "rigid" => Family::Rigid,
        "vehicle" => Family::Vehicle,
        "soft" => Family::Soft,
        "motion" => Family::Motion,
        "terrain" => Family::Terrain,
        _ => return Err(err(format!("family {name}: not placed through SceneDef"))),
    })
}

/// The scene a family's page builds from its URL query (`location.search`): a `SceneDef`
/// (`{ family, ... }`, camelCase) the page may add to (an import's source, the explicit
/// sim's shapes) before `createScene`.
#[wasm_bindgen(js_name = sceneFromQuery)]
pub fn scene_from_query(family_name: &str, query: &str) -> Result<JsValue, JsValue> {
    let def = res(SceneDef::from_query(family(family_name)?, query))?;
    Ok(js(serde_json::to_value(def).map_err(err)?))
}

/// Preset `name`'s scene (`Preset::scene_def`), as `sceneFromQuery` gives it.
#[wasm_bindgen(js_name = presetScene)]
pub fn preset_scene(name: &str) -> Result<JsValue, JsValue> {
    let p = rubble::play::preset(name).ok_or_else(|| err(format!("unknown preset {name}")))?;
    let def = p
        .scene_def()
        .ok_or_else(|| err(format!("preset {name} is not placed through SceneDef")))?;
    Ok(js(serde_json::to_value(res(def)?).map_err(err)?))
}

#[wasm_bindgen]
impl WebWorld {
    /// The world scene `def` needs (`SceneDef::world_def`) on its own device, with
    /// `settings` (`WorldDef` fields, as `create` takes them: features, contacts, solver,
    /// gpuTiming, killBelow, ...) over it; `undefined`: none.
    #[wasm_bindgen(js_name = createScene)]
    pub async fn create_scene(def: JsValue, settings: JsValue) -> Result<WebWorld, JsValue> {
        let scene: SceneDef = from_js("SceneDef", def)?;
        let mut wd = res(scene.world_def())?;
        if !settings.is_undefined() && !settings.is_null() {
            wd = crate::defs::world_def_over(settings, wd)?;
        }
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

    /// Place scene `def` in this (empty) world (`SceneDef::place`). Returns its family's
    /// placement (camelCase); a rigid scene's as typed arrays: `bodies` (handles),
    /// `gpuToScene`, `initial` (each scene body's start, xyz).
    #[wasm_bindgen(js_name = placeScene)]
    pub fn place_scene(&mut self, def: JsValue) -> Result<JsValue, JsValue> {
        let scene: SceneDef = from_js("SceneDef", def)?;
        let placed = res(scene.place(&mut self.world))?;
        if let Placed::Rigid(p) = &placed {
            let o = js_sys::Object::new();
            let bodies: Vec<f64> = p.bodies.iter().map(|b| handle(b.to_bits())).collect();
            let initial: Vec<f32> = p.initial.iter().flatten().copied().collect();
            let set = |k: &str, v: &JsValue| {
                let _ = js_sys::Reflect::set(&o, &k.into(), v);
            };
            set(
                "bodies",
                &js_sys::Float64Array::from(bodies.as_slice()).into(),
            );
            set(
                "gpuToScene",
                &js_sys::Uint32Array::from(p.gpu_to_scene.as_slice()).into(),
            );
            set(
                "initial",
                &js_sys::Float32Array::from(initial.as_slice()).into(),
            );
            return Ok(o.into());
        }
        Ok(js(res(placed.to_json())?))
    }
}
