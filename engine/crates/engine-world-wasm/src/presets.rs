//! The demo scenes placed in Rust (`rubble::play::SceneDef`: rigid stacks, vehicles, soft
//! bodies, motion, terrain): a scene from a page's query or a preset's name, its world on
//! its own device (`createScene`, with the page's settings over the scene's own world
//! definition), and its placement (`placeScene`).

use crate::vehicle::{from_js, js};
use crate::{err, handle, res, WebWorld};
use rubble::play::{api, Placed, SceneDef};
use rubble::*;
use wasm_bindgen::prelude::*;

// The data types every binding shares, generated from the Rust types
// (`rubble::play::api::typescript`), in this module's own declarations.
#[wasm_bindgen(typescript_custom_section)]
const PLAY_TYPES: &str = include_str!("../../engine-webgpu/types/rubble-play.d.ts");

/// The scene a family's page builds from its URL query (`location.search`): a `SceneDef`
/// (`{ family, ... }`, camelCase) the page may add to (an import's source, the explicit
/// sim's shapes) before `createScene` (`play::api::scene_from_query`).
#[wasm_bindgen(js_name = sceneFromQuery, unchecked_return_type = "SceneDef")]
pub fn scene_from_query(family: &str, query: &str) -> Result<JsValue, JsValue> {
    to_js(&res(api::scene_from_query(family, query))?)
}

/// Every preset (`play::api::preset_list`).
#[wasm_bindgen(js_name = presetList, unchecked_return_type = "PresetInfo[]")]
pub fn preset_list() -> Result<JsValue, JsValue> {
    to_js(&api::preset_list())
}

/// Preset `name`'s scene (`play::api::preset_scene`), as `sceneFromQuery` gives it.
#[wasm_bindgen(js_name = presetScene, unchecked_return_type = "SceneDef")]
pub fn preset_scene(name: &str) -> Result<JsValue, JsValue> {
    to_js(&res(api::preset_scene(name))?)
}

fn to_js(v: &impl serde::Serialize) -> Result<JsValue, JsValue> {
    Ok(js(serde_json::to_value(v).map_err(err)?))
}

#[wasm_bindgen]
impl WebWorld {
    /// The world scene `def` needs (`SceneDef::world_def`) on its own device, with
    /// `settings` (`WorldDef` fields, as `create` takes them: features, contacts, solver,
    /// gpuTiming, killBelow, ...) over it; `undefined`: none.
    #[wasm_bindgen(js_name = createScene)]
    pub async fn create_scene(
        #[wasm_bindgen(unchecked_param_type = "SceneDef")] def: JsValue,
        settings: JsValue,
    ) -> Result<WebWorld, JsValue> {
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
        Ok(WebWorld::of(world))
    }

    /// Place scene `def` in this (empty) world (`SceneDef::place`). Returns its family's
    /// placement (camelCase); a rigid scene's as typed arrays: `bodies` (handles),
    /// `gpuToScene`, `initial` (each scene body's start, xyz).
    #[wasm_bindgen(js_name = placeScene)]
    pub fn place_scene(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "SceneDef")] def: JsValue,
    ) -> Result<JsValue, JsValue> {
        let scene: SceneDef = from_js("SceneDef", def)?;
        let (placed, director) = res(api::place(&scene, &mut self.world))?;
        self.director = Some(director);
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

    /// The explicit sim's static shapes (`ExplicitSim::layout().boxes` rows) as world bodies
    /// for rendering, as `placeScene` places them for an explicit scene
    /// (`play::explicit_boxes::place`): the crash lab's scene placed again with other values.
    /// Returns their handles (`destroyBody` removes them).
    #[wasm_bindgen(js_name = placeExplicitBoxes)]
    pub fn place_explicit_boxes(&mut self, rows: JsValue) -> Result<Vec<f64>, JsValue> {
        let rows: Vec<Vec<f64>> = serde_wasm_bindgen::from_value(rows).map_err(|e| crate::err(e.to_string()))?;
        let ids = res(rubble::play::explicit_boxes::place(&mut self.world, &rows))?;
        Ok(ids.iter().map(|b| handle(b.to_bits())).collect())
    }

    /// Do a player action on the placed scene before the next step (`play::Action`, e.g.
    /// `{action: 'fire', aim: {eye, dir}}`, `{action: 'reset'}`).
    pub fn act(
        &mut self,
        #[wasm_bindgen(unchecked_param_type = "Action")] action: JsValue,
    ) -> Result<(), JsValue> {
        let action: rubble::play::Action = from_js("Action", action)?;
        let d = self
            .director
            .as_mut()
            .ok_or_else(|| err("act: no scene placed (placeScene first)"))?;
        res(d.act(&mut self.world, &action))
    }

    /// The placed scene's frame scripts before the frame's steps (the crusher's winch, the
    /// traffic drivers): `dt` the frame's time (s), `userCar` the soft body the player is
    /// steering now (−1: none; its driver leaves it alone).
    #[wasm_bindgen(js_name = beforeFrame)]
    pub fn before_frame(&mut self, dt: f64, user_car: i32) -> Result<(), JsValue> {
        let Some(d) = &mut self.director else {
            return Ok(());
        };
        res(d.before_frame(&mut self.world, dt, u32::try_from(user_car).ok()))
    }

    /// After the frame's steps: start the reads the scripts use next frame (never waits).
    #[wasm_bindgen(js_name = afterFrame)]
    pub fn after_frame(&mut self) -> Result<(), JsValue> {
        let Some(d) = &mut self.director else {
            return Ok(());
        };
        res(d.after_frame(&self.world))
    }

    /// Frame-locked runs: start every read the next `beforeFrame` needs; then wait while
    /// `scriptPending()`.
    #[wasm_bindgen(js_name = prepareFrame)]
    pub fn prepare_frame(&mut self) -> Result<(), JsValue> {
        let Some(d) = &mut self.director else {
            return Ok(());
        };
        res(d.prepare(&self.world))
    }

    /// Whether a script read is still in flight.
    #[wasm_bindgen(js_name = scriptPending)]
    pub fn script_pending(&self) -> bool {
        self.director.as_ref().is_some_and(|d| d.pending(&self.world))
    }

    /// The soft body the player drives (the ram aims at it).
    #[wasm_bindgen(js_name = setPlayer)]
    pub fn set_player(&mut self, soft: u32) {
        if let Some(d) = &mut self.director {
            d.set_player(soft);
        }
    }

    /// The placed scene's script state (`play::Director::state`).
    #[wasm_bindgen(js_name = scriptState)]
    pub fn script_state(&self) -> JsValue {
        js(self.director.as_ref().map(|d| d.state()).unwrap_or_default())
    }

    /// What the placed scene's director did since the last call (`[{step, what}]`, oldest
    /// first): the script's own volleys as well as the player's actions.
    #[wasm_bindgen(js_name = takeEvents, unchecked_return_type = "Event[]")]
    pub fn take_events(&mut self) -> Result<JsValue, JsValue> {
        let ev = self.director.as_mut().map(|d| d.take_events()).unwrap_or_default();
        Ok(js(serde_json::to_value(ev).map_err(err)?))
    }

    /// Whether the placed scene's script fires on its own (checks fire at steps of their own).
    #[wasm_bindgen(js_name = setAutofire)]
    pub fn set_autofire(&mut self, on: bool) {
        if let Some(d) = &mut self.director {
            d.set_autofire(on);
        }
    }
}
