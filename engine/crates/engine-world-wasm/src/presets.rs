//! The demo scenes placed in `rubble::play` (motion, hills, soft bodies): each makes its world on its
//! own device (`create*Scene`) and places its bodies (`place*Scene`), returning what the
//! page draws and drives.

use crate::vehicle::{from_js, js};
use crate::{err, res, WebWorld};
use rubble::play::{hills, motion, softbody};
use rubble::*;
use wasm_bindgen::prelude::*;

async fn world_for(def: WorldDef) -> Result<WebWorld, JsValue> {
    let (info, device, queue) = crate::web_device().await?;
    let world = res(World::new(
        GpuContext::Shared {
            info,
            device,
            queue,
        },
        &def,
    ))?;
    crate::creation_check(&world).await?;
    Ok(WebWorld { world })
}

#[wasm_bindgen]
impl WebWorld {
    /// The world a motion scene needs (`rubble::play::motion::world_def`): `def` is
    /// `MotionSceneDef` (camelCase: `scene`, `lookAhead`).
    #[wasm_bindgen(js_name = createMotionScene)]
    pub async fn create_motion_scene(def: JsValue) -> Result<WebWorld, JsValue> {
        let def: motion::MotionSceneDef = from_js("MotionSceneDef", def)?;
        world_for(res(motion::world_def(&def))?).await
    }

    /// Place a motion scene in this (empty) world (`rubble::play::motion::place`).
    /// Returns `Placed` (camelCase): the bodies to reset, the characters, the gallery's
    /// lanes and shower, the playground's player, NPCs, platform, lift and rockets, the
    /// bullet walls and targets.
    #[wasm_bindgen(js_name = placeMotionScene)]
    pub fn place_motion_scene(&mut self, def: JsValue) -> Result<JsValue, JsValue> {
        let def: motion::MotionSceneDef = from_js("MotionSceneDef", def)?;
        let placed = res(motion::place(&mut self.world, &def))?;
        Ok(js(serde_json::to_value(placed).map_err(err)?))
    }

    /// The world the terrain demo's hills need (`rubble::play::hills::world_def`): `def`
    /// is `{ n }`.
    #[wasm_bindgen(js_name = createHillsScene)]
    pub async fn create_hills_scene(def: JsValue) -> Result<WebWorld, JsValue> {
        let def: hills::HillsDef = from_js("HillsDef", def)?;
        world_for(res(hills::world_def(&def))?).await
    }

    /// Place the hills in this (empty) world (`rubble::play::hills::place`). Returns the
    /// heightfield shape and grid, and each body's handle, slot and starting pose.
    #[wasm_bindgen(js_name = placeHillsScene)]
    pub fn place_hills_scene(&mut self, def: JsValue) -> Result<JsValue, JsValue> {
        let def: hills::HillsDef = from_js("HillsDef", def)?;
        let placed = res(hills::place(&mut self.world, &def))?;
        Ok(js(serde_json::to_value(placed).map_err(err)?))
    }

    /// The world a soft-body scene needs (`rubble::play::softbody::world_def`): `def` is
    /// `SoftSceneDef` (camelCase: scene, car, import, explicit, strength, deform, hz,
    /// iters, threshold, sweeps, rigid, params, runup, cars, target).
    #[wasm_bindgen(js_name = createSoftScene)]
    pub async fn create_soft_scene(def: JsValue) -> Result<WebWorld, JsValue> {
        let def: softbody::SoftSceneDef = from_js("SoftSceneDef", def)?;
        world_for(res(softbody::world_def(&def))?).await
    }

    /// Place a soft-body scene in this (empty) world (`rubble::play::softbody::place`);
    /// on the explicit solver `def.explicitBoxes` are its sim's static shapes. Returns
    /// `Placed` (camelCase): the soft bodies, camera subjects, the ram, the crusher, the
    /// cannonballs and the car's desc when the page needs it.
    #[wasm_bindgen(js_name = placeSoftScene)]
    pub fn place_soft_scene(&mut self, def: JsValue) -> Result<JsValue, JsValue> {
        let def: softbody::SoftSceneDef = from_js("SoftSceneDef", def)?;
        let placed = res(softbody::place(&mut self.world, &def))?;
        Ok(js(serde_json::to_value(placed).map_err(err)?))
    }
}
