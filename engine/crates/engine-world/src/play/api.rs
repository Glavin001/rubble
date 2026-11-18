//! The scene API every host binds (engine/docs/unified-bindings.md): one Rust function per
//! operation, called by the wasm module (`WebWorld`), the C ABI (`rbWorld_*`) and native
//! hosts ([`super::Playground`]) alike. A binding converts its arguments and errors and does
//! nothing else.
//!
//! - [`place`]: a scene placed in a world, with its [`Director`].
//! - [`PresetInfo`]: the registry as data (the shell's menu, `rbPresetList`).
//! - [`HOST_API`]: every scene operation and its name in each binding; each binding's
//!   tests check it has them all.
//! - [`typescript`]: the TypeScript declarations of the data that crosses a binding
//!   (`engine-webgpu/types/rubble-play.d.ts`, checked in; C reads the same JSON).

use super::{Director, Placed, SceneDef};
use crate::World;
use serde::Serialize;

/// Place `scene` in `world` (empty, made from [`SceneDef::world_def`]) and make its
/// director. Hosts run [`Director::before_step`] before every step after this.
pub fn place(scene: &SceneDef, world: &mut World) -> anyhow::Result<(Placed, Director)> {
    let placed = scene.place(world)?;
    let director = Director::new(scene, &placed, world)?;
    Ok((placed, director))
}

/// A preset as data: what a host lists and opens.
#[derive(Clone, Debug, Serialize, PartialEq, ts_rs::TS)]
#[serde(rename_all = "camelCase")]
pub struct PresetInfo {
    /// Unique, URL-safe (`?preset=<name>`).
    pub name: &'static str,
    pub title: &'static str,
    /// [`super::Family::name`]: the `family` tag of its [`SceneDef`].
    pub family: &'static str,
    /// The menu section it is listed under.
    pub hub: &'static str,
    /// Its own page (relative to `engine-webgpu/`) and that page's query.
    pub page: &'static str,
    pub query: &'static str,
    /// Its scene is placed through [`SceneDef`] (any host can run it).
    pub placed: bool,
}

/// Every preset ([`super::presets`]) as [`PresetInfo`].
pub fn preset_list() -> Vec<PresetInfo> {
    super::presets()
        .iter()
        .map(|p| PresetInfo {
            name: p.name,
            title: p.title,
            family: p.family.name(),
            hub: p.family.hub(),
            page: p.page,
            query: p.query,
            placed: super::scene::placed_here(p.family),
        })
        .collect()
}

/// Preset `name`'s scene ([`super::Preset::scene_def`]).
pub fn preset_scene(name: &str) -> anyhow::Result<SceneDef> {
    let p = super::preset(name).ok_or_else(|| anyhow::anyhow!("unknown preset {name}"))?;
    p.scene_def()
        .ok_or_else(|| anyhow::anyhow!("preset {name} is not placed through SceneDef"))?
}

/// The scene a family's page builds from its URL query ([`SceneDef::from_query`]), for a
/// family placed here.
pub fn scene_from_query(family: &str, query: &str) -> anyhow::Result<SceneDef> {
    let f = super::Family::from_name(family)
        .filter(|f| super::scene::placed_here(*f))
        .ok_or_else(|| anyhow::anyhow!("family {family}: not placed through SceneDef"))?;
    SceneDef::from_query(f, query)
}

/// One scene operation and its name in each binding.
#[derive(Clone, Copy, Debug)]
pub struct HostOp {
    /// What it does (the Rust function it calls).
    pub rust: &'static str,
    /// `WebWorld`'s method (or the module's function), as JavaScript sees it.
    pub wasm: &'static str,
    /// The C function.
    pub c: &'static str,
}

const fn op(rust: &'static str, wasm: &'static str, c: &'static str) -> HostOp {
    HostOp { rust, wasm, c }
}

/// Every scene operation a host has. Adding one here fails each binding's coverage test
/// until it has it.
pub const HOST_API: &[HostOp] = &[
    op("play::api::preset_list", "presetList", "rbPresetList"),
    op("play::api::preset_scene", "presetScene", "rbPresetScene"),
    op(
        "play::api::scene_from_query",
        "sceneFromQuery",
        "rbSceneFromQuery",
    ),
    op(
        "SceneDef::world_def + World::new",
        "createScene",
        "rbCreateSceneWorld",
    ),
    op("play::api::place", "placeScene", "rbWorld_PlaceScene"),
    op("Director::act", "act", "rbWorld_Act"),
    op(
        "Director::before_frame",
        "beforeFrame",
        "rbWorld_BeforeFrame",
    ),
    op("Director::after_frame", "afterFrame", "rbWorld_AfterFrame"),
    op("Director::prepare", "prepareFrame", "rbWorld_PrepareFrame"),
    op(
        "Director::pending",
        "scriptPending",
        "rbWorld_ScriptPending",
    ),
    op("Director::set_player", "setPlayer", "rbWorld_SetPlayer"),
    op(
        "Director::set_autofire",
        "setAutofire",
        "rbWorld_SetAutofire",
    ),
    op("Director::state", "scriptState", "rbWorld_GetScriptState"),
    op("Director::take_events", "takeEvents", "rbWorld_TakeEvents"),
    op(
        "World::render_version",
        "renderVersion",
        "rbWorld_GetRenderVersion",
    ),
    op(
        "World::render_manifest",
        "renderManifest",
        "rbWorld_GetRenderManifest",
    ),
];

/// The TypeScript declarations of every type that crosses a binding as data, in one file.
/// Numbers are JavaScript numbers (the bindings send JSON, so `u64` arrives as a number).
pub fn typescript() -> String {
    use ts_rs::{Config, TS};
    let cfg = Config::new().with_large_int("number");
    let mut out = String::from(
        "// Generated from engine-world's Rust types (play::api::typescript). Do not edit:\n\
         // `cargo test -p engine-world --test api_types` regenerates it with BLESS=1.\n\
         // The web gets these as JS objects; C gets the same JSON (rubble.h, rbWorld_*).\n\n\
         export type JsonValue = number | string | boolean | Array<JsonValue> | { [key in string]?: JsonValue } | null;\n",
    );
    macro_rules! decl {
        ($($t:ty),* $(,)?) => {$(
            out.push('\n');
            out.push_str("export ");
            out.push_str(&<$t as TS>::decl(&cfg));
            out.push('\n');
        )*};
    }
    use super::*;
    decl!(
        SceneDef,
        RigidSceneDef,
        vehicles::VehicleSceneDef,
        vehicles::StreetLayout,
        vehicles::LayoutBlock,
        vehicles::LayoutParked,
        vehicles::LayoutMeteor,
        vehicles::RangeLayout,
        vehicles::RangePad,
        softbody::SoftSceneDef,
        softbody::ImportSource,
        motion::MotionSceneDef,
        hills::HillsDef,
        city_scene::CitySceneDef,
        impact::ImpactSceneDef,
        showcase::ShowcaseSceneDef,
        stage::RangeSceneDef,
        Aim,
        Action,
        Event,
        PlayStats,
        PresetInfo,
        crate::RenderManifest,
        crate::RenderGroup,
        crate::RenderPlane,
    );
    out
}
