//! The large-maps city (`city.html`) on [`World`]: the same builder the native benchmark
//! runs (`rubble::city`, engine-world `tests/city_scale.rs`), so the browser builds the
//! same city with the same parameters (house count, layout, house, piece pool, sleep).
//!
//! Placement never blocks: [`WebWorld::city_place`] places as many houses as asked and
//! returns, so the page places a few per frame (within a time budget) and shows progress.
//! Placement's uploads go to the browser's queue, which stages them itself (no wait on
//! the GPU, `World::create_destructible` on wasm32).

use crate::defs::Fields;
use crate::{err, handle, res, WebFrame, WebWorld};
use rubble::city::*;
use rubble::*;
use wasm_bindgen::prelude::*;

/// Fields of a page's `WorldDef` the city takes; the rest are the city's own.
const PAGE_FIELDS: [&str; 5] = [
    "gpuTiming",
    "killBelow",
    "killBounds",
    "readbackFrames",
    "vehicles",
];

/// A city's definition and house, and once begun in a world, its houses.
#[wasm_bindgen]
pub struct WebCity {
    def: CityDef,
    house: DestructibleAssetDesc,
    city: Option<City>,
}

fn city_def(opts: JsValue) -> Result<CityDef, JsValue> {
    let mut d = CityDef::default();
    if opts.is_undefined() || opts.is_null() {
        return Ok(d);
    }
    let mut f = Fields::of("CityDef", opts)?;
    let mut houses = d.houses as u32;
    f.u32("houses", &mut houses)?;
    d.houses = houses as usize;
    f.f32("map", &mut d.map)?;
    f.bool("sleep", &mut d.sleep)?;
    if let Some(v) = f.take("pool") {
        d.pool = match v {
            serde_json::Value::Null => None,
            v => Some(
                v.as_u64()
                    .ok_or_else(|| err("CityDef.pool: expected a slot count or null"))?
                    as u32,
            ),
        };
    }
    f.finish()?;
    if d.houses == 0 {
        return Err(err("CityDef.houses: at least one house"));
    }
    Ok(d)
}

/// The city's `WorldDef` with the page's timing, readback, kill bounds and vehicles.
fn world_def(city: &WebCity, page: JsValue) -> Result<WorldDef, JsValue> {
    let mut def = res(city.def.world_def(&city.house))?;
    if page.is_undefined() || page.is_null() {
        return Ok(def);
    }
    let o = crate::defs::object(page.clone())?;
    if let Some(k) = o.keys().find(|k| !PAGE_FIELDS.contains(&k.as_str())) {
        return Err(err(format!(
            "createCity: WorldDef.{k} is the city's own (the page may set {})",
            PAGE_FIELDS.join(", ")
        )));
    }
    let p = crate::defs::world_def(page)?;
    def.gpu_timing = p.gpu_timing;
    def.kill_bounds = p.kill_bounds;
    def.readback_frames = p.readback_frames;
    def.vehicles = p.vehicles;
    Ok(def)
}

fn js<T: serde::Serialize>(v: &T) -> Result<JsValue, JsValue> {
    serde::Serialize::serialize(v, &serde_wasm_bindgen::Serializer::json_compatible()).map_err(err)
}

#[wasm_bindgen]
impl WebCity {
    /// `opts`: `{ houses, map, sleep, pool }` (`rubble::city::CityDef`; missing fields
    /// keep its defaults: 16 houses, the 2.2 km map, never asleep, no pool).
    #[wasm_bindgen(constructor)]
    pub fn new(opts: JsValue) -> Result<WebCity, JsValue> {
        let def = city_def(opts)?;
        let house = res(city_house())?;
        Ok(WebCity {
            def,
            house,
            city: None,
        })
    }

    pub fn houses(&self) -> u32 {
        self.def.houses as u32
    }

    /// Houses placed so far.
    pub fn placed(&self) -> u32 {
        self.city.as_ref().map_or(0, |c| c.houses.len() as u32)
    }

    /// Chunks and bonds of one house.
    #[wasm_bindgen(js_name = houseCounts)]
    pub fn house_counts(&self) -> Vec<u32> {
        vec![
            self.house.chunks.len() as u32,
            self.house.bonds.len() as u32,
        ]
    }

    /// The world's capacity for this city (`WorldCapacity`, camelCase).
    pub fn capacity(&self) -> Result<JsValue, JsValue> {
        let c = res(self.def.capacity(&self.house))?;
        js(&serde_json::json!({
            "maxBodies": c.max_bodies, "maxColliders": c.max_colliders,
            "maxContacts": c.max_contacts, "maxCandidates": c.max_candidates,
            "maxJoints": c.max_joints, "maxHullVertices": c.max_hull_vertices,
        }))
    }

    /// `{ houses, map, sleep, pool, pitch, ground, street: [x, z], lots: [x, z, ...] }`
    /// (`ground` once begun).
    pub fn layout(&self) -> Result<JsValue, JsValue> {
        let (lots, pitch) = self.def.lots();
        let street = self.city.as_ref().map(|c| c.street());
        js(&serde_json::json!({
            "houses": self.def.houses, "map": self.def.map, "sleep": self.def.sleep,
            "pool": self.def.pool, "pitch": pitch,
            "ground": self.city.as_ref().map(|c| c.ground),
            "street": street.map(|(x, z)| [x, z]),
            "lots": lots.iter().flatten().copied().collect::<Vec<f32>>(),
        }))
    }

    /// The lots a scenario's shots strike, `[x, z, ...]`.
    pub fn targets(&self, scenario: &str) -> Result<Vec<f32>, JsValue> {
        let s = res(CityScenario::from_name(scenario))?;
        let c = self.city.as_ref().ok_or_else(|| err("cityBegin first"))?;
        Ok(c.targets(s).into_iter().flatten().collect())
    }
}

#[wasm_bindgen]
impl WebWorld {
    /// The city's world on a new device: its own `WorldDef` (`CityDef::world_def`),
    /// with the page's `{ gpuTiming, killBelow, killBounds, readbackFrames, vehicles }`.
    /// Fails with the device's limits when the city does not fit them.
    #[wasm_bindgen(js_name = createCity)]
    pub async fn create_city(city: &WebCity, page: JsValue) -> Result<WebWorld, JsValue> {
        let def = world_def(city, page)?;
        let (info, device, queue) = crate::web_device().await?;
        let world = res(World::new(
            GpuContext::Shared {
                info,
                device,
                queue,
            },
            &def,
        ))?;
        Ok(WebWorld { world })
    }

    /// Replace this world with a fresh one for `city` on the same device (a page
    /// rebuilding its city keeps its renderer's device and this object). The old world
    /// is dropped first: two map-sized worlds do not fit the wasm heap at once.
    #[wasm_bindgen(js_name = cityRebuild)]
    pub fn city_rebuild(&mut self, city: &mut WebCity, page: JsValue) -> Result<(), JsValue> {
        let def = world_def(city, page)?;
        // A minimal world in its place, so the old one (and its buffers) goes now.
        let empty = WorldDef {
            capacity: WorldCapacity::small(),
            ..WorldDef::default()
        };
        self.world = res(World::new(self.world.gpu_context(), &empty))?;
        self.world = res(World::new(self.world.gpu_context(), &def))?;
        city.city = None;
        Ok(())
    }

    /// Start `city` in this world: its ground and house asset, no houses yet.
    #[wasm_bindgen(js_name = cityBegin)]
    pub fn city_begin(&mut self, city: &mut WebCity) -> Result<(), JsValue> {
        if city.city.is_some() {
            return Err(err(
                "cityBegin: the city is already begun (make a new WebCity)",
            ));
        }
        city.city = Some(res(City::begin(
            &mut self.world,
            &city.house,
            city.def.clone(),
        ))?);
        Ok(())
    }

    /// Place up to `n` more houses; returns the houses still to place. Never waits.
    #[wasm_bindgen(js_name = cityPlace)]
    pub fn city_place(&mut self, city: &mut WebCity, n: u32) -> Result<u32, JsValue> {
        let c = city.city.as_mut().ok_or_else(|| err("cityBegin first"))?;
        Ok(res(c.place(&mut self.world, n as usize))? as u32)
    }

    /// Start a scenario (`still`, `drive`, `spawn`, `demolish`, `war`) as the native
    /// benchmark does: the handles of the bodies it adds.
    #[wasm_bindgen(js_name = cityStart)]
    pub fn city_start(&mut self, city: &WebCity, scenario: &str) -> Result<Vec<f64>, JsValue> {
        let s = res(CityScenario::from_name(scenario))?;
        let c = city.city.as_ref().ok_or_else(|| err("cityBegin first"))?;
        if c.remaining() > 0 {
            return Err(err(format!(
                "cityStart: {} houses still to place",
                c.remaining()
            )));
        }
        let added = res(c.start(&mut self.world, s))?;
        Ok(added.into_iter().map(|b| handle(b.to_bits())).collect())
    }

    /// Houses' bonds intact, summed (as of the last polled frame).
    #[wasm_bindgen(js_name = cityBondsIntact)]
    pub fn city_bonds_intact(&self, city: &WebCity) -> Result<u32, JsValue> {
        let c = city.city.as_ref().ok_or_else(|| err("cityBegin first"))?;
        let mut n = 0;
        for &d in &c.houses {
            n += res(self.world.destructible_stats(d))?.bonds_intact;
        }
        Ok(n)
    }

    /// The device's buffer limits: `{ maxBufferSize, maxStorageBufferBindingSize }`.
    #[wasm_bindgen(js_name = bufferLimits)]
    pub fn buffer_limits(&self) -> Result<JsValue, JsValue> {
        let l = self.world.device().limits();
        js(&serde_json::json!({
            "maxBufferSize": l.max_buffer_size,
            "maxStorageBufferBindingSize": l.max_storage_buffer_binding_size,
        }))
    }
}

#[wasm_bindgen]
impl WebFrame {
    /// Bonds broken in the folded steps (threshold destructibles' `BondsBroken`).
    #[wasm_bindgen(js_name = bondsBroken)]
    pub fn bonds_broken(&self) -> u32 {
        bonds_broken(self.frame().destruction_events.iter().map(|(_, e)| e))
    }
}
